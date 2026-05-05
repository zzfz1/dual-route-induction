"""
Compute per-head mean activations of o_proj inputs over Pile documents.

For each layer, captures the o_proj input tensor averaged over all tokens
and all documents, and saves:
    activations/{model_name}_pile-10k/mean.ckpt   shape [n_layers, n_heads, head_dim]

This file is required by improbable_bigram_ablation.py and vocablist_ablation.py.

Usage:
    uv run python dual-route-induction/scripts/compute_mean_activations.py --model meta-llama/Llama-3.1-8B
    uv run python dual-route-induction/scripts/compute_mean_activations.py --model Qwen/Qwen3-8B
    uv run python dual-route-induction/scripts/compute_mean_activations.py --model allenai/Olmo-3-1025-7B
    uv run python dual-route-induction/scripts/compute_mean_activations.py --model meta-llama/Llama-3.1-8B --remote
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from nnsight import LanguageModel, save
from tqdm import tqdm

from ndif import load_remote_model
from seed_utils import set_random_seed
from trace_utils import get_o_proj_inputs, is_remote_model

ACT_ROOT = Path(__file__).resolve().parents[1] / "activations"


def build_tok(model):
    def tok(text):
        name = model.config._name_or_path.lower()
        ids = model.tokenizer(text)["input_ids"]
        if "llama" in name:
            return ids[1:]  # strip auto-prepended BOS
        return ids

    return tok


def compute_means(model, texts: list[str], max_tokens_per_doc: int = 256) -> torch.Tensor:
    """
    Returns [n_layers, n_heads, head_dim] mean of o_proj inputs,
    averaged over all tokens across all documents (token-count weighted).
    """
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // n_heads)
    remote = is_remote_model(model)
    tok = build_tok(model)

    running_sum = torch.zeros(n_layers, n_heads * head_dim)
    total_tokens = 0

    for text in tqdm(texts, desc="Computing mean activations"):
        ids = tok(text)
        if not ids:
            continue
        ids = ids[:max_tokens_per_doc]

        with torch.no_grad():
            with model.trace([ids], remote=remote):
                layer_saves = save([])
                for layer_idx in range(n_layers):
                    o_inp = get_o_proj_inputs(model, layer_idx)[0][0]
                    # [1, seq_len, n_heads * head_dim] → mean over seq_len → [n_heads * head_dim]
                    layer_saves.append(o_inp[0].mean(dim=0))

        seq_len = len(ids)
        # In local execution save([]) resolves to a plain list; in remote mode
        # it is a SaveProxy that needs .value — handle both.
        saved_list = getattr(layer_saves, "value", layer_saves)
        for layer_idx, val in enumerate(saved_list):
            v = getattr(val, "value", val)  # unwrap individual proxies if present
            running_sum[layer_idx] += v.float().cpu() * seq_len
        total_tokens += seq_len

    means = running_sum / total_tokens  # [n_layers, n_heads * head_dim]
    return means.view(n_layers, n_heads, head_dim)


def main(args):
    set_random_seed(args.seed)
    model_name = args.model.split("/")[-1]

    if args.remote:
        model = load_remote_model(args.model)
    else:
        model = LanguageModel(args.model, device_map="auto", dispatch=True)

    pile = load_dataset("NeelNanda/pile-10k")["train"]
    texts = [ex["text"] for ex in pile][: args.n]
    print(f"Using {len(texts)} Pile documents for {model_name}")

    means = compute_means(model, texts, max_tokens_per_doc=args.max_tokens_per_doc)
    print(f"Mean activations shape: {tuple(means.shape)}")

    out_dir = ACT_ROOT / f"{model_name}_pile-10k"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mean.ckpt"
    torch.save(means, out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B",
        choices=[
            "meta-llama/Llama-3.1-8B",
            "Qwen/Qwen3-8B",
            "allenai/Olmo-3-1025-7B",
            "allenai/OLMo-2-1124-7B",
            "meta-llama/Meta-Llama-3-8B",
        ],
    )
    parser.add_argument("--n", default=1000, type=int, help="Number of Pile documents")
    parser.add_argument("--max-tokens-per-doc", default=256, type=int)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--seed", default=8, type=int)
    parser.set_defaults(remote=False)
    args = parser.parse_args()
    main(args)
