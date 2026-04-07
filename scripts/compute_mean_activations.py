"""
Compute per-head mean o_proj input activations over Pile documents.
These are used as ablation values in mean-ablation experiments.

Saves: activations/{model_name}_pile-10k/mean.ckpt
  A tensor of shape [n_layers, n_heads, head_dim] (float32, CPU).

Usage:
    uv run python dual-route-induction/scripts/compute_mean_activations.py --remote
    uv run python dual-route-induction/scripts/compute_mean_activations.py --model meta-llama/Llama-3.1-70B --remote
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from nnsight import LanguageModel

from ndif import load_remote_model
from seed_utils import set_random_seed
from trace_utils import RemoteExecutionContext, is_remote_model

CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache"
ACT_ROOT = Path(__file__).resolve().parents[1] / "activations"


def compute_batch_mean(model, token_ids: list[list[int]]) -> torch.Tensor:
    """
    Run one batch and return per-head mean o_proj inputs.
    Returns tensor [n_layers, n_heads, head_dim] (server-side aggregated).
    """
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    remote = is_remote_model(model)

    with torch.no_grad():
        with model.trace(token_ids, remote=remote):
            layer_means = []
            for layer_idx in range(n_layers):
                o_proj_inp = model.model.layers[layer_idx].self_attn.o_proj.inputs[0][0]
                # [bsz, seq_len, n_heads * head_dim] -> mean over bsz and seq_len
                bsz = o_proj_inp.shape[0]
                seq_len = o_proj_inp.shape[1]
                reshaped = o_proj_inp.view(bsz, seq_len, n_heads, head_dim)
                layer_means.append(reshaped.mean(dim=(0, 1)))  # [n_heads, head_dim]

            # Stack and save server-side — one tensor transfer instead of n_layers
            result = torch.stack(layer_means).save()  # [n_layers, n_heads, head_dim]

    return result.detach().cpu().float()


def main(args):
    set_random_seed(args.seed)

    model_name = args.model.split("/")[-1]
    out_dir = ACT_ROOT / f"{model_name}_pile-10k"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mean.ckpt"

    if out_path.exists() and not args.overwrite:
        print(f"Already exists: {out_path}. Use --overwrite to recompute.")
        return

    if args.remote:
        model = load_remote_model(args.model)
        remote_ctx = RemoteExecutionContext(
            args.model,
            max_retries=args.remote_max_retries,
            backoff_base=args.remote_backoff_base,
            backoff_max=args.remote_backoff_max,
        )
    else:
        model = LanguageModel(args.model, device_map="cuda", cache_dir="/share/u/models")
        remote_ctx = None

    # Tokenize pile documents
    pile = load_dataset("NeelNanda/pile-10k")["train"]
    docs = [pile[i]["text"] for i in range(len(pile))]

    def tok(text):
        ids = model.tokenizer(text)["input_ids"]
        if "llama" in model.config._name_or_path.lower():
            ids = ids[1:]  # strip BOS
        return ids[:args.seq_len]

    print("Tokenizing pile documents...")
    tokenized = [tok(d) for d in docs]
    tokenized = [t for t in tokenized if len(t) >= 16]  # skip very short docs
    tokenized = tokenized[:args.n_docs]
    print(f"Using {len(tokenized)} documents, batch_size={args.bsz}")

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    accum = torch.zeros(n_layers, n_heads, head_dim)
    n_batches = 0

    for batch_start in range(0, len(tokenized), args.bsz):
        batch = tokenized[batch_start: batch_start + args.bsz]
        if not batch:
            break

        label = f"mean_act_batch{batch_start//args.bsz}"
        print(f"  batch {batch_start//args.bsz + 1}/{(len(tokenized) + args.bsz - 1)//args.bsz}", flush=True)

        if remote_ctx is not None:
            batch_mean = remote_ctx.request(
                label,
                lambda m, b=batch: compute_batch_mean(m, b),
            )
        else:
            batch_mean = compute_batch_mean(model, batch)

        accum += batch_mean
        n_batches += 1

        # incremental save
        torch.save((accum / n_batches).clone(), out_path)

    final_mean = accum / n_batches
    torch.save(final_mean, out_path)
    print(f"Saved mean activations → {out_path}  shape={tuple(final_mean.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B",
                        choices=["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B"])
    parser.add_argument("--n-docs", default=1024, type=int, help="Number of pile docs to average over")
    parser.add_argument("--bsz", default=8, type=int)
    parser.add_argument("--seq-len", default=128, type=int, help="Max tokens per document")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--remote-max-retries", default=4, type=int)
    parser.add_argument("--remote-backoff-base", default=2.0, type=float)
    parser.add_argument("--remote-backoff-max", default=30.0, type=float)
    parser.add_argument("--seed", default=8, type=int)
    main(parser.parse_args())
