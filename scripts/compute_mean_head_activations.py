"""
Precompute per-(layer, head) mean activations over a Pile-10k sample.

Output: activations/<ModelShortName>_pile-10k/mean.ckpt — a tensor of shape
[n_layers, n_heads, head_dim], the per-head o_proj input averaged over all
(doc, position) pairs in the sample. This is what `vocablist_ablation.py`
loads via `utils.get_mean_head_values(...)` to use as the mean-ablation
baseline.

Usage:
    cd dual-route-induction/scripts
    python compute_mean_head_activations.py --model Qwen/Qwen3-8B --n 1024
"""
from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from nnsight import LanguageModel
from tqdm import tqdm

from seed_utils import set_random_seed
from trace_utils import get_o_proj_input_tensor


def main(args):
    set_random_seed(args.seed)

    model = LanguageModel(args.model, device_map="auto", dispatch=True)
    model_short = args.model.split("/")[-1]

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = getattr(
        model.config, "head_dim", model.config.hidden_size // n_heads
    )

    pile = load_dataset("NeelNanda/pile-10k")["train"]
    pile = pile.shuffle(seed=args.seed).select(range(args.n))

    # Running sum + count to compute the mean without holding all activations.
    running_sum = torch.zeros(n_layers, n_heads, head_dim, dtype=torch.float64)
    running_count = 0

    def tok(text):
        ids = model.tokenizer(text, truncation=True, max_length=args.max_len)[
            "input_ids"
        ]
        return ids

    for example in tqdm(pile, desc="docs"):
        input_ids = tok(example["text"])
        if len(input_ids) < 2:
            continue
        with torch.no_grad():
            with model.trace([input_ids]):
                # Save o_proj inputs for every layer in one trace.
                saves = []
                for layer in range(n_layers):
                    saves.append(get_o_proj_input_tensor(model, layer).save())

        # saves[layer] shape: [bsz=1, seq_len, n_heads * head_dim]
        for layer, saved in enumerate(saves):
            t = saved.detach().to(torch.float64).cpu()  # [1, seq_len, n_heads*head_dim]
            seq_len = t.shape[1]
            t = t.view(1, seq_len, n_heads, head_dim)
            running_sum[layer] += t.sum(dim=(0, 1))
        running_count += seq_len

    if running_count == 0:
        raise RuntimeError("No tokens accumulated; sample was empty?")

    mean = (running_sum / running_count).to(torch.float32)

    out_dir = f"../activations/{model_short}_pile-10k"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/mean.ckpt"
    torch.save(mean, out_path)
    print(
        f"saved {out_path} shape={tuple(mean.shape)} "
        f"({running_count:,} tokens over {args.n} docs)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=1024,
                        help="number of Pile docs to average over")
    parser.add_argument("--max-len", type=int, default=512,
                        help="max tokens per doc (HF truncation cap)")
    parser.add_argument("--seed", type=int, default=8)
    main(parser.parse_args())
