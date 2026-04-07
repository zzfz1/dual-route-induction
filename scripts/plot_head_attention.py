"""
Plot attention matching scores for top-k token and concept copier heads.
Reproduces the dual-bar chart from the paper: Next-tok (NTM) vs Last-tok (LTM)
for the top-k heads ranked by causal copying score.

Usage:
    uv run python dual-route-induction/scripts/plot_head_attention.py
    uv run python dual-route-induction/scripts/plot_head_attention.py --model Llama-3.1-70B --topk 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache"


def load_head_rankings(model_name: str, score_type: str, topk: int) -> list[tuple[int, int]]:
    path = CACHE_ROOT / "head_orderings" / model_name / f"{score_type}_copying.json"
    with path.open() as f:
        rankings = json.load(f)
    return [(r[0], r[1]) for r in rankings[:topk]]


def load_attention_scores(model_name: str, n: int = 2048, seqlen: int = 30, random_tok: bool = False) -> dict[tuple[int,int], dict]:
    suffix = "_randomtokents" if random_tok else ""
    path = CACHE_ROOT / "attention_scores" / model_name / f"n{n}_seqlen{seqlen}{suffix}.json"
    with path.open() as f:
        raw = json.load(f)

    next_tok = {(e["layer"], e["head_idx"]): e["score"] for e in raw["next_tok_attn"]}
    end_tok  = {(e["layer"], e["head_idx"]): e["score"] for e in raw["end_tok_attn"]}
    return next_tok, end_tok


def plot_heads(ax, heads, next_tok, end_tok, title, annotation_head=None, annotation_text=None):
    x = np.arange(len(heads))
    width = 0.35

    next_vals = [next_tok.get(h, 0.0) for h in heads]
    end_vals  = [end_tok.get(h,  0.0) for h in heads]

    bars_next = ax.bar(x - width/2, next_vals, width, color="#5B7FC3", alpha=0.9, label="Next-tok")
    bars_end  = ax.bar(x + width/2, end_vals,  width, color="#C4504A", alpha=0.9, label="Last-tok")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}.{h}" for l, h in heads], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Attention Head", fontsize=10)
    ax.set_ylabel("Attention Matching Score", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(max(next_vals), max(end_vals)) * 1.15 + 0.01)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optional annotation arrow
    if annotation_head is not None and annotation_head in heads:
        idx = heads.index(annotation_head)
        max_end = max(end_vals)
        target_val = end_tok.get(annotation_head, 0.0)
        ax.annotate(
            annotation_text or "max. last-token\nmatching score",
            xy=(idx + width/2, target_val),
            xytext=(idx + width/2 + 1.5, target_val + 0.08),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
            fontsize=8,
        )


def main(args):
    model_name = args.model.split("/")[-1]
    token_heads   = load_head_rankings(model_name, "token",   args.topk)
    concept_heads = load_head_rankings(model_name, "concept", args.topk)

    next_tok, end_tok = load_attention_scores(model_name, n=args.n, seqlen=args.seqlen)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.subplots_adjust(wspace=0.35)

    plot_heads(ax1, token_heads,   next_tok, end_tok, f"{model_name}: Token Copier Heads")
    plot_heads(ax2, concept_heads, next_tok, end_tok, f"{model_name}: Concept Copier Heads",
               annotation_head=concept_heads[1] if len(concept_heads) > 1 else None,
               annotation_text="max. last-token\nmatching score")

    out_path = Path(args.out) if args.out else (
        CACHE_ROOT / "attention_scores" / model_name /
        f"head_attention_plot_top{args.topk}.pdf"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--topk", default=15, type=int, help="Number of top heads to show per type")
    parser.add_argument("--n", default=2048, type=int)
    parser.add_argument("--seqlen", default=30, type=int)
    parser.add_argument("--out", default=None, help="Output path (default: cache/attention_scores/<model>/)")
    main(parser.parse_args())
