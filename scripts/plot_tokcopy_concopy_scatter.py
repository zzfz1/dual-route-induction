"""
Scatter plot: TokCopy vs ConCopy score per head, colored by layer.
Reveals which heads specialize in one route vs. both.

Usage:
    uv run python dual-route-induction/scripts/plot_tokcopy_concopy_scatter.py
    uv run python dual-route-induction/scripts/plot_tokcopy_concopy_scatter.py --model Llama-3.1-8B --topk 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache"


def load_scores(model_name: str, score_type: str) -> dict[tuple[int, int], float]:
    path = CACHE_ROOT / "causal_scores" / model_name / f"{score_type}_len30_n1024.json"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    with path.open() as f:
        return {(r["layer"], r["head_idx"]): r["score"] for r in json.load(f)}


def load_head_rankings(model_name: str, score_type: str, topk: int) -> list[tuple[int, int]]:
    path = CACHE_ROOT / "head_orderings" / model_name / f"{score_type}_copying.json"
    if not path.exists():
        return []
    with path.open() as f:
        return [(r[0], r[1]) for r in json.load(f)[:topk]]


def main(args):
    model_name = args.model.split("/")[-1]

    tok_scores = load_scores(model_name, "token_copying")
    con_scores = load_scores(model_name, "concept_copying")

    heads = sorted(tok_scores.keys())
    layers  = np.array([h[0] for h in heads])
    tok_arr = np.array([tok_scores[h] for h in heads])
    con_arr = np.array([con_scores.get(h, 0.0) for h in heads])

    top_tok = set(load_head_rankings(model_name, "token",   args.topk))
    top_con = set(load_head_rankings(model_name, "concept", args.topk))

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")

    n_layers = layers.max() + 1
    scatter = ax.scatter(tok_arr, con_arr, c=layers, cmap="viridis",
                         s=18, alpha=0.6, linewidths=0, zorder=2)
    cbar = fig.colorbar(scatter, ax=ax, label="Layer")
    cbar.set_ticks(range(0, n_layers, max(1, n_layers // 8)))

    # Label top token heads (blue outline)
    for (layer, head) in top_tok:
        x, y = tok_scores[(layer, head)], con_scores.get((layer, head), 0.0)
        ax.plot(x, y, "o", markersize=9, markerfacecolor="none",
                markeredgecolor="#5B8DB8", markeredgewidth=1.5, zorder=4)
        ax.annotate(f"{layer}.{head}", (x, y), fontsize=7, color="#5B8DB8",
                    xytext=(4, 2), textcoords="offset points")

    # Label top concept heads (red outline)
    for (layer, head) in top_con - top_tok:
        x, y = tok_scores[(layer, head)], con_scores.get((layer, head), 0.0)
        ax.plot(x, y, "s", markersize=9, markerfacecolor="none",
                markeredgecolor="#C4504A", markeredgewidth=1.5, zorder=4)
        ax.annotate(f"{layer}.{head}", (x, y), fontsize=7, color="#C4504A",
                    xytext=(4, 2), textcoords="offset points")

    # Diagonal reference line
    lim = max(tok_arr.max(), con_arr.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=0.8, alpha=0.5, zorder=1)

    ax.set_xlabel("TokCopy score", fontsize=11)
    ax.set_ylabel("ConCopy score", fontsize=11)
    ax.set_title(f"{model_name}: TokCopy vs ConCopy per head\n"
                 f"(○ = top-{args.topk} token heads, □ = top-{args.topk} concept heads)",
                 fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_dir = CACHE_ROOT / "figures" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"plot_tokcopy_concopy_scatter_top{args.topk}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--topk", default=15, type=int)
    main(parser.parse_args())
