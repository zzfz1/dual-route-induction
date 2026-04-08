"""
Plot causal score heatmaps: layer × head grids for TokCopy and ConCopy.

Usage:
    uv run python dual-route-induction/scripts/plot_causal_heatmap.py
    uv run python dual-route-induction/scripts/plot_causal_heatmap.py --model Llama-3.1-8B --topk 10
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


def load_scores_grid(model_name: str, score_type: str) -> np.ndarray:
    """Load scores into a [n_layers, n_heads] array."""
    path = CACHE_ROOT / "causal_scores" / model_name / f"{score_type}_len30_n1024.json"
    if not path.exists():
        raise FileNotFoundError(f"Causal scores not found: {path}")
    with path.open() as f:
        rows = json.load(f)

    n_layers = max(r["layer"] for r in rows) + 1
    n_heads  = max(r["head_idx"] for r in rows) + 1
    grid = np.zeros((n_layers, n_heads))
    for r in rows:
        grid[r["layer"], r["head_idx"]] = r["score"]
    return grid


def load_head_rankings(model_name: str, score_type: str, topk: int) -> list[tuple[int, int]]:
    path = CACHE_ROOT / "head_orderings" / model_name / f"{score_type}_copying.json"
    if not path.exists():
        return []
    with path.open() as f:
        return [(r[0], r[1]) for r in json.load(f)[:topk]]


def plot_heatmap(ax, grid, title, top_heads, cmap):
    n_layers, n_heads = grid.shape
    im = ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest",
                   origin="upper")

    # Mark top-k heads with a white circle
    for layer, head in top_heads:
        ax.plot(head, layer, "o", color="white", markersize=5,
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    ax.set_xlabel("Head index", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_title(title, fontsize=11)

    # Tick every 8 heads and every 4 layers for readability
    ax.set_xticks(range(0, n_heads, max(1, n_heads // 8)))
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 8)))

    return im


def main(args):
    model_name = args.model.split("/")[-1]

    tok_grid = load_scores_grid(model_name, "token_copying")
    con_grid = load_scores_grid(model_name, "concept_copying")

    top_tok = load_head_rankings(model_name, "token", args.topk)
    top_con = load_head_rankings(model_name, "concept", args.topk)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    fig.suptitle(f"{model_name}: Causal Copying Scores (Layer × Head)", fontsize=12)

    im0 = plot_heatmap(axes[0], tok_grid, f"TokCopy (token_copying)\nTop-{args.topk} marked",
                       top_tok, cmap="Blues")
    im1 = plot_heatmap(axes[1], con_grid, f"ConCopy (concept_copying)\nTop-{args.topk} marked",
                       top_con, cmap="Reds")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Score")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Score")

    out_dir = CACHE_ROOT / "figures" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"plot_causal_heatmap_top{args.topk}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--topk", default=15, type=int,
                        help="Number of top heads to mark with circles")
    main(parser.parse_args())
