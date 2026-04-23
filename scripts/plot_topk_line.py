"""
Three-panel line plot:
  Panel 1 — Concept LTM (raw): mean over top-k concept heads, 4 conditions
  Panel 2 — Concept correct-token DLA: mean over top-k concept heads, 4 conditions
  Panel 3 — Hallucinated wrong-token DLA: token-head ranking vs concept-head ranking

Usage:
    uv run python dual-route-induction/scripts/plot_topk_line.py
    uv run python dual-route-induction/scripts/plot_topk_line.py --model Llama-3.1-8B --topk-values 8 16 32 64 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache"

HALLUCINATED_COLOR = "#C4504A"
COPIED_COLOR       = "#5B8DB8"
CONCEPTS_COLOR     = "#8B5CF6"
RANDOM_COLOR       = "#5BAD72"
TOKEN_WRONG_COLOR  = "#F59E0B"
CONCEPT_WRONG_COLOR = "#14B8A6"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_head_rankings(model_name: str, score_type: str,
                       exclude: set[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    path = CACHE_ROOT / "head_orderings" / model_name / f"{score_type}_copying.json"
    with path.open() as f:
        rankings = [(r[0], r[1]) for r in json.load(f)]
    if exclude:
        rankings = [h for h in rankings if h not in exclude]
    return rankings


def topk_flat_indices(rankings: list[tuple[int, int]], k: int, n_heads: int) -> list[int]:
    return [layer * n_heads + head for layer, head in rankings[:k]]


def mean_over_topk(tensor: torch.Tensor, flat_indices: list[int]) -> torch.Tensor:
    """tensor: [n_examples, n_layers, n_heads] → [n_examples]"""
    n_examples, n_layers, n_heads = tensor.shape
    flat = tensor.view(n_examples, n_layers * n_heads)
    return flat[:, flat_indices].mean(dim=1)


def mean_sem(values: torch.Tensor):
    n = values.shape[0]
    return values.mean().item(), values.std().item() / (n ** 0.5)


def build_curve(tensor: torch.Tensor, rankings: list[tuple[int, int]],
                topk_values: list[int], n_heads: int):
    means, sems = [], []
    for k in topk_values:
        idx = topk_flat_indices(rankings, k, n_heads)
        m, s = mean_sem(mean_over_topk(tensor, idx))
        means.append(m)
        sems.append(s)
    return np.array(means), np.array(sems)


# ---------------------------------------------------------------------------
# Load all condition data
# ---------------------------------------------------------------------------

def load_all_conditions(model_name: str, improbable_dir: str):
    base = CACHE_ROOT / "improbable_bigrams" / model_name

    # --- Improbable bigrams ---
    imp_scores = torch.load(base / improbable_dir / "scores" / "per_example_all.pt",
                            map_location="cpu", weights_only=False)
    imp_dla    = torch.load(base / improbable_dir / "dla"    / "per_example_all_p1.pt",
                            map_location="cpu", weights_only=False)

    hall_mask   = torch.tensor([e["second_token_hallucination"] for e in imp_scores["examples"]])
    copied_mask = torch.tensor([e["copy_success"]               for e in imp_scores["examples"]])

    conditions = {
        "hallucinated": {
            "ltm":   imp_scores["ltm_raw"][hall_mask],
            "cor_dla": imp_dla["correct_token_dla"][hall_mask],
            "wrg_dla": imp_dla["predicted_token_dla"][hall_mask],
            "label": "Hallucinated improbable",
            "color": HALLUCINATED_COLOR,
        },
        "copied": {
            "ltm":   imp_scores["ltm_raw"][copied_mask],
            "cor_dla": imp_dla["correct_token_dla"][copied_mask],
            "wrg_dla": imp_dla["predicted_token_dla"][copied_mask],
            "label": "Copied improbable",
            "color": COPIED_COLOR,
        },
    }

    # --- Concept bigrams ---
    con_scores_path = base / "concepts" / "scores" / "per_example_all.pt"
    con_dla_path    = base / "concepts" / "dla"    / "per_example_all_p1.pt"
    if con_scores_path.exists() and con_dla_path.exists():
        con_scores = torch.load(con_scores_path, map_location="cpu", weights_only=False)
        con_dla    = torch.load(con_dla_path,    map_location="cpu", weights_only=False)
        conditions["concepts"] = {
            "ltm":   con_scores["ltm_raw"],
            "cor_dla": con_dla["correct_token_dla"],
            "wrg_dla": con_dla["predicted_token_dla"],
            "label": "2-token concepts",
            "color": CONCEPTS_COLOR,
        }
    else:
        print("  Warning: concepts traces not found; skipping condition.")

    # --- Random tokens ---
    rnd_scores_path = base / "random_tokens" / "scores" / "per_example_all.pt"
    rnd_dla_path    = base / "random_tokens" / "dla"    / "per_example_all_p1.pt"
    if rnd_scores_path.exists() and rnd_dla_path.exists():
        rnd_scores = torch.load(rnd_scores_path, map_location="cpu", weights_only=False)
        rnd_dla    = torch.load(rnd_dla_path,    map_location="cpu", weights_only=False)
        conditions["random"] = {
            "ltm":   rnd_scores["ltm_raw"],
            "cor_dla": rnd_dla["correct_token_dla"],
            "wrg_dla": rnd_dla["predicted_token_dla"],
            "label": "Random phrases",
            "color": RANDOM_COLOR,
        }
    else:
        print("  Warning: random_tokens not found; skipping condition.")

    return conditions


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_panel(ax, conditions, rankings, topk_values, n_heads, metric_key, title, ylabel,
               zero_line=False):
    x = np.array(topk_values)
    for cond in conditions.values():
        means, sems = build_curve(cond[metric_key], rankings, topk_values, n_heads)
        ax.plot(x, means, "o-", color=cond["color"], linewidth=2, markersize=6,
                label=cond["label"], zorder=3)
        ax.fill_between(x, means - sems, means + sems,
                        color=cond["color"], alpha=0.15, zorder=2)

    if zero_line:
        ax.axhline(0, color="black", linewidth=0.7, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in topk_values])
    ax.set_xlabel("Top-k concept-copying heads", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, title="Setups", title_fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def plot_wrong_token_panel(ax, hall_cond, token_rankings, concept_rankings,
                           topk_values, n_heads):
    """Panel 3: wrong-token DLA for hallucinated examples, token vs concept head rankings."""
    x = np.array(topk_values)
    wrg = hall_cond["wrg_dla"]

    for rankings, label, color in [
        (token_rankings,   "Token wrong-token DLA",   TOKEN_WRONG_COLOR),
        (concept_rankings, "Concept wrong-token DLA", CONCEPT_WRONG_COLOR),
    ]:
        means, sems = build_curve(wrg, rankings, topk_values, n_heads)
        ax.plot(x, means, "o-", color=color, linewidth=2, markersize=6,
                label=label, zorder=3)
        ax.fill_between(x, means - sems, means + sems,
                        color=color, alpha=0.15, zorder=2)

    ax.axhline(0, color="black", linewidth=0.7, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in topk_values])
    ax.set_xlabel("Top-k heads within each ranking", fontsize=11)
    ax.set_ylabel("Per-example mean", fontsize=11)
    ax.set_title("Hallucinated wrong-token DLA", fontsize=12)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    model_name = args.model.split("/")[-1]
    improbable_dir = args.improbable_dir
    if improbable_dir is None:
        improbable_dir = "updated_table1_literal" if "8B" in model_name else "table1_literal"

    topk_values = args.topk_values

    exclude = set()
    for spec in (args.exclude_heads or []):
        l, h = spec.split(",")
        exclude.add((int(l), int(h)))

    token_rankings   = load_head_rankings(model_name, "token",   exclude)
    concept_rankings = load_head_rankings(model_name, "concept", exclude)

    conditions = load_all_conditions(model_name, improbable_dir)

    n_heads = next(iter(conditions.values()))["ltm"].shape[2]

    print(f"Model: {model_name}")
    for name, cond in conditions.items():
        print(f"  {name}: ltm={cond['ltm'].shape}, cor_dla={cond['cor_dla'].shape}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")

    plot_panel(ax1, conditions, concept_rankings, topk_values, n_heads,
               "ltm", "Concept LTM (raw)", "Per-example mean", zero_line=False)

    plot_panel(ax2, conditions, concept_rankings, topk_values, n_heads,
               "cor_dla", "Concept correct-token DLA", "Per-example mean", zero_line=True)

    plot_wrong_token_panel(ax3, conditions["hallucinated"],
                           token_rankings, concept_rankings, topk_values, n_heads)

    out_dir = CACHE_ROOT / "figures" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_excl" + "_".join(args.exclude_heads) if args.exclude_heads else ""
    out_path = out_dir / f"plot_topk_line{suffix}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--improbable-dir", default=None)
    parser.add_argument("--topk-values", nargs="+", type=int,
                        default=[8, 16, 32, 64, 128])
    parser.add_argument("--exclude-heads", nargs="*", default=None,
                        metavar="L,H", help="Heads to exclude, e.g. 15,16 13,27")
    main(parser.parse_args())
