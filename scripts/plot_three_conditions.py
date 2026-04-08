"""
Plot 3 and 4 from the paper plan:
  Plot 3 — Three-condition LTM/NTM comparison for top-k token and concept heads
  Plot 4 — Three-condition DLA comparison (correct-token and wrong-token) for top-k heads

Conditions:
  (1) Improbable bigrams
  (2) Random two-token phrases
  (3) Semantic concepts (CounterFact)

Usage:
    uv run python dual-route-induction/scripts/plot_three_conditions.py
    uv run python dual-route-induction/scripts/plot_three_conditions.py --model Llama-3.1-70B --topk 10
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

CONDITION_LABELS = ["Improbable\nBigrams", "Random\nTokens", "Concepts"]
CONDITION_COLORS = ["#C4504A", "#5B8DB8", "#5BAD72"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_head_rankings(model_name: str, score_type: str, topk: int) -> list[tuple[int, int]]:
    path = CACHE_ROOT / "head_orderings" / model_name / f"{score_type}_copying.json"
    with path.open() as f:
        return [(r[0], r[1]) for r in json.load(f)[:topk]]


def load_scores(path: Path) -> dict[tuple[int, int], dict]:
    with path.open() as f:
        rows = json.load(f)
    return {(r["layer"], r["head_idx"]): r for r in rows}


def load_dla(path: Path) -> dict[tuple[int, int], dict]:
    with path.open() as f:
        rows = json.load(f)
    return {(r["layer"], r["head_idx"]): r for r in rows}


def get_condition_scores(model_name: str, improbable_dir: str):
    """Return (scores_dict, dla_dict) for each condition."""
    base = CACHE_ROOT / "improbable_bigrams" / model_name

    scores_paths = [
        base / improbable_dir / "scores" / "per_head_all.json",
        base / "random_tokens" / "scores" / "per_head_all.json",
        base / "concepts" / "scores" / "per_head_all.json",
    ]
    dla_paths = [
        base / improbable_dir / "dla" / "per_head_all_p1.json",
        base / "random_tokens" / "dla" / "per_head_all_p1.json",
        base / "concepts" / "dla" / "per_head_all_p1.json",
    ]

    # Check which conditions have both scores and DLA
    available = []
    for i, (sp, dp) in enumerate(zip(scores_paths, dla_paths)):
        if sp.exists() and dp.exists():
            available.append(i)
        else:
            missing = [p for p in [sp, dp] if not p.exists()]
            print(f"  Skipping condition {CONDITION_LABELS[i].replace(chr(10), ' ')}: missing {missing}")

    scores_list = [load_scores(scores_paths[i]) for i in available]
    dla_list    = [load_dla(dla_paths[i]) for i in available]
    labels      = [CONDITION_LABELS[i] for i in available]
    colors      = [CONDITION_COLORS[i] for i in available]

    return scores_list, dla_list, labels, colors


# ---------------------------------------------------------------------------
# Plot 3: LTM / NTM across three conditions
# ---------------------------------------------------------------------------

def plot_ltm_ntm(ax, heads, scores_list, labels, colors, metric, title, ylabel=True):
    """metric: 'ntm_value_weighted' or 'ltm_value_weighted'"""
    n_heads = len(heads)
    n_conds = len(scores_list)
    width = 0.8 / n_conds
    x = np.arange(n_heads)

    for ci, (scores, label, color) in enumerate(zip(scores_list, labels, colors)):
        vals = [scores.get(h, {}).get(metric, 0.0) for h in heads]
        offset = (ci - n_conds / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}.{h}" for l, h in heads], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Attention Head", fontsize=10)
    if ylabel:
        ax.set_ylabel("Attention Score (value-weighted)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_plot3(model_name, token_heads, concept_heads, scores_list, labels, colors, topk, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{model_name}: LTM/NTM Scores across Three Conditions", fontsize=13, y=1.01)
    fig.subplots_adjust(hspace=0.55, wspace=0.3)

    plot_ltm_ntm(axes[0, 0], token_heads,   scores_list, labels, colors,
                 "ntm_value_weighted", f"Token Heads — NTM (Next-Token Matching)")
    plot_ltm_ntm(axes[0, 1], token_heads,   scores_list, labels, colors,
                 "ltm_value_weighted", f"Token Heads — LTM (Last-Token Matching)", ylabel=False)
    plot_ltm_ntm(axes[1, 0], concept_heads, scores_list, labels, colors,
                 "ntm_value_weighted", f"Concept Heads — NTM (Next-Token Matching)")
    plot_ltm_ntm(axes[1, 1], concept_heads, scores_list, labels, colors,
                 "ltm_value_weighted", f"Concept Heads — LTM (Last-Token Matching)", ylabel=False)

    out_path = out_dir / f"plot3_ltm_ntm_three_conditions_top{topk}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved Plot 3 → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: DLA across three conditions
# ---------------------------------------------------------------------------

def plot_dla(ax, heads, dla_list, labels, colors, metric, title, ylabel=True):
    """metric: 'correct_token_dla' or 'predicted_wrong_token_dla'"""
    n_heads = len(heads)
    n_conds = len(dla_list)
    width = 0.8 / n_conds
    x = np.arange(n_heads)

    for ci, (dla, label, color) in enumerate(zip(dla_list, labels, colors)):
        vals = [dla.get(h, {}).get(metric, 0.0) for h in heads]
        offset = (ci - n_conds / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}.{h}" for l, h in heads], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Attention Head", fontsize=10)
    if ylabel:
        ax.set_ylabel("DLA Score", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")


def make_plot4(model_name, token_heads, concept_heads, dla_list, labels, colors, topk, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{model_name}: DLA Scores across Three Conditions", fontsize=13, y=1.01)
    fig.subplots_adjust(hspace=0.55, wspace=0.3)

    plot_dla(axes[0, 0], token_heads,   dla_list, labels, colors,
             "correct_token_dla",        "Token Heads — Correct Token DLA")
    plot_dla(axes[0, 1], token_heads,   dla_list, labels, colors,
             "predicted_wrong_token_dla","Token Heads — Wrong Token DLA", ylabel=False)
    plot_dla(axes[1, 0], concept_heads, dla_list, labels, colors,
             "correct_token_dla",        "Concept Heads — Correct Token DLA")
    plot_dla(axes[1, 1], concept_heads, dla_list, labels, colors,
             "predicted_wrong_token_dla","Concept Heads — Wrong Token DLA", ylabel=False)

    out_path = out_dir / f"plot4_dla_three_conditions_top{topk}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved Plot 4 → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    model_name = args.model.split("/")[-1]
    improbable_dir = args.improbable_dir

    # Check random_tokens dla exists (may not for 70B)
    base = CACHE_ROOT / "improbable_bigrams" / model_name
    if not (base / "random_tokens").exists():
        print(f"Warning: random_tokens traces not found for {model_name}; condition 2 will be skipped.")

    token_heads   = load_head_rankings(model_name, "token",   args.topk)
    concept_heads = load_head_rankings(model_name, "concept", args.topk)

    # Exclude heads already in token_heads to avoid showing same head in both rows
    token_heads_set = set(token_heads)
    concept_heads_unique = [h for h in concept_heads if h not in token_heads_set]
    n_overlap = len(concept_heads) - len(concept_heads_unique)
    if n_overlap:
        print(f"Note: {n_overlap} head(s) in both token and concept top-{args.topk}; excluded from concept panels.")

    print(f"Loading condition data for {model_name}...")
    scores_list, dla_list, labels, colors = get_condition_scores(model_name, improbable_dir)

    out_dir = CACHE_ROOT / "figures" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    make_plot3(model_name, token_heads, concept_heads_unique, scores_list, labels, colors, args.topk, out_dir)
    make_plot4(model_name, token_heads, concept_heads_unique, dla_list,    labels, colors, args.topk, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--topk", default=15, type=int)
    parser.add_argument(
        "--improbable-dir",
        default=None,
        help="Subdir name under cache/improbable_bigrams/<model>/ for condition 1 "
             "(default: updated_table1_literal for 8B, table1_literal for 70B)",
    )
    args = parser.parse_args()

    if args.improbable_dir is None:
        model_name = args.model.split("/")[-1]
        args.improbable_dir = "updated_table1_literal" if "8B" in model_name else "table1_literal"

    main(args)
