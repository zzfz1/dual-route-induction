"""
Plot ablation curve: hallucination rate vs. number of ablated heads,
comparing token heads vs. concept heads.

Usage:
    uv run python dual-route-induction/scripts/plot_ablation_curve.py
    uv run python dual-route-induction/scripts/plot_ablation_curve.py --model Llama-3.1-70B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache"

TOKEN_COLOR   = "#5B8DB8"
CONCEPT_COLOR = "#C4504A"


def load_ablation_summary(model_name: str) -> list[dict]:
    # find the most recent summary file
    ablation_dir = CACHE_ROOT / "ablation" / model_name
    summaries = sorted(ablation_dir.glob("summary_*.json"))
    print(summaries, "SUMMARIES")
    if not summaries:
        raise FileNotFoundError(f"No ablation summary found in {ablation_dir}")
    with summaries[-1].open() as f:
        return json.load(f)


def print_table(rows: list[dict]) -> None:
    header = f"{'Condition':<25} {'Copy%':>7} {'Hall%':>7} {'PrefFail%':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        copy  = r["copy_success_rate"]
        hall  = r["hallucination_rate"]
        pfail = 1.0 - copy - hall
        print(f"{r['label']:<25} {copy:>7.1%} {hall:>7.1%} {pfail:>10.1%}")


def total_failure(r):
    return 1.0 - r["copy_success_rate"]


def second_token_hall(r):
    return r["hallucination_rate"]


def main(args):
    model_name = args.model.split("/")[-1]
    rows = load_ablation_summary(model_name)

    baseline = next(r for r in rows if r["label"] == "baseline")
    baseline_total = total_failure(baseline)
    baseline_hall  = second_token_hall(baseline)

    token_rows   = [r for r in rows if r["label"].startswith("top") and r["label"].endswith("_token")]
    concept_rows = [r for r in rows if r["label"].startswith("top") and r["label"].endswith("_concept")]

    token_rows.sort(key=lambda r: r["n_ablated"])
    concept_rows.sort(key=lambda r: r["n_ablated"])

    print_table([baseline] + token_rows + concept_rows)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    # Baseline (single line — at baseline prefix failures = 0%, so total = 2nd-token hall.)
    ax.axhline(baseline_total, color="gray", linestyle="--", linewidth=1.0,
               label=f"Baseline ({baseline_total:.0%})", zorder=1)

    tk = [0] + [r["n_ablated"] for r in token_rows]
    ck = [0] + [r["n_ablated"] for r in concept_rows]

    # Token heads — total failure (solid) and 2nd-token hallucination (dashed)
    tt = [baseline_total] + [total_failure(r) for r in token_rows]
    th = [baseline_hall]  + [second_token_hall(r) for r in token_rows]
    ax.plot(tk, tt, "o-", color=TOKEN_COLOR, linewidth=2, markersize=7,
            label="Token ablated — total failure", zorder=3)
    ax.plot(tk, th, "o--", color=TOKEN_COLOR, linewidth=1.5, markersize=5, alpha=0.7,
            label="Token ablated — 2nd-token hall.", zorder=3)

    # Concept heads — total failure (solid) and 2nd-token hallucination (dashed)
    ct = [baseline_total] + [total_failure(r) for r in concept_rows]
    ch = [baseline_hall]  + [second_token_hall(r) for r in concept_rows]
    ax.plot(ck, ct, "s-", color=CONCEPT_COLOR, linewidth=2, markersize=7,
            label="Concept ablated — total failure", zorder=3)
    ax.plot(ck, ch, "s--", color=CONCEPT_COLOR, linewidth=1.5, markersize=5, alpha=0.7,
            label="Concept ablated — 2nd-token hall.", zorder=3)

    # Annotate total failure on token curve
    for k, t in zip(tk[1:], tt[1:]):
        ax.annotate(f"{t:.0%}", (k, t), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color=TOKEN_COLOR)

    ax.set_xlabel("Number of heads ablated (top-k by causal score)", fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title(f"{model_name}: Effect of head ablation\n"
                 f"(solid = total failure, dashed = 2nd-token hallucination)", fontsize=11)
    ax.set_xticks(sorted(set(tk + ck)))
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    legend_handles = [
        mlines.Line2D([], [], color="gray",  linestyle="--", linewidth=1.2,
                      label=f"Baseline ({baseline_total:.0%})"),
        mlines.Line2D([], [], color=TOKEN_COLOR,   linestyle="-",  linewidth=2,
                      marker="o", markersize=6, label="Token ablated — total failure"),
        mlines.Line2D([], [], color=TOKEN_COLOR,   linestyle="--", linewidth=1.5,
                      marker="o", markersize=5, alpha=0.7, label="Token ablated — 2nd-token hall."),
        mlines.Line2D([], [], color=CONCEPT_COLOR, linestyle="-",  linewidth=2,
                      marker="s", markersize=6, label="Concept ablated — total failure"),
        mlines.Line2D([], [], color=CONCEPT_COLOR, linestyle="--", linewidth=1.5,
                      marker="s", markersize=5, alpha=0.7, label="Concept ablated — 2nd-token hall."),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, ncol=2,
              handlelength=3, handleheight=1.2, handletextpad=0.6,
              columnspacing=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    out_dir = CACHE_ROOT / "figures" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "plot_ablation_curve.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    main(parser.parse_args())
