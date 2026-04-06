from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from improbable_bigram_data import DEFAULT_TRACE_ROOT, load_trace_index
from seed_utils import set_random_seed


def load_entries(trace_dir: Path, subset: str):
    entries = load_trace_index(trace_dir)
    if not entries:
        for example_dir in sorted(path for path in trace_dir.iterdir() if path.is_dir()):
            meta_path = example_dir / "meta.json"
            if not meta_path.exists():
                continue
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            entries.append(
                {
                    "task_idx": meta["task_idx"],
                    "bigram": meta["bigram"],
                    "relative_dir": example_dir.name,
                    "copy_success": meta["flags"]["copy_success"],
                    "second_token_hallucination": meta["flags"]["second_token_hallucination"],
                }
            )
    if subset == "all":
        return entries
    if subset == "copied":
        return [entry for entry in entries if entry["copy_success"]]
    if subset == "hallucinated_second_token":
        return [entry for entry in entries if entry["second_token_hallucination"]]
    raise ValueError(f"Unsupported subset: {subset}")


def load_dla_from_state(state: dict, meta: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Read pre-computed DLA scores embedded in the trace state.

    The trace script computes these server-side via the block-diagonal o_proj trick:
      - dla_correct: [n_layers, n_heads] — per-head logit contribution to the target token
      - dla_wrong:   [n_layers, n_heads] — per-head logit contribution to the predicted token

    Raises ValueError if the state was produced by an older trace run that did not
    embed DLA scores. Re-run improbable_bigram_trace.py --overwrite to regenerate.
    """
    if "dla_correct" not in state or "dla_wrong" not in state:
        raise ValueError(
            f"State file for task {meta['task_idx']} ({meta['bigram']!r}) does not contain "
            "pre-computed DLA scores. Re-run improbable_bigram_trace.py --overwrite to "
            "regenerate traces with DLA computation enabled."
        )
    return state["dla_correct"], state["dla_wrong"]


def flatten_rows(score_tensor: torch.Tensor, metric_name: str):
    n_layers, n_heads = score_tensor.shape
    rows = []
    for layer in range(n_layers):
        for head_idx in range(n_heads):
            rows.append(
                {
                    "layer": layer,
                    "head_idx": head_idx,
                    metric_name: float(score_tensor[layer, head_idx].item()),
                }
            )
    return rows


def merge_metric_rows(*metric_groups):
    merged = {}
    for rows in metric_groups:
        for row in rows:
            key = (row["layer"], row["head_idx"])
            if key not in merged:
                merged[key] = {"layer": row["layer"], "head_idx": row["head_idx"]}
            merged[key].update({k: v for k, v in row.items() if k not in ("layer", "head_idx")})
    return [merged[key] for key in sorted(merged.keys())]


def main(args):
    set_random_seed(args.seed)
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir) if args.out_dir else trace_dir / "dla"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_entries(trace_dir, args.subset)
    if not entries:
        raise ValueError(f"No traced examples matched subset={args.subset}.")

    correct_scores = []
    wrong_scores = []
    per_example_meta = []

    state_name = f"{args.pass_name}_state.pt"
    for entry in entries:
        example_dir = trace_dir / entry["relative_dir"]
        with (example_dir / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        state = torch.load(example_dir / state_name, map_location="cpu")

        correct_dla, wrong_dla = load_dla_from_state(state, meta)
        correct_scores.append(correct_dla)
        wrong_scores.append(wrong_dla)
        per_example_meta.append(
            {
                "task_idx": meta["task_idx"],
                "bigram": meta["bigram"],
                "copy_success": meta["flags"]["copy_success"],
                "second_token_hallucination": meta["flags"]["second_token_hallucination"],
                "wrong_token_id": meta["p2"]["predicted_token_id"],
                "correct_token_id": meta["p2"]["target_token_id"],
            }
        )

    correct_scores = torch.stack(correct_scores)
    wrong_scores = torch.stack(wrong_scores)

    payload = {
        "subset": args.subset,
        "pass_name": args.pass_name,
        "examples": per_example_meta,
        "correct_token_dla": correct_scores,
        "predicted_token_dla": wrong_scores,
    }
    torch.save(payload, out_dir / f"per_example_{args.subset}_{args.pass_name}.pt")

    correct_mean = correct_scores.mean(dim=0)
    hallucinated_mask = torch.tensor(
        [meta["second_token_hallucination"] for meta in per_example_meta],
        dtype=torch.bool,
    )
    if hallucinated_mask.any():
        wrong_mean = wrong_scores[hallucinated_mask].mean(dim=0)
    else:
        wrong_mean = torch.full_like(correct_mean, float("nan"))

    rows = merge_metric_rows(
        flatten_rows(correct_mean, "correct_token_dla"),
        flatten_rows(wrong_mean, "predicted_wrong_token_dla"),
    )
    with (out_dir / f"per_head_{args.subset}_{args.pass_name}.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    summary = {
        "subset": args.subset,
        "pass_name": args.pass_name,
        "n_examples": len(entries),
        "n_hallucinated_examples": int(hallucinated_mask.sum().item()),
        "per_example_path": str(
            (out_dir / f"per_example_{args.subset}_{args.pass_name}.pt").resolve()
        ),
        "per_head_path": str(
            (out_dir / f"per_head_{args.subset}_{args.pass_name}.json").resolve()
        ),
    }
    with (out_dir / f"summary_{args.subset}_{args.pass_name}.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_ROOT))
    parser.add_argument(
        "--subset",
        default="all",
        choices=["all", "copied", "hallucinated_second_token"],
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--pass-name", default="p1", choices=["xn", "p1"])
    parser.add_argument("--seed", default=8, type=int)
    main(parser.parse_args())
