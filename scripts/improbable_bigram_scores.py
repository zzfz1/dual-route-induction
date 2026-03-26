from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from improbable_bigram_data import DEFAULT_TRACE_ROOT, load_trace_index
from seed_utils import set_random_seed


def load_entries(trace_dir: Path):
    index_entries = load_trace_index(trace_dir)
    if index_entries:
        return index_entries

    entries = []
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
                "second_token_hallucination": meta["flags"][
                    "second_token_hallucination"
                ],
            }
        )
    return entries


def keep_entry(entry, subset: str):
    if subset == "all":
        return True
    if subset == "copied":
        return bool(entry["copy_success"])
    if subset == "hallucinated_second_token":
        return bool(entry["second_token_hallucination"])
    raise ValueError(f"Unsupported subset: {subset}")


def value_weight_row(attn_row: torch.Tensor, value_norms: torch.Tensor):
    weighted = attn_row * value_norms
    denom = weighted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weighted / denom


def flatten_head_scores(score_tensor: torch.Tensor, metric_name: str):
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
            merged[key].update(
                {k: v for k, v in row.items() if k not in ("layer", "head_idx")}
            )
    return [merged[key] for key in sorted(merged.keys())]


def main(args):
    set_random_seed(args.seed)
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir) if args.out_dir else trace_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        entry for entry in load_entries(trace_dir) if keep_entry(entry, args.subset)
    ]
    if not entries:
        raise ValueError(f"No traced examples matched subset={args.subset}.")

    ltm_raw = []
    ntm_raw = []
    ltm_weighted = []
    ntm_weighted = []
    per_example_meta = []

    for entry in entries:
        example_dir = trace_dir / entry["relative_dir"]
        with (example_dir / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        xn_state = torch.load(example_dir / "xn_state.pt", map_location="cpu")
        p1_state = torch.load(example_dir / "p1_state.pt", map_location="cpu")
        p2_prev_idx = int(meta["positions"]["p2_prev"])

        xn_row_raw = xn_state["attn_row_raw"]
        p1_row_raw = p1_state["attn_row_raw"]
        xn_row_weighted = value_weight_row(xn_row_raw, xn_state["value_norms"])
        p1_row_weighted = value_weight_row(p1_row_raw, p1_state["value_norms"])

        ltm_raw.append(xn_row_raw[:, :, p2_prev_idx])
        ntm_raw.append(p1_row_raw[:, :, p2_prev_idx])
        ltm_weighted.append(xn_row_weighted[:, :, p2_prev_idx])
        ntm_weighted.append(p1_row_weighted[:, :, p2_prev_idx])
        per_example_meta.append(
            {
                "task_idx": meta["task_idx"],
                "bigram": meta["bigram"],
                "copy_success": meta["flags"]["copy_success"],
                "second_token_hallucination": meta["flags"][
                    "second_token_hallucination"
                ],
            }
        )

    ltm_raw = torch.stack(ltm_raw)
    ntm_raw = torch.stack(ntm_raw)
    ltm_weighted = torch.stack(ltm_weighted)
    ntm_weighted = torch.stack(ntm_weighted)

    payload = {
        "subset": args.subset,
        "examples": per_example_meta,
        "ltm_raw": ltm_raw,
        "ntm_raw": ntm_raw,
        "ltm_value_weighted": ltm_weighted,
        "ntm_value_weighted": ntm_weighted,
    }
    torch.save(payload, out_dir / f"per_example_{args.subset}.pt")

    ltm_raw_mean = ltm_raw.mean(dim=0)
    ntm_raw_mean = ntm_raw.mean(dim=0)
    ltm_weighted_mean = ltm_weighted.mean(dim=0)
    ntm_weighted_mean = ntm_weighted.mean(dim=0)

    metric_rows = merge_metric_rows(
        flatten_head_scores(ltm_raw_mean, "ltm_raw"),
        flatten_head_scores(ntm_raw_mean, "ntm_raw"),
        flatten_head_scores(ltm_weighted_mean, "ltm_value_weighted"),
        flatten_head_scores(ntm_weighted_mean, "ntm_value_weighted"),
    )
    with (out_dir / f"per_head_{args.subset}.json").open("w", encoding="utf-8") as f:
        json.dump(metric_rows, f, ensure_ascii=False, indent=2)

    summary = {
        "subset": args.subset,
        "n_examples": len(entries),
        "per_example_path": str((out_dir / f"per_example_{args.subset}.pt").resolve()),
        "per_head_path": str((out_dir / f"per_head_{args.subset}.json").resolve()),
    }
    with (out_dir / f"summary_{args.subset}.json").open("w", encoding="utf-8") as f:
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
    parser.add_argument("--seed", default=8, type=int)
    main(parser.parse_args())
