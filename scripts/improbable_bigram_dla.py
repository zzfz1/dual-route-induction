from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.utils import logging as hf_logging

from improbable_bigram_data import DEFAULT_TRACE_ROOT, load_trace_index
from seed_utils import set_random_seed


def parse_dtype(name: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


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


def load_model(args):
    hf_token = os.environ.get("HF_TOKEN")
    kwargs = {
        "dtype": parse_dtype(args.dtype),
        "low_cpu_mem_usage": True,
        "token": hf_token,
    }
    if args.device_map != "none":
        kwargs["device_map"] = args.device_map

    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    finally:
        hf_logging.set_verbosity(previous_verbosity)
    if args.device_map == "none":
        model.to(torch.device(args.device))
    model.eval()
    return model


def final_norm_scale(resid_pre: torch.Tensor, rmsnorm) -> torch.Tensor:
    eps = getattr(rmsnorm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(rmsnorm, "eps")
    return torch.rsqrt(resid_pre.pow(2).mean() + eps)


def materialize_module_tensor(module, tensor_name: str, device: torch.device, dtype: torch.dtype):
    tensor = getattr(module, tensor_name)
    if isinstance(tensor, torch.nn.Parameter):
        tensor = tensor.detach()
    if not getattr(tensor, "is_meta", False):
        return tensor.to(device=device, dtype=dtype)

    hook = getattr(module, "_hf_hook", None)
    weights_map = getattr(hook, "weights_map", None)
    if weights_map is None:
        raise RuntimeError(
            f"Tensor {module.__class__.__name__}.{tensor_name} is on the meta device "
            "and no Accelerate weights_map is available to materialize it."
        )

    realized = weights_map[tensor_name]
    if isinstance(realized, torch.nn.Parameter):
        realized = realized.detach()
    if getattr(realized, "is_meta", False):
        raise RuntimeError(
            f"Tensor {module.__class__.__name__}.{tensor_name} is still meta after "
            "looking it up in Accelerate weights_map."
        )
    return realized.to(device=device, dtype=dtype)


def compute_example_dla(model, state, meta, compute_device: torch.device):
    core_model = model.model
    rmsnorm = core_model.norm
    lm_head = model.lm_head

    norm_weight = materialize_module_tensor(
        rmsnorm, "weight", compute_device, torch.float32
    )
    resid_pre = state["resid_pre_final_norm"].to(compute_device, dtype=torch.float32)
    scale = final_norm_scale(resid_pre, rmsnorm)

    correct_token_id = int(meta["p2"]["target_token_id"])
    wrong_token_id = int(meta["p2"]["predicted_token_id"])

    lm_head_weight = materialize_module_tensor(
        lm_head, "weight", compute_device, torch.float32
    )
    correct_unembed = lm_head_weight[correct_token_id]
    wrong_unembed = lm_head_weight[wrong_token_id]

    head_inputs = state["head_o_proj_in"]
    n_layers, n_heads, head_dim = head_inputs.shape
    hidden_size = model.config.hidden_size

    correct_scores = torch.empty((n_layers, n_heads), dtype=torch.float32)
    wrong_scores = torch.empty((n_layers, n_heads), dtype=torch.float32)

    for layer_idx in range(n_layers):
        o_proj = materialize_module_tensor(
            core_model.layers[layer_idx].self_attn.o_proj,
            "weight",
            compute_device,
            torch.float32,
        )
        o_proj_blocks = o_proj.view(hidden_size, n_heads, head_dim).permute(1, 0, 2)
        layer_head_inputs = head_inputs[layer_idx].to(compute_device, dtype=torch.float32)
        head_resid = torch.einsum("hod,hd->ho", o_proj_blocks, layer_head_inputs)
        normalized = head_resid * scale * norm_weight.unsqueeze(0)
        correct_scores[layer_idx] = normalized @ correct_unembed
        wrong_scores[layer_idx] = normalized @ wrong_unembed

    return correct_scores.cpu(), wrong_scores.cpu()


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

    model = load_model(args)
    compute_device = torch.device(args.compute_device)

    correct_scores = []
    wrong_scores = []
    per_example_meta = []

    state_name = f"{args.pass_name}_state.pt"
    for entry in entries:
        example_dir = trace_dir / entry["relative_dir"]
        with (example_dir / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        state = torch.load(example_dir / state_name, map_location="cpu")

        correct_dla, wrong_dla = compute_example_dla(model, state, meta, compute_device)
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
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B",
        choices=["meta-llama/Llama-3.1-8B"],
    )
    parser.add_argument("--pass-name", default="p1", choices=["xn", "p1"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-device", default="cpu")
    parser.add_argument("--seed", default=8, type=int)
    main(parser.parse_args())
