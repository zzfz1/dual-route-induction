"""
Mean-ablation experiment on improbable bigrams.

For each bigram prompt, replaces the o_proj inputs of specified attention heads
with their mean activations (computed over Pile by compute_mean_activations.py),
then measures whether hallucination is reduced.

Reports per-bigram and aggregate:
  - baseline copy_success / hallucination rate
  - ablated copy_success / hallucination rate for each head set

Usage:
    # Ablate top-k concept heads (default):
    uv run python dual-route-induction/scripts/improbable_bigram_ablation.py --remote

    # Ablate specific heads (e.g. 13,27 and 15,16):
    uv run python dual-route-induction/scripts/improbable_bigram_ablation.py --remote --heads 13,27 15,16

    # Ablate top-k token heads:
    uv run python dual-route-induction/scripts/improbable_bigram_ablation.py --remote --head-type token --topk 10

    # 70B model:
    uv run python dual-route-induction/scripts/improbable_bigram_ablation.py --remote --model meta-llama/Llama-3.1-70B
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from improbable_bigram_data import (
    DEFAULT_TASKS_PATH,
    DEFAULT_TRACE_ROOT,
    load_bigram_tasks,
    validate_prompt_layouts,
)
from ndif import load_remote_model
from seed_utils import set_random_seed
from trace_utils import RemoteExecutionContext, is_remote_model

CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache"
ACT_ROOT = Path(__file__).resolve().parents[1] / "activations"


def load_mean_activations(model_name: str) -> torch.Tensor:
    path = ACT_ROOT / f"{model_name}_pile-10k" / "mean.ckpt"
    if not path.exists():
        raise FileNotFoundError(
            f"Mean activations not found at {path}. "
            "Run compute_mean_activations.py --remote first."
        )
    return torch.load(path, map_location="cpu")  # [n_layers, n_heads, head_dim]


def load_head_rankings(model_name: str, score_type: str, topk: int) -> list[tuple[int, int]]:
    path = CACHE_ROOT / "head_orderings" / model_name / f"{score_type}_copying.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Head rankings not found at {path}. "
            "Run causal_scores.py first."
        )
    with path.open() as f:
        return [(r[0], r[1]) for r in json.load(f)[:topk]]


def run_ablated_pass(model, input_ids: list[int], heads_to_ablate: list[tuple[int, int]],
                     head_means: torch.Tensor) -> torch.Tensor:
    """
    Run a forward pass with specified heads replaced by their mean activations.
    Returns logits for the last token [vocab_size].
    """
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    remote = is_remote_model(model)

    layers_to_patch = sorted(set(l for l, _ in heads_to_ablate))

    with torch.no_grad():
        with model.trace(input_ids, remote=remote):
            for layer_idx in layers_to_patch:
                o_proj = model.model.layers[layer_idx].self_attn.o_proj
                o_proj_inp = o_proj.inputs[0][0]  # [bsz, seq_len, n_heads * head_dim]
                bsz = o_proj_inp.shape[0]
                seq_len = o_proj_inp.shape[1]
                head_acts = o_proj_inp.view(bsz, seq_len, n_heads, head_dim)

                for _, h in [(l, h) for l, h in heads_to_ablate if l == layer_idx]:
                    mean_val = head_means[layer_idx, h].to(o_proj_inp.device)
                    head_acts[:, :, h, :] = mean_val

                patched = head_acts.reshape(bsz, seq_len, n_heads * head_dim)
                o_proj.inputs = ((patched,), {})

            logits = model.output.logits[0, -1, :].save()

    return logits.detach().cpu()


def score_layout(model, layout, heads_to_ablate, head_means, remote_ctx, label):
    def _run(m):
        # xn pass: predict prefix token
        xn_logits = run_ablated_pass(m, layout.input_ids_xn, heads_to_ablate, head_means)
        p1_pred = int(xn_logits.argmax().item())
        p1_correct = p1_pred == layout.prefix_token_id

        # p1 pass (teacher-forced): predict suffix token
        p1_logits = run_ablated_pass(m, layout.input_ids_p1, heads_to_ablate, head_means)
        p2_pred = int(p1_logits.argmax().item())
        p2_correct = p2_pred == layout.suffix_token_id

        return {
            "task_idx": layout.task_idx,
            "bigram": layout.bigram,
            "p1_correct": p1_correct,
            "p2_correct": p2_correct,
            "copy_success": p1_correct and p2_correct,
            "hallucination": p1_correct and not p2_correct,
            "p1_pred": p1_pred,
            "p2_pred": p2_pred,
        }

    if remote_ctx is not None:
        return remote_ctx.request(label, _run)
    return _run(model)


def evaluate_head_set(model, layouts, heads_to_ablate, head_means, remote_ctx, workers, label):
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                score_layout, model, layout, heads_to_ablate, head_means,
                remote_ctx, f"{label}_task{layout.task_idx}"
            ): layout
            for layout in layouts
        }
        for future in as_completed(futures):
            layout = futures[future]
            exc = future.exception()
            if exc:
                print(f"ERROR task {layout.task_idx}: {exc}", flush=True)
                raise exc
            results.append(future.result())

    results.sort(key=lambda r: r["task_idx"])
    n = len(results)
    copy_rate = sum(r["copy_success"] for r in results) / n
    hall_rate  = sum(r["hallucination"] for r in results) / n
    return results, copy_rate, hall_rate


def build_tok(model):
    def tok(text, bos=False):
        ids = model.tokenizer(text)["input_ids"]
        return ids if bos else ids[1:]
    return tok


def main(args):
    set_random_seed(args.seed)

    model_name = args.model.split("/")[-1]
    out_dir = CACHE_ROOT / "ablation" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if args.remote:
        model = load_remote_model(args.model)
        remote_ctx = RemoteExecutionContext(
            args.model,
            max_retries=args.remote_max_retries,
            backoff_base=args.remote_backoff_base,
            backoff_max=args.remote_backoff_max,
        )
    else:
        from nnsight import LanguageModel
        model = LanguageModel(args.model, device_map="cuda")
        remote_ctx = None

    tok = build_tok(model)

    # Load mean activations
    head_means = load_mean_activations(model_name)  # [n_layers, n_heads, head_dim]
    print(f"Loaded mean activations: {tuple(head_means.shape)}")

    # Load bigram tasks and layouts
    tasks = load_bigram_tasks(args.tasks_path)
    layouts, mismatches = validate_prompt_layouts(tasks, tok)
    if mismatches:
        print(f"Warning: {len(mismatches)} tokenization mismatches skipped.")
    layouts = layouts[args.start: args.stop]
    print(f"Evaluating {len(layouts)} bigrams on {model_name}")

    # Build head sets to evaluate
    head_sets = {}

    # Baseline: no ablation
    head_sets["baseline"] = []

    # Custom heads from --heads arg: e.g. --heads 13,27 15,16
    if args.heads:
        custom = [tuple(int(x) for x in h.split(",")) for h in args.heads]
        head_sets[f"custom_{'_'.join(args.heads)}"] = custom
        print(f"Custom heads: {custom}")

    # Top-k from head rankings
    for head_type in args.head_types:
        for k in args.topk_values:
            heads = load_head_rankings(model_name, head_type, k)
            head_sets[f"top{k}_{head_type}"] = heads

    # Run evaluations
    all_results = {}
    summary_rows = []

    for label, heads_to_ablate in head_sets.items():
        n_ablated = len(heads_to_ablate)
        print(f"\n--- {label}: ablating {n_ablated} heads ---", flush=True)
        if n_ablated > 0:
            print(f"    heads: {heads_to_ablate[:10]}{'...' if n_ablated > 10 else ''}")

        results, copy_rate, hall_rate = evaluate_head_set(
            model, layouts, heads_to_ablate, head_means, remote_ctx,
            args.workers if args.remote else 1, label
        )
        all_results[label] = results
        summary_rows.append({
            "label": label,
            "n_ablated": n_ablated,
            "heads": heads_to_ablate,
            "copy_success_rate": round(copy_rate, 4),
            "hallucination_rate": round(hall_rate, 4),
            "n_examples": len(results),
        })
        print(f"    copy_success={copy_rate:.1%}  hallucination={hall_rate:.1%}")

    # Save outputs
    fname_suffix = f"start{args.start}_stop{args.stop or 'end'}"
    with (out_dir / f"summary_{fname_suffix}.json").open("w") as f:
        json.dump(summary_rows, f, indent=2)
    torch.save(all_results, out_dir / f"per_example_{fname_suffix}.pt")
    print(f"\nSaved results → {out_dir}")

    # Print summary table
    print("\n=== SUMMARY ===")
    print(f"{'Condition':<35} {'Copy%':>8} {'Hall%':>8} {'N heads':>8}")
    print("-" * 62)
    for row in summary_rows:
        print(f"{row['label']:<35} {row['copy_success_rate']:>8.1%} {row['hallucination_rate']:>8.1%} {row['n_ablated']:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B",
                        choices=["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B"])
    parser.add_argument("--tasks-path", default=str(DEFAULT_TASKS_PATH))
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--stop", default=None, type=int)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument(
        "--head-types", nargs="+", default=["concept", "token"],
        choices=["concept", "token"],
        help="Which ranked head sets to sweep over"
    )
    parser.add_argument(
        "--topk-values", nargs="+", type=int, default=[1, 2, 4, 8, 16],
        help="Values of k for top-k ablation sweep"
    )
    parser.add_argument(
        "--heads", nargs="*", default=None,
        help="Specific heads to ablate, e.g. --heads 13,27 15,16"
    )
    parser.add_argument("--remote-max-retries", default=4, type=int)
    parser.add_argument("--remote-backoff-base", default=2.0, type=float)
    parser.add_argument("--remote-backoff-max", default=30.0, type=float)
    parser.add_argument("--seed", default=8, type=int)
    main(parser.parse_args())
