from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


CONDITION_ORDER = [
    "Hallucinated improbable",
    "Copied improbable",
    "2-token concepts",
    "Random phrases",
]

CONDITION_COLORS = {
    "Hallucinated improbable": "#c0392b",
    "Copied improbable": "#2e86de",
    "2-token concepts": "#8e44ad",
    "Random phrases": "#27ae60",
}

WRONG_DLA_COLORS = {
    "Token wrong-token DLA": "#e67e22",
    "Concept wrong-token DLA": "#16a085",
}


def _condition_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS[condition],
            marker="o",
            linewidth=2,
            markersize=6,
            label=condition,
        )
        for condition in CONDITION_ORDER
    ]


def _add_condition_legend(axis):
    axis.legend(
        handles=_condition_legend_handles(),
        title="Setups",
        frameon=False,
        loc="upper right",
        fontsize=9,
        title_fontsize=9,
    )


def _add_top_figure_condition_legend(fig):
    fig.legend(
        handles=_condition_legend_handles(),
        title="Setups",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
    )


def configure_matplotlib():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "font.family": "serif",
        }
    )


def find_repo_root(start: Path) -> Path:
    candidates: list[Path] = []
    for candidate in (start, *start.parents):
        candidates.extend(
            [
                candidate,
                candidate / "dual-route-induction",
                candidate / "improbable-bigram-causality" / "dual-route-induction",
            ]
        )

    for candidate in candidates:
        if (candidate / "cache").exists() and (candidate / "scripts").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate the dual-route-induction repo root from the current working directory."
    )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_if_exists(path: Path):
    if path.exists():
        return load_json(path)
    return None


def load_cache(
    root: Path | str,
    model_name: str = "Llama-3.1-8B",
    improbable_run: str = "updated_table1_literal",
    concept_run: str = "selected_two_token_concepts",
    random_run: str = "random_tokens",
):
    root = Path(root).resolve()
    head_dir = root / "cache" / "head_orderings" / model_name
    improbable_dir = root / "cache" / "improbable_bigrams" / model_name / improbable_run
    concept_dir = root / "cache" / "improbable_bigrams" / model_name / concept_run
    random_dir = root / "cache" / "improbable_bigrams" / model_name / random_run

    cache = {
        "root": root,
        "model_name": model_name,
        "improbable_run": improbable_run,
        "concept_run": concept_run,
        "random_run": random_run,
        "token_ranking": [tuple(head) for head in load_json(head_dir / "token_copying.json")],
        "concept_ranking": [tuple(head) for head in load_json(head_dir / "concept_copying.json")],
        "improbable_summary": load_json_if_exists(
            root
            / "cache"
            / "improbable_bigrams"
            / model_name
            / "table1_literal_improbable_current_prompt_summary.json"
        ),
        "random_summary": load_json_if_exists(
            root
            / "cache"
            / "improbable_bigrams"
            / model_name
            / "table1_literal_random_tokens_current_prompt_summary.json"
        ),
        "improbable_manifest": load_json(improbable_dir / "manifest.json"),
        "concept_manifest": load_json(concept_dir / "manifest.json"),
        "random_manifest": load_json(random_dir / "manifest.json"),
        "improbable_index": [
            json.loads(line)
            for line in (improbable_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        "concept_index": [
            json.loads(line)
            for line in (concept_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        "random_index": [
            json.loads(line)
            for line in (random_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        "improbable_scores_all": torch.load(
            improbable_dir / "scores" / "per_example_all.pt", map_location="cpu"
        ),
        "improbable_scores_hall": torch.load(
            improbable_dir / "scores" / "per_example_hallucinated_second_token.pt",
            map_location="cpu",
        ),
        "improbable_dla_all": torch.load(
            improbable_dir / "dla" / "per_example_all_p1.pt", map_location="cpu"
        ),
        "improbable_dla_hall": torch.load(
            improbable_dir / "dla" / "per_example_hallucinated_second_token_p1.pt",
            map_location="cpu",
        ),
        "concept_scores_all": torch.load(
            concept_dir / "scores" / "per_example_all.pt", map_location="cpu"
        ),
        "concept_dla_all": torch.load(
            concept_dir / "dla" / "per_example_all_p1.pt", map_location="cpu"
        ),
        "random_scores_all": torch.load(
            random_dir / "scores" / "per_example_all.pt", map_location="cpu"
        ),
        "random_dla_all": torch.load(
            random_dir / "dla" / "per_example_all_p1.pt", map_location="cpu"
        ),
    }

    score_examples = cache["improbable_scores_all"]["examples"]
    dla_examples = cache["improbable_dla_all"]["examples"]
    score_task_ids = [entry["task_idx"] for entry in score_examples]
    dla_task_ids = [entry["task_idx"] for entry in dla_examples]
    if score_task_ids != dla_task_ids:
        raise ValueError("Per-example score and DLA payloads are not aligned by task_idx.")

    cache["improbable_task_ids"] = score_task_ids
    cache["hallucinated_mask"] = torch.tensor(
        [bool(entry["second_token_hallucination"]) for entry in score_examples],
        dtype=torch.bool,
    )
    cache["copied_mask"] = torch.tensor(
        [bool(entry["copy_success"]) for entry in score_examples],
        dtype=torch.bool,
    )
    concept_score_examples = cache["concept_scores_all"]["examples"]
    concept_dla_examples = cache["concept_dla_all"]["examples"]
    concept_score_task_ids = [entry["task_idx"] for entry in concept_score_examples]
    concept_dla_task_ids = [entry["task_idx"] for entry in concept_dla_examples]
    if concept_score_task_ids != concept_dla_task_ids:
        raise ValueError("Per-example concept score and DLA payloads are not aligned by task_idx.")

    cache["concept_task_ids"] = concept_score_task_ids
    cache["concept_count"] = len(cache["concept_scores_all"]["examples"])
    cache["random_count"] = len(cache["random_scores_all"]["examples"])
    return cache


def dataset_overview(cache) -> pd.DataFrame:
    summary = cache["improbable_summary"]
    random_summary = cache["random_summary"]
    traced_hall = int(cache["hallucinated_mask"].sum().item())
    traced_copy = int(cache["copied_mask"].sum().item())
    concept_hall = sum(
        1 for entry in cache["concept_index"] if entry["second_token_hallucination"]
    )
    concept_copy = sum(1 for entry in cache["concept_index"] if entry["copy_success"])

    improbable_examples = len(cache["improbable_task_ids"])
    concept_examples = cache["concept_count"]
    random_examples = cache["random_count"]
    if summary is not None:
        improbable_examples = summary["n_tasks"]
        improbable_hall = summary["n_second_token_hallucination"]
        improbable_copy = summary["n_copy_success"]
    else:
        improbable_hall = traced_hall
        improbable_copy = traced_copy

    if random_summary is not None:
        random_hall = random_summary["n_second_token_hallucination"]
        random_copy = random_summary["n_copy_success"]
    else:
        random_hall = sum(
            1 for entry in cache["random_index"] if entry["second_token_hallucination"]
        )
        random_copy = sum(1 for entry in cache["random_index"] if entry["copy_success"])

    return pd.DataFrame(
        [
            {
                "dataset": "Improbable summary",
                "examples": improbable_examples,
                "hallucinated_second_token": improbable_hall,
                "copied": improbable_copy,
            },
            {
                "dataset": "Updated trace cache",
                "examples": len(cache["improbable_task_ids"]),
                "hallucinated_second_token": traced_hall,
                "copied": traced_copy,
            },
            {
                "dataset": "2-token concept cache",
                "examples": concept_examples,
                "hallucinated_second_token": concept_hall,
                "copied": concept_copy,
            },
            {
                "dataset": "Random phrase cache",
                "examples": random_examples,
                "hallucinated_second_token": random_hall,
                "copied": random_copy,
            },
        ]
    )


def trace_consistency_markdown(cache) -> str:
    if cache["improbable_summary"] is None:
        return (
            "### Trace Consistency\n"
            "The prompt-summary JSON is not present in this repo snapshot. "
            "The notebook therefore uses the traced cache directly."
        )

    summary_failures = set(cache["improbable_summary"]["second_token_hallucination_failures"])
    traced_failures = {
        entry["task_idx"]
        for entry in cache["improbable_index"]
        if entry["second_token_hallucination"]
    }
    missing = sorted(summary_failures - traced_failures)
    if not missing and len(summary_failures) == len(traced_failures):
        return (
            "### Trace Consistency\n"
            "The generation summary and the traced cache agree on the hallucinated examples."
        )

    return (
        "### Trace Consistency\n"
        f"- Generation summary hallucinations: `{len(summary_failures)}`\n"
        f"- Traced hallucinations in `{cache['improbable_run']}`: `{len(traced_failures)}`\n"
        f"- Missing from traced cache: `{missing}`\n"
        "The notebook uses the traced cache for all statistics and figures."
    )


def head_mask(heads, n_layers: int = 32, n_heads: int = 32) -> torch.Tensor:
    mask = torch.zeros((n_layers, n_heads), dtype=torch.bool)
    for layer, head_idx in heads:
        mask[layer, head_idx] = True
    return mask


def aggregate_metric(score_tensor: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    return score_tensor[:, mask].mean(dim=1).cpu().numpy()


def bootstrap_mean_ci(
    values,
    n_bootstrap: int = 4000,
    seed: int = 8,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means[idx] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_diff_ci(
    left,
    right,
    n_bootstrap: int = 4000,
    seed: int = 8,
) -> tuple[float, float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        left_sample = rng.choice(left, size=len(left), replace=True)
        right_sample = rng.choice(right, size=len(right), replace=True)
        diffs[idx] = left_sample.mean() - right_sample.mean()
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def _token_mask(cache, k: int) -> torch.Tensor:
    return head_mask(cache["token_ranking"][:k])


def _concept_mask(cache, k: int) -> torch.Tensor:
    return head_mask(cache["concept_ranking"][:k])


def _condition_series(
    improbable_tensor: torch.Tensor,
    concept_tensor: torch.Tensor,
    random_tensor: torch.Tensor,
    mask: torch.Tensor,
    cache,
) -> dict[str, np.ndarray]:
    return {
        "Hallucinated improbable": aggregate_metric(
            improbable_tensor[cache["hallucinated_mask"]], mask
        ),
        "Copied improbable": aggregate_metric(
            improbable_tensor[cache["copied_mask"]], mask
        ),
        "2-token concepts": aggregate_metric(concept_tensor, mask),
        "Random phrases": aggregate_metric(random_tensor, mask),
    }


def token_ntm_series(cache, k: int, weighted: bool = False) -> dict[str, np.ndarray]:
    token_heads = _token_mask(cache, k)
    field = "ntm_value_weighted" if weighted else "ntm_raw"
    improbable = cache["improbable_scores_all"][field]
    concepts = cache["concept_scores_all"][field]
    random_scores = cache["random_scores_all"][field]
    return _condition_series(improbable, concepts, random_scores, token_heads, cache)


def concept_ltm_series(cache, k: int, weighted: bool = False) -> dict[str, np.ndarray]:
    concept_heads = _concept_mask(cache, k)
    field = "ltm_value_weighted" if weighted else "ltm_raw"
    improbable = cache["improbable_scores_all"][field]
    concepts = cache["concept_scores_all"][field]
    random_scores = cache["random_scores_all"][field]
    return _condition_series(improbable, concepts, random_scores, concept_heads, cache)


def token_correct_dla_series(cache, k: int) -> dict[str, np.ndarray]:
    token_heads = _token_mask(cache, k)
    improbable = cache["improbable_dla_all"]["correct_token_dla"]
    concepts = cache["concept_dla_all"]["correct_token_dla"]
    random_scores = cache["random_dla_all"]["correct_token_dla"]
    return _condition_series(improbable, concepts, random_scores, token_heads, cache)


def concept_correct_dla_series(cache, k: int) -> dict[str, np.ndarray]:
    concept_heads = _concept_mask(cache, k)
    improbable = cache["improbable_dla_all"]["correct_token_dla"]
    concepts = cache["concept_dla_all"]["correct_token_dla"]
    random_scores = cache["random_dla_all"]["correct_token_dla"]
    return _condition_series(improbable, concepts, random_scores, concept_heads, cache)


def token_wrong_dla_hall_series(cache, k: int) -> np.ndarray:
    token_heads = _token_mask(cache, k)
    hall_wrong = cache["improbable_dla_all"]["predicted_token_dla"][cache["hallucinated_mask"]]
    return aggregate_metric(hall_wrong, token_heads)


def concept_wrong_dla_hall_series(cache, k: int) -> np.ndarray:
    concept_heads = _concept_mask(cache, k)
    hall_wrong = cache["improbable_dla_all"]["predicted_token_dla"][cache["hallucinated_mask"]]
    return aggregate_metric(hall_wrong, concept_heads)


def build_pairwise_summary(cache, k: int) -> pd.DataFrame:
    metric_getters = [
        ("Token NTM (raw)", token_ntm_series(cache, k, weighted=False)),
        ("Token NTM (value-weighted)", token_ntm_series(cache, k, weighted=True)),
        ("Token correct-token DLA", token_correct_dla_series(cache, k)),
    ]

    rows = []
    for metric, series in metric_getters:
        hall = series["Hallucinated improbable"]
        copied = series["Copied improbable"]
        concepts = series["2-token concepts"]
        random_phrases = series["Random phrases"]
        for comparison, right in (
            ("Hallucinated - Copied", copied),
            ("Hallucinated - 2-token concepts", concepts),
            ("Hallucinated - Random", random_phrases),
        ):
            ci_low, ci_high = bootstrap_diff_ci(hall, right)
            rows.append(
                {
                    "metric": metric,
                    "comparison": comparison,
                    "left_mean": hall.mean(),
                    "right_mean": right.mean(),
                    "diff": hall.mean() - right.mean(),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    hall_correct = token_correct_dla_series(cache, k)["Hallucinated improbable"]
    hall_wrong = token_wrong_dla_hall_series(cache, k)
    ci_low, ci_high = bootstrap_diff_ci(hall_wrong, hall_correct)
    rows.append(
        {
            "metric": "Hallucinated token DLA",
            "comparison": "Wrong-token - Correct-token",
            "left_mean": hall_wrong.mean(),
            "right_mean": hall_correct.mean(),
            "diff": hall_wrong.mean() - hall_correct.mean(),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )
    return pd.DataFrame(rows)


def build_concept_pairwise_summary(cache, k: int) -> pd.DataFrame:
    metric_getters = [
        ("Concept LTM (raw)", concept_ltm_series(cache, k, weighted=False)),
        ("Concept LTM (value-weighted)", concept_ltm_series(cache, k, weighted=True)),
        ("Concept correct-token DLA", concept_correct_dla_series(cache, k)),
    ]

    rows = []
    for metric, series in metric_getters:
        hall = series["Hallucinated improbable"]
        copied = series["Copied improbable"]
        concepts = series["2-token concepts"]
        random_phrases = series["Random phrases"]
        for comparison, right in (
            ("Hallucinated - Copied", copied),
            ("Hallucinated - 2-token concepts", concepts),
            ("Hallucinated - Random", random_phrases),
        ):
            ci_low, ci_high = bootstrap_diff_ci(hall, right)
            rows.append(
                {
                    "metric": metric,
                    "comparison": comparison,
                    "left_mean": hall.mean(),
                    "right_mean": right.mean(),
                    "diff": hall.mean() - right.mean(),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    hall_correct = concept_correct_dla_series(cache, k)["Hallucinated improbable"]
    hall_wrong = concept_wrong_dla_hall_series(cache, k)
    ci_low, ci_high = bootstrap_diff_ci(hall_wrong, hall_correct)
    rows.append(
        {
            "metric": "Hallucinated concept DLA",
            "comparison": "Wrong-token - Correct-token",
            "left_mean": hall_wrong.mean(),
            "right_mean": hall_correct.mean(),
            "diff": hall_wrong.mean() - hall_correct.mean(),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )
    return pd.DataFrame(rows)


def interpretation_markdown(cache, k: int = 32) -> str:
    summary = build_pairwise_summary(cache, k)

    def row(metric: str, comparison: str):
        match = summary[
            (summary["metric"] == metric) & (summary["comparison"] == comparison)
        ]
        return match.iloc[0]

    ntm_hc = row("Token NTM (raw)", "Hallucinated - Copied")
    ntm_h2c = row("Token NTM (raw)", "Hallucinated - 2-token concepts")
    ntm_hr = row("Token NTM (raw)", "Hallucinated - Random")
    dla_hc = row("Token correct-token DLA", "Hallucinated - Copied")
    dla_h2c = row("Token correct-token DLA", "Hallucinated - 2-token concepts")
    dla_hr = row("Token correct-token DLA", "Hallucinated - Random")
    wrong_vs_correct = row("Hallucinated token DLA", "Wrong-token - Correct-token")

    if ntm_hc["ci_high"] < 0 and ntm_h2c["ci_high"] < 0:
        verdict = (
            "The top token-copying heads show lower NTM on hallucinated prompts than on both "
            "copied improbable prompts and coherent 2-token concepts. That is direct evidence "
            "for a hallucination-specific token-head attention failure, which matches "
            "Possibility 1-1."
        )
    elif ntm_hc["ci_low"] <= 0 <= ntm_hc["ci_high"] and ntm_h2c["ci_low"] <= 0 <= ntm_h2c["ci_high"] and (
        dla_hc["ci_high"] < 0 or dla_h2c["ci_high"] < 0 or wrong_vs_correct["ci_low"] > 0
    ):
        verdict = (
            "The token heads attend similarly on hallucinated prompts, copied improbable prompts, "
            "and 2-token concepts, but their DLA looks worse on hallucinated cases. That pattern "
            "is more consistent with Possibility 1-2 than with Possibility 1-1."
        )
    else:
        verdict = (
            "Within the improbable-bigram set, the traced cache does not show a clear hallucination-"
            "specific token-head breakdown. The stronger claim that token-head behavior is similar "
            "across all conditions is harder to defend, because the 2-token concept and random-"
            "phrase controls are both stronger on the token-head metrics."
        )

    return "\n".join(
        [
            f"### Hypothesis 1 Readout at Top-{k}",
            (
                f"- Hallucinated vs copied NTM: `{ntm_hc['diff']:.4f}` "
                f"(95% CI `{ntm_hc['ci_low']:.4f}` to `{ntm_hc['ci_high']:.4f}`)"
            ),
            (
                f"- Hallucinated vs copied correct-token DLA: `{dla_hc['diff']:.4f}` "
                f"(95% CI `{dla_hc['ci_low']:.4f}` to `{dla_hc['ci_high']:.4f}`)"
            ),
            (
                f"- Hallucinated vs 2-token concepts NTM: `{ntm_h2c['diff']:.4f}` "
                f"(95% CI `{ntm_h2c['ci_low']:.4f}` to `{ntm_h2c['ci_high']:.4f}`)"
            ),
            (
                f"- Hallucinated vs 2-token concepts correct-token DLA: `{dla_h2c['diff']:.4f}` "
                f"(95% CI `{dla_h2c['ci_low']:.4f}` to `{dla_h2c['ci_high']:.4f}`)"
            ),
            (
                f"- Hallucinated vs random NTM: `{ntm_hr['diff']:.4f}` "
                f"(95% CI `{ntm_hr['ci_low']:.4f}` to `{ntm_hr['ci_high']:.4f}`)"
            ),
            (
                f"- Hallucinated vs random correct-token DLA: `{dla_hr['diff']:.4f}` "
                f"(95% CI `{dla_hr['ci_low']:.4f}` to `{dla_hr['ci_high']:.4f}`)"
            ),
            (
                f"- Hallucinated wrong-token minus correct-token DLA: `{wrong_vs_correct['diff']:.4f}` "
                f"(95% CI `{wrong_vs_correct['ci_low']:.4f}` to `{wrong_vs_correct['ci_high']:.4f}`)"
            ),
            "",
            verdict,
        ]
    )


def concept_interpretation_markdown(cache, k: int = 32) -> str:
    summary = build_concept_pairwise_summary(cache, k)

    def row(metric: str, comparison: str):
        match = summary[
            (summary["metric"] == metric) & (summary["comparison"] == comparison)
        ]
        return match.iloc[0]

    ltm_hc = row("Concept LTM (raw)", "Hallucinated - Copied")
    ltm_h2c = row("Concept LTM (raw)", "Hallucinated - 2-token concepts")
    ltm_hr = row("Concept LTM (raw)", "Hallucinated - Random")
    dla_hc = row("Concept correct-token DLA", "Hallucinated - Copied")
    dla_h2c = row("Concept correct-token DLA", "Hallucinated - 2-token concepts")
    dla_hr = row("Concept correct-token DLA", "Hallucinated - Random")
    wrong_vs_correct = row("Hallucinated concept DLA", "Wrong-token - Correct-token")

    verdict = (
        "The current cache includes hallucinated improbable bigrams, copied improbable prompts, "
        "coherent 2-token concepts, and random two-token phrases. That allows a direct check of "
        "whether concept-head behavior on hallucinated prompts looks more like the coherent concept "
        "condition or like the random-token control."
    )

    if ltm_hc["ci_high"] < 0 and ltm_h2c["ci_high"] < 0:
        verdict += (
            " Hallucinated prompts show lower concept-head LTM than both copied improbable prompts "
            "and coherent 2-token concepts. That supports a hallucination-specific concept-head "
            "attention failure."
        )
    elif ltm_hc["ci_low"] <= 0 <= ltm_hc["ci_high"] and ltm_h2c["ci_low"] <= 0 <= ltm_h2c["ci_high"] and (
        dla_hc["ci_high"] < 0 or dla_h2c["ci_high"] < 0 or wrong_vs_correct["ci_low"] > 0
    ):
        verdict += (
            " Hallucinated prompts, copied improbable prompts, and 2-token concepts have similar "
            "concept-head LTM, but the DLA looks worse on hallucinated cases. That is more "
            "compatible with output-signaling differences than with attention failures."
        )
    else:
        verdict += (
            " In the current traced cache, concept heads do not cleanly separate hallucinated "
            "prompts from both copied improbable prompts and coherent 2-token concepts, and they "
            "do not strongly favor the wrong token over the correct token in hallucinated cases."
        )

    return "\n".join(
        [
            f"### Concept-Head Readout at Top-{k}",
            (
                f"- Hallucinated vs copied LTM: `{ltm_hc['diff']:.4f}` "
                f"(95% CI `{ltm_hc['ci_low']:.4f}` to `{ltm_hc['ci_high']:.4f}`)"
            ),
            (
                f"- Hallucinated vs copied correct-token DLA: `{dla_hc['diff']:.4f}` "
                f"(95% CI `{dla_hc['ci_low']:.4f}` to `{dla_hc['ci_high']:.4f}`)"
            ),
            (
                f"- Hallucinated vs 2-token concepts LTM: `{ltm_h2c['diff']:.4f}` "
                f"(95% CI `{ltm_h2c['ci_low']:.4f}` to `{ltm_h2c['ci_high']:.4f}`)"
            ),
            (
                f"- Hallucinated vs 2-token concepts correct-token DLA: `{dla_h2c['diff']:.4f}` "
                f"(95% CI `{dla_h2c['ci_low']:.4f}` to `{dla_h2c['ci_high']:.4f}`)"
            ),
            (
                f"- Hallucinated vs random LTM: `{ltm_hr['diff']:.4f}` "
                f"(95% CI `{ltm_hr['ci_low']:.4f}` to `{ltm_hr['ci_high']:.4f}`)"
            ),
            (
                f"- Hallucinated vs random correct-token DLA: `{dla_hr['diff']:.4f}` "
                f"(95% CI `{dla_hr['ci_low']:.4f}` to `{dla_hr['ci_high']:.4f}`)"
            ),
            (
                f"- Hallucinated wrong-token minus correct-token DLA: `{wrong_vs_correct['diff']:.4f}` "
                f"(95% CI `{wrong_vs_correct['ci_low']:.4f}` to `{wrong_vs_correct['ci_high']:.4f}`)"
            ),
            "",
            verdict,
        ]
    )


def _metric_sweep_rows(cache, ks, label: str, getter):
    rows = []
    for k in ks:
        series = getter(cache, k)
        for condition in CONDITION_ORDER:
            values = series[condition]
            ci_low, ci_high = bootstrap_mean_ci(values)
            rows.append(
                {
                    "metric": label,
                    "condition": condition,
                    "k": k,
                    "mean": values.mean(),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return rows


def _single_metric_sweep_rows(
    cache,
    ks,
    label: str,
    getter,
    condition: str = "Hallucinated improbable",
):
    rows = []
    for k in ks:
        values = getter(cache, k)
        ci_low, ci_high = bootstrap_mean_ci(values)
        rows.append(
            {
                "metric": label,
                "condition": condition,
                "k": k,
                "mean": values.mean(),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return rows


def _plot_wrong_dla_comparison(axis, summary: pd.DataFrame, ks):
    for metric in ("Token wrong-token DLA", "Concept wrong-token DLA"):
        wrong_df = summary[
            (summary["metric"] == metric) & (summary["condition"] == "Hallucinated improbable")
        ].sort_values("k")
        x = wrong_df["k"].to_numpy(dtype=float)
        mean = wrong_df["mean"].to_numpy(dtype=float)
        lower = wrong_df["ci_low"].to_numpy(dtype=float)
        upper = wrong_df["ci_high"].to_numpy(dtype=float)
        color = WRONG_DLA_COLORS[metric]
        axis.plot(x, mean, marker="o", linewidth=2, color=color, label=metric)
        axis.fill_between(x, lower, upper, color=color, alpha=0.15)

    axis.set_title("Hallucinated wrong-token DLA")
    axis.set_xlabel("Top-k heads within each ranking")
    axis.set_ylabel("Per-example mean")
    axis.set_xticks(list(ks))
    axis.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axis.legend(frameon=False, loc="best")


def plot_k_sweep(cache, ks=(8, 16, 32, 64, 128), save_path: Path | None = None):
    rows = []
    rows.extend(
        _metric_sweep_rows(cache, ks, "Token NTM (raw)", lambda c, k: token_ntm_series(c, k, False))
    )
    rows.extend(
        _metric_sweep_rows(cache, ks, "Token correct-token DLA", token_correct_dla_series)
    )
    rows.extend(
        _single_metric_sweep_rows(
            cache,
            ks,
            "Token wrong-token DLA",
            token_wrong_dla_hall_series,
        )
    )
    rows.extend(
        _single_metric_sweep_rows(
            cache,
            ks,
            "Concept wrong-token DLA",
            concept_wrong_dla_hall_series,
        )
    )
    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for axis, metric in zip(axes[:2], ["Token NTM (raw)", "Token correct-token DLA"]):
        metric_df = summary[summary["metric"] == metric]
        for condition in CONDITION_ORDER:
            condition_df = metric_df[metric_df["condition"] == condition].sort_values("k")
            x = condition_df["k"].to_numpy(dtype=float)
            mean = condition_df["mean"].to_numpy(dtype=float)
            lower = condition_df["ci_low"].to_numpy(dtype=float)
            upper = condition_df["ci_high"].to_numpy(dtype=float)
            color = CONDITION_COLORS[condition]
            axis.plot(x, mean, marker="o", linewidth=2, color=color, label=condition)
            axis.fill_between(x, lower, upper, color=color, alpha=0.15)
        axis.set_title(metric)
        axis.set_xlabel("Top-k token-copying heads")
        axis.set_ylabel("Per-example mean")
        axis.set_xticks(list(ks))
        if metric == "Token correct-token DLA":
            axis.axhline(0.0, color="black", linewidth=1, alpha=0.5)

    _add_condition_legend(axes[0])
    _add_condition_legend(axes[1])
    _plot_wrong_dla_comparison(axes[2], summary, ks)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes, summary


def plot_concept_k_sweep(cache, ks=(8, 16, 32, 64, 128), save_path: Path | None = None):
    rows = []
    rows.extend(
        _metric_sweep_rows(
            cache, ks, "Concept LTM (raw)", lambda c, k: concept_ltm_series(c, k, False)
        )
    )
    rows.extend(
        _metric_sweep_rows(cache, ks, "Concept correct-token DLA", concept_correct_dla_series)
    )
    rows.extend(
        _single_metric_sweep_rows(
            cache,
            ks,
            "Concept wrong-token DLA",
            concept_wrong_dla_hall_series,
        )
    )
    rows.extend(
        _single_metric_sweep_rows(
            cache,
            ks,
            "Token wrong-token DLA",
            token_wrong_dla_hall_series,
        )
    )
    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for axis, metric in zip(axes[:2], ["Concept LTM (raw)", "Concept correct-token DLA"]):
        metric_df = summary[summary["metric"] == metric]
        for condition in CONDITION_ORDER:
            condition_df = metric_df[metric_df["condition"] == condition].sort_values("k")
            x = condition_df["k"].to_numpy(dtype=float)
            mean = condition_df["mean"].to_numpy(dtype=float)
            lower = condition_df["ci_low"].to_numpy(dtype=float)
            upper = condition_df["ci_high"].to_numpy(dtype=float)
            color = CONDITION_COLORS[condition]
            axis.plot(x, mean, marker="o", linewidth=2, color=color, label=condition)
            axis.fill_between(x, lower, upper, color=color, alpha=0.15)
        axis.set_title(metric)
        axis.set_xlabel("Top-k concept-copying heads")
        axis.set_ylabel("Per-example mean")
        axis.set_xticks(list(ks))
        if metric == "Concept correct-token DLA":
            axis.axhline(0.0, color="black", linewidth=1, alpha=0.5)

    _add_condition_legend(axes[0])
    _add_condition_legend(axes[1])
    _plot_wrong_dla_comparison(axes[2], summary, ks)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes, summary


def _box_and_strip(axis, values_by_condition, ylabel: str, title: str):
    ordered = [values_by_condition[condition] for condition in CONDITION_ORDER]
    positions = np.arange(1, len(CONDITION_ORDER) + 1)
    axis.boxplot(
        ordered,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#f6f6f6", "edgecolor": "#666666"},
        medianprops={"color": "#111111", "linewidth": 1.6},
        whiskerprops={"color": "#666666"},
        capprops={"color": "#666666"},
    )
    rng = np.random.default_rng(8)
    for idx, condition in enumerate(CONDITION_ORDER, start=1):
        values = values_by_condition[condition]
        jitter = rng.normal(loc=idx, scale=0.045, size=len(values))
        axis.scatter(
            jitter,
            values,
            s=18,
            alpha=0.55,
            color=CONDITION_COLORS[condition],
            edgecolor="none",
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(CONDITION_ORDER, rotation=15, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)


def plot_distribution_panels(cache, k: int = 32, save_path: Path | None = None):
    ntm = token_ntm_series(cache, k, weighted=False)
    dla = token_correct_dla_series(cache, k)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=False)
    _box_and_strip(axes[0], ntm, "Per-example mean", f"Top-{k} token-head NTM")
    _box_and_strip(axes[1], dla, "Per-example mean", f"Top-{k} token-head correct-token DLA")
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    _add_top_figure_condition_legend(fig)
    fig.subplots_adjust(top=0.80, wspace=0.26)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


def plot_concept_distribution_panels(cache, k: int = 32, save_path: Path | None = None):
    ltm = concept_ltm_series(cache, k, weighted=False)
    dla = concept_correct_dla_series(cache, k)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=False)
    _box_and_strip(axes[0], ltm, "Per-example mean", f"Top-{k} concept-head LTM")
    _box_and_strip(
        axes[1], dla, "Per-example mean", f"Top-{k} concept-head correct-token DLA"
    )
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    _add_top_figure_condition_legend(fig)
    fig.subplots_adjust(top=0.80, wspace=0.26)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


def plot_hallucinated_dla_pair(cache, k: int = 32, save_path: Path | None = None):
    hall_correct = token_correct_dla_series(cache, k)["Hallucinated improbable"]
    hall_wrong = token_wrong_dla_hall_series(cache, k)
    diff = hall_correct - hall_wrong

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].scatter(hall_correct, hall_wrong, color=CONDITION_COLORS["Hallucinated improbable"], alpha=0.7)
    diagonal_min = min(hall_correct.min(), hall_wrong.min())
    diagonal_max = max(hall_correct.max(), hall_wrong.max())
    axes[0].plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], color="black", linewidth=1)
    axes[0].set_xlabel("Correct-token DLA")
    axes[0].set_ylabel("Predicted wrong-token DLA")
    axes[0].set_title(f"Hallucinated examples only, top-{k} token heads")

    axes[1].hist(diff, bins=12, color="#555555", alpha=0.8)
    axes[1].axvline(diff.mean(), color=CONDITION_COLORS["Copied improbable"], linewidth=2, linestyle="--")
    axes[1].set_xlabel("Correct-token DLA minus wrong-token DLA")
    axes[1].set_ylabel("Number of hallucinated examples")
    axes[1].set_title("Positive values favor the correct token")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


def plot_concept_hallucinated_dla_pair(cache, k: int = 32, save_path: Path | None = None):
    hall_correct = concept_correct_dla_series(cache, k)["Hallucinated improbable"]
    hall_wrong = concept_wrong_dla_hall_series(cache, k)
    diff = hall_correct - hall_wrong

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].scatter(
        hall_correct,
        hall_wrong,
        color=CONDITION_COLORS["Hallucinated improbable"],
        alpha=0.7,
    )
    diagonal_min = min(hall_correct.min(), hall_wrong.min())
    diagonal_max = max(hall_correct.max(), hall_wrong.max())
    axes[0].plot(
        [diagonal_min, diagonal_max],
        [diagonal_min, diagonal_max],
        color="black",
        linewidth=1,
    )
    axes[0].set_xlabel("Correct-token DLA")
    axes[0].set_ylabel("Predicted wrong-token DLA")
    axes[0].set_title(f"Hallucinated examples only, top-{k} concept heads")

    axes[1].hist(diff, bins=12, color="#555555", alpha=0.8)
    axes[1].axvline(
        diff.mean(),
        color=CONDITION_COLORS["Copied improbable"],
        linewidth=2,
        linestyle="--",
    )
    axes[1].set_xlabel("Correct-token DLA minus wrong-token DLA")
    axes[1].set_ylabel("Number of hallucinated examples")
    axes[1].set_title("Positive values favor the correct token")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


def plot_top_head_heatmap(cache, top_k: int = 16, save_path: Path | None = None):
    heads = cache["token_ranking"][:top_k]
    ntm_tensor = cache["improbable_scores_all"]["ntm_raw"]
    concept_ntm = cache["concept_scores_all"]["ntm_raw"]
    random_ntm = cache["random_scores_all"]["ntm_raw"]
    correct_dla = cache["improbable_dla_all"]["correct_token_dla"]
    wrong_dla = cache["improbable_dla_all"]["predicted_token_dla"]
    concept_correct_dla = cache["concept_dla_all"]["correct_token_dla"]
    random_correct_dla = cache["random_dla_all"]["correct_token_dla"]

    ntm_matrix = []
    dla_matrix = []
    labels = []

    for layer, head_idx in heads:
        labels.append(f"{layer}.{head_idx}")
        ntm_matrix.append(
            [
                float(ntm_tensor[cache["hallucinated_mask"], layer, head_idx].mean().item()),
                float(ntm_tensor[cache["copied_mask"], layer, head_idx].mean().item()),
                float(concept_ntm[:, layer, head_idx].mean().item()),
                float(random_ntm[:, layer, head_idx].mean().item()),
            ]
        )
        dla_matrix.append(
            [
                float(correct_dla[cache["hallucinated_mask"], layer, head_idx].mean().item()),
                float(wrong_dla[cache["hallucinated_mask"], layer, head_idx].mean().item()),
                float(correct_dla[cache["copied_mask"], layer, head_idx].mean().item()),
                float(concept_correct_dla[:, layer, head_idx].mean().item()),
                float(random_correct_dla[:, layer, head_idx].mean().item()),
            ]
        )

    ntm_matrix = np.asarray(ntm_matrix, dtype=float)
    dla_matrix = np.asarray(dla_matrix, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(5, 0.35 * top_k)), constrained_layout=True)
    ntm_image = axes[0].imshow(ntm_matrix, aspect="auto", cmap="viridis")
    axes[0].set_title(f"Top-{top_k} token heads: NTM")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(["Hall", "Copied", "Concepts", "Random"])
    axes[0].set_yticks(range(top_k))
    axes[0].set_yticklabels(labels)
    fig.colorbar(ntm_image, ax=axes[0], shrink=0.9)

    abs_max = max(abs(dla_matrix.min()), abs(dla_matrix.max()))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    dla_image = axes[1].imshow(dla_matrix, aspect="auto", cmap="coolwarm", norm=norm)
    axes[1].set_title(f"Top-{top_k} token heads: DLA")
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(
        ["Hall correct", "Hall wrong", "Copied correct", "Concepts correct", "Random correct"]
    )
    plt.setp(axes[1].get_xticklabels(), rotation=18, ha="right")
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels(labels)
    fig.colorbar(dla_image, ax=axes[1], shrink=0.9)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


def plot_concept_head_heatmap(cache, top_k: int = 16, save_path: Path | None = None):
    heads = cache["concept_ranking"][:top_k]
    ltm_tensor = cache["improbable_scores_all"]["ltm_raw"]
    concept_ltm = cache["concept_scores_all"]["ltm_raw"]
    random_ltm = cache["random_scores_all"]["ltm_raw"]
    correct_dla = cache["improbable_dla_all"]["correct_token_dla"]
    wrong_dla = cache["improbable_dla_all"]["predicted_token_dla"]
    concept_correct_dla = cache["concept_dla_all"]["correct_token_dla"]
    random_correct_dla = cache["random_dla_all"]["correct_token_dla"]

    ltm_matrix = []
    dla_matrix = []
    labels = []

    for layer, head_idx in heads:
        labels.append(f"{layer}.{head_idx}")
        ltm_matrix.append(
            [
                float(ltm_tensor[cache["hallucinated_mask"], layer, head_idx].mean().item()),
                float(ltm_tensor[cache["copied_mask"], layer, head_idx].mean().item()),
                float(concept_ltm[:, layer, head_idx].mean().item()),
                float(random_ltm[:, layer, head_idx].mean().item()),
            ]
        )
        dla_matrix.append(
            [
                float(correct_dla[cache["hallucinated_mask"], layer, head_idx].mean().item()),
                float(wrong_dla[cache["hallucinated_mask"], layer, head_idx].mean().item()),
                float(correct_dla[cache["copied_mask"], layer, head_idx].mean().item()),
                float(concept_correct_dla[:, layer, head_idx].mean().item()),
                float(random_correct_dla[:, layer, head_idx].mean().item()),
            ]
        )

    ltm_matrix = np.asarray(ltm_matrix, dtype=float)
    dla_matrix = np.asarray(dla_matrix, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(5, 0.35 * top_k)), constrained_layout=True)
    ltm_image = axes[0].imshow(ltm_matrix, aspect="auto", cmap="viridis")
    axes[0].set_title(f"Top-{top_k} concept heads: LTM")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(["Hall", "Copied", "Concepts", "Random"])
    axes[0].set_yticks(range(top_k))
    axes[0].set_yticklabels(labels)
    fig.colorbar(ltm_image, ax=axes[0], shrink=0.9)

    abs_max = max(abs(dla_matrix.min()), abs(dla_matrix.max()))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    dla_image = axes[1].imshow(dla_matrix, aspect="auto", cmap="coolwarm", norm=norm)
    axes[1].set_title(f"Top-{top_k} concept heads: DLA")
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(
        ["Hall correct", "Hall wrong", "Copied correct", "Concepts correct", "Random correct"]
    )
    plt.setp(axes[1].get_xticklabels(), rotation=18, ha="right")
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels(labels)
    fig.colorbar(dla_image, ax=axes[1], shrink=0.9)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


def _dla_source_tensor(cache, source: str) -> tuple[torch.Tensor, str]:
    if source == "hallucinated_correct":
        tensor = cache["improbable_dla_all"]["correct_token_dla"][cache["hallucinated_mask"]]
        return tensor, "Hallucinated correct-token DLA"
    if source == "hallucinated_wrong":
        tensor = cache["improbable_dla_all"]["predicted_token_dla"][cache["hallucinated_mask"]]
        return tensor, "Hallucinated wrong-token DLA"
    if source == "copied_correct":
        tensor = cache["improbable_dla_all"]["correct_token_dla"][cache["copied_mask"]]
        return tensor, "Copied-improbable correct-token DLA"
    if source == "concepts_correct":
        tensor = cache["concept_dla_all"]["correct_token_dla"]
        return tensor, "2-token-concept correct-token DLA"
    if source == "random_correct":
        tensor = cache["random_dla_all"]["correct_token_dla"]
        return tensor, "Random-phrase correct-token DLA"
    raise ValueError(f"Unsupported DLA source: {source}")


def mean_dla_by_head(cache, source: str) -> pd.DataFrame:
    tensor, label = _dla_source_tensor(cache, source)
    mean_scores = tensor.mean(dim=0).cpu().numpy()
    rows = []
    for layer in range(mean_scores.shape[0]):
        for head_idx in range(mean_scores.shape[1]):
            rows.append(
                {
                    "layer": layer,
                    "head_idx": head_idx,
                    "head": f"{layer}.{head_idx}",
                    "mean_dla": float(mean_scores[layer, head_idx]),
                    "source": label,
                }
            )
    return pd.DataFrame(rows).sort_values("mean_dla", ascending=False).reset_index(drop=True)


def top_dla_membership_table(
    cache,
    source: str = "hallucinated_correct",
    top_n: int = 20,
    compare_k: int = 32,
) -> pd.DataFrame:
    token_top = set(cache["token_ranking"][:compare_k])
    concept_top = set(cache["concept_ranking"][:compare_k])
    token_rank = {tuple(head): idx + 1 for idx, head in enumerate(cache["token_ranking"])}
    concept_rank = {tuple(head): idx + 1 for idx, head in enumerate(cache["concept_ranking"])}

    top_df = mean_dla_by_head(cache, source).head(top_n).copy()
    top_df["in_token_top_k"] = [
        (row.layer, row.head_idx) in token_top for row in top_df.itertuples()
    ]
    top_df["in_concept_top_k"] = [
        (row.layer, row.head_idx) in concept_top for row in top_df.itertuples()
    ]
    top_df["token_rank"] = [
        token_rank[(row.layer, row.head_idx)] for row in top_df.itertuples()
    ]
    top_df["concept_rank"] = [
        concept_rank[(row.layer, row.head_idx)] for row in top_df.itertuples()
    ]
    top_df["membership_label"] = np.select(
        [
            top_df["in_token_top_k"] & top_df["in_concept_top_k"],
            top_df["in_token_top_k"],
            top_df["in_concept_top_k"],
        ],
        [
            "Both",
            "Token only",
            "Concept only",
        ],
        default="Neither",
    )
    return top_df


def summarize_top_dla_membership(
    cache,
    sources=(
        "hallucinated_correct",
        "hallucinated_wrong",
        "copied_correct",
        "concepts_correct",
        "random_correct",
    ),
    top_n: int = 20,
    compare_k: int = 32,
) -> pd.DataFrame:
    rows = []
    for source in sources:
        table = top_dla_membership_table(
            cache, source=source, top_n=top_n, compare_k=compare_k
        )
        rows.append(
            {
                "source": table["source"].iloc[0],
                "top_n": top_n,
                "compare_k": compare_k,
                "token_overlap": int(table["in_token_top_k"].sum()),
                "concept_overlap": int(table["in_concept_top_k"].sum()),
                "both_overlap": int((table["in_token_top_k"] & table["in_concept_top_k"]).sum()),
                "neither_overlap": int((~table["in_token_top_k"] & ~table["in_concept_top_k"]).sum()),
                "mean_token_rank": float(table["token_rank"].mean()),
                "mean_concept_rank": float(table["concept_rank"].mean()),
            }
        )
    return pd.DataFrame(rows)
