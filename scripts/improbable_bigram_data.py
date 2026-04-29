from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


DUAL_ROUTE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DUAL_ROUTE_ROOT.parent


def _first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DEFAULT_TASKS_PATH = _first_existing_path(
    PROJECT_ROOT / "data" / "llama3.1_tasks.json",
    PROJECT_ROOT / "improbable-bigram-causality" / "data" / "llama3.1_tasks.json",
)
DEFAULT_GENERATIONS_PATH = _first_existing_path(
    PROJECT_ROOT / "data" / "llama3.1_base_generations.csv",
    PROJECT_ROOT
    / "improbable-bigram-causality"
    / "data"
    / "llama3.1_base_generations.csv",
)
DEFAULT_TRACE_ROOT = (
    DUAL_ROUTE_ROOT
    / "cache"
    / "improbable_bigrams"
    / "Llama-3.1-8B"
    / "table1_literal"
)
DEFAULT_RANDOM_TASKS_PATH = (
    DUAL_ROUTE_ROOT / "data" / "llama3.1_random_two_token_tasks.json"
)
PROMPT_STYLE = "table1_literal"
EXPECTED_BIGRAM_OCCURRENCES = 9


@dataclass(frozen=True)
class BigramTask:
    task_idx: int
    decoded: str
    prefix_token_id: int
    suffix_token_id: int
    quote_string: str


@dataclass(frozen=True)
class PromptLayout:
    task_idx: int
    bigram: str
    prefix_token_id: int
    suffix_token_id: int
    quote_string: str
    prompt_style: str
    prompt_text: str
    input_ids_xn: list[int]
    input_ids_p1: list[int]
    p2_prev_idx: int
    p2_context_indices: list[int]
    x_n_idx: int
    p1_idx: int
    final_prev_span_start: int
    final_prev_span_end: int

    def to_dict(self):
        return asdict(self)


def build_table1_prompt_lines(bigram: str, quote_string: str = "") -> list[str]:
    quoted_bigram = f"{quote_string}{bigram}{quote_string}"
    return [
        f"I will repeat the phrase {quoted_bigram} three times\n",
        f"{bigram}\n",
        f"{bigram}\n",
        f"{bigram}\n",
        f"I will repeat the phrase {quoted_bigram} five times\n",
        f"{bigram}\n",
        f"{bigram}\n",
        f"{bigram}\n",
        f"{bigram}\n",
    ]


def build_table1_prompt(bigram: str, quote_string: str = "") -> str:
    return "".join(build_table1_prompt_lines(bigram, quote_string))


def load_bigram_tasks(tasks_path: Path | str = DEFAULT_TASKS_PATH) -> list[BigramTask]:
    tasks_path = Path(tasks_path)
    with tasks_path.open("r", encoding="utf-8") as f:
        raw_tasks = json.load(f)

    tasks = []
    for idx, raw in enumerate(raw_tasks):
        quote_string = raw.get("quote_string", "")
        if quote_string is False:
            continue
        if not isinstance(quote_string, str):
            raise ValueError(
                f"Task {idx} has unsupported quote_string={quote_string!r}."
            )
        tasks.append(
            BigramTask(
                task_idx=idx,
                decoded=raw["decoded"],
                prefix_token_id=int(raw["prefix_i"]),
                suffix_token_id=int(raw["suffix_i"]),
                quote_string=quote_string,
            )
        )
    return tasks


def _prefix_token_span(
    tok, text: str, start_char: int, end_char: int
) -> tuple[int, int]:
    start_tok = len(tok(text[:start_char], bos=True))
    end_tok = len(tok(text[:end_char], bos=True))
    return start_tok, end_tok


def build_prompt_layout(task: BigramTask, tok) -> tuple[PromptLayout | None, list[str]]:
    """
    Build a prompt layout for a given bigram task.
    This includes constructing the prompt text, tokenizing it, and verifying that the bigram tokens appear in the expected locations.
    """
    lines = build_table1_prompt_lines(task.decoded, task.quote_string)
    prompt_text = "".join(lines)
    input_ids_xn = tok(prompt_text, bos=True)
    input_ids_p1 = input_ids_xn + [task.prefix_token_id]

    errors = []
    bigram_tokens = tok(task.decoded, bos=False)
    expected = [task.prefix_token_id, task.suffix_token_id]
    if bigram_tokens != expected:
        errors.append(
            f"Standalone bigram tokenization mismatch: expected {expected}, got {bigram_tokens}"
        )

    prefix_count = input_ids_xn.count(task.prefix_token_id)
    suffix_count = input_ids_xn.count(task.suffix_token_id)
    if (
        prefix_count < EXPECTED_BIGRAM_OCCURRENCES
        or suffix_count < EXPECTED_BIGRAM_OCCURRENCES
    ):
        errors.append(
            "Prompt token count mismatch: "
            f"expected prefix/suffix counts to both be at least {EXPECTED_BIGRAM_OCCURRENCES}, "
            f"got prefix_count={prefix_count}, suffix_count={suffix_count}"
        )
    p2_context_indices = [
        idx
        for idx, token_id in enumerate(input_ids_xn)
        if token_id == task.suffix_token_id
    ]

    # Check the final occurrence of the bigram in the prompt,
    # which should be immediately before x_n.
    final_prev_start_char = sum(len(line) for line in lines[:-1])
    final_prev_end_char = final_prev_start_char + len(task.decoded)
    span_start, span_end = _prefix_token_span(
        tok, prompt_text, final_prev_start_char, final_prev_end_char
    )

    repeated_tokens = input_ids_xn[span_start:span_end]
    if repeated_tokens != expected:
        errors.append(
            f"Final repeated occurrence mismatch: expected {expected}, got {repeated_tokens}"
        )

    x_n_idx = len(input_ids_xn) - 1
    if span_end != x_n_idx:
        errors.append(
            f"Expected final repeated occurrence to end immediately before x_n; "
            f"got span_end={span_end}, x_n_idx={x_n_idx}"
        )
    if not p2_context_indices or p2_context_indices[-1] != span_end - 1:
        errors.append(
            "Expected the final previous-context p2 token to be the suffix token "
            f"at index {span_end - 1}; got p2_context_indices={p2_context_indices}"
        )

    if input_ids_p1[-1] != task.prefix_token_id:
        errors.append(
            "Teacher-forced p1 pass does not end with the correct prefix token."
        )

    if errors:
        return None, errors

    return (
        PromptLayout(
            task_idx=task.task_idx,
            bigram=task.decoded,
            prefix_token_id=task.prefix_token_id,
            suffix_token_id=task.suffix_token_id,
            quote_string=task.quote_string,
            prompt_style=PROMPT_STYLE,
            prompt_text=prompt_text,
            input_ids_xn=input_ids_xn,
            input_ids_p1=input_ids_p1,
            p2_prev_idx=span_end - 1,
            p2_context_indices=p2_context_indices,
            x_n_idx=x_n_idx,
            p1_idx=len(input_ids_p1) - 1,
            final_prev_span_start=span_start,
            final_prev_span_end=span_end,
        ),
        [],
    )


def validate_prompt_layouts(tasks, tok):
    layouts = []
    mismatches = []
    for task in tasks:
        layout, errors = build_prompt_layout(task, tok)
        if errors:
            mismatches.append(
                {
                    "task_idx": task.task_idx,
                    "bigram": task.decoded,
                    "prefix_token_id": task.prefix_token_id,
                    "suffix_token_id": task.suffix_token_id,
                    "quote_string": task.quote_string,
                    "errors": errors,
                }
            )
        else:
            layouts.append(layout)
    return layouts, mismatches


def load_trace_index(trace_dir: Path | str) -> list[dict]:
    trace_dir = Path(trace_dir)
    index_path = trace_dir / "index.jsonl"
    if not index_path.exists():
        return []

    entries = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
