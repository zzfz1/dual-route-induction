from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_PATH = PROJECT_ROOT / "data" / "llama3.1_tasks.json"
DEFAULT_GENERATIONS_PATH = PROJECT_ROOT / "data" / "llama3.1_70b_base_generations.csv"
DEFAULT_TRACE_ROOT = (
    PROJECT_ROOT
    / "dual-route-induction"
    / "cache"
    / "improbable_bigrams"
    / "Llama-3.1-70B"
    / "table1_literal"
)
DEFAULT_RANDOM_TASKS_PATH = (
    PROJECT_ROOT
    / "dual-route-induction"
    / "data"
    / "llama3.1_random_two_token_tasks.json"
)
PROMPT_STYLE = "table1_literal"


@dataclass(frozen=True)
class BigramTask:
    task_idx: int
    decoded: str
    prefix_token_id: int
    suffix_token_id: int


@dataclass(frozen=True)
class PromptLayout:
    task_idx: int
    bigram: str
    prefix_token_id: int
    suffix_token_id: int
    prompt_style: str
    prompt_text: str
    input_ids_xn: list[int]
    input_ids_p1: list[int]
    p2_prev_idx: int
    x_n_idx: int
    p1_idx: int
    final_prev_span_start: int
    final_prev_span_end: int

    def to_dict(self):
        return asdict(self)


def build_table1_prompt_lines(bigram: str) -> list[str]:
    return [
        f"I will repeat the phrase {bigram} three times\n",
        f"{bigram}\n",
        f"{bigram}\n",
        f"{bigram}\n",
        f"I will repeat the phrase {bigram} five times\n",
        f"{bigram}\n",
        f"{bigram}\n",
        f"{bigram}\n",
        f"{bigram}\n",
    ]


def build_table1_prompt(bigram: str) -> str:
    return "".join(build_table1_prompt_lines(bigram))


def load_bigram_tasks(tasks_path: Path | str = DEFAULT_TASKS_PATH) -> list[BigramTask]:
    tasks_path = Path(tasks_path)
    with tasks_path.open("r", encoding="utf-8") as f:
        raw_tasks = json.load(f)

    tasks = []
    for idx, raw in enumerate(raw_tasks):
        tasks.append(
            BigramTask(
                task_idx=idx,
                decoded=raw["decoded"],
                prefix_token_id=int(raw["prefix_i"]),
                suffix_token_id=int(raw["suffix_i"]),
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
    lines = build_table1_prompt_lines(task.decoded)
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
    # Check the final occurrence of the bigram in the prompt
    # Which should be immediately before x_n and thus p2's previous token (p2_prev).
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
            prompt_style=PROMPT_STYLE,
            prompt_text=prompt_text,
            input_ids_xn=input_ids_xn,
            input_ids_p1=input_ids_p1,
            p2_prev_idx=span_end - 1,
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
