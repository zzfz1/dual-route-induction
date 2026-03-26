from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer

from improbable_bigram_data import DEFAULT_RANDOM_TASKS_PATH
from seed_utils import set_random_seed

try:
    from wordfreq import zipf_frequency
except ImportError as exc:
    raise ImportError(
        "improbable_bigram_generate_tasks.py requires the `wordfreq` package. "
        "Install dual-route-induction/requirements.txt first."
    ) from exc


QUOTE_STRING = '"'
WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*\Z")
VOWEL_RE = re.compile(r"[aeiouy]")
MIN_WORD_ZIPF = 3.0


@dataclass(frozen=True)
class TokenCandidate:
    token_id: int
    token_text: str
    word: str


def decode_ids(tokenizer, token_ids: list[int] | tuple[int, ...]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def english_word_from_token_text(
    token_text: str, *, require_leading_space: bool
) -> str | None:
    if not token_text or token_text != token_text.rstrip():
        return None
    if any(ch in token_text for ch in "\r\n\t"):
        return None

    if require_leading_space:
        if not token_text.startswith(" ") or token_text.startswith("  "):
            return None
        surface = token_text[1:]
    else:
        if token_text.startswith(" "):
            return None
        surface = token_text

    if not surface or surface.startswith(" "):
        return None
    if not WORD_RE.fullmatch(surface):
        return None
    if surface != surface.lower():
        return None
    if VOWEL_RE.search(surface) is None:
        return None
    if zipf_frequency(surface, "en") < MIN_WORD_ZIPF:
        return None
    return surface


def build_candidate_pools(tokenizer) -> tuple[list[TokenCandidate], list[TokenCandidate]]:
    prefixes: list[TokenCandidate] = []
    suffixes: list[TokenCandidate] = []
    special_ids = set(tokenizer.all_special_ids)

    for token_id in range(len(tokenizer)):
        if token_id in special_ids:
            continue

        token_text = decode_ids(tokenizer, [token_id])

        prefix_word = english_word_from_token_text(
            token_text, require_leading_space=False
        )
        if prefix_word is not None:
            prefixes.append(
                TokenCandidate(
                    token_id=token_id,
                    token_text=token_text,
                    word=prefix_word,
                )
            )

        suffix_word = english_word_from_token_text(
            token_text, require_leading_space=True
        )
        if suffix_word is not None:
            suffixes.append(
                TokenCandidate(
                    token_id=token_id,
                    token_text=token_text,
                    word=suffix_word,
                )
            )

    return prefixes, suffixes


def build_task(tokenizer, prefix: TokenCandidate, suffix: TokenCandidate, quote_token_id: int):
    expected_pair = [prefix.token_id, suffix.token_id]
    decoded = decode_ids(tokenizer, expected_pair)

    if encode_text(tokenizer, decoded) != expected_pair:
        return None

    quoted = f'{QUOTE_STRING}{decoded}{QUOTE_STRING}'
    expected_quoted = [quote_token_id, *expected_pair, quote_token_id]
    if encode_text(tokenizer, quoted) != expected_quoted:
        return None

    return {
        "decoded": decoded,
        "prefix": prefix.token_text,
        "suffix": suffix.token_text,
        "prefix_word": prefix.word,
        "suffix_word": suffix.word,
        "prefix_i": prefix.token_id,
        "suffix_i": suffix.token_id,
        "multiscript": False,
        "quote_string": QUOTE_STRING,
    }


def sample_tasks(
    tokenizer,
    prefixes: list[TokenCandidate],
    suffixes: list[TokenCandidate],
    n_tasks: int,
    seed: int,
    max_attempts: int,
) -> list[dict]:
    quote_ids = encode_text(tokenizer, QUOTE_STRING)
    if len(quote_ids) != 1:
        raise ValueError(
            f'Expected {QUOTE_STRING!r} to be a single token, got ids={quote_ids}.'
        )
    quote_token_id = quote_ids[0]

    rng = random.Random(seed)
    seen_pairs: set[tuple[int, int]] = set()
    seen_decoded: set[str] = set()
    tasks: list[dict] = []
    attempts = 0

    while len(tasks) < n_tasks and attempts < max_attempts:
        attempts += 1
        prefix = rng.choice(prefixes)
        suffix = rng.choice(suffixes)
        pair = (prefix.token_id, suffix.token_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        task = build_task(tokenizer, prefix, suffix, quote_token_id)
        if task is None or task["decoded"] in seen_decoded:
            continue

        seen_decoded.add(task["decoded"])
        tasks.append(task)

    if len(tasks) < n_tasks:
        raise RuntimeError(
            f"Only generated {len(tasks)} tasks after {attempts} attempts; "
            f"requested {n_tasks}. Increase --max-attempts or relax the filters."
        )

    return tasks


def main(args):
    set_random_seed(args.seed)

    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)

    prefixes, suffixes = build_candidate_pools(tokenizer)
    if not prefixes or not suffixes:
        raise RuntimeError("Failed to find any valid English one-token prefix/suffix words.")

    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = max(args.n * 100, 10_000)

    tasks = sample_tasks(
        tokenizer=tokenizer,
        prefixes=prefixes,
        suffixes=suffixes,
        n_tasks=args.n,
        seed=args.seed,
        max_attempts=max_attempts,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(tasks)} tasks to {out_path.resolve()}")
    print(
        f"Candidate pools: {len(prefixes)} prefix words, {len(suffixes)} suffix words."
    )
    if tasks:
        example = tasks[0]
        print(
            "Example:",
            repr(example["decoded"]),
            [example["prefix"], example["suffix"]],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate random two-token English phrases that stay exactly two tokens "
            "standalone and inside double quotes."
        )
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out-path", default=str(DEFAULT_RANDOM_TASKS_PATH))
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=8)
    main(parser.parse_args())
