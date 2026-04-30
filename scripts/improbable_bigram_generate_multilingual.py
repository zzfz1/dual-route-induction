"""
Generate random two-token multi-script bigrams.

Each bigram is a (prefix, suffix) token pair where:
  - The two tokens come from different Unicode scripts
  - The combined bigram contains characters from ≥2 scripts
  - The pair is tokenization-stable (encodes to exactly those 2 token IDs
    both standalone and inside double quotes)

Usage:
    python multiscript_bigram_generate_tasks.py --model meta-llama/Llama-3.1-8B --n 100
"""

import argparse
import json
import os
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer


QUOTE_STRING = '"'

# Scripts we consider interesting (non-Common, non-Inherited)
TARGET_SCRIPTS = {
    "ARABIC", "ARMENIAN", "BENGALI", "CYRILLIC", "DEVANAGARI",
    "ETHIOPIC", "GEORGIAN", "GREEK", "GUJARATI", "GURMUKHI",
    "HANGUL", "HAN", "HEBREW", "HIRAGANA", "KANNADA",
    "KATAKANA", "KHMER", "LAO", "MALAYALAM", "MYANMAR",
    "ORIYA", "SINHALA", "TAMIL", "TELUGU", "THAI", "TIBETAN",
}

# Latin is also valid as one half of a multi-script pair
ALL_CANDIDATE_SCRIPTS = TARGET_SCRIPTS | {"LATIN"}

# Minimum number of "real" script characters in a token
MIN_SCRIPT_CHARS = 2


@dataclass(frozen=True)
class TokenCandidate:
    token_id: int
    token_text: str
    script: str          # dominant script of this token
    has_leading_space: bool


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def decode_ids(tokenizer, token_ids: list[int] | tuple[int, ...]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Unicode script detection
# ---------------------------------------------------------------------------

def _script_from_name(ch: str) -> str:
    """Heuristic script detection from unicodedata.name() for Python < 3.13."""
    try:
        name = unicodedata.name(ch, "")
    except ValueError:
        return "UNKNOWN"

    name_upper = name.upper()

    script_prefixes = [
        ("CJK", "HAN"), ("HANGUL", "HANGUL"), ("HIRAGANA", "HIRAGANA"),
        ("KATAKANA", "KATAKANA"), ("ARABIC", "ARABIC"), ("CYRILLIC", "CYRILLIC"),
        ("GREEK", "GREEK"), ("HEBREW", "HEBREW"), ("DEVANAGARI", "DEVANAGARI"),
        ("BENGALI", "BENGALI"), ("THAI", "THAI"), ("GEORGIAN", "GEORGIAN"),
        ("ARMENIAN", "ARMENIAN"), ("ETHIOPIC", "ETHIOPIC"), ("TAMIL", "TAMIL"),
        ("TELUGU", "TELUGU"), ("KANNADA", "KANNADA"), ("MALAYALAM", "MALAYALAM"),
        ("GUJARATI", "GUJARATI"), ("GURMUKHI", "GURMUKHI"), ("ORIYA", "ORIYA"),
        ("SINHALA", "SINHALA"), ("KHMER", "KHMER"), ("LAO", "LAO"),
        ("MYANMAR", "MYANMAR"), ("TIBETAN", "TIBETAN"),
        ("LATIN", "LATIN"),
    ]
    for prefix, script in script_prefixes:
        if prefix in name_upper:
            return script

    # Latin letters sometimes lack "LATIN" in their name
    cat = unicodedata.category(ch)
    if cat.startswith("L"):
        cp = ord(ch)
        if (0x0041 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
            return "LATIN"

    if cat in ("Nd", "No", "Nl"):
        return "COMMON"
    if cat in ("Zs", "Zl", "Zp", "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po",
               "Sm", "Sc", "Sk", "So", "Cc", "Cf"):
        return "COMMON"

    return "UNKNOWN"


def get_char_script(ch: str) -> str:
    """Return the Unicode script for a single character."""
    # Python 3.13+ has unicodedata.script(); fall back to heuristic
    if hasattr(unicodedata, "script"):
        s = unicodedata.script(ch)
        return s.upper().replace(" ", "_")
    return _script_from_name(ch)


def get_script_counts(text: str) -> dict[str, int]:
    """Count characters per Unicode script, ignoring Common/Inherited/Unknown."""
    counts: dict[str, int] = {}
    for ch in text:
        script = get_char_script(ch)
        if script in ("COMMON", "INHERITED", "UNKNOWN"):
            continue
        counts[script] = counts.get(script, 0) + 1
    return counts


def dominant_script(text: str) -> str | None:
    """Return the script with the most characters, or None."""
    counts = get_script_counts(text)
    if not counts:
        return None
    return max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# Token filtering
# ---------------------------------------------------------------------------

def is_valid_token_text(token_text: str) -> bool:
    """Basic sanity: no control chars, not pure whitespace/punctuation."""
    stripped = token_text.strip()
    if "\n" in token_text or "\t" in token_text or "\r" in token_text:
        return False
    if not stripped:
        return False
    # Reject tokens with control characters (except newline)
    if any(unicodedata.category(ch).startswith("C") and ch != "\n"
           for ch in stripped):
        return False
    return True


def build_candidate_pool(tokenizer) -> dict[str, list[TokenCandidate]]:
    """
    Scan the tokenizer vocab and collect tokens grouped by dominant script.

    Each token must:
      - Have ≥ MIN_SCRIPT_CHARS characters from a real script
      - Be dominated (≥80%) by a single script in ALL_CANDIDATE_SCRIPTS
      - Not be a special token
    """
    pool: dict[str, list[TokenCandidate]] = {}
    special_ids = set(tokenizer.all_special_ids)

    for token_id in range(len(tokenizer)):
        if token_id in special_ids:
            continue

        token_text = decode_ids(tokenizer, [token_id])
        if not is_valid_token_text(token_text):
            continue

        has_leading_space = (
            token_text.startswith(" ") and not token_text.startswith("  ")
        )
        content = token_text.lstrip(" ")

        if len(content) < 1:
            continue

        script_counts = get_script_counts(content)
        total_script_chars = sum(script_counts.values())
        if total_script_chars < MIN_SCRIPT_CHARS:
            continue

        dom = max(script_counts, key=script_counts.get)
        if dom not in ALL_CANDIDATE_SCRIPTS:
            continue

        # Require dominant script to account for ≥80% of script characters
        if script_counts[dom] < total_script_chars * 0.8:
            continue

        candidate = TokenCandidate(
            token_id=token_id,
            token_text=token_text,
            script=dom,
            has_leading_space=has_leading_space,
        )
        pool.setdefault(dom, []).append(candidate)

    return pool


# ---------------------------------------------------------------------------
# Task construction and validation
# ---------------------------------------------------------------------------

def bigram_is_multiscript(prefix: TokenCandidate, suffix: TokenCandidate) -> bool:
    """Check that the combined bigram truly has ≥2 Unicode scripts."""
    combined = prefix.token_text + suffix.token_text
    scripts = set(get_script_counts(combined).keys())
    return len(scripts) >= 2


def build_task(
    tokenizer,
    prefix: TokenCandidate,
    suffix: TokenCandidate,
    quote_token_id: int,
) -> dict | None:
    """Validate tokenization stability and build the task dict."""
    expected_pair = [prefix.token_id, suffix.token_id]
    decoded = decode_ids(tokenizer, expected_pair)

    # Stability check: decoded text must re-encode to the same 2 tokens
    if encode_text(tokenizer, decoded) != expected_pair:
        return None

    # Also stable inside quotes
    quoted = f'{QUOTE_STRING}{decoded}{QUOTE_STRING}'
    expected_quoted = [quote_token_id, *expected_pair, quote_token_id]
    if encode_text(tokenizer, quoted) != expected_quoted:
        return None

    # Final multi-script verification on the decoded form
    scripts_found = set(get_script_counts(decoded).keys())
    if len(scripts_found) < 2:
        return None

    return {
        "decoded": decoded,
        "prefix": prefix.token_text,
        "suffix": suffix.token_text,
        "prefix_script": prefix.script,
        "suffix_script": suffix.script,
        "scripts": sorted(scripts_found),
        "prefix_i": prefix.token_id,
        "suffix_i": suffix.token_id,
        "multiscript": True,
        "quote_string": QUOTE_STRING,
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_tasks(
    tokenizer,
    pool: dict[str, list[TokenCandidate]],
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

    scripts = list(pool.keys())
    if len(scripts) < 2:
        raise RuntimeError(
            f"Need tokens from ≥2 scripts, but only found: {scripts}"
        )

    rng = random.Random(seed)
    seen_pairs: set[tuple[int, int]] = set()
    seen_decoded: set[str] = set()
    tasks: list[dict] = []
    attempts = 0

    while len(tasks) < n_tasks and attempts < max_attempts:
        attempts += 1

        # Pick two different scripts
        script_a, script_b = rng.sample(scripts, 2)

        # Randomly decide which is prefix vs suffix
        if rng.random() < 0.5:
            prefix = rng.choice(pool[script_a])
            suffix = rng.choice(pool[script_b])
        else:
            prefix = rng.choice(pool[script_b])
            suffix = rng.choice(pool[script_a])

        pair = (prefix.token_id, suffix.token_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        if not bigram_is_multiscript(prefix, suffix):
            continue

        task = build_task(tokenizer, prefix, suffix, quote_token_id)
        if task is None or task["decoded"] in seen_decoded:
            continue

        seen_decoded.add(task["decoded"])
        tasks.append(task)

    if len(tasks) < n_tasks:
        raise RuntimeError(
            f"Only generated {len(tasks)} tasks after {attempts} attempts; "
            f"requested {n_tasks}. Increase --max-attempts or relax filters."
        )

    return tasks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    random.seed(args.seed)

    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)

    print(f"Building candidate pool from {args.model} vocabulary...")
    pool = build_candidate_pool(tokenizer)

    total = sum(len(v) for v in pool.values())
    print(f"Found {total} candidate tokens across {len(pool)} scripts:")
    for script, candidates in sorted(pool.items(), key=lambda x: -len(x[1])):
        print(f"  {script}: {len(candidates)} tokens")

    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = max(args.n * 200, 50_000)

    tasks = sample_tasks(
        tokenizer=tokenizer,
        pool=pool,
        n_tasks=args.n,
        seed=args.seed,
        max_attempts=max_attempts,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(tasks)} multi-script bigram tasks to {out_path.resolve()}")

    if tasks:
        ex = tasks[0]
        print(
            f"Example: {ex['decoded']!r}  "
            f"scripts={ex['scripts']}  "
            f"tokens=[{ex['prefix']!r}, {ex['suffix']!r}]"
        )

    # Show script-pair distribution
    pair_counts: dict[str, int] = {}
    for t in tasks:
        key = f"{t['prefix_script']}+{t['suffix_script']}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    print("\nScript-pair distribution:")
    for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
        print(f"  {pair}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate random two-token multi-script bigrams. Each bigram "
            "contains characters from ≥2 Unicode scripts and is "
            "tokenization-stable (standalone and inside double quotes)."
        )
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out-path", default="multiscript_bigram_tasks.json")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=8)
    main(parser.parse_args())