"""
Smoke-test tokenization behavior for causal_scores.py across model families.

Reproduces the tok() closure from causal_scores.main and runs it through the
exact call sites (entity bucketing, pile_chunk, generate_seq_batch) so we can
verify that OLMo / Qwen / Pythia outputs match Llama's tokenization invariants
*before* running the full pipeline.

Run with:
    uv run python scripts/debug_tokenization.py --model allenai/Olmo-3-1025-7B
    uv run python scripts/debug_tokenization.py --model Qwen/Qwen3-8B
    uv run python scripts/debug_tokenization.py --model meta-llama/Llama-3.1-8B   # baseline
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer


def make_tok(tokenizer, name):
    def tok(s, bos=False):
        if "llama" in name.lower():
            if not bos:
                return tokenizer(s)["input_ids"][1:]
            return tokenizer(s)["input_ids"]
        if (
            "olmo" in name.lower()
            or "pythia" in name.lower()
            or "qwen" in name.lower()
        ):
            ids = tokenizer(s)["input_ids"]
            if not bos:
                return ids
            bos_id = tokenizer.bos_token_id
            return [bos_id] + ids if bos_id is not None else ids
        raise ValueError(f"Unsupported model family: {name}")
    return tok


def check_invariants(tokenizer, tok, name):
    bos_id = tokenizer.bos_token_id
    print(f"\n=== {name} ===")
    print(f"bos_token_id          = {bos_id}")
    print(f"add_bos_token (attr)  = {getattr(tokenizer, 'add_bos_token', '<absent>')}")
    print(f"raw tokenizer('')     = {tokenizer('')['input_ids']}")
    print(f"raw tokenizer(' New York') = {tokenizer(' New York')['input_ids']}")
    print(f"raw tokenizer('\\n')  = {tokenizer(chr(10))['input_ids']}")

    print(f"tok('', bos=True)     = {tok('', bos=True)}     (expect [BOS] of length 1)")
    print(f"tok('\\n', bos=False) = {tok(chr(10), bos=False)} (expect single newline id)")
    ids_no_bos = tok(" New York", bos=False)
    print(f"tok(' New York')      = {ids_no_bos} -> {[tokenizer.decode([i]) for i in ids_no_bos]}")

    # red flag: BOS leaking into bos=False
    if bos_id is not None and ids_no_bos and ids_no_bos[0] == bos_id:
        print("  ⚠ BOS LEAK: tok(s, bos=False) starts with BOS — entity bucketing will be wrong.")
    bos_only = tok("", bos=True)
    if bos_id is not None and len(bos_only) > 1:
        print("  ⚠ DOUBLE BOS: tok('', bos=True) longer than 1 — generate_seq_batch will inject [BOS, BOS, ...].")


def emulate_build_work_items(tok, entities):
    """Mirrors build_work_items lines 423-433: bucket entities by len(tok(ent))."""
    buckets = {2: [], 3: [], 4: [], 5: []}
    for ent in entities:
        toks = tok(ent)
        n = len(toks)
        if n in buckets:
            buckets[n].append((ent, toks))
    return buckets


def emulate_clean_prompt(tok, ent_toks, sequence_len=12):
    """Mirrors generate_seq_batch lines 120-127 with a deterministic 'pile' chunk."""
    newline = tok("\n", bos=False)[-1]
    bos_ids = tok("", bos=True)
    fake_pile_chunk = list(range(sequence_len - len(ent_toks)))  # deterministic placeholder
    ent_chunk_full = fake_pile_chunk + ent_toks
    ent_chunk_trunc = fake_pile_chunk + [ent_toks[0]]
    return bos_ids + ent_chunk_full + [newline] + ent_chunk_trunc


def bucket_counterfact(tok, csv_path):
    """Run the build_work_items entity-bucketing pass over the full CounterFact subjects."""
    subjects = list(pd.read_csv(csv_path)["subject"])
    lengths = Counter()
    sample_per_bucket = {2: [], 3: [], 4: [], 5: []}
    for ent in subjects:
        n = len(tok(ent))
        lengths[n] += 1
        if n in sample_per_bucket and len(sample_per_bucket[n]) < 3:
            sample_per_bucket[n].append(ent)
    return len(subjects), lengths, sample_per_bucket


def report_top_heads(model_name, head_orderings_root, num_hidden_layers, top_k=10):
    short = model_name.split("/")[-1]
    base = Path(head_orderings_root) / short
    print(f"\n--- top-{top_k} ranked heads for {short} ({base}) ---")
    for fname, label in [("concept_copying.json", "concept (m1 promote)"), ("token_copying.json", "token (m2 promote)")]:
        path = base / fname
        if not path.exists():
            print(f"  {label:<22} {fname} -> MISSING")
            continue
        ranking = json.load(open(path))
        top = ranking[:top_k]
        layers = [pair[0] for pair in top]
        layer_hist = Counter(layers)
        # bucket layers into early (0..1/3), mid (1/3..2/3), late (2/3..)
        third = num_hidden_layers / 3
        depth_bucket = Counter()
        for l in layers:
            if l < third:
                depth_bucket["early"] += 1
            elif l < 2 * third:
                depth_bucket["mid"] += 1
            else:
                depth_bucket["late"] += 1
        print(f"  {label:<22} top-{top_k}: {top}")
        print(f"    layer histogram (top-{top_k})    : {dict(sorted(layer_hist.items()))}")
        print(f"    depth buckets (n_layers={num_hidden_layers}): early<{int(third)} mid<{int(2*third)} late>= -> {dict(depth_bucket)}")


def report_config(name):
    cfg = AutoConfig.from_pretrained(name)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print(f"\n--- model config for {name} ---")
    print(f"  hidden_size            = {cfg.hidden_size}")
    print(f"  num_hidden_layers      = {cfg.num_hidden_layers}")
    print(f"  num_attention_heads    = {cfg.num_attention_heads}")
    print(f"  num_key_value_heads    = {getattr(cfg, 'num_key_value_heads', '<absent>')}")
    print(f"  head_dim               = {head_dim}")
    print(
        f"  o_proj.inputs expected = [bsz, seq, {cfg.hidden_size}]"
        f"  (n_heads * head_dim = {cfg.num_attention_heads * head_dim})"
    )


def report_cached_results(model_name, cache_root, sequence_len, n):
    # ChunkOutputSaver was pickled from causal_scores.__main__; expose it under
    # __main__ here so pickle.load can resolve it.
    import sys
    import causal_scores as cs
    sys.modules["__main__"].ChunkOutputSaver = cs.ChunkOutputSaver

    short = model_name.split("/")[-1]
    base = Path(cache_root) / short
    fname = f"len{sequence_len}_n{n}"
    print(f"\n--- cached causal_scores results for {short} ({base}) ---")
    for suffix, label in [("", "concept (counterfact)"), ("_randoments", "token (random)")]:
        pkl = base / f"{fname}{suffix}.pkl"
        if not pkl.exists():
            print(f"  {label:<22} {pkl.name} -> MISSING")
            continue
        with open(pkl, "rb") as f:
            clean, corrupt, patched = pickle.load(f)
        n_examples = clean.n
        clean_acc = clean.get_acc().item() if torch.is_tensor(clean.get_acc()) else float(clean.get_acc())
        corrupt_acc = corrupt.get_acc().item() if torch.is_tensor(corrupt.get_acc()) else float(corrupt.get_acc())
        m1_diff = patched.get_m1() - corrupt.get_m1()
        m2_diff = patched.get_m2() - corrupt.get_m2()
        # diff tensors are [n_layers * n_heads]; flat top-k summary
        top_m1 = torch.topk(m1_diff, k=5)
        top_m2 = torch.topk(m2_diff, k=5)
        print(f"  {label:<22} n={n_examples:5d}")
        print(f"    clean acc            = {clean_acc:.4f}")
        print(f"    corrupt acc          = {corrupt_acc:.4f}")
        print(f"    max(patched - corrupt) m1_prob = {m1_diff.max().item():+.4f}  (top-5 vals: {top_m1.values.tolist()})")
        print(f"    max(patched - corrupt) m2_prob = {m2_diff.max().item():+.4f}  (top-5 vals: {top_m2.values.tolist()})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--probe-entities",
        nargs="+",
        default=["New York", "iPhone", "Microsoft", "Eiffel Tower", "Cristiano Ronaldo"],
    )
    p.add_argument(
        "--counterfact-csv",
        default=str(Path(__file__).resolve().parents[1] / "data" / "counterfact_expanded.csv"),
    )
    p.add_argument(
        "--cache-root",
        default=str(Path(__file__).resolve().parents[1] / "cache" / "causal_scores"),
        help="Root of cached causal_scores results (per-model subdirs).",
    )
    p.add_argument(
        "--head-orderings-root",
        default=str(Path(__file__).resolve().parents[1] / "cache" / "head_orderings"),
        help="Root of cached head-ordering JSONs (per-model subdirs).",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--sequence-len", type=int, default=30)
    p.add_argument(
        "--skip-cached",
        action="store_true",
        help="Skip the cached-results comparison block.",
    )
    p.add_argument(
        "--skip-config",
        action="store_true",
        help="Skip the model-config block (avoid HF auth for gated repos).",
    )
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tok = make_tok(tokenizer, args.model)
    check_invariants(tokenizer, tok, args.model)

    print("\n--- entity bucketing (mirrors build_work_items) ---")
    buckets = emulate_build_work_items(tok, args.probe_entities)
    for n, items in buckets.items():
        if items:
            print(f"  len={n}: {items}")

    # build a fake clean prompt with a 2-token entity (or first nonempty bucket)
    bigrams = buckets[2]
    if not bigrams:
        print("  ⚠ no 2-token entities in probe set; pick longer entities or check BOS leak.")
    else:
        ent_str, ent_toks = bigrams[0]
        print(f"\n--- emulated clean prompt for entity {ent_str!r} ({ent_toks}) ---")
        seq = emulate_clean_prompt(tok, ent_toks, sequence_len=10)
        print(f"sequence ids ({len(seq)} toks): {seq}")
        print("decoded per-position:")
        for i, t in enumerate(seq):
            print(f"  [{i:3d}] {t:7d}  {tokenizer.decode([t])!r}")
        print(f"\nentities[:, 0] for stats_from_logits would be {ent_toks[0]} ({tokenizer.decode([ent_toks[0]])!r})")
        print(f"entities[:, 1] for stats_from_logits would be {ent_toks[1]} ({tokenizer.decode([ent_toks[1]])!r})")

    csv_path = Path(args.counterfact_csv)
    if not csv_path.exists():
        print(f"\n⚠ counterfact CSV not found at {csv_path}; skipping full-bucket pass.")
        return

    print(f"\n--- full counterfact_expanded.csv bucketing for {args.model} ---")
    total, lengths, samples = bucket_counterfact(tok, csv_path)
    print(f"total subjects: {total}")
    for n in sorted(lengths):
        marker = " ← used by build_work_items" if n in (2, 3, 4, 5) else ""
        print(f"  len={n:2d}: {lengths[n]:5d} entities{marker}")
    print("samples per bucket used by causal_scores:")
    for n in (2, 3, 4, 5):
        print(f"  len={n}: {samples.get(n, [])}")

    cfg_num_layers = None
    if not args.skip_config:
        cfg = AutoConfig.from_pretrained(args.model)
        cfg_num_layers = cfg.num_hidden_layers
        report_config(args.model)

    if not args.skip_cached:
        report_cached_results(args.model, args.cache_root, args.sequence_len, args.n)
        # Need num_hidden_layers for the depth bucketing; fall back to a config read.
        if cfg_num_layers is None:
            cfg_num_layers = AutoConfig.from_pretrained(args.model).num_hidden_layers
        report_top_heads(args.model, args.head_orderings_root, cfg_num_layers, args.top_k)


if __name__ == "__main__":
    main()
