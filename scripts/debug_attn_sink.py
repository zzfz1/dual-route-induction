"""Two-in-one probe for the cached `attention_scores.json` puzzle.

Hypothesis 1 — off-by-one indexing
  In `attention_scores.py`:
      sequences = bos_ids + rand1 + ent + rand2 + \n + rand1
      start_idxs = position + 1
      end_idxs   = position + len(ent)
  The intended targets are ent[0] ("next-token induction") and
  ent[-1] ("chunk-skip"). Those land at sequence indices
  len(bos_ids)+position and len(bos_ids)+position+len(ent)-1. So
  start_idxs / end_idxs are correct iff len(bos_ids)==1.
    - Llama-3.1-8B: bos auto-prepended (1 token)  → correct
    - OLMo-3-1025-7B / Qwen3-8B: no BOS (0 tokens) → off by ONE
  We probe by reporting attention at BOTH the "script" position and
  the "corrected" position so we can see if the bug masks the real
  induction signal.

Hypothesis 2 — attention sink
  Maybe q_norm / k_norm in OLMo-3 / Qwen3 makes the first real token
  (or position 0) absorb a lot of mass even on heads that nominally
  do induction. We log attention to the first real token alongside
  the targets.

Usage:
    python debug_attn_sink.py --model meta-llama/Llama-3.1-8B
    python debug_attn_sink.py --model allenai/Olmo-3-1025-7B
    python debug_attn_sink.py --model Qwen/Qwen3-8B
"""

import argparse
import os
import random
import torch
import pandas as pd
from collections import defaultdict
from datasets import load_dataset
from nnsight import LanguageModel

import utils
from utils import pile_chunk
from seed_utils import set_random_seed

# Cached top-1 induction head per model (next_tok_attn, n2048_seqlen30):
TOP_INDUCTION = {
    "meta-llama/Llama-3.1-8B": (15, 30),
    "allenai/Olmo-3-1025-7B": (13, 4),
    "Qwen/Qwen3-8B": (20, 21),
}


def make_tok(model):
    name = model.config._name_or_path

    def tok(s, bos=False, pad_mask=False):
        if pad_mask:
            return model.tokenizer.pad(
                {"input_ids": s}, return_tensors="pt"
            )["attention_mask"]
        if "llama" in name.lower():
            ids = model.tokenizer(s)["input_ids"]
            return ids if bos else ids[1:]
        ids = model.tokenizer(s, add_special_tokens=False)["input_ids"]
        if not bos:
            return ids
        bos_id = model.tokenizer.bos_token_id
        return [bos_id] + ids if bos_id is not None else ids

    return tok


def get_extractor(model):
    name = model.config._name_or_path
    table = {
        "meta-llama/Llama-3.1-8B": utils.get_l3_attn_weights,
        "allenai/Olmo-3-1025-7B": utils.get_olmo2_attn_weights,
    }
    if name in table:
        return table[name]
    if "qwen" in name.lower():
        return utils.get_qwen3_attn_weights
    raise KeyError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", default=64, type=int)
    p.add_argument("--bsz", default=8, type=int)
    p.add_argument("--seq-len", default=30, type=int)
    p.add_argument("--seed", default=8, type=int)
    args = p.parse_args()

    set_random_seed(args.seed)
    torch.set_grad_enabled(False)

    if args.model not in TOP_INDUCTION:
        raise SystemExit(f"No cached top induction head for {args.model}")
    L, H = TOP_INDUCTION[args.model]

    model = LanguageModel(args.model, device_map="cuda")
    fn = get_extractor(model)
    tok = make_tok(model)

    pile = load_dataset("JeanKaddour/minipile")["test"]
    str_entities = list(pd.read_csv("../data/counterfact_expanded.csv")["subject"])
    bigrams = []
    for ent in str_entities:
        ids = tok(ent)
        if len(ids) == 2:
            bigrams.append(ids)
        if len(bigrams) >= args.n:
            break
    print(f"[{args.model}] using top induction head L{L}.H{H}")
    print(f"  collected {len(bigrams)} bigram entities")

    bos_ids = tok("", bos=True)
    newline = tok("\n", bos=False)[-1]
    n_batches = len(bigrams) // args.bsz

    # Collect: for each example, attention from last position
    # to (start_idx, end_idx, pos 0, pos 1, sum elsewhere, total).
    rows = []
    for bi in range(n_batches):
        batch_ents = bigrams[bi * args.bsz : (bi + 1) * args.bsz]
        sequences, start_idxs, end_idxs = [], [], []
        for ent in batch_ents:
            position = random.choice(
                range(args.seq_len // 2, args.seq_len - len(ent) + 1)
            )
            rand1 = pile_chunk(position, pile, tok)
            rand2 = pile_chunk(args.seq_len - position - len(ent), pile, tok)
            seq = bos_ids + rand1 + ent + rand2 + [newline] + rand1
            # start_idx = position of (ent[0]+1) in unpadded sequence,
            # adjusted by len(bos_ids) since the script counts position
            # from the start of `bos_ids + rand1 + ...`.
            start_idxs.append(len(bos_ids) + position + 1)
            end_idxs.append(len(bos_ids) + position + len(ent))
            sequences.append(seq)

        # Same logic the driver uses for left padding:
        masks = tok(sequences, pad_mask=True)
        flipped_masks = [m - 1 for m in masks]
        pad_offsets = torch.tensor([-sum(f).item() for f in flipped_masks])

        # Per-example bookkeeping. We need the position of ent[0] and
        # ent[-1] in the UNPADDED sequence — that's what we actually want
        # to measure. The script gets these right only when
        # len(bos_ids) == 1.
        bos_len = len(bos_ids)
        ent_first_unpad = []
        ent_last_unpad = []
        script_next = []  # what the driver actually queries
        script_end = []
        for i, ent in enumerate(batch_ents):
            position_i = (
                # recover from start_idxs (which we set with +bos_len above)
                start_idxs[i] - bos_len - 1
            )
            ent_first_unpad.append(bos_len + position_i)            # corrected
            ent_last_unpad.append(bos_len + position_i + len(ent) - 1)  # corrected
            script_next.append(position_i + 1)                       # ← driver
            script_end.append(position_i + len(ent))                 # ← driver

        attn = fn(model, sequences, L, value_weighting=False)  # [bsz, H, S, S]
        head_attn = attn[:, H]  # [bsz, S, S]
        last_row = head_attn[:, -1]  # [bsz, S]
        bsz_i = last_row.shape[0]

        for i in range(bsz_i):
            row = last_row[i]  # [S]
            n_pad = pad_offsets[i].item()
            S = row.shape[0]
            real_first = n_pad  # left padding → real seq starts here
            row_sum = row.sum().item()

            def at(unpad_idx):
                p = unpad_idx + n_pad
                return row[p].item() if 0 <= p < S else 0.0

            rows.append(
                {
                    "first_real": at(0),
                    "next_corrected": at(ent_first_unpad[i]),
                    "end_corrected": at(ent_last_unpad[i]),
                    "next_script": at(script_next[i]),
                    "end_script": at(script_end[i]),
                    "pad": row[:real_first].sum().item(),
                    "row_sum": row_sum,
                }
            )

    df = pd.DataFrame(rows)
    print(f"\n[{args.model}] L{L}.H{H} bos_len={len(bos_ids)} — over {len(df)} examples")
    print(f"  Compare 'next_script' (driver) vs 'next_corrected' (ent[0]).")
    print(f"  If bos_len != 1 they will disagree by one position.\n")
    for col in ["first_real", "next_corrected", "end_corrected",
                "next_script", "end_script", "pad", "row_sum"]:
        s = df[col]
        print(f"  {col:16s} mean={s.mean():.4f}  median={s.median():.4f}"
              f"  max={s.max():.4f}  >0.5: {(s>0.5).sum()}")

    print("\n  Per-example (first 12):")
    print(df.head(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
