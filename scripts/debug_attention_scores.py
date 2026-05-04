"""Verify the attention-score extractors in `utils.py` against HF's own
`output_attentions=True` forward pass.

The `attention_scores.py` driver computes per-head attention weights via
custom extractors (`get_l3_attn_weights`, `get_olmo2_attn_weights`,
`get_qwen3_attn_weights`). Llama-3.1-8B shows induction heads with
next_tok_attn ≈ 0.98 while OLMo-3 / Qwen3 cap out near 0.3-0.4. Before
attributing this to architecture (q_norm / GQA), we want to rule out
extractor bugs.

For each model:
  1. Build a repetition prompt:
        [bos?] foo bar quack emmy plink doer Ed.mont.on \n foo bar quack emmy plink doer
  2. Run a single forward pass with `attn_implementation="eager"` and
     `output_attentions=True`. Save HF's per-layer attention matrices.
  3. Run the custom `retrieve_attention(...)` extractor with
     `value_weighting=False` so the comparison stays at the raw softmax
     level. (Value-weighting is a post-hoc reweight that doesn't
     change which heads dominate.)
  4. For each layer / head, compare the final-row attention vector
     (i.e. attn[..., -1, :]) between HF and the custom extractor.
  5. Print max / mean abs diff per layer, plus the top heads by HF
     `attn[-1, start_idx]` so we can see whether the induction signal
     exists at all in the raw HF attentions.

Usage:
    cd dual-route-induction/scripts
    HF_HOME=/work/nvme/bfzp/hf_cache python debug_attention_scores.py \
        --model meta-llama/Llama-3.1-8B
    HF_HOME=/work/nvme/bfzp/hf_cache python debug_attention_scores.py \
        --model allenai/Olmo-3-1025-7B
    HF_HOME=/work/nvme/bfzp/hf_cache python debug_attention_scores.py \
        --model Qwen/Qwen3-8B
"""

import argparse
import os
import random
import sys
import torch
import numpy as np
from typing import List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel

import utils
from seed_utils import set_random_seed


def build_prompt_ids(tokenizer, model_name: str, entity_str: str,
                     prefix_str: str, sequence_len: int = 30,
                     ) -> Tuple[List[int], int, int]:
    """Replicate the structure used by `attention_scores.generate_ragged_batch`.

    Returns (input_ids, start_idx, end_idx) where:
      start_idx points at the first token AFTER the entity in block 1
                (an induction head should attend here from the last token).
      end_idx   points at the LAST token of the entity in block 1
                (a chunk-skip head should attend here).
    """
    # Tokenize entity (no leading space — entity sits mid-document).
    ent_ids = tokenizer(entity_str, add_special_tokens=False)["input_ids"]
    pre_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
    newline_id = tokenizer("\n", add_special_tokens=False)["input_ids"][-1]

    # Place entity so that block 1 length = position + len(ent), then
    # block 2 = pre_ids only (truncated/padded to match prefix length).
    # We just use a simple layout: prefix + entity + \n + prefix.
    seq = []
    if "llama" in model_name.lower():
        # Llama tokenizer has bos but add_bos=False for raw call. We add it.
        seq.append(tokenizer.bos_token_id)
    elif tokenizer.bos_token_id is not None:
        seq.append(tokenizer.bos_token_id)

    block1_start = len(seq)
    seq.extend(pre_ids)
    ent_start_in_seq = len(seq)
    seq.extend(ent_ids)
    ent_end_in_seq = len(seq) - 1
    seq.append(newline_id)
    seq.extend(pre_ids)

    # In `attention_scores.py`, start_idxs = position + 1 where
    # `position` is the index of the FIRST entity token; this points at
    # the SECOND token of the entity (i.e. the "next" token to attend
    # to, given that the second block ends right where the entity began
    # in block 1). Adopting the same convention here.
    start_idx = ent_start_in_seq + 1
    end_idx = ent_end_in_seq

    return seq, start_idx, end_idx


def hf_attentions(model_name: str, input_ids: List[int]) -> torch.Tensor:
    """Forward pass with `output_attentions=True` and eager attention.

    Returns a tensor of shape [n_layers, n_heads, seq, seq].
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    ids = torch.tensor([input_ids], device="cuda")
    with torch.no_grad():
        out = model(ids, output_attentions=True, use_cache=False)
    # tuple of len n_layers, each [bsz=1, n_heads, seq, seq]
    attns = torch.stack([a[0].float().cpu() for a in out.attentions], dim=0)
    del model
    torch.cuda.empty_cache()
    return attns  # [L, H, S, S]


def custom_attentions(model_name: str, input_ids: List[int]) -> torch.Tensor:
    """Run the custom extractor for every layer with value_weighting=False.

    The extractors assume a non-None `attention_mask` from
    `self_attn.inputs[1]["attention_mask"]`. For a single un-padded
    sequence HF passes None. We force the mask path by giving nnsight
    a pre-padded batch of two identical-length sequences (one is a
    pad-truncation of the other) so the tokenizer.pad() call attaches a
    real mask. We then drop the dummy row before returning.
    """
    model = LanguageModel(model_name, device_map="cuda")
    n_layers = model.config.num_hidden_layers
    # Add a 1-token-shorter twin so tokenizer.pad() returns a real mask.
    twin = list(input_ids[:-1])  # one shorter, will be padded by 1
    batch = [list(input_ids), twin]

    # The retrieve_attention dispatcher in attention_scores.py:
    name = model.config._name_or_path
    table = {
        "meta-llama/Llama-3.1-8B": utils.get_l3_attn_weights,
        "allenai/Olmo-3-1025-7B": utils.get_olmo2_attn_weights,
        "allenai/OLMo-2-1124-7B": utils.get_olmo2_attn_weights,
    }
    if name in table:
        fn = table[name]
    elif "qwen" in name.lower():
        fn = utils.get_qwen3_attn_weights
    elif "llama" in name.lower():
        fn = utils.get_l3_attn_weights
    else:
        raise KeyError(name)

    # Tokenizer pads on the LEFT for some configs and RIGHT for others;
    # the script's driver uses default (right) but we make it explicit
    # so downstream slicing is predictable.
    model.tokenizer.padding_side = "right"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    out_layers = []
    for layer in range(n_layers):
        a = fn(model, batch, layer, value_weighting=False)  # [bsz, H, S, S]
        # Real sequence is row 0; full length matches input_ids exactly.
        out_layers.append(a[0, :, : len(input_ids), : len(input_ids)].float())
    del model
    torch.cuda.empty_cache()
    return torch.stack(out_layers, dim=0)  # [L, H, S, S]


def compare(name: str, hf: torch.Tensor, ours: torch.Tensor,
            start_idx: int, end_idx: int, top_k: int = 10):
    print(f"\n[{name}] HF vs custom — shapes {tuple(hf.shape)} vs {tuple(ours.shape)}")
    if hf.shape != ours.shape:
        print(f"  ⚠  shape mismatch — cannot compare directly")
        return
    diff = (hf - ours).abs()
    print(f"  abs-diff (all):      max={diff.max().item():.4e}  mean={diff.mean().item():.4e}")
    # Focus on the LAST row (the query position used by attention_scores.py).
    diff_last = (hf[..., -1, :] - ours[..., -1, :]).abs()
    print(f"  abs-diff (last row): max={diff_last.max().item():.4e}  mean={diff_last.mean().item():.4e}")

    # Top heads by HF attention[last, start_idx] (= "next_tok_attn"):
    L, H = hf.shape[0], hf.shape[1]
    next_attn_hf = hf[..., -1, start_idx].reshape(-1)
    end_attn_hf  = hf[..., -1, end_idx].reshape(-1)
    next_attn_us = ours[..., -1, start_idx].reshape(-1)
    end_attn_us  = ours[..., -1, end_idx].reshape(-1)

    print(f"  HF max next/end:     {next_attn_hf.max().item():.4f} / {end_attn_hf.max().item():.4f}")
    print(f"  US max next/end:     {next_attn_us.max().item():.4f} / {end_attn_us.max().item():.4f}")

    # Top-K induction heads per HF
    topk = torch.topk(next_attn_hf, top_k)
    print(f"  Top-{top_k} induction heads (HF, attn[last->start_idx]):")
    for v, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        l, h = idx // H, idx % H
        v_us = next_attn_us[idx].item()
        print(f"    L{l:2d}.H{h:2d}  hf={v:.4f}  us={v_us:.4f}")

    # Sanity: row sums for both
    row_sum_hf = hf[..., -1, :].sum(-1)
    row_sum_us = ours[..., -1, :].sum(-1)
    print(f"  Row-sum sanity HF: min={row_sum_hf.min():.4f} max={row_sum_hf.max():.4f}")
    print(f"  Row-sum sanity US: min={row_sum_us.min():.4f} max={row_sum_us.max():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--entity", default="Edmonton")
    parser.add_argument("--prefix", default="foo bar quack emmy plink doer")
    parser.add_argument("--seed", default=8, type=int)
    parser.add_argument("--top-k", default=10, type=int)
    args = parser.parse_args()

    set_random_seed(args.seed)
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids, start_idx, end_idx = build_prompt_ids(
        tokenizer, args.model, args.entity, args.prefix
    )
    print(f"Model: {args.model}")
    print(f"Prompt length: {len(ids)}, start_idx={start_idx} end_idx={end_idx}")
    print(f"  start tok: {tokenizer.decode([ids[start_idx]])!r}")
    print(f"  end tok:   {tokenizer.decode([ids[end_idx]])!r}")
    print(f"  full:      {tokenizer.decode(ids)!r}")

    print("\nRunning HF eager forward (output_attentions=True)...")
    hf = hf_attentions(args.model, ids)
    print("Running custom extractor over all layers...")
    ours = custom_attentions(args.model, ids)
    compare(args.model, hf, ours, start_idx, end_idx, top_k=args.top_k)


if __name__ == "__main__":
    main()
