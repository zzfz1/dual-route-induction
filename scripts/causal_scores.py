"""
Patch outputs of individual heads across examples at the -2 position,
and save how much each head:
    - promotes `Ed` at the -2 position (logit diff, prob. diff)
    - promotes `mont` at the -1 position (logit diff, prob. diff)

Clean: foo bar [qux] Ed.mont.on\nfoo bar [qux] Ed
Corr:  elf min Apple MININGquee\nfoo bar [qux] Ed

You can also replace CounterFact entities with random tokens using `--random_tok_entities`
Then, run `convert_causal_scores.py` to convert the resulting .pkl files into readable json scores.
"""

import os
import json
import pickle
import argparse
import random
import sys
import threading
import time
import torch
import numpy as np
import pandas as pd
from random import shuffle
from tqdm import tqdm
from collections import defaultdict
from nnsight import LanguageModel
from datasets import load_dataset

import utils
from ndif import load_remote_model
from utils import pile_chunk, json_tuple_keys, flatidx_to_grididx

# Empirically, NDIF can sustain 16-example remote microbatches with 5 patched heads
# per trace on Llama-3.1-8B; larger settings such as 16x6 or 32x3 hit OOM on later layers.
REMOTE_MAX_BATCH_SIZE = 16
REMOTE_PATCH_BATCH_TARGET = 80


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except ValueError:
                pass


configure_utf8_stdio()


def is_remote_transport_error(exc):
    text_markers = (
        "Server disconnected without sending a response",
        "RemoteProtocolError",
        "Bad Gateway",
        "WriteTimeout",
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectionError",
        "DisconnectedError",
        "socket closed",
        "timed out",
    )
    class_markers = {
        "RemoteProtocolError",
        "WriteTimeout",
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectionError",
        "DisconnectedError",
    }

    stack = [exc]
    seen = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        current_text = f"{type(current).__name__}: {current}"
        if type(current).__name__ in class_markers:
            return True
        if any(marker in current_text for marker in text_markers):
            return True

        stack.extend(
            [getattr(current, "__cause__", None), getattr(current, "__context__", None)]
        )

    return False


class RemoteExecutionContext:
    def __init__(
        self,
        model_name,
        max_retries=4,
        backoff_base=2.0,
        backoff_max=30.0,
    ):
        self.model_name = model_name
        self.max_retries = max(1, max_retries)
        self.backoff_base = float(backoff_base)
        self.backoff_max = float(backoff_max)
        self._local = threading.local()

    def reset_model(self):
        self._local.model = None

    def get_model(self):
        model = getattr(self._local, "model", None)
        if model is None:
            model = load_remote_model(self.model_name, utils)
            model._ndif_remote = True
            self._local.model = model
        return model

    def request(self, label, fn):
        attempt = 1
        while True:
            try:
                return fn(self.get_model())
            except Exception as exc:
                if not is_remote_transport_error(exc) or attempt >= self.max_retries:
                    raise

                delay = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)
                self.reset_model()
                print(
                    f"NDIF transport error during {label} "
                    f"(attempt {attempt}/{self.max_retries}): {type(exc).__name__}; "
                    f"retrying in {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
                attempt += 1


def flat_to_dict(flattensor, n_heads=32):
    d = {}
    for idx in range(len(flattensor)):
        d[flatidx_to_grididx(idx, n_heads)] = flattensor[idx].item()
    return d


def flat_to_ranking(flattensor, n_heads=32):
    _, idxs = torch.topk(flattensor, k=len(flattensor))
    return [flatidx_to_grididx(i, n_heads) for i in idxs]


def get_scoretype(args):
    return "token" if args.random_tok_entities else "concept"


def get_output_paths(args):
    model_name = args.model.split("/")[-1]
    scoretype = get_scoretype(args)
    fname = f"len{args.sequence_len}_n{args.n}"
    token_suffix = "_randoments" if args.random_tok_entities else ""
    shard_suffix = (
        f"_shard{args.work_shard_index}of{args.work_shard_count}"
        if args.work_shard_count > 1
        else ""
    )

    score_dir = f"../cache/causal_scores/{model_name}/"
    score_dir += f"{args.ckpt}/" if args.ckpt is not None else ""

    rank_dir = f"../cache/head_orderings/{model_name}/"
    rank_dir += f"{args.ckpt}/" if args.ckpt is not None else ""

    ranking_name = f"{scoretype}_copying"
    ranking_name += (
        f"_{fname}.json" if (args.n != 1024 or args.sequence_len != 30) else ".json"
    )

    return {
        "model_name": model_name,
        "scoretype": scoretype,
        "fname": fname,
        "score_dir": score_dir,
        "rank_dir": rank_dir,
        "results_pkl": score_dir + f"{fname}{token_suffix}{shard_suffix}.pkl",
        "scores_json": score_dir + f"{scoretype}_copying_{fname}.json",
        "ranking_json": rank_dir + ranking_name,
        "resume_pkl": score_dir + f"{fname}{token_suffix}{shard_suffix}_resume.pkl",
        "shared_work_items_pkl": score_dir + f"{fname}{token_suffix}_workitems.pkl",
        "base_results_pkl": score_dir + f"{fname}{token_suffix}.pkl",
    }


def get_run_config(args):
    return {
        "model": args.model,
        "ckpt": args.ckpt,
        "n": args.n,
        "bsz": args.bsz,
        "sequence_len": args.sequence_len,
        "remote": args.remote,
        "random_tok_entities": args.random_tok_entities,
        "work_shard_index": args.work_shard_index,
        "work_shard_count": args.work_shard_count,
    }


def atomic_pickle_dump(obj, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def save_resume_state(path, args, work_items, results, next_work_idx):
    state = {
        "version": 1,
        "run_config": get_run_config(args),
        "work_items": work_items,
        "results": results,
        "next_work_idx": next_work_idx,
    }
    atomic_pickle_dump(state, path)


def load_resume_state(path, args):
    with open(path, "rb") as f:
        state = pickle.load(f)

    if state.get("run_config") != get_run_config(args):
        raise ValueError(
            f"Resume state at {path} does not match the current run configuration."
        )

    return state


class ChunkOutputSaver:
    def __init__(self, name, n_heads):
        self.name = name
        self.n_heads = n_heads
        self.n = 0
        self.correct = torch.zeros(n_heads)  # specifically it's whether m1 is correct
        self.m1_prob = torch.zeros(n_heads)
        self.m2_prob = torch.zeros(n_heads)
        self.m1_logit = torch.zeros(n_heads)
        self.m2_logit = torch.zeros(n_heads)

    def __repr__(self):
        return f"ChunkOutputSaver-{self.name}-heads{self.n_heads}"

    def update(self, correct, m1_prob, m2_prob, m1_logit, m2_logit):
        """
        Each input array should be shape (bsz, n_heads)
        """
        assert len(correct) == len(m2_prob)
        assert len(m2_prob) == len(m1_prob)
        self.n += len(correct)
        self.correct += correct.sum(dim=0)
        self.m1_prob += m1_prob.sum(dim=0)
        self.m2_prob += m2_prob.sum(dim=0)
        self.m1_logit += m1_logit.sum(dim=0)
        self.m2_logit += m2_logit.sum(dim=0)

    def update_sums(self, count, correct, m1_prob, m2_prob, m1_logit, m2_logit):
        """
        Each input array should be shape (n_heads,)
        """
        tensors = [
            torch.as_tensor(x, dtype=torch.float32).reshape(-1).cpu()
            for x in (correct, m1_prob, m2_prob, m1_logit, m2_logit)
        ]
        assert all(t.shape == (self.n_heads,) for t in tensors)

        self.n += count
        self.correct += tensors[0]
        self.m1_prob += tensors[1]
        self.m2_prob += tensors[2]
        self.m1_logit += tensors[3]
        self.m2_logit += tensors[4]

    def get_acc(self):
        return self.correct / self.n

    def get_m1(self):
        return self.m1_prob / self.n

    def get_m2(self):
        return self.m2_prob / self.n

    def get_m1_logit(self):
        return self.m1_logit / self.n

    def get_m2_logit(self):
        return self.m2_logit / self.n


def clone_chunk_output_saver(saver):
    cloned = ChunkOutputSaver(saver.name, saver.n_heads)
    cloned.n = saver.n
    cloned.correct = saver.correct.clone()
    cloned.m1_prob = saver.m1_prob.clone()
    cloned.m2_prob = saver.m2_prob.clone()
    cloned.m1_logit = saver.m1_logit.clone()
    cloned.m2_logit = saver.m2_logit.clone()
    return cloned


def merge_chunk_output_savers(savers):
    if not savers:
        raise ValueError("Expected at least one ChunkOutputSaver to merge.")

    merged = clone_chunk_output_saver(savers[0])
    for saver in savers[1:]:
        merged.n += saver.n
        merged.correct += saver.correct
        merged.m1_prob += saver.m1_prob
        merged.m2_prob += saver.m2_prob
        merged.m1_logit += saver.m1_logit
        merged.m2_logit += saver.m2_logit
    return merged


def merge_result_sets(result_sets):
    if not result_sets:
        raise ValueError("Expected at least one result set to merge.")
    return [
        merge_chunk_output_savers([results[idx] for results in result_sets])
        for idx in range(len(result_sets[0]))
    ]


# generate clean and corrupted prompts
def generate_seq_batch(entities, pile, tok, sequence_len):
    clean, corrupt = [], []
    for ent in entities:
        rand = pile_chunk(sequence_len - len(ent), pile, tok, shuf_pile=True)
        ent_chunk_full = rand + ent
        ent_chunk_trunc = rand + [ent[0]]
        clean_prompt = [1] + ent_chunk_full + [13] + ent_chunk_trunc

        rand2 = pile_chunk(sequence_len, pile, tok, shuf_pile=True)
        corrupt_prompt = [1] + rand2 + [13] + ent_chunk_trunc
        # print(repr(tokenizer.decode(clean_prompt)))
        # print(repr(tokenizer.decode(corrupt_prompt)))
        clean.append(clean_prompt)
        corrupt.append(corrupt_prompt)
    return clean, corrupt


def is_remote_model(model):
    return getattr(model, "_ndif_remote", False)


def get_o_proj_inputs(model, layer):
    if model.config._name_or_path == "EleutherAI/pythia-6.9b":
        return model.gpt_neox.layers[layer].attention.dense.inputs
    return model.model.layers[layer].self_attn.o_proj.inputs


def set_o_proj_inputs(model, layer, new_tup):
    if model.config._name_or_path == "EleutherAI/pythia-6.9b":
        model.gpt_neox.layers[layer].attention.dense.inputs = new_tup
    else:
        model.model.layers[layer].self_attn.o_proj.inputs = new_tup


def get_o_proj_input_tensor(model, layer):
    return get_o_proj_inputs(model, layer)[0][0]


def get_remote_head_chunk_size(batch_size, heads_per_layer):
    return max(
        1,
        min(heads_per_layer, REMOTE_PATCH_BATCH_TARGET // max(1, batch_size)),
    )


def inference_logits(model, sequences):
    with torch.no_grad():
        with model.trace(
            sequences,
            remote=is_remote_model(model)
            and not getattr(model, "_ndif_session_active", False),
        ):
            logits = model.output.logits.save()
    return logits.detach().cpu()


def stats_from_logits(logits, entities):
    if not torch.is_tensor(entities):
        entities = torch.tensor(entities, dtype=torch.long, device=logits.device)
    else:
        entities = entities.to(logits.device)

    # trying to predict the second token of the entity
    batch = torch.arange(entities.shape[0], device=logits.device)
    last_logits = logits[:, -1, :]
    prev_logits = logits[:, -2, :]

    correct = (last_logits.argmax(dim=-1) == entities[:, 1]).float()
    m1_logits = last_logits[batch, entities[:, 1]]
    m2_logits = prev_logits[batch, entities[:, 0]]
    m1_probs = (m1_logits - last_logits.logsumexp(dim=-1)).exp()
    m2_probs = (m2_logits - prev_logits.logsumexp(dim=-1)).exp()

    return correct, m1_probs, m2_probs, m1_logits, m2_logits


def sum_stats(stats, n_groups=None):
    if n_groups is None:
        return tuple(stat.sum() for stat in stats)

    group_size = stats[0].shape[0] // n_groups
    return tuple(stat.reshape(n_groups, group_size).sum(dim=1) for stat in stats)


def saved_to_cpu(saved):
    tensor = saved.detach().cpu()
    if tensor.ndim == 0:
        return tensor.reshape(1)
    return tensor


def add_stats(accum, stats):
    stats = tuple(torch.as_tensor(stat, dtype=torch.float32).cpu() for stat in stats)
    if accum is None:
        return tuple(stat.clone() for stat in stats)
    return tuple(a + b for a, b in zip(accum, stats))


def stats_from_saved_components(saved_stats, entities):
    if not torch.is_tensor(entities):
        entities = torch.tensor(entities, dtype=torch.long)
    else:
        entities = entities.to(dtype=torch.long)

    pred_ids, m1_logits, m2_logits, last_lse, prev_lse = saved_stats
    entities = entities.to(pred_ids.device)
    pred_ids = pred_ids.to(dtype=torch.long).reshape(-1)
    m1_logits = m1_logits.to(dtype=torch.float32).reshape(-1)
    m2_logits = m2_logits.to(dtype=torch.float32).reshape(-1)
    last_lse = last_lse.to(dtype=torch.float32).reshape(-1)
    prev_lse = prev_lse.to(dtype=torch.float32).reshape(-1)

    correct = (pred_ids == entities[:, 1]).float()
    m1_probs = (m1_logits - last_lse).exp()
    m2_probs = (m2_logits - prev_lse).exp()
    return correct, m1_probs, m2_probs, m1_logits, m2_logits


def _remote_sum_stats_request(model, sequences, entities):
    with torch.no_grad():
        with model.trace(
            sequences,
            remote=is_remote_model(model)
            and not getattr(model, "_ndif_session_active", False),
        ):
            logits = model.output.logits[:, -2:, :]
            ent = entities.to(logits.device)
            batch = torch.arange(ent.shape[0], device=logits.device)
            last_logits = logits[:, -1, :]
            prev_logits = logits[:, -2, :]
            pred_ids_saved = last_logits.argmax(dim=-1).save()
            m1_logits_saved = last_logits[batch, ent[:, 1]].save()
            m2_logits_saved = prev_logits[batch, ent[:, 0]].save()
            last_lse_saved = last_logits.logsumexp(dim=-1).save()
            prev_lse_saved = prev_logits.logsumexp(dim=-1).save()

    saved_stats_cpu = tuple(
        saved_to_cpu(saved)
        for saved in (
            pred_ids_saved,
            m1_logits_saved,
            m2_logits_saved,
            last_lse_saved,
            prev_lse_saved,
        )
    )
    return sum_stats(stats_from_saved_components(saved_stats_cpu, entities))


def remote_sum_stats(model, sequences, entities, remote_ctx=None, label=None):
    if remote_ctx is None:
        return _remote_sum_stats_request(model, sequences, entities)

    return remote_ctx.request(
        label or f"sum_stats_b{len(sequences)}",
        lambda trace_model: _remote_sum_stats_request(trace_model, sequences, entities),
    )


def is_remote_oom(exc):
    msg = str(exc)
    return "OutOfMemoryError" in msg or "CUDA out of memory" in msg


def no_patching(model, sequences, entities):
    logits = inference_logits(model, sequences)
    return stats_from_logits(logits, entities)


def _get_head_activations_request(model, prompt, layer):
    with torch.no_grad():
        with model.trace(
            prompt,
            remote=is_remote_model(model)
            and not getattr(model, "_ndif_session_active", False),
        ):
            o_proj_inp = get_o_proj_input_tensor(model, layer).save()

        # [bsz, seq_len, model_dim] -> [bsz, seq_len, n_heads, head_dim]
        heads_per_layer = model.config.num_attention_heads
        head_dim = model.config.hidden_size // heads_per_layer
        return o_proj_inp.view(*o_proj_inp.shape[:-1], heads_per_layer, head_dim)


def get_head_activations(model, prompt, layer, remote_ctx=None, label=None):
    if remote_ctx is None:
        return _get_head_activations_request(model, prompt, layer)

    return remote_ctx.request(
        label or f"clean_heads_l{layer}_b{len(prompt)}",
        lambda trace_model: _get_head_activations_request(trace_model, prompt, layer),
    )


def _remote_patch_chunk_stats_request(
    model,
    layer,
    corr_seq,
    entities,
    clean_heads,
    batch_size,
    heads_per_layer,
    head_dim,
    chunk_heads,
):
    chunk_size = chunk_heads.shape[0]
    expanded_corr_seq = corr_seq * chunk_size
    expanded_entities = entities.repeat(chunk_size, 1)

    with torch.no_grad():
        with model.trace(
            expanded_corr_seq,
            remote=is_remote_model(model)
            and not getattr(model, "_ndif_session_active", False),
        ):
            # Grab the traced input tuple for this layer's output projection.
            tup = get_o_proj_inputs(model, layer)
            # The first tensor in that tuple is the actual o_proj input we will patch.
            original = tup[0][0]
            original_shape = original.shape
            chunk_batch = original_shape[0]  # should be equal to batch_size * chunk_size.

            # View model_dim as [n_heads, head_dim] so we can patch a specific head slice.
            sub = original.view(
                chunk_batch, original_shape[1], heads_per_layer, head_dim
            )
            # Row indices over the expanded batch dimension.
            batch_idx = torch.arange(chunk_batch, device=original.device)
            # For each repeated example, choose which head in this chunk should be patched.
            head_idx = chunk_heads.to(original.device).repeat_interleave(batch_size)
            patch_values = (
                clean_heads[:, chunk_heads, :]
                .transpose(0, 1)
                .reshape(chunk_batch, head_dim)
                .to(original.device)
            )
            # Patch the selected heads for all repeated examples in the expanded batch at once.
            sub[batch_idx, -2, head_idx, :] = patch_values
            set_o_proj_inputs(model, layer, ((sub.view(original_shape),), tup[1]))
            logits = model.output.logits[:, -2:, :]
            ent = expanded_entities.to(logits.device)
            batch = torch.arange(ent.shape[0], device=logits.device)
            last_logits = logits[:, -1, :]
            prev_logits = logits[:, -2, :]
            pred_ids_saved = last_logits.argmax(dim=-1).save()
            m1_logits_saved = last_logits[batch, ent[:, 1]].save()
            m2_logits_saved = prev_logits[batch, ent[:, 0]].save()
            last_lse_saved = last_logits.logsumexp(dim=-1).save()
            prev_lse_saved = prev_logits.logsumexp(dim=-1).save()

    return sum_stats(
        stats_from_saved_components(
            tuple(
                saved_to_cpu(saved)
                for saved in (
                    pred_ids_saved,
                    m1_logits_saved,
                    m2_logits_saved,
                    last_lse_saved,
                    prev_lse_saved,
                )
            ),
            expanded_entities,
        ),
        n_groups=chunk_size,
    )


def remote_patch_chunk_stats(
    model,
    layer,
    corr_seq,
    entities,
    clean_heads,
    batch_size,
    heads_per_layer,
    head_dim,
    chunk_heads,
    remote_ctx=None,
):
    chunk_size = chunk_heads.shape[0]
    try:
        if remote_ctx is None:
            return _remote_patch_chunk_stats_request(
                model,
                layer,
                corr_seq,
                entities,
                clean_heads,
                batch_size,
                heads_per_layer,
                head_dim,
                chunk_heads,
            )
        return remote_ctx.request(
            f"patch_l{layer}_h{int(chunk_heads[0])}-{int(chunk_heads[-1])}_b{batch_size}",
            lambda trace_model: _remote_patch_chunk_stats_request(
                trace_model,
                layer,
                corr_seq,
                entities,
                clean_heads,
                batch_size,
                heads_per_layer,
                head_dim,
                chunk_heads,
            ),
        )
    except Exception as exc:
        # If we get an OOM error from NDIF, it's likely because the expanded batch size after patching is too large.
        # In that case, we can split the head chunk into smaller pieces and recursively call this function on each piece to reduce the batch size.
        if is_remote_oom(exc) and chunk_size > 1:
            mid = chunk_size // 2
            left = remote_patch_chunk_stats(
                model,
                layer,
                corr_seq,
                entities,
                clean_heads,
                batch_size,
                heads_per_layer,
                head_dim,
                chunk_heads[:mid],
                remote_ctx=remote_ctx,
            )
            right = remote_patch_chunk_stats(
                model,
                layer,
                corr_seq,
                entities,
                clean_heads,
                batch_size,
                heads_per_layer,
                head_dim,
                chunk_heads[mid:],
                remote_ctx=remote_ctx,
            )
            return tuple(torch.cat((l, r)) for l, r in zip(left, right))
        raise

# patches the -2 index for a batch of sequences
def patch_head_m2(model, clean_seq, corr_seq, entities):
    heads_per_layer = model.config.num_attention_heads
    head_dim = model.config.hidden_size // heads_per_layer

    # will be 1024-dimensional lists
    correct = []
    m1_probs = []
    m2_probs = []
    m1_logits = []
    m2_logits = []
    for layer in range(model.config.num_hidden_layers):
        # (bsz, seq_len, n_heads, head_dim)
        clean_heads = get_head_activations(model, clean_seq, layer)

        for head_idx in range(heads_per_layer):
            remote = is_remote_model(model)
            with torch.no_grad():
                with model.trace(corr_seq, remote=remote):
                    tup = get_o_proj_inputs(model, layer)
                    original_shape = tup[0][0].shape  # [bsz, seq_len, model_dim]
                    sub = tup[0][0].view(
                        original_shape[0], original_shape[1], heads_per_layer, head_dim
                    )

                    sub[:, -2, head_idx, :] = clean_heads[:, -2, head_idx, :]
                    sub = sub.view(original_shape)
                    new_tup = ((sub,), tup[1])
                    set_o_proj_inputs(model, layer, new_tup)

                    logits = model.output.logits.save()

            # [bsz] bits that we want to collect
            c, m1, m2, m1l, m2l = stats_from_logits(logits.detach().cpu(), entities)
            correct.append(c)
            m1_probs.append(m1)
            m2_probs.append(m2)
            m1_logits.append(m1l)
            m2_logits.append(m2l)

    # wait i want (bsz, 1024) tensors
    return (
        torch.stack(correct).T,
        torch.stack(m1_probs).T,
        torch.stack(m2_probs).T,
        torch.stack(m1_logits).T,
        torch.stack(m2_logits).T,
    )


def remote_layer_scores(
    model,
    clean_seq,
    corr_seq,
    entities,
    layer,
    head_chunk_size,
    remote_ctx=None,
):
    heads_per_layer = model.config.num_attention_heads
    head_dim = model.config.hidden_size // heads_per_layer
    clean_heads = get_head_activations(
        model,
        clean_seq,
        layer,
        remote_ctx=remote_ctx,
        label=f"clean_heads_l{layer}_b{len(clean_seq)}",
    )[:, -2, :, :]

    layer_chunks = [[], [], [], [], []]
    for head_start in range(0, heads_per_layer, head_chunk_size):
        head_stop = min(head_start + head_chunk_size, heads_per_layer)
        chunk_heads = torch.arange(head_start, head_stop, dtype=torch.long)
        chunk_stats_cpu = remote_patch_chunk_stats(
            model,
            layer,
            corr_seq,
            entities,
            clean_heads,
            len(clean_seq),
            heads_per_layer,
            head_dim,
            chunk_heads,
            remote_ctx=remote_ctx,
        )
        for chunk_list, stat in zip(layer_chunks, chunk_stats_cpu):
            chunk_list.append(stat)

    return tuple(torch.cat(chunks) for chunks in layer_chunks)


def remote_batch_scores(model, clean_seq, corr_seq, entities, remote_ctx=None):
    """
    Run a causal-score batch on NDIF using compact remote stat saves.

    This avoids downloading full vocab logits and patches multiple heads at
    once by expanding the corrupt batch over larger head chunks.
    """
    batch_size = len(clean_seq)
    entities = torch.as_tensor(entities, dtype=torch.long)
    # if the batch size is too large, break it into smaller batches to avoid NDIF OOM errors.
    if batch_size > REMOTE_MAX_BATCH_SIZE:
        clean_total = None
        corrupt_total = None
        patched_total = None
        for start in range(0, batch_size, REMOTE_MAX_BATCH_SIZE):
            stop = min(start + REMOTE_MAX_BATCH_SIZE, batch_size)
            clean_stats, corrupt_stats, patched_stats = remote_batch_scores(
                model,
                clean_seq[start:stop],
                corr_seq[start:stop],
                entities[start:stop],
                remote_ctx=remote_ctx,
            )
            clean_total = add_stats(clean_total, clean_stats)
            corrupt_total = add_stats(corrupt_total, corrupt_stats)
            patched_total = add_stats(patched_total, patched_stats)
        return clean_total, corrupt_total, patched_total

    heads_per_layer = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    head_chunk_size = get_remote_head_chunk_size(batch_size, heads_per_layer)

    clean = remote_sum_stats(
        model, clean_seq, entities, remote_ctx=remote_ctx, label=f"clean_b{batch_size}"
    )
    corrupt = remote_sum_stats(
        model,
        corr_seq,
        entities,
        remote_ctx=remote_ctx,
        label=f"corrupt_b{batch_size}",
    )
    layer_stats = [
        remote_layer_scores(
            model,
            clean_seq,
            corr_seq,
            entities,
            layer,
            head_chunk_size,
            remote_ctx=remote_ctx,
        )
        for layer in range(n_layers)
    ]

    if not layer_stats:
        raise RuntimeError(
            "remote_batch_scores produced no patched chunks; "
            "the patched remote traces did not execute."
        )
    patched = tuple(torch.cat([layer[layer_idx] for layer in layer_stats]) for layer_idx in range(5))
    return clean, corrupt, patched


def build_work_items(args, pile, tok):
    # dummy entities for comparison
    sorted_entities = defaultdict(list)
    if args.random_tok_entities:
        for i in range(args.n):
            doc_toks = []
            while len(doc_toks) < 5:
                doc = pile.shuffle()[0]["text"]
                doc_toks = tok(doc)

            shuffle(doc_toks)
            if i % 4 == 0:
                sorted_entities["bigram"].append(doc_toks[:2])
            elif i % 4 == 1:
                sorted_entities["trigram"].append(doc_toks[:3])
            elif i % 4 == 2:
                sorted_entities["fourgram"].append(doc_toks[:4])
            elif i % 4 == 3:
                sorted_entities["fivegram"].append(doc_toks[:5])

    # load and sort entities of different token lengths
    else:
        str_entities = list(pd.read_csv("../data/counterfact_expanded.csv")["subject"])
        for ent in str_entities:
            toks = tok(ent)
            if len(toks) == 2:
                sorted_entities["bigram"].append(toks)
            elif len(toks) == 3:
                sorted_entities["trigram"].append(toks)
            elif len(toks) == 4:
                sorted_entities["fourgram"].append(toks)
            elif len(toks) == 5:
                sorted_entities["fivegram"].append(toks)

    work_items = []
    for label, ents in sorted_entities.items():
        selected_ents = ents[: args.n // 4]
        n_batches = len(selected_ents) // args.bsz
        for batch_idx in range(n_batches):
            batch_ents = selected_ents[
                batch_idx * args.bsz : (batch_idx + 1) * args.bsz
            ]
            batch_clean, batch_corr = generate_seq_batch(
                batch_ents, pile, tok, args.sequence_len
            )
            work_items.append(
                {
                    "label": label,
                    "batch_idx": batch_idx,
                    "batch_ents": batch_ents,
                    "batch_clean": batch_clean,
                    "batch_corr": batch_corr,
                }
            )

    return work_items


def load_or_build_work_items(args, paths, tok):
    if os.path.exists(paths["shared_work_items_pkl"]):
        with open(paths["shared_work_items_pkl"], "rb") as f:
            return pickle.load(f)

    pile = load_dataset("NeelNanda/pile-10k")["train"]
    work_items = build_work_items(args, pile, tok)
    atomic_pickle_dump(work_items, paths["shared_work_items_pkl"])
    return work_items


def filter_work_items_for_shard(work_items, args):
    if args.work_shard_count <= 1:
        return work_items
    return [
        item
        for idx, item in enumerate(work_items)
        if idx % args.work_shard_count == args.work_shard_index
    ]


def save_final_outputs(paths, model, clean_results, corrupt_results, patched_results):
    all_results = [clean_results, corrupt_results, patched_results]

    print(paths["base_results_pkl"])
    with open(paths["base_results_pkl"], "wb") as f:
        pickle.dump(all_results, f)

    scoretype = paths["scoretype"]
    if scoretype == "token":
        diff = patched_results.get_m2() - corrupt_results.get_m2()
    else:
        diff = patched_results.get_m1() - corrupt_results.get_m1()

    copying_scores = flat_to_dict(diff, n_heads=model.config.num_attention_heads)
    with open(paths["scores_json"], "w") as f:
        json.dump(json_tuple_keys(copying_scores), f)

    copying_rankings = flat_to_ranking(diff, n_heads=model.config.num_attention_heads)
    with open(paths["ranking_json"], "w") as f:
        json.dump(copying_rankings, f)


def merge_shard_outputs(args):
    paths = get_output_paths(args)
    shard_results = []
    for shard_idx in range(args.work_shard_count):
        shard_args = argparse.Namespace(**vars(args))
        shard_args.work_shard_index = shard_idx
        shard_paths = get_output_paths(shard_args)
        with open(shard_paths["results_pkl"], "rb") as f:
            shard_results.append(pickle.load(f))

    clean_results, corrupt_results, patched_results = merge_result_sets(shard_results)

    if args.ckpt is not None:
        assert args.model in ["allenai/OLMo-2-1124-7B", "EleutherAI/pythia-6.9b"]
        model = LanguageModel(
            args.model, device_map="auto", dispatch=(not args.remote), revision=args.ckpt
        )
    elif args.remote:
        model = load_remote_model(args.model, utils)
    else:
        model = LanguageModel(
            args.model,
            device_map="auto",
            dispatch=(not args.remote),
            cache_dir="/share/u/models",
        )
    model._ndif_remote = args.remote
    save_final_outputs(paths, model, clean_results, corrupt_results, patched_results)


def main(args):
    random.seed(8)
    torch.manual_seed(8)
    np.random.seed(8)
    assert args.bsz <= args.n // 4
    if args.work_shard_count < 1:
        raise ValueError("--work-shard-count must be at least 1.")
    if not (0 <= args.work_shard_index < args.work_shard_count):
        raise ValueError("--work-shard-index must be in [0, --work-shard-count).")
    paths = get_output_paths(args)

    if args.merge_shards:
        if args.work_shard_count <= 1:
            raise ValueError("--merge-shards requires --work-shard-count > 1.")
        merge_shard_outputs(args)
        return

    remote = args.remote
    remote_ctx = None

    if remote:
        if args.ckpt is not None:
            raise ValueError(
                "NDIF remote execution does not support --ckpt in this script."
            )
        model = load_remote_model(args.model, utils)
    elif args.ckpt is not None:
        assert args.model in ["allenai/OLMo-2-1124-7B", "EleutherAI/pythia-6.9b"]
        model = LanguageModel(
            args.model, device_map="auto", dispatch=(not remote), revision=args.ckpt
        )
    else:
        model = LanguageModel(
            args.model,
            device_map="auto",
            dispatch=(not remote),
            cache_dir="/share/u/models",
        )
    model._ndif_remote = remote
    tokenizer = model.tokenizer
    if remote:
        remote_ctx = RemoteExecutionContext(
            args.model,
            max_retries=args.remote_max_retries,
            backoff_base=args.remote_backoff_base,
            backoff_max=args.remote_backoff_max,
        )

    # tokenization function for any model
    def tok(s, bos=False, model=model):
        if "llama" in model.config._name_or_path:
            if not bos:
                return model.tokenizer(s)["input_ids"][1:]
            else:
                return model.tokenizer(s)["input_ids"]
        elif (
            "OLMo" in model.config._name_or_path
            or "pythia" in model.config._name_or_path
        ):
            if not bos:
                return model.tokenizer(s)["input_ids"]
            else:
                return [model.tokenizer.bos_token_id] + model.tokenizer(s)["input_ids"]

    # accumulators. patching experiments will log across each attention head
    n_heads = model.config.num_attention_heads * model.config.num_hidden_layers
    os.makedirs(paths["score_dir"], exist_ok=True)
    os.makedirs(paths["rank_dir"], exist_ok=True)

    if args.prepare_work_items:
        work_items = load_or_build_work_items(args, paths, tok)
        shard_work_items = filter_work_items_for_shard(work_items, args)
        print(
            f"prepared {len(work_items)} shared work items; "
            f"shard {args.work_shard_index}/{args.work_shard_count} would run "
            f"{len(shard_work_items)} batches"
        )
        return

    if args.resume:
        state = load_resume_state(paths["resume_pkl"], args)
        work_items = state["work_items"]
        clean_results, corrupt_results, patched_results = state["results"]
        next_work_idx = state["next_work_idx"]
        print(f"resuming from batch {next_work_idx}/{len(work_items)}")
    else:
        work_items = load_or_build_work_items(args, paths, tok)
        work_items = filter_work_items_for_shard(work_items, args)
        clean_results = ChunkOutputSaver("clean", 1)
        corrupt_results = ChunkOutputSaver("corrupt", 1)
        patched_results = ChunkOutputSaver("patched", n_heads)
        next_work_idx = 0
        print(
            f"starting shard {args.work_shard_index}/{args.work_shard_count} "
            f"with {len(work_items)} batches"
        )
        save_resume_state(
            paths["resume_pkl"],
            args,
            work_items,
            [clean_results, corrupt_results, patched_results],
            next_work_idx,
        )

    last_label = None
    try:
        for work_idx in tqdm(
            range(next_work_idx, len(work_items)),
            initial=next_work_idx,
            total=len(work_items),
        ):
            item = work_items[work_idx]
            if item["label"] != last_label:
                print(item["label"], tokenizer.decode(item["batch_ents"][0]))
                last_label = item["label"]

            batch_ents = torch.tensor(item["batch_ents"], dtype=torch.long)
            if remote:
                clean_stats, corrupt_stats, patched_stats = remote_batch_scores(
                    model,
                    item["batch_clean"],
                    item["batch_corr"],
                    batch_ents,
                    remote_ctx=remote_ctx,
                )
                clean_results.update_sums(len(batch_ents), *clean_stats)
                corrupt_results.update_sums(len(batch_ents), *corrupt_stats)
                patched_results.update_sums(len(batch_ents), *patched_stats)
            else:
                clean_results.update(
                    *no_patching(model, item["batch_clean"], batch_ents)
                )
                corrupt_results.update(
                    *no_patching(model, item["batch_corr"], batch_ents)
                )

                # patch outputs of each head from [-2] index
                patched_results.update(
                    *patch_head_m2(
                        model, item["batch_clean"], item["batch_corr"], batch_ents
                    )
                )
            save_resume_state(
                paths["resume_pkl"],
                args,
                work_items,
                [clean_results, corrupt_results, patched_results],
                work_idx + 1,
            )
    except Exception:
        print(f"resume state saved to {paths['resume_pkl']}")
        raise

    all_results = [clean_results, corrupt_results, patched_results]
    if args.work_shard_count > 1:
        print(paths["results_pkl"])
        with open(paths["results_pkl"], "wb") as f:
            pickle.dump(all_results, f)
    else:
        save_final_outputs(
            paths, model, clean_results, corrupt_results, patched_results
        )

    if os.path.exists(paths["resume_pkl"]):
        os.remove(paths["resume_pkl"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-hf",
        choices=[
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Meta-Llama-3-8B",
            "allenai/OLMo-2-1124-7B",
            "EleutherAI/pythia-6.9b",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Meta-Llama-3.1-70B",
            "meta-llama/Llama-3.2-3B",
            "allenai/OLMo-2-0425-1B",
        ],
    )
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--n", default=1024, type=int)
    parser.add_argument("--bsz", default=32, type=int)
    parser.add_argument("--sequence_len", default=30, type=int)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--random_tok_entities", action="store_true")
    parser.add_argument("--remote-max-retries", default=4, type=int)
    parser.add_argument("--remote-backoff-base", default=2.0, type=float)
    parser.add_argument("--remote-backoff-max", default=30.0, type=float)
    parser.add_argument("--work-shard-index", default=0, type=int)
    parser.add_argument("--work-shard-count", default=1, type=int)
    parser.add_argument("--prepare-work-items", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    parser.set_defaults(random_tok_entities=False, remote=False, resume=False)
    args = parser.parse_args()
    main(args)
