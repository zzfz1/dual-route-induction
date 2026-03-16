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
        "results_pkl": score_dir + f"{fname}{token_suffix}.pkl",
        "scores_json": score_dir + f"{scoretype}_copying_{fname}.json",
        "ranking_json": rank_dir + ranking_name,
        "resume_pkl": score_dir + f"{fname}{token_suffix}_resume.pkl",
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


def inference_logits(model, sequences):
    remote = is_remote_model(model)
    with torch.no_grad():
        with model.trace(sequences, remote=remote):
            logits = model.output.logits.save()
    return logits.detach().cpu()


def stats_from_logits(logits, entities):
    # trying to predict the second token of the entity
    correct = logits[:, -1].argmax(dim=-1) == entities[:, 1]

    probs = logits.softmax(dim=-1)
    m1_probs = probs[torch.arange(len(entities)), -1, entities[:, 1]]
    m2_probs = probs[torch.arange(len(entities)), -2, entities[:, 0]]

    m1_logits = logits[torch.arange(len(entities)), -1, entities[:, 1]]
    m2_logits = logits[torch.arange(len(entities)), -2, entities[:, 0]]

    return correct, m1_probs, m2_probs, m1_logits, m2_logits


def no_patching(model, sequences, entities):
    logits = inference_logits(model, sequences)
    return stats_from_logits(logits, entities)


def get_head_activations(model, prompt, layer):
    remote = is_remote_model(model)
    with torch.no_grad():
        if model.config._name_or_path == "EleutherAI/pythia-6.9b":
            with model.trace(prompt, remote=remote):
                o_proj_inp = (
                    model.gpt_neox.layers[layer].attention.dense.inputs[0][0].save()
                )
        else:
            with model.trace(prompt, remote=remote):
                o_proj_inp = (
                    model.model.layers[layer].self_attn.o_proj.inputs[0][0].save()
                )

        # [bsz, seq_len, model_dim] -> [bsz, seq_len, n_heads, head_dim]
        heads_per_layer = model.config.num_attention_heads
        head_dim = model.config.hidden_size // heads_per_layer
        return o_proj_inp.view(*o_proj_inp.shape[:-1], heads_per_layer, head_dim)


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
                    if model.config._name_or_path == "EleutherAI/pythia-6.9b":
                        tup = model.gpt_neox.layers[layer].attention.dense.inputs
                    else:
                        tup = model.model.layers[layer].self_attn.o_proj.inputs

                    original_shape = tup[0][0].shape  # [bsz, seq_len, model_dim]
                    sub = tup[0][0].view(
                        original_shape[0], original_shape[1], heads_per_layer, head_dim
                    )

                    sub[:, -2, head_idx, :] = clean_heads[:, -2, head_idx, :]
                    sub = sub.view(original_shape)
                    new_tup = ((sub,), tup[1])

                    if model.config._name_or_path == "EleutherAI/pythia-6.9b":
                        model.gpt_neox.layers[layer].attention.dense.inputs = new_tup
                    else:
                        model.model.layers[layer].self_attn.o_proj.inputs = new_tup

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


def main(args):
    random.seed(8)
    torch.manual_seed(8)
    np.random.seed(8)
    assert args.bsz <= args.n // 4
    paths = get_output_paths(args)

    remote = args.remote

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

    if args.resume:
        state = load_resume_state(paths["resume_pkl"], args)
        work_items = state["work_items"]
        clean_results, corrupt_results, patched_results = state["results"]
        next_work_idx = state["next_work_idx"]
        print(f"resuming from batch {next_work_idx}/{len(work_items)}")
    else:
        pile = load_dataset("NeelNanda/pile-10k")["train"]
        work_items = build_work_items(args, pile, tok)
        clean_results = ChunkOutputSaver("clean", 1)
        corrupt_results = ChunkOutputSaver("corrupt", 1)
        patched_results = ChunkOutputSaver("patched", n_heads)
        next_work_idx = 0
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

    # save all results just in case
    print(paths["results_pkl"])
    with open(paths["results_pkl"], "wb") as f:
        pickle.dump(all_results, f)

    # save important results as json
    # convert to dictionaries with (layer, head_idx) tuples
    scoretype = paths["scoretype"]

    diff = patched_results.get_m1() - corrupt_results.get_m1()
    copying_scores = flat_to_dict(diff, n_heads=model.config.num_attention_heads)
    with open(paths["scores_json"], "w") as f:
        json.dump(json_tuple_keys(copying_scores), f)

    copying_rankings = flat_to_ranking(diff, n_heads=model.config.num_attention_heads)
    with open(paths["ranking_json"], "w") as f:
        json.dump(copying_rankings, f)

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
    parser.set_defaults(random_tok_entities=False, remote=False, resume=False)
    args = parser.parse_args()
    main(args)
