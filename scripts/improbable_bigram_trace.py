from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from nnsight import LanguageModel, save

import utils
from improbable_bigram_data import (
    DEFAULT_TASKS_PATH,
    DEFAULT_TRACE_ROOT,
    PROMPT_STYLE,
    load_bigram_tasks,
    validate_prompt_layouts,
)
from ndif import load_remote_model
from seed_utils import set_random_seed
from trace_utils import RemoteExecutionContext, is_remote_model


def atomic_write_json(path: Path, payload):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def atomic_torch_save(path: Path, payload):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def build_tok(model):
    def tok(text, bos=False, model=model):
        if "llama" in model.config._name_or_path.lower():
            ids = model.tokenizer(text)["input_ids"]
            return ids if bos else ids[1:]
        if (
            "olmo" in model.config._name_or_path.lower()
            or "pythia" in model.config._name_or_path.lower()
        ):
            ids = model.tokenizer(text)["input_ids"]
            return [model.tokenizer.bos_token_id] + ids if bos else ids
        raise ValueError(f"Unsupported model family: {model.config._name_or_path}")

    return tok


def load_model(args):
    if args.remote:
        model = load_remote_model(args.model, utils)
    else:
        model = LanguageModel(
            args.model,
            device_map="auto",
            dispatch=True,
            cache_dir="/share/u/models",
        )
    model._ndif_remote = args.remote
    return model


def _capture_pass_state(model, input_ids):
    if model.config._name_or_path != "meta-llama/Llama-3.1-8B":
        raise ValueError(
            "improbable_bigram_trace.py currently only supports meta-llama/Llama-3.1-8B."
        )

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    with torch.no_grad():
        with model.trace(
            [input_ids],
            remote=is_remote_model(model)
            and not getattr(model, "_ndif_session_active", False),
        ):
            # head's o_proj input for final token
            head_saves = save([])
            # attn row for final token against all previous tokens
            attn_row_saves = save([])
            # value norms for all tokens and heads
            value_norm_saves = save([])

            for layer_idx in range(n_layers):
                # Manually reconstruct the attention computation to extract the relevant intermediate values.
                attn = model.model.layers[layer_idx].self_attn
                n_kv_groups = attn.num_key_value_groups

                position = attn.inputs[1]["position_embeddings"]
                attention_mask = attn.inputs[1]["attention_mask"]

                query_states = attn.q_proj.output
                key_states = attn.k_proj.output
                value_states = attn.v_proj.output
                o_proj_inp = attn.o_proj.inputs[0][0]

                bsz = query_states.shape[0]
                seq_len = query_states.shape[1]

                value_states = value_states.view(bsz, seq_len, -1, head_dim).transpose(
                    1, 2
                )
                value_states = torch.repeat_interleave(value_states, n_kv_groups, dim=1)
                value_norms = torch.linalg.vector_norm(value_states, dim=-1)

                query_states = query_states.view(bsz, seq_len, -1, head_dim).transpose(
                    1, 2
                )
                key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
                cos = position[0].unsqueeze(1)
                sin = position[1].unsqueeze(1)

                q1 = query_states[..., : query_states.shape[-1] // 2]
                q2 = query_states[..., query_states.shape[-1] // 2 :]
                k1 = key_states[..., : key_states.shape[-1] // 2]
                k2 = key_states[..., key_states.shape[-1] // 2 :]

                query_states = (query_states * cos) + (
                    torch.cat((-q2, q1), dim=-1) * sin
                )
                key_states = (key_states * cos) + (torch.cat((-k2, k1), dim=-1) * sin)
                key_states = torch.repeat_interleave(key_states, n_kv_groups, dim=1)

                attn_weights = torch.matmul(
                    query_states, key_states.transpose(2, 3)
                ) / (head_dim**0.5)
                if attention_mask is None:
                    # For the final token row we save below, there are no future
                    # positions to mask in an unpadded single-sequence trace.
                    attn_probs = attn_weights.softmax(dim=-1)
                else:
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_probs = (attn_weights + causal_mask).softmax(dim=-1)

                head_saves.append(o_proj_inp[0, -1, :].view(n_heads, head_dim).save())
                attn_row_saves.append(attn_probs[0, :, -1, :].save())
                value_norm_saves.append(value_norms[0].save())

            resid_saved = model.model.layers[n_layers - 1].output[0][-1, :].save()
            logits_saved = model.output.logits[0, -1, :].save()

    state = {
        "logits": logits_saved.detach().cpu().to(torch.float32),
        "resid_pre_final_norm": resid_saved.detach().cpu().to(torch.float32),
        "head_o_proj_in": torch.stack(
            [saved.detach().cpu().to(torch.float32) for saved in head_saves]
        ),
        "attn_row_raw": torch.stack(
            [saved.detach().cpu().to(torch.float32) for saved in attn_row_saves]
        ),
        "value_norms": torch.stack(
            [saved.detach().cpu().to(torch.float32) for saved in value_norm_saves]
        ),
    }
    return state


def capture_pass_state(model, input_ids, remote_ctx=None, label=None):
    if remote_ctx is None:
        return _capture_pass_state(model, input_ids)
    return remote_ctx.request(
        label or f"trace_len{len(input_ids)}",
        lambda trace_model: _capture_pass_state(trace_model, input_ids),
    )


def score_target(logits: torch.Tensor, target_token_id: int):
    probs = torch.softmax(logits, dim=-1)
    pred_token_id = int(torch.argmax(logits).item())
    return {
        "target_token_id": int(target_token_id),
        "target_logit": float(logits[target_token_id].item()),
        "target_prob": float(probs[target_token_id].item()),
        "predicted_token_id": pred_token_id,
        "predicted_logit": float(logits[pred_token_id].item()),
        "predicted_prob": float(probs[pred_token_id].item()),
    }


def is_example_complete(example_dir: Path):
    return all(
        (example_dir / name).exists()
        for name in ("meta.json", "xn_state.pt", "p1_state.pt")
    )


def rebuild_index(out_dir: Path):
    entries = []
    for example_dir in sorted(path for path in out_dir.iterdir() if path.is_dir()):
        meta_path = example_dir / "meta.json"
        if not meta_path.exists():
            continue
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        entries.append(
            {
                "task_idx": meta["task_idx"],
                "bigram": meta["bigram"],
                "relative_dir": example_dir.name,
                "copy_success": meta["flags"]["copy_success"],
                "second_token_hallucination": meta["flags"][
                    "second_token_hallucination"
                ],
            }
        )

    index_path = out_dir / "index.jsonl"
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    tmp_path.replace(index_path)


def write_manifest(out_dir: Path, args, total_tasks: int):
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "remote": args.remote,
        "prompt_style": PROMPT_STYLE,
        "tasks_path": str(Path(args.tasks_path).resolve()),
        "total_tasks": total_tasks,
    }
    atomic_write_json(out_dir / "manifest.json", manifest)


def main(args):
    set_random_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args)
    tok = build_tok(model)

    tasks = load_bigram_tasks(args.tasks_path)
    layouts, mismatches = validate_prompt_layouts(tasks, tok)
    if mismatches:
        atomic_write_json(out_dir / "mismatches.json", mismatches)
        raise ValueError(
            f"Found {len(mismatches)} prompt/tokenization mismatches. "
            f"See {out_dir / 'mismatches.json'}."
        )

    write_manifest(out_dir, args, len(layouts))

    layout_slice = layouts[args.start : args.stop if args.stop is not None else None]
    remote_ctx = None
    if args.remote:
        remote_ctx = RemoteExecutionContext(
            args.model,
            max_retries=args.remote_max_retries,
            backoff_base=args.remote_backoff_base,
            backoff_max=args.remote_backoff_max,
        )

    for layout in layout_slice:
        example_dir = out_dir / f"{layout.task_idx:03d}"
        if is_example_complete(example_dir) and not args.overwrite:
            print(f"skip task {layout.task_idx:03d} {layout.bigram!r}")
            continue

        print(f"trace task {layout.task_idx:03d} {layout.bigram!r}")
        example_dir.mkdir(parents=True, exist_ok=True)

        xn_state = capture_pass_state(
            model,
            layout.input_ids_xn,
            remote_ctx=remote_ctx,
            label=f"xn_{layout.task_idx}",
        )
        p1_state = capture_pass_state(
            model,
            layout.input_ids_p1,
            remote_ctx=remote_ctx,
            label=f"p1_{layout.task_idx}",
        )

        p1_stats = score_target(xn_state["logits"], layout.prefix_token_id)
        p2_stats = score_target(p1_state["logits"], layout.suffix_token_id)

        meta = {
            "task_idx": layout.task_idx,
            "bigram": layout.bigram,
            "prompt_style": layout.prompt_style,
            "prompt_text": layout.prompt_text,
            "prefix_token_id": layout.prefix_token_id,
            "suffix_token_id": layout.suffix_token_id,
            "input_ids_xn": layout.input_ids_xn,
            "input_ids_p1": layout.input_ids_p1,
            "positions": {
                "p2_prev": layout.p2_prev_idx,
                "x_n": layout.x_n_idx,
                "p1": layout.p1_idx,
            },
            "p1": p1_stats,
            "p2": p2_stats,
            "flags": {
                "p1_correct": p1_stats["predicted_token_id"] == layout.prefix_token_id,
                "p2_correct": p2_stats["predicted_token_id"] == layout.suffix_token_id,
                "copy_success": (
                    p1_stats["predicted_token_id"] == layout.prefix_token_id
                    and p2_stats["predicted_token_id"] == layout.suffix_token_id
                ),
                "second_token_hallucination": (
                    p1_stats["predicted_token_id"] == layout.prefix_token_id
                    and p2_stats["predicted_token_id"] != layout.suffix_token_id
                ),
            },
        }

        atomic_torch_save(example_dir / "xn_state.pt", xn_state)
        atomic_torch_save(example_dir / "p1_state.pt", p1_state)
        atomic_write_json(example_dir / "meta.json", meta)

    rebuild_index(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B",
        choices=["meta-llama/Llama-3.1-8B"],
    )
    parser.add_argument("--tasks-path", default=str(DEFAULT_TASKS_PATH))
    parser.add_argument("--out-dir", default=str(DEFAULT_TRACE_ROOT))
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--stop", default=None, type=int)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--remote-max-retries", default=4, type=int)
    parser.add_argument("--remote-backoff-base", default=2.0, type=float)
    parser.add_argument("--remote-backoff-max", default=30.0, type=float)
    parser.add_argument("--seed", default=8, type=int)
    parser.set_defaults(remote=False, overwrite=False)
    main(parser.parse_args())
