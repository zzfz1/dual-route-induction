import sys
import threading
import time

import torch

import utils
from ndif import load_remote_model
from utils import pile_chunk

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
        "Error submitting request to model deployment",
        "RemoteProtocolError",
        "Bad Gateway",
        "Service Unavailable",
        "WriteTimeout",
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectError",
        "ConnectionError",
        "DisconnectedError",
        "Connection reset by peer",
        "socket closed",
        "timed out",
    )
    class_markers = {
        "RemoteProtocolError",
        "WriteTimeout",
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectError",
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


def generate_seq_batch(entities, pile, tok, sequence_len):
    clean, corrupt = [], []
    for ent in entities:
        rand = pile_chunk(sequence_len - len(ent), pile, tok, shuf_pile=True)
        ent_chunk_full = rand + ent
        ent_chunk_trunc = rand + [ent[0]]
        clean_prompt = [1] + ent_chunk_full + [13] + ent_chunk_trunc

        rand2 = pile_chunk(sequence_len, pile, tok, shuf_pile=True)
        corrupt_prompt = [1] + rand2 + [13] + ent_chunk_trunc
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
            tup = get_o_proj_inputs(model, layer)
            original = tup[0][0]
            original_shape = original.shape
            chunk_batch = original_shape[0]

            sub = original.view(
                chunk_batch, original_shape[1], heads_per_layer, head_dim
            )
            batch_idx = torch.arange(chunk_batch, device=original.device)
            head_idx = chunk_heads.to(original.device).repeat_interleave(batch_size)
            patch_values = (
                clean_heads[:, chunk_heads, :]
                .transpose(0, 1)
                .reshape(chunk_batch, head_dim)
                .to(original.device)
            )
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
