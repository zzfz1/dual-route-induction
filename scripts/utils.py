import random
import torch
import math
from nnsight import save

import llama
import pythia


def json_tuple_keys(mapping):
    return [{"layer": k[0], "head_idx": k[1], "score": v} for k, v in mapping.items()]


def pile_chunk(random_len, pile, tok, shuf_pile=True):
    sample = []
    while len(sample) < random_len:
        doc = pile.shuffle()[0]["text"]  # sample from huggingface
        sample = tok(doc, bos=False)[:random_len]
        if shuf_pile:
            random.shuffle(sample)
    return sample


def flatidx_to_grididx(flat_idx, n_heads):
    if isinstance(flat_idx, torch.Tensor):
        flat_idx = flat_idx.item()
    layer = flat_idx // n_heads
    head = flat_idx % n_heads
    return (layer, head)


def grididx_to_flatidx(grid_idx, n_heads):
    layer, head = grid_idx
    return layer * n_heads + head


# function to load in the cached head means for ablation purposes
def get_mean_head_values(model_name):
    dir = f"../activations/{model_name}_pile-10k/"
    return torch.load(dir + "mean.ckpt")


def get_l2_attn_weights(model, tokenized, layer, value_weighting):
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    with torch.no_grad():
        with model.trace(tokenized):
            # positional_embeddings (cos, sin) each shape [bsz, seq_len, head_dim]
            position = model.model.layers[layer].self_attn.inputs[1][
                "position_embeddings"
            ]
            attention_mask = model.model.layers[layer].self_attn.inputs[1][
                "attention_mask"
            ]

            # [bsz, seq_len, model_size]
            query_states = model.model.layers[layer].self_attn.q_proj.output
            key_states = model.model.layers[layer].self_attn.k_proj.output

            bsz = query_states.shape[0]
            seq_len = query_states.shape[1]
            if value_weighting:
                # [bsz, seq_len, model_size] -> [bsz, seq_len, n_heads, head_dim]
                value_states = model.model.layers[layer].self_attn.v_proj.output
                value_states = value_states.view(bsz, seq_len, n_heads, head_dim).save()

            # from modeling_llama, convert to [bsz, n_heads, seq_len, head_dim] and rotate
            query_states = query_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            query_states, key_states = llama.apply_rotary_pos_emb(
                query_states, key_states, position[0], position[1]
            )

            # not needed because num_key_value_heads == num_attention_heads
            # key_states = repeat_kv(key_states, self.num_key_value_groups)
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)

            # has to be eager implementation
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.save()

    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else:
        # get l2 norm of each head value vector [bsz, seq_len, n_heads] -> [bsz, n_heads, seq_len]
        value_norms = (
            torch.linalg.vector_norm(value_states, dim=-1)
            .detach()
            .cpu()
            .transpose(1, 2)
        )

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

        # then multiply by softmax values and normalize
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective


def get_l3_attn_weights(model, tokenized, layer, value_weighting):
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    with torch.no_grad():
        with model.trace(tokenized):
            # self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
            n_kv_groups = model.model.layers[layer].self_attn.num_key_value_groups

            # positional_embeddings (cos, sin) each shape [bsz, seq_len, head_dim]
            position = model.model.layers[layer].self_attn.inputs[1][
                "position_embeddings"
            ]
            attention_mask = model.model.layers[layer].self_attn.inputs[1][
                "attention_mask"
            ]

            # grouped query means that we have more queries than we do keys/values
            query_states = model.model.layers[
                layer
            ].self_attn.q_proj.output  # [bsz, seq_len, model_size=4096]
            key_states = model.model.layers[
                layer
            ].self_attn.k_proj.output  # [bsz, seq_len, 1024]
            bsz = query_states.shape[0]
            seq_len = query_states.shape[1]

            if value_weighting:
                value_states = model.model.layers[
                    layer
                ].self_attn.v_proj.output  # [bsz, seq_len, 1024]
                value_states = value_states.view(bsz, seq_len, -1, head_dim).transpose(
                    1, 2
                )  # [bsz, seq_len, 8, head_dim] -> [bsz, 8, seq_len, head_dim]
                value_states = llama.repeat_kv(value_states, n_kv_groups).save()

            # from modeling_llama, convert to [bsz, n_heads, seq_len, head_dim] and rotate
            query_states = query_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            query_states, key_states = llama.apply_rotary_pos_emb(
                query_states, key_states, position[0], position[1]
            )

            # not needed because num_key_value_heads == num_attention_heads
            key_states = llama.repeat_kv(key_states, n_kv_groups)
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)

            # has to be eager implementation
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.save()

    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else:
        # get l2 norm of each head value vector [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu()

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

        # then multiply by softmax values and normalize
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective


def get_olmo2_attn_weights(model, tokenized, layer, value_weighting):
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    with torch.no_grad():
        with model.trace(tokenized):
            # positional_embeddings (cos, sin) each shape [bsz, seq_len, head_dim]
            position = model.model.layers[layer].self_attn.inputs[1][
                "position_embeddings"
            ]
            attention_mask = model.model.layers[layer].self_attn.inputs[1][
                "attention_mask"
            ]

            # [bsz, seq_len, model_size]
            query_states = model.model.layers[layer].self_attn.q_norm.output
            key_states = model.model.layers[layer].self_attn.k_norm.output
            bsz = query_states.shape[0]
            seq_len = query_states.shape[1]

            if value_weighting:
                # [bsz, seq_len, model_size] -> [bsz, seq_len, n_heads, head_dim]
                value_states = model.model.layers[layer].self_attn.v_proj.output
                value_states = value_states.view(bsz, seq_len, n_heads, head_dim).save()

            # from modeling_llama, convert to [bsz, n_heads, seq_len, head_dim] and rotate
            bsz = query_states.shape[0]
            seq_len = query_states.shape[1]
            query_states = query_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            query_states, key_states = llama.apply_rotary_pos_emb(
                query_states, key_states, position[0], position[1]
            )

            scaling = model.model.layers[
                layer
            ].self_attn.scaling  # it's just 1/math.sqrt(head_dim)
            attn_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
            )
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.save()

    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else:
        # get l2 norm of each head value vector [bsz, seq_len, n_heads] -> [bsz, n_heads, seq_len]
        value_norms = (
            torch.linalg.vector_norm(value_states, dim=-1)
            .detach()
            .cpu()
            .transpose(1, 2)
        )

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

        # then multiply by softmax values and normalize
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective


def get_pythia_attn_weights(model, tokenized, layer, value_weighting):
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    with torch.no_grad():
        with model.trace(tokenized):
            attention_mask = model.gpt_neox.layers[layer].attention.inputs[1][
                "attention_mask"
            ]
            pos = model.gpt_neox.layers[layer].attention.inputs[1][
                "position_embeddings"
            ]

            qkv = model.gpt_neox.layers[layer].attention.query_key_value.output
            bsz = qkv.shape[0]
            seq_len = qkv.shape[1]
            qkv = (
                qkv.view((bsz, seq_len, n_heads, 3 * head_dim))
                .transpose(1, 2)
                .chunk(3, dim=-1)
            )

            query_states = qkv[0]
            key_states = qkv[1]
            value_states = qkv[2]
            query_states, key_states = pythia.apply_rotary_pos_emb(
                query_states, key_states, pos[0], pos[1]
            )

            if value_weighting:
                # [bsz, seq_len, model_size] -> [bsz, seq_len, n_heads, head_dim]
                value_states = value_states.reshape(
                    bsz, seq_len, n_heads, head_dim
                ).save()

            scaling = model.gpt_neox.layers[
                layer
            ].attention.scaling  # it's just 1/math.sqrt(head_dim)
            attn_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
            )
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.save()

    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else:
        # get l2 norm of each head value vector [bsz, seq_len, n_heads] -> [bsz, n_heads, seq_len]
        value_norms = (
            torch.linalg.vector_norm(value_states, dim=-1)
            .detach()
            .cpu()
            .transpose(1, 2)
        )

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

        # then multiply by softmax values and normalize
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective


def collect_attention_sums(
    model,
    tokenized,
    source_positions,
    target_positions,
    value_weighting=True,
    remote=False,
):
    if not remote:
        raise ValueError(
            "collect_attention_sums is only intended for remote execution in this minimal patch."
        )
    if model.config._name_or_path != "meta-llama/Llama-3.1-8B":
        raise ValueError(
            "Remote execution is only supported for meta-llama/Llama-3.1-8B."
        )

    source_positions = torch.as_tensor(source_positions, dtype=torch.long)
    target_positions = torch.as_tensor(target_positions, dtype=torch.long)
    next_saves = []
    end_saves = []

    with torch.no_grad():
        with model.session(remote=True):
            next_saves = save([])
            end_saves = save([])
            for layer in range(model.config.num_hidden_layers):
                attn = model.model.layers[layer].self_attn
                with model.trace(tokenized):
                    n_heads = model.config.num_attention_heads
                    head_dim = model.config.hidden_size // n_heads
                    n_kv_groups = attn.num_key_value_groups

                    position = attn.inputs[1]["position_embeddings"]
                    query_states = attn.q_proj.output
                    key_states = attn.k_proj.output

                    bsz = query_states.shape[0]
                    seq_len = query_states.shape[1]
                    if value_weighting:
                        value_states = attn.v_proj.output
                        value_states = value_states.view(
                            bsz, seq_len, -1, head_dim
                        ).transpose(1, 2)
                        # repeat for grouped query and key heads
                        value_states = torch.repeat_interleave(
                            value_states, n_kv_groups, dim=1
                        )

                    query_states = query_states.view(
                        bsz, seq_len, -1, head_dim
                    ).transpose(1, 2)
                    key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(
                        1, 2
                    )

                    cos = position[0].unsqueeze(1)
                    sin = position[1].unsqueeze(1)

                    q1 = query_states[..., : query_states.shape[-1] // 2]
                    q2 = query_states[..., query_states.shape[-1] // 2 :]
                    k1 = key_states[..., : key_states.shape[-1] // 2]
                    k2 = key_states[..., key_states.shape[-1] // 2 :]

                    query_states = (query_states * cos) + (
                        torch.cat((-q2, q1), dim=-1) * sin
                    )
                    key_states = (key_states * cos) + (
                        torch.cat((-k2, k1), dim=-1) * sin
                    )

                    key_states = torch.repeat_interleave(key_states, n_kv_groups, dim=1)
                    attn_weights = torch.matmul(
                        query_states, key_states.transpose(2, 3)
                    ) / math.sqrt(head_dim)

                    if value_weighting:
                        value_norms = torch.linalg.vector_norm(value_states, dim=-1)
                        attn_weights = attn_weights.softmax(dim=-1)
                        attn_weights = attn_weights * value_norms.unsqueeze(2).expand(
                            attn_weights.shape
                        )
                        attn_weights /= torch.sum(attn_weights, dim=-1, keepdim=True)
                    else:
                        attn_weights = attn_weights.softmax(dim=-1)

                    device = attn_weights.device
                    final_query = attn_weights[:, :, -1, :]
                    source_idx = (
                        source_positions.to(device)
                        .view(-1, 1, 1)
                        .expand(-1, final_query.shape[1], 1)
                    )
                    target_idx = (
                        target_positions.to(device)
                        .view(-1, 1, 1)
                        .expand(-1, final_query.shape[1], 1)
                    )

                    next_saves.append(
                        torch.gather(final_query, 2, source_idx)
                        .squeeze(-1)
                        .sum(dim=0)
                        .save()
                    )
                    end_saves.append(
                        torch.gather(final_query, 2, target_idx)
                        .squeeze(-1)
                        .sum(dim=0)
                        .save()
                    )

    next_sums = torch.stack([tensor.detach().cpu() for tensor in next_saves])
    end_sums = torch.stack([tensor.detach().cpu() for tensor in end_saves])
    return next_sums, end_sums
