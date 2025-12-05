'''
Apply OV matrices of concept (or token) heads to "see" the concept stored 
inside a hidden state at a particular layer/token position. 
'''
import os 
import torch 
import json 
import argparse 
import matplotlib.pyplot as plt 
import seaborn as sns 
from nnsight import LanguageModel 
from matplotlib import font_manager 


def logit_lens(concept_vec, model, k=10):
    with torch.no_grad():
        if 'pythia' in model.config._name_or_path:
            logits = model.embed_out(model.gpt_neox.final_layer_norm(concept_vec.cuda())) 
        else:
            logits = model.lm_head(model.model.norm(concept_vec.cuda()))
        return torch.topk(logits.softmax(dim=-1).cpu(), k=k)

# for llama 3 
def gqa_repeat(attn_module, config):
    ''' 
    You only need to do this for v_proj and k_proj, not query or output. 
    128 * 8 # = 1024, so 8 is the number of actual value vectors there are. `num_key_value_heads`
    32 / 8 # = 4, so there are now four query vectors per value vector. `query_group_size` (misnomer num_key_value_groups)
    There are only eight v_proj heads. To match with the 32 query heads, each of these eight guys is re-used four times.
    '''
    query_group_size = attn_module.num_key_value_groups
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    # v_proj is shape (out=1024, in=4096) for llama3
    original = attn_module.v_proj.weight.reshape((num_kv_heads, head_dim, config.hidden_size))
    repeated = original.repeat_interleave(query_group_size, dim=0)
    return repeated.reshape((config.hidden_size, config.hidden_size))

def get_ov_sum(model, k, head_ordering='concept'):
    model_name = model.config._name_or_path.split('/')[-1]
    model_dim = model.config.hidden_size 
    head_dim = model_dim // model.config.num_attention_heads

    if head_ordering == 'all':
        to_sum = [(l, h) for l in range(model.config.num_hidden_layers) for h in range(model.config.num_attention_heads)]
    elif head_ordering == 'function':
        with open(f'../cache/head_orderings/{model_name}/fv.json', 'r') as f: 
            tups = json.load(f)
        to_sum = [(l, h) for l, h in tups][:k]
    else: 
        with open(f'../cache/causal_scores/{model_name}/{head_ordering}_copying_len30_n1024.json', 'r') as f: 
            temp = json.load(f)
        tups = sorted([(d['layer'], d['head_idx'], d['score']) for d in temp], key=lambda t: t[2], reverse=True)
        to_sum = [(l, h) for l, h, _ in tups][:k]
    layerset = set([l for l, _ in to_sum])

    with torch.no_grad():
        ov_sum = torch.zeros((model_dim, model_dim), device='cuda')
        for layer in layerset:
            for l, h in to_sum:
                if l == layer:
                    if model_name == 'pythia-6.9b':
                        # TODO we are ignoring the bias, I think this throws it off (output, input)
                        _, _, v_proj = model.gpt_neox.layers[l].attention.query_key_value.weight.split(model.config.hidden_size, dim=0)
                        o_proj = model.gpt_neox.layers[1].attention.dense.weight

                    else:
                        this_attn = model.model.layers[l].self_attn 
                        v_proj = gqa_repeat(this_attn, model.config)
                        o_proj = this_attn.o_proj.weight

                    # (out_features, in_features). 
                    V = v_proj[h * head_dim : (h+1) * head_dim] # select rows so that (128, 4096) projects hidden state down. 
                    O = o_proj[:, h * head_dim : (h+1) * head_dim] # select columns so that (4096, 128) converts value back up
                    ov_sum += torch.matmul(O, V) # 4096, 4096
        return ov_sum

# given a concept string w project hidden state from layer_idx/offset using summed OV matrices 
def proj_onto_ov(w, model, layer_idx, offset=-1, head_ordering='concept', k=80):
    ov_sum = get_ov_sum(model, k, head_ordering)
    with model.trace(w):
        if 'pythia' in model.config._name_or_path:
            state = model.gpt_neox.layers[layer_idx].output[0].squeeze()[offset].save()
        else:
            state = model.model.layers[layer_idx].output[0].squeeze()[offset].save()
    return torch.matmul(ov_sum, state), state

def ov_lens(w, model, k=80, head_ordering='concept', print_k=5, offset=-1, raw=False, max_layer=None):
    cjk_font = font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')

    cmap = {
        'concept' : 'Reds',
        'token' : 'Blues',
        'all' : 'Purples',
        'function' : 'Greens'
    }[head_ordering]

    to_print = [model.tokenizer.decode(t) for t in model.tokenizer(w)['input_ids']]
    if 'llama' in model.config._name_or_path:
        to_print = to_print[1:]

    probss, idxss = [], []
    for layer_idx in range(0, model.config.num_hidden_layers):
        projd, original = proj_onto_ov(w, model, layer_idx, offset, head_ordering, k)

        if raw:
            probs, idxs = logit_lens(original, model, k=print_k)
        else:
            probs, idxs = logit_lens(projd, model, k=print_k)

        probss.append(probs)
        idxss.append([model.tokenizer.decode(t) for t in idxs])
    
    if max_layer is not None:
        probss = probss[:max_layer + 1]
        idxss = idxss[:max_layer + 1]
    
    # Adjust figure height based on number of layers
    num_layers = len(probss)
    fig_height = max(3, num_layers * 0.3)  # Scale height with layers, minimum 3
    fig_width = max(5, print_k * 2)
    _, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(probss, annot=idxss, fmt='', ax=ax, vmin=0, vmax=1.0, cmap=cmap)
    for text in ax.texts:
        text.set_fontproperties(cjk_font)
    # ax.add_patch(Rectangle((0, 3), 5, 16, fill=False, edgecolor='blue', lw=3))
    title = str(to_print) + f' - lens({to_print[offset]})'
    if raw:
        title = f'Top-{print_k} Raw Logit Lens Outputs\n' + title
    else:
        title = f'Top-{print_k} {head_ordering.title()} Lens Outputs\n' + title
    plt.title(title)
    plt.xlabel('Top Ranked Tokens (Higher->Lower)')
    plt.ylabel('Layer')

    model_name = model.config._name_or_path.split('/')[-1]
    if raw:
        head_ordering = 'raw'
    save_dir = f'../figures/concept_lens/{model_name}/{head_ordering}/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + f'{w}_k{k}_offset{offset}_print{print_k}.png')


def main(args):
    model = LanguageModel(args.model, device_map='cuda')

    ov_lens(
        args.word, model, k=args.k, head_ordering=args.head_ordering, print_k=args.print_k, 
        offset=args.offset, raw=args.raw
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', default='The Dangerous Blizzard', type=str)
    parser.add_argument('--k', default=80, type=int)
    parser.add_argument('--print_k', default=5, type=int) # top-k, how many logit lens outputs to see 
    parser.add_argument('--offset', default=-1, type=int)
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--head_ordering', default='concept', type=str,
                        choices=[
                            'concept',
                            'token',
                            'all'
                        ])
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', 
                        choices=[
                            'meta-llama/Llama-2-7b-hf',
                            'meta-llama/Meta-Llama-3-8B',
                            'allenai/OLMo-2-1124-7B',
                            'EleutherAI/pythia-6.9b'
                        ])
    parser.set_defaults(raw=False)
    args = parser.parse_args()
    main(args)

