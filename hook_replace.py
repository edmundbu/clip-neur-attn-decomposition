from functools import partial

import einops
import torch
import numpy as np

from utils.get_dims import get_dims


class ReplaceHook():
    def __init__(self, model):
        self.model = model
        self.out = None
        
        self.bias_scaled = None
        self.pcs = None
        self.means = None  

        self.neurons_zero_mask = None
        
    def _project_neurons_reconstruct(self, ret):
        full_out_repr = ret + self.bias_scaled - self.means
        reconstruction = 0
        for pc in self.pcs:
            dots = torch.einsum('boc,oc->bc', full_out_repr, pc)
            single_reconstruction = dots.unsqueeze(1) * pc.unsqueeze(0) 
            reconstruction += single_reconstruction
        reconstruction += self.means
        self.out = reconstruction.sum(axis=-1)
        return ret
    
    def _project_neurons_heads_reconstruct(self, ret):
        full_out_repr = ret + self.bias_scaled - self.means
        reconstruction = 0
        for pc in self.pcs:
            dots = torch.einsum('bhoc,hoc->bhc', full_out_repr, pc) 
            single_reconstruction = dots.unsqueeze(2) * pc.unsqueeze(0) 
            reconstruction += single_reconstruction
        reconstruction += self.means 
        self.out = reconstruction.sum(axis=(1, -1))
        return ret

    def get_output(self):
        return self.out

    def _replace_attn_sink(self, ret):
        ret[:, :, 0, -1] = 0.0
        return ret
    
    def _replace_neurons_zero(self, ret, extra):
        orig_sink = ret[-1:, :, :]
        ret = ret * self.neurons_zero_mask  # NBC * C
        if extra:
            ret = torch.cat([ret, orig_sink], dim=0)
        return ret
    
    def _replace_neurons_mean(self, ret, mask, means):
        ret = torch.where(mask, means, ret)
        return ret


def hook_register_reconstruct(model, pcs, means):
    assert isinstance(pcs, list)
    assert model.visual.attnpool.forward_mode == 'per_head_per_neuron', model.visual.attnpool.forward_mode
    replace_hook = ReplaceHook(model)
    rn_manager = model.hook_manager.forks['visual']
    num_heads, num_tokens, embed_dim, out_dim = get_dims(model)
    replace_hook.means = means

    replace_hook.pcs = pcs
    if len(pcs[0].shape) == 2:  # neurons
        bias_scaled = model.visual.attnpool.c_proj.bias / embed_dim
        replace_hook.bias_scaled = einops.repeat(bias_scaled, 'o -> o c', c=embed_dim)
        rn_manager.register('out.post_head_collapse', replace_hook._project_neurons_reconstruct)
        print("Set up neuron reconstruction!")
    elif len(pcs[0].shape) == 3:  # neurons+heads
        bias_scaled = model.visual.attnpool.c_proj.bias / (embed_dim * num_heads)
        replace_hook.bias_scaled = einops.repeat(bias_scaled, 'o -> h o c', h=num_heads, c=embed_dim)
        rn_manager.register('out.pre_collapse', replace_hook._project_neurons_heads_reconstruct)
        print("Set up neuron+head reconstruction!")
    return replace_hook


def hook_register_attn_sink(model, mode='post'):
    replace_hook = ReplaceHook(model)
    rn_manager = model.hook_manager.forks['visual']
    if mode =='post':
        rn_manager.register('attn.post_softmax', replace_hook._replace_attn_sink)
    elif mode == 'pre':
        rn_manager.register('attn.pre_softmax', replace_hook._replace_attn_sink)
    return replace_hook


def hook_register_neurons_zero(model, neurons_sink_diffs_path, top_k, extra, device='cuda'):
    num_heads, num_tokens, embed_dim, out_dim = get_dims(model)
    replace_hook = ReplaceHook(model)
    rn_manager = model.hook_manager.forks['visual']
    hook_fn = partial(replace_hook._replace_neurons_zero, extra=extra)
    rn_manager.register('input_tokens', hook_fn)
    
    diff_mat = torch.from_numpy(np.load(neurons_sink_diffs_path))
    mean_diff = diff_mat.mean(axis=0)  
    _, indices = torch.topk(mean_diff, top_k)
    print(indices)
    mask = torch.ones(embed_dim, dtype=torch.float32, device=device)
    for i in indices:
        mask[i] *= 0.0
    replace_hook.neurons_zero_mask = mask

    if extra:
        model.visual.attnpool.MHA_decomposed.extra = True

    return replace_hook


def hook_register_neurons_mean(model, mask, means):
    replace_hook = ReplaceHook(model)
    rn_manager = model.hook_manager.forks['visual']
    hook_fn = partial(replace_hook._replace_neurons_mean, mask=mask, means=means)
    rn_manager.register('input_tokens', hook_fn)
    return replace_hook
