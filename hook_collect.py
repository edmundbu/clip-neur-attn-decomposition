from functools import partial

import einops

from utils.get_dims import get_dims


class CollectHook():
    def __init__(self):
        self.out_mat = None
    
    def _collect_neurons_plus_heads(self, ret, bias_scaled):
        self.out_mat = ret + bias_scaled  # BHOC
        return ret
    
    def _collect_attn(self, ret):
        self.out_mat = ret  # BHNN 
        return ret
    
    def _collect_out_tokens_cls(self, ret, bias_scaled, collapse_heads):
        if collapse_heads:
            self.out_mat = ret[:, 0, :, :, :].sum(axis=-2) + bias_scaled  # BNMHO -> BMO
        else: 
            self.out_mat = ret[:, 0, ...] + bias_scaled  # BNMHO -> BMHO
        return ret

    def _collect_out_tokens(self, ret, bias_scaled, collapse_heads):
        if collapse_heads: 
            self.out_mat = ret.squeeze(0).sum(axis=[1, 2]) + bias_scaled  # BNNHO -> NO
        else:
            raise NotImplementedError()
        return ret
    
    def _collect_in_tokens(self, ret):
        self.out_mat = ret.permute(1, 0, 2)  # NBC -> BNC
        return ret
    
    def _collect_conv_maps(self, ret, embed_dim):
        if ret.shape[1] == embed_dim:
            self.out_mat.append(ret)
        return ret
    
    def reset_conv_maps(self):
        self.out_mat = []

    def _collect_neurons_spatial(self, ret, bias_scaled):
        self.out_mat = ret + bias_scaled
        return ret

    def get_out_mat(self, as_np=False):
        return self.out_mat.cpu().numpy() if as_np else self.out_mat
    

def hook_register_neurons_heads(model):
    """Returns output decomposition of shape BHOC."""
    assert model.visual.attnpool.forward_mode == 'per_head_per_neuron'
    collect_hook = CollectHook()
    rn_manager = model.hook_manager.forks['visual']
    num_heads, num_tokens, embed_dim, out_dim = get_dims(model)
    bias_scaled = model.visual.attnpool.c_proj.bias / (embed_dim * num_heads)
    bias_scaled = einops.repeat(bias_scaled, 'o -> h o c', h=num_heads, c=embed_dim)
    hook_fn = partial(collect_hook._collect_neurons_plus_heads, bias_scaled=bias_scaled)
    rn_manager.register('out.pre_collapse', hook_fn)
    return collect_hook


def hook_register_attention(model, mode='post'):
    assert model.visual.attnpool.forward_mode != 'default'
    collect_hook = CollectHook()
    rn_manager = model.hook_manager.forks['visual']
    if mode == 'pre':
        rn_manager.register('attn.pre_softmax', collect_hook._collect_attn)
    elif mode =='post':
        rn_manager.register('attn.post_softmax', collect_hook._collect_attn)
    return collect_hook


def hook_register_tokens(model, mode: str, collapse_heads=True):
    """
    Collects output or input tokens.
    
    mode:
    - 'cls_spatial' returns output decomposition of shape BMO or BMHO 
    - 'no_spatial' TODO
    - 'input returns input post-pos-embed of shape BNC
    """
    collect_hook = CollectHook()
    rn_manager = model.hook_manager.forks['visual']
    num_heads, num_tokens, embed_dim, out_dim = get_dims(model)
    if mode == 'cls_spatial':
        assert model.visual.attnpool.forward_mode == 'per_head'
        bias_scaled = model.visual.attnpool.c_proj.bias / num_tokens  # output is summed
        if not collapse_heads:
            bias_scaled = bias_scaled / num_heads
        hook_fn = partial(collect_hook._collect_out_tokens_cls, bias_scaled=bias_scaled, collapse_heads=collapse_heads)
        rn_manager.register('out.pre_collapse', hook_fn)
    elif mode =='no_spatial':
        assert model.visual.attnpool.forward_mode == 'per_head'
        bias_scaled = model.visual.attnpool.c_proj.bias  # output is indexed
        if not collapse_heads:
            bias_scaled = bias_scaled / num_heads
        hook_fn = partial(collect_hook._collect_out_tokens, bias_scaled=bias_scaled, collapse_heads=collapse_heads)
        rn_manager.register('out.pre_collapse', hook_fn)
    elif mode =='input':
        rn_manager.register('input_tokens', collect_hook._collect_in_tokens)
    return collect_hook


def hook_register_conv_maps(model, embed_dim):
    """Returns a list of per-block activation maps from layer 4 of ResNet."""
    collect_hook = CollectHook()
    collect_hook.out_mat = []
    rn_manager = model.hook_manager.forks['visual']
    hook_fn = partial(collect_hook._collect_conv_maps, embed_dim=embed_dim)
    rn_manager.register('post_act', hook_fn)
    return collect_hook


def hook_register_neurons_spatial(model):
    """Returns output decomposition of shape BNCO."""
    assert model.visual.attnpool.forward_mode == 'per_neuron_spatial'
    collect_hook = CollectHook()
    rn_manager = model.hook_manager.forks['visual']
    num_heads, num_tokens, embed_dim, out_dim = get_dims(model)
    bias_scaled = model.visual.attnpool.c_proj.bias / (embed_dim * num_tokens)
    bias_scaled = einops.repeat(bias_scaled, 'o -> n c o', n=num_tokens, c=embed_dim)
    hook_fn = partial(collect_hook._collect_neurons_spatial, bias_scaled=bias_scaled)
    rn_manager.register('out.pre_collapse', hook_fn)
    return collect_hook