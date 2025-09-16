from typing import Optional

import torch


def get_topk_nh_pairs(pcs, text_embeddings, k, toggle_print=False):
    """Returns top heads and top neurons that correspond to the top neuron-head pairs."""
    num_classes = text_embeddings.shape[0]
    cached_heads = torch.empty(num_classes, k, dtype=torch.int64, device='cuda')
    cached_neurons = torch.empty(num_classes, k, dtype=torch.int64, device='cuda')
    top_scores = torch.empty(num_classes, k, dtype=torch.float32, device='cuda')
    num_heads, embed_dim, out_dim = pcs.shape
    pcs = pcs.reshape(num_heads*embed_dim, out_dim)
    for idx, text_embedding in enumerate(text_embeddings):
        dots = torch.matmul(pcs, text_embedding)
        topk_sims, topk_indices = torch.topk(dots, k)
        top_scores[idx] = topk_sims
        if toggle_print:
            print(f"Similarity scores: {top_scores}")
        topk_h = torch.div(topk_indices, embed_dim, rounding_mode='floor')
        topk_c = topk_indices % embed_dim
        cached_heads[idx] = topk_h
        if toggle_print:
            print(f"Head indices: {cached_heads[idx]}")
        cached_neurons[idx] = topk_c
        if toggle_print:
            print(f"Neuron indices: {cached_neurons[idx]}")
    return top_scores, cached_heads, cached_neurons


def threshold_pairs(scores, heads, neurons, divide_by):  
    """Keep only the tensors above (top similarity / divide_by) and pad the rest."""
    # should threshold the neuron-head pairs, pad the tensors, and return a count of the non-zero elements
    num_classes = scores.shape[0]
    new_lengths = torch.empty(num_classes, dtype=torch.int64, device=heads.device)
    new_heads, new_neurons = torch.zeros_like(heads), torch.zeros_like(neurons)
    for idx in range(num_classes):
        threshold = scores[idx][0] / divide_by
        filtered = scores[idx][scores[idx] > threshold]
        keep_shape = filtered.shape[0]
        new_lengths[idx] = keep_shape
        new_heads[idx, 0:keep_shape] = heads[idx, 0:keep_shape]
        new_neurons[idx, 0:keep_shape] = neurons[idx, 0:keep_shape]
    return new_lengths, new_heads, new_neurons


def interpolate_sink(heatmap, h, w):
    """Replaces sink patch with mean of patches around it. Expects BNHO or BNC."""
    west_patch = heatmap[:, -2:-1, ...]
    north_patch = heatmap[:, -(w+1):-w, ...]
    northwest_patch = heatmap[:, -(w+2):-(w+1), ...]
    heatmap[:, -1, ...] = (west_patch + north_patch + northwest_patch) / 3
    return heatmap
