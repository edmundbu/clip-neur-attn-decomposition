def get_dims(model):
    """Returns (num_heads, num_tokens, embed_dim, out_dim)."""
    mha = model.visual.attnpool
    return mha.num_heads, (model.visual.image_size // 32) ** 2 + 1, mha.k_proj.bias.shape[0], mha.c_proj.bias.shape[0]