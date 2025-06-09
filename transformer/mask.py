import torch


def reshape_mask(mask):
    # b t -> b 1 1 t (to be broadcastable to b h t t)
    return mask.unsqueeze(1).unsqueeze(1)

def build_padding_mask(x, pad_token):
    # x: b t shape
    mask = torch.ones_like(x)
    return mask.masked_fill(x == pad_token, 0)

def build_causal_mask(x):
    # x: b t shape
    m = torch.ones_like(x)
    return torch.tril(m)

def merge_masks(m1, m2):
    return m1 * m2