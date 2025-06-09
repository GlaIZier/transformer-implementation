import torch

from transformer.mask import build_padding_mask, reshape_mask, build_causal_mask, merge_masks


def test_reshape_mask():
    x = torch.zeros((2, 3))
    mask = reshape_mask(x)
    print(mask)
    assert mask.shape == (2, 1, 1, 3)

def test_build_padding_mask():
    x = torch.rand(5, 6)
    x[0, -3:] = 100
    x[1, -2:] = 100
    x[2, -1] = 100
    x[3, :] = 100
    print(x)
    mask = build_padding_mask(x, 100)
    print(mask)
    assert mask[0, -3:].sum() == 0
    assert mask[0, :-3].sum() == 3
    assert mask[1, -2:].sum() == 0
    assert mask[1, :-2].sum() == 4
    assert mask[2, -1:].sum() == 0
    assert mask[2, :-1].sum() == 5
    assert mask[3, :].sum() == 0

def test_build_causal_mask():
    x = torch.rand(3, 3)
    mask = build_causal_mask(x)
    print(mask)
    assert mask[0, 1:].sum() == 0
    assert mask[0, 0] != 0
    assert mask[1, 2:].sum() == 0
    assert mask[1, :2].sum() != 0
    assert mask[2, 2] != 0

def test_merge_masks():
    m1 = torch.tensor([[1, 1, 1], [0, 1, 1]])
    m2 = torch.tensor([[1, 1, 0], [0, 0, 0]])
    merged = merge_masks(m1, m2)
    print(merged)
    assert merged[0, 2] == 0
    assert merged[0, :2].sum() == 2
    assert merged[1, :].sum() == 0