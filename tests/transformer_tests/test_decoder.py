import torch

from transformer.decoder import DecoderLayer
from transformer.mask import build_padding_mask, build_causal_mask, merge_masks, reshape_mask


def test_decoder_layer():
    decoder_layer = DecoderLayer(h=2, d=16)
    x = torch.rand(3, 3, 16)
    y = torch.rand(3, 3, 16)
    self_mask1 = build_padding_mask(torch.tensor([[2, 2, 0], [2, 0, 0], [2, 2, 0]]), pad_token=0)
    self_mask2 = build_causal_mask(torch.tensor([[2, 2, 0], [2, 0, 0], [2, 2, 0]]))
    self_mask = merge_masks(self_mask1, self_mask2)
    print(f"self_mask: \n {self_mask}")
    self_mask = reshape_mask(self_mask)

    cross_mask = build_padding_mask(torch.tensor([[2, 2, 2], [2, 0, 0], [2, 2, 0]]), pad_token=0)
    cross_mask = reshape_mask(cross_mask)
    print(f"cross_mask: \n {cross_mask}")
    decoded = decoder_layer(x, y, self_mask=self_mask, cross_mask=cross_mask)
    print(decoded.shape)
    assert decoded.shape == x.shape

    y = torch.rand(3, 2, 16) # different length of encoded seq
    cross_mask = build_padding_mask(torch.tensor([[2, 2], [2, 0], [2, 2]]), pad_token=0)
    cross_mask = reshape_mask(cross_mask)
    print(f"cross_mask shorter: \n {cross_mask}")
    decoded = decoder_layer(x, y, self_mask=self_mask, cross_mask=cross_mask)
    print(decoded.shape)
    assert decoded.shape == x.shape

