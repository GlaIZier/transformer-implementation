import torch

from transformer.decoder import DecoderLayer, Decoder
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

def test_decoder():
    decoder = Decoder(vocab_size=32, n=2, d=16, h=2)
    # x = torch.randint(0, 32, (2, 3))
    x = torch.tensor([[15, 7, 0], [10, 0, 0], [1, 3, 0]])
    y = torch.rand(3, 3, 16)

    self_mask1 = build_padding_mask(x, pad_token=0)
    self_mask2 = build_causal_mask(x)
    self_mask = merge_masks(self_mask1, self_mask2)
    print(f"self_mask: \n {self_mask}")
    self_mask = reshape_mask(self_mask)

    cross_mask = build_padding_mask(torch.tensor([[2, 2, 2], [2, 0, 0], [2, 2, 0]]), pad_token=0)
    cross_mask = reshape_mask(cross_mask)
    print(f"cross_mask: \n {cross_mask}")
    decoded = decoder(x, y, self_mask=self_mask, cross_mask=cross_mask)
    print(decoded.shape)
    assert decoded.shape == (3, 3, 16)

    y = torch.rand(3, 4, 16) # longer than decoder's sequence
    cross_mask = build_padding_mask(torch.tensor([[2, 2, 2, 2], [2, 0, 0, 0], [2, 2, 0, 0]]), pad_token=0)
    cross_mask = reshape_mask(cross_mask)
    print(f"cross_mask longer: \n {cross_mask}")
    decoded = decoder(x, y, self_mask=self_mask, cross_mask=cross_mask)
    print(decoded.shape)
    assert decoded.shape == (3, 3, 16)
