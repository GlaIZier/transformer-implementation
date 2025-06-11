import torch

from transformer.mask import build_padding_mask, reshape_mask, build_causal_mask, merge_masks
from transformer.transformer import Transformer


def test_transformer():
    transformer = Transformer(vocab_size=32, n=2, d=16, h=2, embed_tying=False)
    enc_x = torch.tensor([[15, 7, 3], [10, 10, 0], [1, 0, 0]])
    dec_x = torch.tensor([[21, 8, 0, 0], [25, 0, 0, 0], [8, 1, 2, 3]])
    # dec_x = torch.tensor([[21, 8], [25, 0], [8, 1]])

    enc_mask = build_padding_mask(enc_x, pad_token=0)
    print(f"enc_mask: \n {enc_mask}")
    enc_mask = reshape_mask(enc_mask)

    dec_mask1 = build_padding_mask(dec_x, pad_token=0)
    dec_mask2 = build_causal_mask(dec_x)
    dec_mask = merge_masks(dec_mask1, dec_mask2)
    print(f"dec_mask: \n {dec_mask}")
    dec_mask = reshape_mask(dec_mask)

    transformed = transformer(enc_x, dec_x, enc_mask=enc_mask, dec_mask=dec_mask)
    print(transformed.shape)
    assert transformed.shape == (3, 4, 32)