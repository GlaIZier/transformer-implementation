import torch

from transformer.encoder import EncoderUnmasked, EncoderLayerUnmasked, EncoderLayer, Encoder
from transformer.mask import build_padding_mask, reshape_mask
from transformer.tokenizer import TransformerTokenizer


def test_encoder_inference():
    texts = ["Bonjour, ça va?", "This is a small text for testing", "👌"]
    vocab_size = 384
    tokenizer = TransformerTokenizer(vocab_size=vocab_size)
    ids = tokenizer.encode(texts, max_len=32)
    encoder = EncoderUnmasked()
    assert encoder(ids).size() == torch.Size([3, 32, 512])


def test_encoder_layer():
    encoder_layer = EncoderLayer()
    self_mask = build_padding_mask(torch.tensor([[2, 2, 0], [2, 0, 0]]), pad_token=0)
    self_mask = reshape_mask(self_mask)
    x = torch.rand(2, 3, 512)

    encoded = encoder_layer(x, self_mask=self_mask)
    print(encoded.shape)
    assert encoded.shape == x.shape

def test_encoder():
    encoder = Encoder(d=512)
    x = torch.randint(0, 2 ** 13, (2, 3))
    self_mask = build_padding_mask(torch.tensor([[2, 2, 0], [2, 0, 0]]), pad_token=0)
    self_mask = reshape_mask(self_mask)
    encoded = encoder(x, self_mask)
    print(encoded.shape)
    assert encoded.shape == (2, 3, 512)