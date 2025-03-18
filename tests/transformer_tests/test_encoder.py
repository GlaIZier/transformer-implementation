import torch

from transformer.encoder import Encoder
from transformer.tokenizer import TransformerTokenizer


def test_a():
    assert 1 == 1

def test_encoder_inference():
    texts = ["Bonjour, ça va?", "This is a small text for testing", "👌"]
    vocab_size = 384
    tokenizer = TransformerTokenizer(vocab_size=vocab_size)
    ids = tokenizer.encode(texts, max_len=32)
    encoder = Encoder()
    assert encoder(ids).size() == torch.Size([3, 32, 512])
