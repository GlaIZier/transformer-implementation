import torch
from minbpe_tokenizer.tokenizer import SpecialTokenizer, RegexTokenizer


class TransformerTokenizer:

    def __init__(self, vocab_size: int = 512):
        self._tokenizer = SpecialTokenizer.default_trained(vocab_size=vocab_size, tokenizer=RegexTokenizer())

    def encode(self, texts: list[str], start: bool = True, end: bool = True, pad: bool = True, max_len: int = 512):
        list_ids = []
        for text in texts:
            list_ids.append(self._tokenizer.encode(text=text, start=start, end=end, pad=pad, max_len=max_len))
        return torch.tensor(list_ids)