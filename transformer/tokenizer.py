from abc import abstractmethod
from typing import Sequence

import torch
from minbpe_tokenizer import data
from minbpe_tokenizer.tokenizer import SpecialTokenizer, RegexTokenizer


class Tokenizer:

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def texify(self, tokens: Sequence[int]) -> str:
        pass

    @abstractmethod
    def vocab_size(self):
        pass

    @abstractmethod
    def get_special_tokens(self):
        pass


class SpecialTokens:
    def __init__(self, tokenizer: Tokenizer):
        if isinstance(tokenizer, Tiktokenizer):
            self.start = "START "
            self.end = tokenizer.encoding.eot_token
            self.pad = " PAD"
            self.start_num = 23380
            self.end_num = 100257
            self.pad_num = 62854
        elif isinstance(tokenizer, MinBpeTokenizer):
            self.start = SpecialTokenizer.START_TOKEN
            self.end = SpecialTokenizer.END_TOKEN
            self.pad = SpecialTokenizer.PAD_TOKEN
            self.start_num = tokenizer.tokenizer._special_vocab_inverted[self.start]
            self.end_num = tokenizer.tokenizer._special_vocab_inverted[self.end]
            self.pad_num = tokenizer.tokenizer._special_vocab_inverted[self.pad]
        else:
            raise ValueError


class Tiktokenizer(Tokenizer):

    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.special_tokens = SpecialTokens(self)

    def tokenize(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def texify(self, tokens: Sequence[int]) -> str:
        return self.encoding.decode(tokens)

    def vocab_size(self):
        return self.encoding.n_vocab

    def get_special_tokens(self):
        return self.special_tokens

class MinBpeTokenizer(Tokenizer):

    def __init__(self):
        self.tokenizer = SpecialTokenizer(tokenizer=RegexTokenizer())
        self.tokenizer.train(data.training_text)
        self.special_tokens = SpecialTokens(self)

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def texify(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(tokens)

    def vocab_size(self):
        return len(self.tokenizer)

    def get_special_tokens(self):
        return self.special_tokens

class TransformerTokenizer:

    def __init__(self, vocab_size: int = 512):
        self._tokenizer = SpecialTokenizer.default_trained(vocab_size=vocab_size, tokenizer=RegexTokenizer())

    def encode(self, texts: list[str], start: bool = True, end: bool = True, pad: bool = True, max_len: int = 512):
        list_ids = []
        for text in texts:
            list_ids.append(self._tokenizer.encode(text=text, start=start, end=end, pad=pad, max_len=max_len))
        return torch.tensor(list_ids)