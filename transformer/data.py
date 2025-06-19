# Dataset
import csv
from abc import abstractmethod
from enum import Enum
from pathlib import Path

import torch
from minbpe_tokenizer import data
from minbpe_tokenizer.tokenizer import SpecialTokenizer, RegexTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from transformer.mask import build_padding_mask


class Partition(Enum):
    TRAIN = "train"
    VAL = "val"

class Tokenizer:

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        pass

    def vocab_size(self):
        pass

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

    def vocab_size(self):
        return len(self.tokenizer)

    def get_special_tokens(self):
        return self.special_tokens


class EnFrDataset(Dataset):

    def __init__(self, file: Path | str, tokenizer: Tokenizer = Tiktokenizer("cl100k_base"), partition: Partition = Partition.TRAIN, val_ratio: float = 0.1):
        # partition = TRAIN | VAL
        self._partition = partition
        self._val_ratio = val_ratio

        self._data = []
        self._train_map: dict[int, int] = {}
        self._val_map: dict[int, int] = {}
        train_id = 0
        val_id = 0
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            # we want data indexes start from 0, but filter out the first header row
            for i, row in enumerate(reader, start=-1):
                if i == -1:
                    continue
                en = row[0]
                fr = tokenizer.get_special_tokens().start + row[1]
                self._data.append(tuple([en, fr]))
                if int(i * val_ratio) == int((i - 1) * val_ratio):
                    self._train_map[train_id] = i
                    train_id += 1
                else:
                    self._val_map[val_id] = i
                    val_id += 1

    class Iterator:

        def __init__(self, outer):
            self.cur = 0
            self.outer = outer

        def __next__(self):
            if self.cur == len(self.outer._data):
                raise StopIteration()
            cur = self.outer._data[self.cur]
            self.cur += 1
            return cur

    def __iter__(self):
        return EnFrDataset.Iterator(self)

    @property
    def partition(self):
        return self._partition

    @partition.setter
    def partition(self, partition):
        self._partition = partition

    def __len__(self):
        return len(self._train_map) if self._partition == Partition.TRAIN else len(self._val_map)

    def __getitem__(self, idx):
        return self._data[self._train_map[idx]] if self._partition == Partition.TRAIN else self._data[
            self._val_map[idx]]


class TokEnFrDataset(Dataset):

    @staticmethod
    def build_train_sample(en_str: str, dec_str: str, tokenizer):
        en_encoded = tokenizer.tokenize(en_str)
        dec_encoded = tokenizer.tokenize(dec_str)
        dec_encoded.append(tokenizer.get_special_tokens().end_num)
        en_sents = []
        dec_sents = []
        target_sents = []

        for i in range(1, len(dec_encoded)):
            dec_sents.append(dec_encoded[:i])
            target_sents.append(dec_encoded[1: i + 1])
        en_sents.extend([en_encoded] * len(dec_sents))
        return list(zip(en_sents, dec_sents, target_sents))

    def __init__(self, file: Path | str, tokenizer: Tokenizer = Tiktokenizer("cl100k_base"), partition: Partition = Partition.TRAIN, val_ratio: float = 0.1):
        self._dataset = EnFrDataset(file, tokenizer=tokenizer, partition=partition, val_ratio=0)
        # partition = TRAIN | VAL
        self._partition = partition
        self._val_ratio = val_ratio

        self._data = []
        self._train_map: dict[int, int] = {}
        self._val_map: dict[int, int] = {}
        train_id = 0
        val_id = 0
        i = 0
        for en, fr in self._dataset:
            for sample in self.build_train_sample(en, fr, tokenizer):
                self._data.append(sample)
                if int(i * val_ratio) == int((i - 1) * val_ratio):
                    self._train_map[train_id] = i
                    train_id += 1
                else:
                    self._val_map[val_id] = i
                    val_id += 1
                i += 1

    @property
    def partition(self):
        return self._partition

    @partition.setter
    def partition(self, partition):
        self._partition = partition

    def __len__(self):
        return len(self._train_map) if self._partition == Partition.TRAIN else len(self._val_map)

    def __getitem__(self, idx):
        return self._data[self._train_map[idx]] if self._partition == Partition.TRAIN else self._data[
            self._val_map[idx]]

# for partial usage since pickle doesn't handle closures
def collate(batch, pad_num):
    # print(batch)
    _x, _y, _label = list(zip(*batch))
    enc_x = pad_sequence([torch.tensor(t) for t in _x], batch_first=True, padding_value=pad_num)
    dec_x = pad_sequence([torch.tensor(t) for t in _y], batch_first=True, padding_value=pad_num)
    label = pad_sequence([torch.tensor(t) for t in _label], batch_first=True, padding_value=pad_num)
    enc_mask = build_padding_mask(enc_x, pad_token=pad_num)
    dec_mask = build_padding_mask(dec_x, pad_token=pad_num)
    return enc_x, dec_x, label, enc_mask, dec_mask
