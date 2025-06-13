# Dataset
import csv
from abc import abstractmethod
from enum import Enum
from pathlib import Path

from torch.utils.data import Dataset


class Partition(Enum):
    TRAIN = "train"
    VAL = "val"


class Tokenizer:

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        pass


class Tiktokenizer(Tokenizer):

    def __init__(self, encoding_name: str = ""):
        import tiktoken
        self.encoding = tiktoken.get_encoding(encoding_name)

    def tokenize(self, text: str) -> list[int]:
        return self.encoding.encode(text)


class Tokens(Enum):
    START = "START "
    END = "<|endoftext|>"
    PAD = " PAD"
    START_NUM = 23380
    END_NUM = 100257
    PAD_NUM = 62854


class EnFrDataset(Dataset):

    def __init__(self, file: Path | str, partition: Partition = Partition.TRAIN, val_ratio: float = 0.1):
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
                fr = Tokens.START.value + row[1]
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
    def build_train_sample(tokenizer, en_str: str, dec_str: str):
        en_encoded = tokenizer.tokenize(en_str)
        dec_encoded = tokenizer.tokenize(dec_str)
        dec_encoded.append(Tokens.END_NUM.value)
        en_sents = []
        dec_sents = []
        target_sents = []

        for i in range(1, len(dec_encoded)):
            dec_sents.append(dec_encoded[:i])
            target_sents.append(dec_encoded[1: i + 1])
        en_sents.extend([en_encoded] * len(dec_sents))
        return list(zip(en_sents, dec_sents, target_sents))

    def __init__(self, file: Path | str, tokenizer: Tokenizer = Tiktokenizer("cl100k_base"), partition: Partition = Partition.TRAIN, val_ratio: float = 0.1):
        self._dataset = EnFrDataset(file, partition, val_ratio=0)
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
            for sample in self.build_train_sample(tokenizer, en, fr):
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
