import torch.nn.functional as F
from torch import nn

from transformer.attention import MHSA
from transformer.pe import PE


class EncoderLayerUnmasked(nn.Module):

    def __init__(self, d: int = 512, h: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mhsa = MHSA(d, h)
        self.norm1 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, d * 4)
        self.ff2 = nn.Linear(d * 4, d)
        self.norm2 = nn.LayerNorm(d)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, d = x.size()
        x = x + self.attn_dropout(self.mhsa(x, x, x))
        x = self.norm1(x)
        x = x + self.resid_dropout(self.ff2(F.relu(self.ff1(x))))
        x = self.norm2(x)
        return x

class EncoderUnmasked(nn.Module):

    def __init__(self, vocab_size: int = 2**13, max_len: int = 1024, n: int = 6, d: int = 512, h: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pe = PE(d=d, max_len=max_len)
        self.layers = [EncoderLayer(d, h) for _ in range(n)] # bug: without nn.ModuleList, can't push to device

    def forward(self, x):
        b, t = x.size()
        x = self.embed(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        return x

class EncoderLayer(nn.Module):

    def __init__(self, d: int = 512, h: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mhsa = MHSA(d, h)
        self.norm1 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, d * 4)
        self.ff2 = nn.Linear(d * 4, d)
        self.norm2 = nn.LayerNorm(d)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, self_mask=None):
        # b, t, d = x.size()
        x = x + self.attn_dropout(self.mhsa(x, x, x, mask=self_mask))
        x = self.norm1(x)
        x = x + self.resid_dropout(self.ff2(F.relu(self.ff1(x))))
        x = self.norm2(x)
        return x

class Encoder(nn.Module):

    def __init__(self, vocab_size: int = 2**13, n: int = 6, d: int = 512, h: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pe = PE(d=d)
        self.layers = nn.ModuleList([EncoderLayer(d, h) for _ in range(n)])

    def forward(self, x, self_mask = None):
        # b, t = x.size()
        x = self.embed(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, self_mask=self_mask)
        return x
