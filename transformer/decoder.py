import torch.nn.functional as F
from torch import nn

from transformer.attention import MHSA
from transformer.pe import PE


class DecoderLayer(nn.Module):

    def __init__(self, d: int = 512, h: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mhsa = MHSA(d=d, h=h)
        self.attn_norm = nn.LayerNorm(d)
        self.attn_dropout = nn.Dropout(dropout)

        self.mhca = MHSA(d=d, h=h)
        self.cross_attn_norm = nn.LayerNorm(d)
        self.cross_attn_dropout = nn.Dropout(dropout)

        self.ff1 = nn.Linear(d, d * 4)
        self.ff2 = nn.Linear(d * 4, d)
        self.resid_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, dec_x, enc_x, self_mask=None, cross_mask=None):
        # self_mask is merged decoders padding and causal masks
        # cross_mask is equal to encoders padding mask because we don't want to attend to encoded padded tokens
        # b, t, d = dec_x.size()
        x = dec_x + self.attn_dropout(self.mhsa(dec_x, dec_x, dec_x, mask=self_mask))
        x = self.attn_norm(x)

        x = x + self.cross_attn_dropout(self.mhca(x, enc_x, enc_x, mask=cross_mask))
        x = self.cross_attn_norm(x)

        x = x + self.resid_dropout(self.ff2(F.relu(self.ff1(x))))
        x = self.norm(x)
        return x

class Decoder(nn.Module):

    def __init__(self, vocab_size: int = 2**13, n: int = 6, d: int = 512, h: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pe = PE(d=d)
        self.layers = nn.ModuleList([DecoderLayer(d, h) for _ in range(n)])

    def forward(self, dec_x, enc_x, self_mask=None, cross_mask=None):
        b, t = dec_x.size()
        x = self.embed(dec_x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, enc_x, self_mask=self_mask, cross_mask=cross_mask)
        return x

    def get_embed_weights(self):
        return self.embed.weight