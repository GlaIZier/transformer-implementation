from torch import nn

from transformer.decoder import Decoder
from transformer.encoder import Encoder


class Output(nn.Module):

    def __init__(self, vocab_size: int = 2**13, d: int = 512, ff_weight = None):
        super().__init__()
        self.ff = nn.Linear(d, vocab_size)
        # weight tying with the decoder embedding
        if ff_weight is not None:
            self.ff.weight = ff_weight

    def forward(self, x):
        return self.ff(x)


class Transformer(nn.Module):

    def __init__(self, vocab_size: int = 2 ** 13, n: int = 6, d: int = 512, h: int = 8, embed_tying=True):
        super().__init__()
        self.encoder = Encoder(vocab_size=vocab_size, n=n, d=d, h=h)
        self.decoder = Decoder(vocab_size=vocab_size, n=n, d=d, h=h)
        if embed_tying:
            self.output = Output(vocab_size=vocab_size, d=d, ff_weight=self.decoder.get_embed_weights())
        else:
            self.output = Output(vocab_size=vocab_size, d=d)

    def forward(self, enc_x, dec_x, enc_mask=None, dec_mask=None):
        encoded = self.encoder(enc_x, enc_mask)
        decoded = self.decoder(dec_x=dec_x, enc_x=encoded, self_mask=dec_mask, cross_mask=enc_mask)
        return self.output(decoded)