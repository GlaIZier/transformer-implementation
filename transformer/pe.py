import torch
from torch import nn


class PE(nn.Module):

    def __init__(self, d: int = 512, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.dropout = nn.Dropout(p=dropout)

        twoi = torch.arange(0, self.d, 2)
        pow_ = torch.pow(10000, twoi / self.d)
        position = torch.arange(0, max_len).unsqueeze(1)
        sin_p = torch.sin(position / pow_)
        cos_p = torch.cos(position / pow_)
        pe = torch.zeros(max_len, self.d, requires_grad=False) # Explicit, register buffer insures requires grad = False
        pe[:, 0::2] = sin_p
        pe[:, 1::2] = cos_p
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        b, t, d = x.size()
        # print(x.size())
        x = x + self.pe[:, :t, :]
        # print(self.pe.size())
        return self.dropout(x)


class PEEmbed(nn.Module):

    def __init__(self, d: int = 512, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.pe = nn.Embedding(max_len, d)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        b, t, d = x.size()
        print(x.size())
        pos = self.pe(torch.arange(t))
        x = x + pos
        return self.dropout(x)


#
# class PE1():
#     def __init__(self, d: int = 512):
#         self.d = d
#     # -> d vector
#     def __call__(self, pos):
#         pow = torch.pow(10000, torch.arange(0, self.d) / self.d)
#         return torch.sin(torch.arange(0, self.d) / pow)
#
# print(PE1()(1).size()) # torch.Size([512])
#
# class PEScalar():
#     def __init__(self, d: int = 512):
#         self.d = d
#     # 1 -> d vector
#     def __call__(self, pos):
#         twoi = torch.arange(0, self.d, 2)
#         pow_ = torch.pow(10000, twoi / self.d)
#         sin_p = torch.sin(pos / pow_)
#         cos_p = torch.cos(pos / pow_)
#         # a = torch.arange(0, 12, 2)
#         # b = torch.arange(1, 12, 2)
#         # torch.stack((a, b), dim=1).view(-1)
#         return torch.stack((sin_p, cos_p), dim=-1).view(-1)
#
# print(PEScalar()(1).size()) # torch.Size([1, 512])
#
# class PEVector():
#     def __init__(self, d: int = 512):
#         self.d = d
#     # 1 -> 1 d
#     # t 1 -> t d
#     def __call__(self, pos):
#         twoi = torch.arange(0, self.d, 2)
#         pow_ = torch.pow(10000, twoi / self.d)
#         sin_p = torch.sin(pos / pow_)
#         cos_p = torch.cos(pos / pow_)
#         # a = torch.arange(0, 12, 2).view(-1, 2)
#         # b = torch.arange(1, 12, 2).view(-1, 2)
#         # torch.stack((a, b), dim=-1).view(-1, 4)
#         return torch.stack((sin_p, cos_p), dim=-1).view(-1, self.d)
#
# print(PEVector()(1).size()) # torch.Size([1, 512])
# print(PEVector()(torch.arange(3).view(-1, 1)).size()) # torch.Size([3, 512])
#
# class PE():
#     def __init__(self, d: int = 512):
#         self.d = d
#     # b t 1 -> b t d
#     def __call__(self, pos):
#         b, t, _ = pos.size()
#         twoi = torch.arange(0, self.d, 2)
#         pow_ = torch.pow(10000, twoi / self.d)
#         sin_p = torch.sin(pos / pow_)
#         cos_p = torch.cos(pos / pow_)
#         # a = torch.arange(0, 12, 2).view(-1, 2)
#         # b = torch.arange(1, 12, 2).view(-1, 2)
#         # torch.stack((a, b), dim=-1).view(-1, 4)
#         return torch.stack((sin_p, cos_p), dim=-1).view(-1, t, self.d)
#
# print(PE()(torch.arange(6).view(-1, 3, 1)).size()) # torch.Size([2, 3, 512])
#
# class PEAnotherImpl():
#     def __init__(self, d: int = 512):
#         self.d = d
#     # b t 1 -> b t d
#     def __call__(self, pos):
#         b, t, _ = pos.size()
#         twoi = torch.arange(0, self.d, 2)
#         pow_ = torch.pow(10000, twoi / self.d)
#         max_len = 1024
#         position = torch.arange(0, max_len).unsqueeze(1)
#         sin_p = torch.sin(position / pow_)
#         cos_p = torch.cos(position / pow_)
#         pe = torch.zeros(max_len, self.d)
#         pe[:, 0::2] = sin_p
#         pe[:, 1::2] = cos_p
#         return pe[:t, :].unsqueeze(0).repeat(b, 1, 1)
#
# print(PEAnotherImpl()(torch.arange(6).view(-1, 3, 1)).size()) # torch.Size([2, 3, 512])
#
# class PEModule(nn.Module):
#
#     def __init__(self, d: int = 512, max_len: int = 1024, dropout: float = 0.1):
#         super().__init__()
#         self.d = d
#         self.dropout = nn.Dropout(p=dropout)
#
#         twoi = torch.arange(0, self.d, 2)
#         pow_ = torch.pow(10000, twoi / self.d)
#         pos = torch.arange(max_len).unsqueeze(1)
#         print(pos.size())
#         sin_p = torch.sin(pos / pow_)
#         cos_p = torch.cos(pos / pow_)
#         print(sin_p.size())
#         print(cos_p.size())
#         pe = torch.stack((sin_p, cos_p), dim=-1).view(-1, self.d) # downside sin, cos don't alternate
#         print(pe.size())
#         pe = pe.unsqueeze(0)
#         print(pe.size())
#         self.register_buffer("pe", pe)
#
#     def forward(self, x):
#         # x.size: b, t, d
#         x = x + self.pe[:, : t, :]
#         return self.dropout(x)
# print(PEModule(d=4)(torch.arange(24).view(-1, 3, 4)).size()) # torch.Size([2, 3, 4])
#
# class PositionalEncodingAnnotatedTransformerModule(nn.Module):
#     "Implement the PE function."
#
#     def __init__(self, d_model, dropout, max_len=5000):
#         super(PositionalEncodingAnnotatedTransformer, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         print(position.size())
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
#         )
#         print(div_term.size())
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         print(pe.size())
#         self.register_buffer("pe", pe)
#
#     def forward(self, x):
#         print(self.pe[:, : x.size(1)].size())
#         x = x + self.pe[:, : x.size(1)].requires_grad_(False)
#         return self.dropout(x)
#
# print(PositionalEncodingAnnotatedTransformerModule(512, 0.1)(torch.arange(6).view(-1, 3, 1)).size()) # torch.Size([2, 3, 512])