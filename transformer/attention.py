import torch
import torch.nn.functional as F
from torch import nn

from transformer.mask import reshape_mask


def self_attention_unmasked(q, k, v):
    # if 3 dim: b t d
    # prod = Q.bmm(K.permute(0, 2, 1))
    # or
    # prod = torch.einsum("btd, bsd -> bts", q, k)
    # if 4 dim: b h t dh
    prod = torch.einsum("bhtd, bhsd -> bhts", q, k)
    scaled_prod = prod / torch.sqrt(torch.tensor(q.shape[-1]))
    softmaxed_prod = F.softmax(scaled_prod, dim=-1)
    # print(softmaxed_prod.shape)
    # print(softmaxed_prod)
    return softmaxed_prod @ v

def self_attention(q, k, v, mask=None, verbose=False):
    # if 3 dim: b t d
    #prod = Q.bmm(K.permute(0, 2, 1))
    # or
    # prod = torch.einsum("btd, bsd -> bts", q, k)
    # if 4 dim: b t h dh:
    prod = torch.einsum("bthd, bshd -> bhts", q, k)
    scaled_prod = prod/torch.sqrt(torch.tensor(q.shape[-1]))
    if verbose:
        print(f"scaled_prod.shape: \n {scaled_prod.shape}")
    # mask should be in shape to be broadcastable to bhts and lead to masked keys only (last s dim), e.g. # b t -> b 1 1 t
    if mask is not None:
        mask = mask if scaled_prod.dim() == mask.dim() else reshape_mask(mask)
        scaled_prod = scaled_prod.masked_fill(mask == 0, float("-inf"))
    if verbose:
        print(f"scaled_prod: \n {scaled_prod}")
    softmaxed_prod = F.softmax(scaled_prod, dim=-1)
    if verbose:
        print(f"softmaxed_prod: \n {softmaxed_prod}")
    # swap h and t in v
    return softmaxed_prod @ v.permute(0, 2, 1, 3)



class MHSAUnmasked(nn.Module):
    def __init__(self, d: int = 512, h: int = 8):
        super().__init__()
        assert d % h == 0
        self.d = d
        self.dh = d // h
        self.h = h
        self.wq = nn.Linear(self.d, self.d)
        self.wk = nn.Linear(self.d, self.d)
        self.wv = nn.Linear(self.d, self.d)
        self.wo = nn.Linear(self.d, self.d)

    def forward(self, q, k, v):
        # b, t, d
        b, t, d = q.size()
        wq = self.wq(q)
        wk = self.wk(k)
        wv = self.wv(v)
        wq = wq.view(b, t, self.h, self.dh)
        wk = wk.view(b, t, self.h, self.dh)
        wv = wv.view(b, t, self.h, self.dh)
        # b, t, h, dh
        # if changing from 4 dim -> 3 dim: b*h, t, dh
        # wq = wq.permute(0, 2, 1, 3).reshape(b * self.h, t, self.dh)
        # wk = wk.permute(0, 2, 1, 3).reshape(b * self.h, t, self.dh)
        # wv = wv.permute(0, 2, 1, 3).reshape(b * self.h, t, self.dh)
        # another option 4 dim -> 3 dim
        # wq = wq.transpose(1, 2).contiguous().view(b * self.h, t, self.dh)
        # wk = wk.transpose(1, 2).contiguous().view(b * self.h, t, self.dh)
        # wv = wv.transpose(1, 2).contiguous().view(b * self.h, t, self.dh)
        # changing the number of dims is not necessary as @ supports 4 dims
        attn = self_attention(wq, wk, wv)
        # b * h, t, dh
        # attn = attn.view(b, self.h, t, self.dh).permute(0, 2, 1, 3).reshape(b, t, d)
        attn = attn.view(b, self.h, t, self.dh).transpose(1, 2).contiguous().view(b, t, d)
        wo = self.wo(attn)
        # -> b t d
        return wo
        # # 1 2 3 4
        # x = F.relu(self.conv1(x))
        # return F.relu(self.conv2(x))


class MHSA(nn.Module):
    def __init__(self, d: int = 512, h: int = 8):
        super().__init__()
        assert d % h == 0
        self.d = d
        self.dh = d // h
        self.h = h
        self.wq = nn.Linear(self.d, self.d)
        self.wk = nn.Linear(self.d, self.d)
        self.wv = nn.Linear(self.d, self.d)
        self.wo = nn.Linear(self.d, self.d)

    def forward(self, q, k, v, mask=None):
        # q and k/v might be of different sizes if lengths of decoder and encoders inputs are different
        bq, tq, dq = q.size()
        bk, tk, dk = k.size()
        wq = self.wq(q)
        wk = self.wk(k)
        wv = self.wv(v)
        wq = wq.view(bq, tq, self.h, self.dh)
        wk = wk.view(bk, tk, self.h, self.dh)
        wv = wv.view(bk, tk, self.h, self.dh)
        # b, t, h, dh
        # if changing from 4 dim -> 3 dim: b*h, t, dh
        # wq = wq.permute(0, 2, 1, 3).reshape(b * self.h, t, self.dh)
        # wk = wk.permute(0, 2, 1, 3).reshape(b * self.h, t, self.dh)
        # wv = wv.permute(0, 2, 1, 3).reshape(b * self.h, t, self.dh)
        # another option 4 dim -> 3 dim
        # wq = wq.transpose(1, 2).contiguous().view(b * self.h, t, self.dh)
        # wk = wk.transpose(1, 2).contiguous().view(b * self.h, t, self.dh)
        # wv = wv.transpose(1, 2).contiguous().view(b * self.h, t, self.dh)
        # changing the number of dims is not necessary as @ supports 4 dims
        attn = self_attention(wq, wk, wv, mask=mask)
        # b * h, t, dh
        # attn = attn.view(b, self.h, t, self.dh).permute(0, 2, 1, 3).reshape(b, t, d)
        attn = attn.view(bq, self.h, tq, self.dh).transpose(1, 2).contiguous().view(bq, tq, dq)
        wo = self.wo(attn)
        return wo
        # # 1 2 3 4
        # x = F.relu(self.conv1(x))
        # return F.relu(self.conv2(x))

