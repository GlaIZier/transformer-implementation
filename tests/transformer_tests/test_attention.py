import torch

from transformer.attention import self_attention


def test_attention():
    # play with mask

    x = torch.rand([2, 3, 2, 4])
    print(x)
    # mask 2 batches 3 timeseries
    mask = torch.ones([2, 3])
    mask[0, 2] = 0
    mask[1, 2] = 0
    mask[1, 1] = 0
    print(f"mask: \n {mask}")
    # add head dim to make mask broatcastable to q x k.T prod. mask shape 2, 1, 3
    mask = mask.unsqueeze(1)

    # mask = mask.permute(0, 2, 1)
    # is the mask that I need? keys are ignored?
    print(f"wrong mask: \n {mask}")
    #  mask = 2 1 3 -> b prepended before broadcasting (1!!!) h (remains since already 2) t (broadcasted from 1) d (remains since already 3)
    print(f"wrong mask broadcast: \n {mask.broadcast_to([2, 2, 3, 3])}")
    a = self_attention(x, x, x, mask=mask, verbose=True)
    print(f"wrong a: \n {a}")
    print(f"wrong a.shape: \n {a.shape}")
    # leads to wrong attention since the shape of mask is wrong 2 1 3
    assert a.shape != (2, 2, 3, 4)

    # correct mask
    # mask 2 batches 3 timeseries
    mask = torch.ones([2, 3])
    mask[0, 2] = 0
    mask[1, 2] = 0
    mask[1, 1] = 0
    mask = mask.unsqueeze(1).unsqueeze(1)

    print(f"mask: \n {mask}")
    #  mask = 2 1 1 3 -> b (remains already 2) h (broadcasted from 1) t (broadcasted from 1) d (remains since already 3)
    print(f"mask broadcast: \n {mask.broadcast_to([2, 2, 3, 3])}")
    a = self_attention(x, x, x, mask=mask, verbose=True)
    print(f"a: \n {a}")
    print(f"a.shape: \n {a.shape}")
    assert a.shape == (2, 2, 3, 4) # b h t d

def test_attention_mask():
    # mask is equal to making keys on masked places 0:
    # the result in terms of masked symbols is the same
    x = torch.rand([2, 3, 2, 4])
    k = x.clone()
    k[0, 2, 0, :] = float("-inf")
    k[0, 2, 1, :] = float("-inf")
    k[1, 2, 0, :] = float("-inf")
    k[1, 1, 0, :] = float("-inf")
    k[1, 2, 1, :] = float("-inf")
    k[1, 1, 1, :] = float("-inf")
    print(f"k: \n {k}")
    a = self_attention(x, k, x, verbose=True)
    print(f"a: \n {a}")
    print(f"a.shape: \n {a.shape}")
    assert a.shape == (2, 2, 3, 4)  # b h t d
    # a is the same shape as if mask was applied in q * k:

    test = torch.rand([2, 3, 4])
    test[0, 2, :] = 0
    test[1, 1, :] = 0
    test[1, 2, :] = 0

    print(f"test: \n {test}")
    test_v = test.view(2, 3, 2, 2)
    print(f"test_v: \n {test_v}")
    test_perm = test_v.permute(0, 2, 1, 3)
    print(f"test_perm: \n {test_perm}")

    # or like that:
    test_q = torch.rand([2, 3, 4])
    test_k = test_q.clone()
    test_k[0, 2, :] = float("-inf")
    test_k[1, 1, :] = float("-inf")
    test_k[1, 2, :] = float("-inf")
    print(f"test_k: \n {test_k}")

    test_q_view = test_q.view(2, 3, 2, 2)
    test_k_view = test_k.view(2, 3, 2, 2)
    print(f"test_k_view: \n {test_k_view}")
    test_q_perm = test_q_view.permute(0, 2, 1, 3)
    test_k_perm = test_k_view.permute(0, 2, 1, 3)
    print(f"test_k_perm: \n {test_k_perm}")
    qk = torch.einsum("bhtd, bhsd -> bhts", test_q_perm, test_k_perm)
    print(f"q * k: \n {qk}")
    assert qk.shape == (2, 2, 3, 3)  # b h t s