
def reshape_mask(mask):
    # b t -> b 1 1 t (to be broadcastable to b h t t)
    return mask.unsqueeze(1).unsqueeze(1)