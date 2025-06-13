import tiktoken
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformer.mask import build_padding_mask, reshape_mask, build_causal_mask, merge_masks
from transformer.transformer import Transformer


def test_transformer():
    transformer = Transformer(vocab_size=32, n=2, d=16, h=2, embed_tying=False)
    enc_x = torch.tensor([[15, 7, 3], [10, 10, 0], [1, 0, 0]])
    dec_x = torch.tensor([[21, 8, 0, 0], [25, 0, 0, 0], [8, 1, 2, 3]])
    # dec_x = torch.tensor([[21, 8], [25, 0], [8, 1]])

    enc_mask = build_padding_mask(enc_x, pad_token=0)
    print(f"enc_mask: \n {enc_mask}")
    enc_mask = reshape_mask(enc_mask)

    dec_mask1 = build_padding_mask(dec_x, pad_token=0)
    dec_mask2 = build_causal_mask(dec_x)
    dec_mask = merge_masks(dec_mask1, dec_mask2)
    print(f"dec_mask: \n {dec_mask}")
    dec_mask = reshape_mask(dec_mask)

    transformed = transformer(enc_x, dec_x, enc_mask=enc_mask, dec_mask=dec_mask)
    print(transformed.shape)
    assert transformed.shape == (3, 4, 32)

def test_transformer_inference():
    encoding = tiktoken.get_encoding("cl100k_base")
    sents = ["Hello World", "This is a simple sentence", "Me"]
    encoded_sents = [encoding.encode(s) for s in sents]
    enc_x = pad_sequence([torch.tensor(es) for es in encoded_sents], batch_first=True, padding_value=encoding.eot_token)
    print(enc_x)
    dec_sents = ["Bonjour", "C'est une phrase", "START"]
    dec_encoded_sents = [encoding.encode(s) for s in dec_sents]
    dec_x = pad_sequence([torch.tensor(es) for es in dec_encoded_sents], batch_first=True,
                         padding_value=encoding.eot_token)
    print(dec_x)

    transformer = Transformer(vocab_size=encoding.n_vocab, n=3, d=256, h=4)

    enc_mask = build_padding_mask(enc_x, pad_token=100257)
    print(f"enc_mask: \n {enc_mask}")
    enc_mask = reshape_mask(enc_mask)

    dec_mask1 = build_padding_mask(dec_x, pad_token=100257)
    dec_mask2 = build_causal_mask(dec_x)
    dec_mask = merge_masks(dec_mask1, dec_mask2)
    print(f"dec_mask: \n {dec_mask}")
    dec_mask = reshape_mask(dec_mask)

    output = transformer(enc_x, dec_x, enc_mask=enc_mask, dec_mask=dec_mask)
    print(f"output shape: {output.shape}")
    softmaxed = F.softmax(output, dim=-1)
    print(f"softmaxed[0, 0, :10]: {softmaxed[0, 0, :10]}")
    predicted = softmaxed.argmax(dim=-1)
    print(f"predicted: \n {predicted}")
    assert predicted.shape == dec_x.shape

    predicted_list = predicted.tolist()
    predicted_decoded = [encoding.decode(l) for l in predicted_list]
    print(f"predicted decoded: \n {predicted_decoded}")

def test_transformer_generate():
    # Predicting next words
    encoding = tiktoken.get_encoding("cl100k_base")
    sent = "This is a simple sentence"
    encoded_sent = encoding.encode(sent)
    enc_x = torch.tensor(encoded_sent).unsqueeze(0)
    dec_x = torch.tensor(encoding.encode("C")).unsqueeze(0)

    transformer = Transformer(vocab_size=encoding.n_vocab, n=3, d=256, h=4)

    predicted_tokens = []
    for _ in range(5):
        output = transformer(enc_x=enc_x, dec_x=dec_x)
        softmaxed = F.softmax(output, dim=-1)
        predicted = softmaxed.argmax(dim=-1)
        predicted_tokens.append(predicted.tolist()[-1][-1])
        dec_x = torch.cat((dec_x, predicted), dim=-1)

    print(predicted_tokens)
    assert len(predicted_tokens) == 5
    print(f"predicted sentence: \n {encoding.decode(predicted_tokens)}")