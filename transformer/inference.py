import logging

import torch
import torch.nn.functional as F
import typer

from transformer.tokenizer import Tiktokenizer, MinBpeTokenizer
from transformer.transformer import Transformer

log = logging.getLogger(__name__)

app = typer.Typer()

@torch.no_grad()
@app.command()
def translate(
        sentence: str = typer.Option("Ask Tom", help="Sentence to translate"),
        model_weights_path: str = typer.Option(default="..model/model.pt", help="Path to the trained model"),
        d: int = typer.Option(default=64),
        n: int = typer.Option(default=2),
        h: int = typer.Option(default=4),
        device: str = typer.Option(default="mps", help="Device to run inference on"),
        tiktokenizer: bool = typer.Option(default=False),
):
    """
    Model params should be the same as in the training script.
    """
    tokenizer = Tiktokenizer() if tiktokenizer else MinBpeTokenizer()
    transformer = Transformer(vocab_size=tokenizer.vocab_size(), n=n, d=d, h=h)
    transformer.eval()
    transformer.load_state_dict(torch.load(model_weights_path))
    log.info(f"Number of model's params: {sum(p.numel() for p in transformer.parameters())}")
    transformer = transformer.to(device)
    transformer.eval()

    encoded_sent = tokenizer.tokenize(sentence)
    enc_x = torch.tensor(encoded_sent).unsqueeze(0)
    enc_x = enc_x.to(device)
    dec_x = torch.tensor(tokenizer.tokenize(tokenizer.get_special_tokens().start)).unsqueeze(0)
    dec_x = dec_x.to(device)

    predicted_tokens = []
    # TODO add params initialization?
    for _ in range(int(len(sentence) * 1.5)):
        output = transformer(enc_x=enc_x, dec_x=dec_x)
        softmaxed = F.softmax(output, dim=-1)
        predicted = softmaxed.argmax(dim=-1)
        predicted_tokens.append(predicted.tolist()[-1][-1])
        # add the last predicted token to the decoder input
        dec_x = torch.cat((dec_x, predicted[:, -1].unsqueeze(0)), dim=-1)
        if predicted[-1, -1] == tokenizer.get_special_tokens().end_num:
            break

    print(predicted_tokens)
    print(f"predicted sentence: \n {tokenizer.texify(predicted_tokens)}")


if __name__ == "__main__":
    _format = "%(levelname)s %(asctime)s - %(name)s in %(funcName)s() line %(lineno)d: %(message)s"
    logging.basicConfig(format=_format, level=logging.INFO, handlers=[logging.StreamHandler()])
    app()