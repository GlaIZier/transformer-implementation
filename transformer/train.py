import logging
import math
from functools import partial
from pathlib import Path

import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer.data import TokEnFrDataset, Tiktokenizer, MinBpeTokenizer, Partition, Tokenizer, \
    collate
from transformer.transformer import Transformer

log = logging.getLogger(__name__)

app = typer.Typer()


class Trainer:

    def __init__(self, transformer: Transformer, tokenizer: Tokenizer, dataloader: DataLoader, loss_fn, optimizer,
                 device: str, max_epochs: int, model_dir: Path, log_epoch_freq: int = 10):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs
        self.model_final_path = model_dir.joinpath("model-final-med.pt")
        # self.model_intermediate_path = model_dir.joinpath("model-intermediate-small.pt")
        self.log_epoch_freq = log_epoch_freq

    def save_model(self, model_path):
        torch.save(self.transformer.state_dict(), model_path)

    def _calc_batch_loss(self, batch):
        enc_x, dec_x, label, enc_mask, dec_mask = batch
        enc_x, dec_x, label, enc_mask, dec_mask = (enc_x.to(self.device), dec_x.to(self.device),
                                                   label.to(self.device), enc_mask.to(self.device),
                                                   dec_mask.to(self.device))
        output = self.transformer(enc_x, dec_x, enc_mask=enc_mask, dec_mask=dec_mask)
        return self.loss_fn(output.view(-1, self.tokenizer.vocab_size()), label.view(-1))

    def _train_epoch(self):
        running_loss = 0.
        self.transformer.train(True)
        self.dataloader.dataset.partition = Partition.TRAIN

        for i, data in enumerate(self.dataloader, 1):
            # Clear grads
            self.optimizer.zero_grad()
            loss = self._calc_batch_loss(data)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return 0 if running_loss == 0 else running_loss / i

    def _validate_epoch(self):
        running_vloss = 0.
        self.transformer.eval()
        self.dataloader.dataset.partition = Partition.VAL

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(self.dataloader, 1):
                running_vloss += self._calc_batch_loss(vdata)

        return 0 if running_vloss == 0 else running_vloss / i

    def train(self):
        best_val_loss = math.inf
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            avg_batch_train_loss = self._train_epoch()
            if epoch % self.log_epoch_freq == 0:
                log.info('Average epoch {} batch loss: {}'.format(epoch, avg_batch_train_loss))

            avg_batch_val_loss = self._validate_epoch()
            if epoch % self.log_epoch_freq == 0:
                log.info('Average epoch {} batch val loss: {}'.format(epoch, avg_batch_val_loss))
            if avg_batch_val_loss < best_val_loss:
                best_val_loss = avg_batch_val_loss
                self.save_model(self.model_final_path)


@app.command()
def train(
        data: Path = typer.Option(default="data/eng_-french-small.csv"),
        model_dir: Path = typer.Option(default="model"),
        d: int = typer.Option(default=64),
        n: int = typer.Option(default=2),
        h: int = typer.Option(default=4),
        device: str = typer.Option(default="mps"),
        val_ratio: float = typer.Option(default=0.1),
        batch_size: int = typer.Option(default=16),
        num_workers: int = typer.Option(default=1),
        max_epochs: int = typer.Option(default=500),
        tiktokenizer: bool = typer.Option(default=False),
        log_epoch_freq: int = typer.Option(default=10)
):
    tokenizer = Tiktokenizer() if tiktokenizer else MinBpeTokenizer()
    # batch = 16 if memory is not enough
    dataloader_params = {
        'collate_fn': partial(collate, pad_num=tokenizer.get_special_tokens().pad_num),
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
    }

    dataset = TokEnFrDataset(file=data, tokenizer=tokenizer, val_ratio=val_ratio)
    dataloader = DataLoader(dataset=dataset, **dataloader_params)
    transformer = Transformer(vocab_size=tokenizer.vocab_size(), n=n, d=d, h=h)
    log.info(f"Number of model's params: {sum(p.numel() for p in transformer.parameters())}")
    transformer = transformer.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.get_special_tokens().pad_num)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=3e-4, weight_decay=1e-4)
    trainer = Trainer(
        transformer=transformer,
        tokenizer=tokenizer,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        max_epochs=max_epochs,
        model_dir=model_dir,
        log_epoch_freq=log_epoch_freq
    )
    trainer.train()


if __name__ == "__main__":
    _format = "%(levelname)s %(asctime)s - %(name)s in %(funcName)s() line %(lineno)d: %(message)s"
    logging.basicConfig(format=_format, level=logging.INFO, handlers=[logging.StreamHandler()])
    app()
