import logging
import math
from pathlib import Path

import torch
import typer
from torch.utils.data import DataLoader

from transformer.data import get_collate_fn, TokEnFrDataset, Tiktokenizer, MinBpeTokenizer, Partition, Tokenizer
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
        self.model_final_path = model_dir.joinpath("model-final-small.pt"),
        self.model_intermediate_path = model_dir.joinpath("model-intermediate-small.pt")
        self.log_epoch_freq = log_epoch_freq

    def save_model(self, model_path):
        torch.save(self.transformer.state_dict(), model_path)

    def _train_epoch(self, epoch):
        running_loss = 0.
        last_loss = 0.
        best_loss = math.inf

        self.transformer.train(True)
        self.dataloader.dataset.partition = Partition.TRAIN

        for i, data in enumerate(self.dataloader, 1):
            enc_x, dec_x, label, enc_mask, dec_mask = data
            enc_x, dec_x, label, enc_mask, dec_mask = (enc_x.to(self.device), dec_x.to(self.device),
                                                       label.to(self.device), enc_mask.to(self.device),
                                                       dec_mask.to(self.device))
            # Clear grads
            self.optimizer.zero_grad()

            output = self.transformer(enc_x, dec_x, enc_mask=enc_mask, dec_mask=dec_mask)
            loss = self.loss_fn(output.view(-1, self.tokenizer.vocab_size()), label.view(-1))
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 0:
                last_loss = running_loss / 1000
                log.info('Average batch loss: {}'.format(last_loss))
                running_loss = 0.
                # if last_loss < best_loss:
                #     save_model()

            # helps to reduce memory consumption?
            # if i % 5000 == 0:
            #     torch.mps.empty_cache()

        if epoch % self.log_epoch_freq == 0:
            log.info('Average epoch loss: {}'.format(last_loss if last_loss > 0 else running_loss / i))
        return last_loss

    def _validate_epoch(self, epoch):
        running_vloss = 0.0
        self.transformer.eval()
        self.dataloader.dataset.partition = Partition.VAL

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(self.dataloader):
                enc_x, dec_x, label, enc_mask, dec_mask = vdata
                enc_x, dec_x, label, enc_mask, dec_mask = (enc_x.to(self.device), dec_x.to(self.device),
                                                           label.to(self.device), enc_mask.to(self.device),
                                                           dec_mask.to(self.device))
                output = self.transformer(enc_x, dec_x, enc_mask=enc_mask, dec_mask=dec_mask)
                vloss = self.loss_fn(output.view(-1, self.tokenizer.vocab_size()), label.view(-1))
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        if epoch % self.log_epoch_freq == 0:
            log.info('Average valid loss {}'.format(avg_vloss))

        return avg_vloss

    def train(self):
        best_val_loss = math.inf
        for epoch in range(self.max_epochs + 1, 1):
            if epoch % self.log_epoch_freq == 0:
                log.info('EPOCH {}:'.format(epoch))
            avg_train_loss = self._train_epoch(epoch)

            avg_val_loss = self._validate_epoch(epoch)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(self.model_final_path)


def train(
        data: Path = typer.Option(default="../../data/eng_-french-small.csv"),
        model_dir: Path = typer.Option(default="../../model/"),
        d: int = typer.Option(default=64),
        n: int = typer.Option(default=2),
        h: int = typer.Option(default=4),
        device: str = typer.Option(default="mps"),
        val_ratio: float = typer.Option(default=0.1),
        batch_size: int = typer.Option(default=16),
        num_workers: int = typer.Option(default=1),
        max_epochs: int = typer.Option(default=500),
        tiktokenizer: bool = typer.Option(default=False),
):
    tokenizer = Tiktokenizer() if tiktokenizer else MinBpeTokenizer()
    # batch = 16 if memory is not enough
    dataloader_params = {
        'collate_fn': get_collate_fn(tokenizer.get_special_tokens().pad_num),
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


if __name__ == "__main__":
    _format = "%(levelname)s %(asctime)s - %(name)s in %(funcName)s() line %(lineno)d: %(message)s"
    logging.basicConfig(format=_format, level=logging.INFO, handlers=[logging.StreamHandler()])
    app()
