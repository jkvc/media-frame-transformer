from os.path import join

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, RobertaForSequenceClassification

from media_frame_transformer.utils import DEVICE

BATCHSIZE = 30
N_DATALOADER_WORKER = 2


def train(
    train_dataset,
    valid_dataset,
    logdir,
    n_epoch,
    batchsize=BATCHSIZE,
    n_dataloader_worker=N_DATALOADER_WORKER,
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=n_dataloader_worker,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=n_dataloader_worker,
    )

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=15,
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    writer = SummaryWriter(logdir)
    lowest_valid_loss = float("inf")
    best_model_checkpoint_path = join(logdir, "best.pth")

    for e in range(n_epoch):
        # train
        for i, batch in enumerate(tqdm(train_loader, desc=f"{e}, train")):
            xs, ys = batch
            xs, ys = xs.to(DEVICE), ys.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(xs)
            loss = F.cross_entropy(outputs.logits, ys)
            loss.backward()
            optimizer.step()

            # tensorboard
            step_idx = e * len(train_loader) + i
            train_loss = loss.item() / ys.shape[0]
            writer.add_scalar("train loss", train_loss, step_idx)
            is_correct = torch.argmax(outputs.logits, dim=-1) == ys
            train_acc = is_correct.sum() / ys.shape[0]
            writer.add_scalar("train acc", train_acc, step_idx)

        # valid
        with torch.no_grad():
            total_n_samples = 0
            total_n_correct = 0
            total_loss = 0
            for i, batch in enumerate(tqdm(valid_loader, desc=f"{e}, valid")):
                xs, ys = batch
                xs, ys = xs.to(DEVICE), ys.to(DEVICE)
                outputs = model(xs)
                loss = F.cross_entropy(outputs.logits, ys)
                total_loss += loss
                is_correct = torch.argmax(outputs.logits, dim=-1) == ys
                total_n_correct += is_correct.sum()
                total_n_samples += ys.shape[0]
            valid_acc = total_n_correct / total_n_samples
            writer.add_scalar("valid acc", valid_acc, e)
            valid_loss = total_loss / total_n_samples
            if valid_loss < lowest_valid_loss:
                print(
                    ">> new best valid loss",
                    round(valid_loss.item(), 5),
                    "save checkpoint",
                )
                lowest_valid_loss = valid_loss
                torch.save(model.state_dict(), best_model_checkpoint_path)

    writer.close()
