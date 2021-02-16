from collections import defaultdict
from os.path import exists, join

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, AutoModelForSequenceClassification

from media_frame_transformer.utils import DEVICE

N_DATALOADER_WORKER = 6
TRAIN_BATCHSIZE = 25


def train(
    model,
    train_dataset,
    valid_dataset,
    logdir,
    n_epoch,
    batchsize=TRAIN_BATCHSIZE,
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

    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    writer = SummaryWriter(logdir)
    lowest_valid_loss = float("inf")
    # best_model_checkpoint_path = join(logdir, "best.pth")

    for e in range(n_epoch):
        # train
        model.train()
        for i, batch in enumerate(tqdm(train_loader, desc=f"{e}, train")):
            xs, ys, _ = batch  # todo
            xs, ys = xs.to(DEVICE), ys.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(xs)
            loss = F.cross_entropy(outputs.logits, ys, reduction="mean")
            loss.backward()
            optimizer.step()

            # tensorboard
            step_idx = e * len(train_loader) + i
            train_loss = loss.item()
            writer.add_scalar("train loss", train_loss, step_idx)
            is_correct = torch.argmax(outputs.logits, dim=-1) == ys
            train_acc = is_correct.sum() / ys.shape[0]
            writer.add_scalar("train acc", train_acc, step_idx)

        # valid
        model.eval()
        with torch.no_grad():
            total_n_samples = 0
            total_n_correct = 0
            total_loss = 0
            for i, batch in enumerate(tqdm(valid_loader, desc=f"{e}, valid")):
                xs, ys, _ = batch
                xs, ys = xs.to(DEVICE), ys.to(DEVICE)
                outputs = model(xs)
                loss = F.cross_entropy(outputs.logits, ys, reduction="sum")
                total_loss += loss
                is_correct = torch.argmax(outputs.logits, dim=-1) == ys
                total_n_correct += is_correct.sum()
                total_n_samples += ys.shape[0]

            valid_acc = total_n_correct / total_n_samples
            writer.add_scalar("valid acc", valid_acc, e)
            valid_loss = total_loss / total_n_samples
            writer.add_scalar("valid loss", valid_loss.item(), e)

            if valid_loss < lowest_valid_loss:
                print(
                    ">> new best valid loss",
                    round(valid_loss.item(), 4),
                    "valid acc",
                    round(valid_acc.item(), 4),
                    "save checkpoint",
                )
                lowest_valid_loss = valid_loss
                model.save_pretrained(logdir)

    writer.close()


VALID_BATCHSIZE = 100


def valid(
    model,
    valid_dataset,
    train_dataset=None,
    batchsize=VALID_BATCHSIZE,
    n_dataloader_worker=N_DATALOADER_WORKER,
):

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=n_dataloader_worker,
    )
    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=n_dataloader_worker,
        )
        if train_dataset
        else None
    )
    model = model.to(DEVICE)

    name0loaders = [("valid", valid_loader)]
    if train_loader:
        name0loaders.append(("train", train_loader))

    metrics = {}
    with torch.no_grad():
        model.eval()
        for splitname, dataloader in name0loaders:
            total_n_samples = 0
            total_n_correct = 0
            total_loss = 0
            for i, batch in enumerate(tqdm(dataloader, desc=splitname)):
                xs, ys = batch
                xs, ys = xs.to(DEVICE), ys.to(DEVICE)
                outputs = model(xs)
                loss = F.cross_entropy(outputs.logits, ys, reduction="sum")
                total_loss += loss
                is_correct = torch.argmax(outputs.logits, dim=-1) == ys
                total_n_correct += is_correct.sum()
                total_n_samples += ys.shape[0]

            valid_loss = (total_loss / total_n_samples).item()
            valid_acc = (total_n_correct / total_n_samples).item()
            print(
                ">> valid loss",
                round(valid_loss, 4),
                "valid acc",
                round(valid_acc, 4),
            )
            metrics[f"{splitname}_acc"] = valid_acc
            metrics[f"{splitname}_loss"] = valid_loss

    return metrics


def get_kfold_metrics(
    issues, task, kfold, kfold_models_root, valid_on_train_also=False
):
    for ki in range(kfold):
        assert exists(join(kfold_models_root, f"fold_{ki}"))

    issue_metrics = defaultdict(list)
    kfold_datasets = load_kfold(issues, task, kfold)

    for ki, datasets in enumerate(kfold_datasets):
        valid_dataset = datasets["valid"]
        train_dataset = datasets["train"] if valid_on_train_also else None

        model = AutoModelForSequenceClassification.from_pretrained(
            join(kfold_models_root, f"fold_{ki}")
        )

        metrics = valid(model, valid_dataset, train_dataset)
        for k, v in metrics.items():
            issue_metrics[k].append(v)

    metrics = {k: sum(vs) / len(vs) for k, vs in issue_metrics.items()}
    return metrics
