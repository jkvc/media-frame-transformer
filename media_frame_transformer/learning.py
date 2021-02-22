from collections import defaultdict
from os.path import exists, join

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, AutoModelForSequenceClassification

from media_frame_transformer.dataset import get_kfold_primary_frames_datasets
from media_frame_transformer.utils import DEVICE, save_json

N_DATALOADER_WORKER = 6
TRAIN_BATCHSIZE = 25
MAX_EPOCH = 15
NUM_EARLY_STOP_NON_IMPROVE_EPOCH = 3


def train(
    model,
    train_dataset,
    valid_dataset,
    logdir,
    max_epochs=MAX_EPOCH,
    num_early_stop_non_improve_epoch=NUM_EARLY_STOP_NON_IMPROVE_EPOCH,
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
    # lowest_valid_loss = float("inf")

    metrics = {
        "train_acc": 0,
        "train_loss": float("inf"),
        "valid_acc": 0,
        "valid_loss": float("inf"),
    }
    num_non_improve_epoch = 0

    for e in range(max_epochs):
        # train
        # model.train()
        # for i, batch in enumerate(tqdm(train_loader, desc=f"{e}, train")):
        #     xs, ys, weights = batch
        #     xs, ys, weights = xs.to(DEVICE), ys.to(DEVICE), weights.to(DEVICE)

        #     optimizer.zero_grad()
        #     outputs = model(xs)
        #     loss = F.cross_entropy(outputs.logits, ys, reduction="none")
        #     loss = loss * weights
        #     loss = loss.mean()
        #     loss.backward()
        #     optimizer.step()

        #     # tensorboard
        #     step_idx = e * len(train_loader) + i
        #     train_loss = loss.item()
        #     writer.add_scalar("train loss", train_loss, step_idx)
        #     is_correct = torch.argmax(outputs.logits, dim=-1) == ys
        #     train_acc = is_correct.sum() / ys.shape[0]
        #     writer.add_scalar("train acc", train_acc, step_idx)
        train_acc, train_loss = _train_epoch(model, optimizer, train_loader, writer, e)
        metrics["train_acc"] = train_acc
        metrics["train_loss"] = train_loss

        # valid
        # model.eval()
        # with torch.no_grad():
        #     total_n_samples = 0
        #     total_n_correct = 0
        #     total_loss = 0
        #     for i, batch in enumerate(tqdm(valid_loader, desc=f"{e}, valid")):
        #         xs, ys, _ = batch
        #         xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        #         outputs = model(xs)
        #         loss = F.cross_entropy(outputs.logits, ys, reduction="sum")
        #         total_loss += loss
        #         is_correct = torch.argmax(outputs.logits, dim=-1) == ys
        #         total_n_correct += is_correct.sum()
        #         total_n_samples += ys.shape[0]

        #     valid_acc = total_n_correct / total_n_samples
        #     writer.add_scalar("valid acc", valid_acc, e)
        #     valid_loss = total_loss / total_n_samples
        #     writer.add_scalar("valid loss", valid_loss.item(), e)

        valid_acc, valid_loss = _valid_epoch(model, valid_loader, writer, e)

        if valid_loss < metrics["valid_loss"]:
            print(">> new best valid loss save checkpoint")
            metrics["valid_loss"] = valid_loss
            metrics["valid_acc"] = valid_acc
            model.save_pretrained(logdir)
            num_non_improve_epoch = 0
            save_json(metrics, join(logdir, "leaf_metrics.json"))
        else:
            num_non_improve_epoch += 1
            print(">> not improved epoch", num_non_improve_epoch)
            if num_non_improve_epoch >= num_early_stop_non_improve_epoch:
                print(">> early stop")
                break

    save_json(metrics, join(logdir, "leaf_metrics.json"))
    writer.close()


def _valid_epoch(model, valid_loader, writer=None, epoch_idx=None):
    model.eval()
    total_n_samples = 0
    total_n_correct = 0
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(
                valid_loader,
                desc=f"{epoch_idx if epoch_idx is not None else '?'}, valid",
            )
        ):
            xs, ys, _ = batch
            xs, ys = xs.to(DEVICE), ys.to(DEVICE)
            outputs = model(xs)
            loss = F.cross_entropy(outputs.logits, ys, reduction="sum")
            total_loss += loss
            is_correct = torch.argmax(outputs.logits, dim=-1) == ys
            total_n_correct += is_correct.sum()
            total_n_samples += ys.shape[0]

        valid_acc = (total_n_correct / total_n_samples).item()
        valid_loss = (total_loss / total_n_samples).item()

        if writer is not None and epoch_idx is not None:
            writer.add_scalar("valid acc", valid_acc, epoch_idx)
            writer.add_scalar("valid loss", valid_loss, epoch_idx)

    print(">> valid loss", round(valid_loss, 4), "valid acc", round(valid_acc, 4))
    return valid_acc, valid_loss


def _train_epoch(model, optimizer, train_loader, writer=None, epoch_idx=None):
    total_n_samples = total_n_correct = total_loss = 0

    model.train()
    for i, batch in enumerate(
        tqdm(train_loader, desc=f"{epoch_idx if epoch_idx is not None else '?'}, train")
    ):
        xs, ys, weights = batch
        xs, ys, weights = xs.to(DEVICE), ys.to(DEVICE), weights.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(xs)
        loss = F.cross_entropy(outputs.logits, ys, reduction="none")
        loss = loss * weights
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        is_correct = torch.argmax(outputs.logits, dim=-1) == ys
        num_correct = is_correct.sum()
        train_acc = num_correct / ys.shape[0]

        if writer is not None and epoch_idx is not None:
            # tensorboard
            step_idx = epoch_idx * len(train_loader) + i
            writer.add_scalar("train loss", loss.item(), step_idx)
            writer.add_scalar("train acc", train_acc, step_idx)

        total_n_samples += ys.shape[0]
        total_loss += (loss * ys.shape[0]).item()
        total_n_correct += num_correct

    acc = (total_n_correct / total_n_samples).item()
    loss = total_loss / total_n_samples

    print(">> train loss", round(loss, 4), "train acc", round(acc, 4))
    return acc, loss


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
                xs, ys, _ = batch
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
    issues,
    kfold,
    kfold_models_root,
    valid_on_train_also=False,
    zeroth_fold_only=False,
):
    for ki in range(kfold):
        assert exists(join(kfold_models_root, f"fold_{ki}"))
        if zeroth_fold_only:
            break

    issue_metrics = defaultdict(list)
    kfold_datasets = get_kfold_primary_frames_datasets(issues, kfold)

    for ki, datasets in enumerate(kfold_datasets):
        valid_dataset = datasets["valid"]
        train_dataset = datasets["train"] if valid_on_train_also else None

        model = AutoModelForSequenceClassification.from_pretrained(
            join(kfold_models_root, f"fold_{ki}")
        )

        metrics = valid(model, valid_dataset, train_dataset)
        for k, v in metrics.items():
            issue_metrics[k].append(v)

        if zeroth_fold_only:
            break

    metrics = {k: sum(vs) / len(vs) for k, vs in issue_metrics.items()}
    return metrics
