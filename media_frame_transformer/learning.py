from collections import defaultdict
from os.path import exists, join
from pprint import pprint
from typing import Dict

import numpy as np
import torch
from numpy.lib.function_base import iterable
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, AutoModelForSequenceClassification

from media_frame_transformer.dataset import PrimaryFrameDataset
from media_frame_transformer.utils import DEVICE, save_json

N_DATALOADER_WORKER = 6
TRAIN_BATCHSIZE = 50
MAX_EPOCH = 30
NUM_EARLY_STOP_NON_IMPROVE_EPOCH = 3
VALID_BATCHSIZE = 150


def train(
    model,
    train_dataset,
    valid_dataset,
    logdir,
    additional_valid_datasets: Dict[str, PrimaryFrameDataset] = None,
    max_epochs=MAX_EPOCH,
    num_early_stop_non_improve_epoch=NUM_EARLY_STOP_NON_IMPROVE_EPOCH,
    batchsize=TRAIN_BATCHSIZE,
    n_dataloader_worker=N_DATALOADER_WORKER,
    save_model=True,
    keep_latest=False,
    skip_train_zeroth_epoch=False,
    valid_every_n_epoch=1,
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=n_dataloader_worker,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=VALID_BATCHSIZE,
        shuffle=True,
        num_workers=n_dataloader_worker,
    )
    additional_valid_loaders = None
    if additional_valid_datasets is not None:
        additional_valid_loaders = {}
        for name, dataset in additional_valid_datasets.items():
            additional_valid_loaders[name] = DataLoader(
                dataset,
                batch_size=VALID_BATCHSIZE,
                shuffle=False,
                num_workers=n_dataloader_worker,
            )

    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    writer = SummaryWriter(logdir)
    # lowest_valid_loss = float("inf")

    metrics = {
        # "train_acc": 0,
        "train_loss": float("inf"),
        "train_f1": 0,
        # "valid_acc": 0,
        "valid_loss": float("inf"),
        "valid_f1": 0,
    }
    num_non_improve_epoch = 0

    for e in range(max_epochs):
        print(">> begin epoch", e)

        # train
        if skip_train_zeroth_epoch and e == 0:
            pass
        else:
            train_metrics = train_epoch(model, optimizer, train_loader, writer, e)
            for k, v in train_metrics.items():
                metrics[f"train_{k}"] = v

        # valid
        if e % valid_every_n_epoch != 0 and e != max_epochs - 1:
            print(">> skip valid at epoch", e)
        else:
            valid_metrics = valid_epoch(model, valid_loader, writer, e)
            valid_f1 = valid_metrics["f1"]

            if valid_f1 > metrics["valid_f1"] or keep_latest:
                # new best, save stuff
                is_this_epoch_valid_improve = True
                if keep_latest:
                    print("** keeping latest")
                else:
                    print("++ new best valid f1")
                for k, v in valid_metrics.items():
                    metrics[f"valid_{k}"] = v
                num_non_improve_epoch = 0
                if save_model:
                    print(
                        f'++ save model checkpoint to {join(logdir, "checkpoint.pth")}'
                    )
                    torch.save(model, join(logdir, "checkpoint.pth"))
            else:
                # not improving
                is_this_epoch_valid_improve = False
                num_non_improve_epoch += 1
                print("~~ not improved epoch #", num_non_improve_epoch)
                if num_non_improve_epoch >= num_early_stop_non_improve_epoch:
                    print(">> early stop")
                    break

            # additional valid
            if additional_valid_loaders is not None:
                for set_name, set_valid_loader in additional_valid_loaders.items():
                    additional_valid_metrics = valid_epoch(
                        model, set_valid_loader, writer, e, set_name
                    )
                    if is_this_epoch_valid_improve:
                        for k, v in additional_valid_metrics.items():
                            metrics[f"{set_name}_{k}"] = v

            save_json(metrics, join(logdir, "leaf_metrics.json"))
            save_json(metrics, join(logdir, f"leaf_epoch_{e}.json"))

        print(">> end epoch", e, "\n")

    writer.close()


def valid_epoch(model, valid_loader, writer=None, epoch_idx=None, valid_set_name=None):
    model.eval()
    all_logits = []
    all_labels = []

    total_n_samples = 0
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(
                valid_loader,
                desc=f"{epoch_idx if epoch_idx is not None else '?'}, {'valid' if valid_set_name is None else valid_set_name}",
            )
        ):
            outputs = model(batch)
            loss = outputs["loss"]
            total_loss += loss

            logits = outputs["logits"]
            all_logits.append(logits.detach().cpu().numpy())
            labels = outputs["labels"]
            all_labels.append(labels.detach().cpu().numpy())

            num_samples = outputs["logits"].shape[0]
            total_n_samples += num_samples
            total_loss += (loss * num_samples).item()

        valid_loss = (total_loss / total_n_samples).item()

        f1, precision, recall = calc_f1(all_logits, all_labels)

        if writer is not None and epoch_idx is not None:
            writer.add_scalar(
                f"{valid_set_name if valid_set_name else 'valid'} f1",
                f1,
                epoch_idx,
            )
            writer.add_scalar(
                f"{valid_set_name if valid_set_name else 'valid'} loss",
                valid_loss,
                epoch_idx,
            )

    metrics = {
        "loss": valid_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    _print_metrics(metrics)
    return metrics


def train_epoch(model, optimizer, train_loader, writer=None, epoch_idx=None):
    model.train()

    total_n_samples = total_loss = 0
    all_logits = []
    all_labels = []

    for i, batch in enumerate(
        tqdm(train_loader, desc=f"{epoch_idx if epoch_idx is not None else '?'}, train")
    ):
        optimizer.zero_grad()
        outputs = model(batch)

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        logits = outputs["logits"]
        all_logits.append(logits.detach().cpu().numpy())
        labels = outputs["labels"]
        all_labels.append(labels.detach().cpu().numpy())

        if writer is not None and epoch_idx is not None:
            # tensorboard
            step_idx = epoch_idx * len(train_loader) + i
            writer.add_scalar("train loss", loss.item(), step_idx)

        num_samples = outputs["logits"].shape[0]
        total_n_samples += num_samples
        total_loss += (loss * num_samples).item()

    loss = total_loss / total_n_samples

    f1, precision, recall = calc_f1(all_logits, all_labels)
    metrics = {
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    _print_metrics(metrics)
    return metrics


def _print_metrics(metrics):
    for k, v in metrics.items():
        print("--", k.rjust(10), ":", v)


def calc_f1(all_logits, all_labels):
    if isinstance(all_logits, list):
        all_logits = np.concatenate(all_logits, axis=0)
    if isinstance(all_labels, list):
        all_labels = np.concatenate(all_labels, axis=0)

    if all_labels.ndim == 1:
        # one-hot label case
        y_pred = np.argmax(all_logits, axis=-1)
        y_true = all_labels
    else:
        # K-hot label case
        y_pred = 1 / (1 + np.exp(-all_logits))  # sigmoid
        y_pred = y_pred > 0.5
        y_true = all_labels

    f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    precision = precision_score(y_true=y_true, y_pred=y_pred, average="micro")
    recall = recall_score(y_true=y_true, y_pred=y_pred, average="micro")
    return f1, precision, recall


# def valid(
#     model,
#     valid_dataset,
#     train_dataset=None,
#     batchsize=VALID_BATCHSIZE,
#     n_dataloader_worker=N_DATALOADER_WORKER,
# ):

#     valid_loader = DataLoader(
#         valid_dataset,
#         batch_size=batchsize,
#         shuffle=True,
#         num_workers=n_dataloader_worker,
#     )
#     train_loader = (
#         DataLoader(
#             train_dataset,
#             batch_size=batchsize,
#             shuffle=True,
#             num_workers=n_dataloader_worker,
#         )
#         if train_dataset
#         else None
#     )
#     model = model.to(DEVICE)

#     name0loaders = [("valid", valid_loader)]
#     if train_loader:
#         name0loaders.append(("train", train_loader))

#     metrics = {}
#     with torch.no_grad():
#         model.eval()
#         for splitname, dataloader in name0loaders:
#             total_n_samples = 0
#             total_n_correct = 0
#             total_loss = 0
#             for i, batch in enumerate(tqdm(dataloader, desc=splitname)):
#                 xs, ys, _ = batch
#                 xs, ys = xs.to(DEVICE), ys.to(DEVICE)
#                 outputs = model(xs)
#                 loss = F.cross_entropy(outputs.logits, ys, reduction="sum")
#                 total_loss += loss
#                 is_correct = torch.argmax(outputs.logits, dim=-1) == ys
#                 total_n_correct += is_correct.sum()
#                 total_n_samples += ys.shape[0]

#             valid_loss = (total_loss / total_n_samples).item()
#             valid_acc = (total_n_correct / total_n_samples).item()
#             print(
#                 ">> valid loss",
#                 round(valid_loss, 4),
#                 "valid acc",
#                 round(valid_acc, 4),
#             )
#             metrics[f"{splitname}_acc"] = valid_acc
#             metrics[f"{splitname}_loss"] = valid_loss
#     print(metrics)
#     return metrics


# def get_kfold_metrics(
#     issues,
#     kfold,
#     kfold_models_root,
#     valid_on_train_also=False,
#     zeroth_fold_only=False,
# ):
#     for ki in range(kfold):
#         assert exists(join(kfold_models_root, f"fold_{ki}"))
#         if zeroth_fold_only:
#             break

#     issue_metrics = defaultdict(list)
#     kfold_datasets = get_kfold_primary_frames_datasets(issues, kfold)

#     for ki, datasets in enumerate(kfold_datasets):
#         valid_dataset = datasets["valid"]
#         train_dataset = datasets["train"] if valid_on_train_also else None

#         model = AutoModelForSequenceClassification.from_pretrained(
#             join(kfold_models_root, f"fold_{ki}")
#         )

#         metrics = valid(model, valid_dataset, train_dataset)
#         for k, v in metrics.items():
#             issue_metrics[k].append(v)

#         if zeroth_fold_only:
#             break

#     metrics = {k: sum(vs) / len(vs) for k, vs in issue_metrics.items()}
#     return metrics
