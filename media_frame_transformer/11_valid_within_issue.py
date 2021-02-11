from collections import defaultdict
from glob import glob
from os.path import exists, join

import pandas as pd
import torch
from config import ISSUES, MODELS_DIR
from torch.nn import functional as F
from torch.utils.data import DataLoader, dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, RobertaForSequenceClassification

from media_frame_transformer.dataset import load_kfold
from media_frame_transformer.utils import DEVICE, mkdir_overwrite

EXPERIMENT_NAME = "1.1-f"
KFOLD = 8

N_DATALOADER_WORKER = 2
BATCHSIZE = 100


def valid(
    train_dataset,
    valid_dataset,
    model_path,
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
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)

    metrics = {}
    with torch.no_grad():
        for splitname, dataloader in [
            ("train", train_loader),
            ("valid", valid_loader),
        ]:
            total_n_samples = 0
            total_n_correct = 0
            total_loss = 0
            for i, batch in enumerate(tqdm(dataloader, desc=splitname)):
                xs, ys = batch
                xs, ys = xs.to(DEVICE), ys.to(DEVICE)
                outputs = model(xs)
                loss = F.cross_entropy(outputs.logits, ys)
                total_loss += loss
                is_correct = torch.argmax(outputs.logits, dim=-1) == ys
                total_n_correct += is_correct.sum()
                total_n_samples += ys.shape[0]

            metrics[f"{splitname}_acc"] = (total_n_correct / total_n_samples).item()
            metrics[f"{splitname}_loss"] = (total_loss / total_n_samples).item()
    return metrics


if __name__ == "__main__":
    root_path = join(MODELS_DIR, EXPERIMENT_NAME)
    assert exists(
        root_path
    ), f"{root_path} does not exist, choose the correct experiment name"

    issue2means = {}
    for issue in ISSUES:
        print(issue)
        issue_path = join(root_path, issue)

        kfold_datasets = load_kfold([issue], "primary_frame", KFOLD)
        kfold_model_paths = [
            join(issue_path, f"fold_{ki}", "best.pth") for ki in range(KFOLD)
        ]
        for path in kfold_model_paths:
            assert exists(path)

        issue_metrics = defaultdict(list)
        for ki, (datasets, model_path) in enumerate(
            zip(kfold_datasets, kfold_model_paths)
        ):
            train_dataset = datasets["train"]
            valid_dataset = datasets["valid"]
            metrics = valid(train_dataset, valid_dataset, model_path)
            for k, v in metrics.items():
                issue_metrics[k].append(v)

        issue_means = {k: sum(vs) / len(vs) for k, vs in issue_metrics.items()}
        issue2means[issue] = issue_means

    df = pd.DataFrame.from_dict(issue2means, orient="index")
    df.loc["all"] = df.mean()
    print(df)
    df.to_csv(join(root_path, "metrics.csv"))
