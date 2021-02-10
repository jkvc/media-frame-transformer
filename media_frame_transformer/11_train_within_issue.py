from os.path import exists, join

import torch
from config import ISSUES, MODELS_DIR
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, BertConfig, RobertaForSequenceClassification
from transformers.file_utils import TF_TOKEN_CLASSIFICATION_SAMPLE

from media_frame_transformer.dataset import load_kfold
from media_frame_transformer.utils import DEVICE, mkdir_overwrite

EXPERIMENT_NAME = "1.1-f"

KFOLD = 8
N_EPOCH = 8
BATCHSIZE = 30
N_DATALOADER_WORKER = 2


def train(train_dataset, valid_dataset, logdir):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCHSIZE,
        shuffle=True,
        num_workers=N_DATALOADER_WORKER,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCHSIZE,
        shuffle=True,
        num_workers=N_DATALOADER_WORKER,
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

    for e in range(N_EPOCH):
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


if __name__ == "__main__":
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    assert not exists(
        save_root
    ), f"{save_root} already exists, remove existing or choose another experment name"
    mkdir_overwrite(save_root)

    for issue in ISSUES:
        print(issue)
        save_issue = join(save_root, issue)
        mkdir_overwrite(save_issue)

        kfold_datasets = load_kfold([issue], "primary_frame", KFOLD)
        for ki, datasets in enumerate(kfold_datasets):
            save_fold = join(save_issue, f"fold_{ki}")
            mkdir_overwrite(save_fold)

            train_dataset = datasets["train"]
            valid_dataset = datasets["valid"]
            train(train_dataset, valid_dataset, save_fold)
