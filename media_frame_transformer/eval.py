from collections import defaultdict
from glob import glob
from os.path import dirname, exists, join

import pandas as pd
import torch
from config import MODELS_DIR
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, AutoModelForSequenceClassification

from media_frame_transformer.dataset import get_kfold_primary_frames_datasets
from media_frame_transformer.learning import valid
from media_frame_transformer.utils import DEVICE, load_json, save_json


def do_valid_model(pretrained_model_dir):
    valid_config = load_json(join(pretrained_model_dir, "valid_config.json"))

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_dir)
    kfold_datasets = get_kfold_primary_frames_datasets(
        issues=valid_config["issues"], k=valid_config["kfold"]
    )
    valid_dataset = kfold_datasets[valid_config["ki"]]["valid"]
    metrics = valid(model, valid_dataset)
    return {"mean": metrics}


def eval_pretrained_model(pretrained_model_dir):
    metrics_json_path = join(pretrained_model_dir, "metrics.json")
    if not exists(metrics_json_path):
        metrics = do_valid_model(pretrained_model_dir)
        save_json(metrics, metrics_json_path)
    else:
        metrics = load_json(metrics_json_path)

    df = pd.DataFrame.from_dict(metrics, orient="index")
    df.to_csv(join(pretrained_model_dir, "metrics.csv"))


def eval_all_leaves(experiment_dir):
    completed_leaf_paths = sorted(
        glob(join(experiment_dir, "**", "_complete"), recursive=True)
    )
    completed_leaf_dirs = [dirname(p) for p in completed_leaf_paths]
    for d in completed_leaf_dirs:
        print("--", d.replace(experiment_dir, ""))
        eval_pretrained_model(d)


if __name__ == "__main__":
    eval_all_leaves(join(MODELS_DIR, "1.1.roberta_half.best"))
