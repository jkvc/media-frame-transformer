# deprecated?

from collections import defaultdict
from glob import glob
from os.path import dirname, exists, join
from pprint import pprint

import pandas as pd
import torch
from config import MODELS_DIR
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, AutoModelForSequenceClassification

from media_frame_transformer.learning import valid_epoch
from media_frame_transformer.utils import DEVICE, load_json, save_json

# def do_valid_model(pretrained_model_dir):
#     valid_config = load_json(join(pretrained_model_dir, "valid_config.json"))

#     model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_dir).to(
#         DEVICE
#     )
#     kfold_datasets = get_kfold_primary_frames_datasets(
#         issues=valid_config["issues"], k=valid_config["kfold"]
#     )
#     valid_dataset = kfold_datasets[valid_config["ki"]]["valid"]
#     valid_loader = DataLoader(valid_dataset, batch_size=500, shuffle=True)
#     valid_acc, valid_loss = valid_epoch(model, valid_loader)
#     return {
#         "mean": {
#             "valid_acc": valid_acc,
#             "valid_loss": valid_loss,
#         },
#     }


def eval_pretrained_model(pretrained_model_dir):
    metrics_json_path = join(pretrained_model_dir, "leaf_metrics.json")
    if not exists(metrics_json_path):
        metrics = do_valid_model(pretrained_model_dir)
        save_json(metrics, metrics_json_path)
    else:
        metrics = load_json(metrics_json_path)

    df = pd.DataFrame.from_dict(metrics, orient="index")
    df.to_csv(join(pretrained_model_dir, "leaf_metrics.csv"))


def eval_all_leaves(experiment_dir):
    completed_leaf_paths = sorted(
        glob(join(experiment_dir, "**", "_complete"), recursive=True)
    )
    completed_leaf_dirs = [dirname(p) for p in completed_leaf_paths]
    for d in completed_leaf_dirs:
        print("--", d.replace(experiment_dir, ""))
        eval_pretrained_model(d)


def reduce_tree_inplace(tree):
    root = tree
    metrics = defaultdict(list)
    for child in root.values():
        if "mean" not in child:
            reduce_tree_inplace(child)
        child_mean = child["mean"]
        for k, v in child_mean.items():
            metrics[k].append(v)
    mean_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    root["mean"] = mean_metrics


def save_tree(rootdir, tree, save_filename):
    save_json(tree, join(rootdir, save_filename))

    meanrows = {}
    for childname, subtree in tree.items():
        if childname != "mean":
            save_tree(join(rootdir, childname), subtree, save_filename)
            meanrows[childname] = subtree["mean"]

    meanrows["mean"] = tree["mean"]
    df = pd.DataFrame.from_dict(meanrows, orient="index")
    df.to_csv(join(rootdir, "mean_metrics.csv"))


def reduce_and_save_metrics(
    rootdir,
    leaf_metric_filename="leaf_metrics.json",
    save_filename="mean_metrics.json",
):
    leaf_metric_paths = sorted(
        glob(join(rootdir, "**", leaf_metric_filename), recursive=True)
    )
    if len(leaf_metric_paths) == 0:
        return
    pprint(leaf_metric_paths)

    # build tree to leaf metrics
    tree = {}
    for p in leaf_metric_paths:
        parent = tree
        toks = p.replace(rootdir, "").strip("/").split("/")
        for i, child in enumerate(toks[:-1]):
            if child not in parent:
                if i == len(toks) - 2:
                    # FIXME
                    # parent[child] = {"mean": load_json(p)}
                    leaf_metrics = load_json(p)
                    processed_leaf = {}
                    for k, v in leaf_metrics.items():
                        if isinstance(v, list):
                            v = v[-1]
                        processed_leaf[k] = v
                    parent[child] = {"mean": processed_leaf}
                else:
                    parent[child] = {}
            parent = parent[child]
    pprint(tree)

    reduce_tree_inplace(tree)
    pprint(tree)

    save_tree(rootdir, tree, save_filename)


if __name__ == "__main__":
    exp_dir = join(MODELS_DIR, "1.1.roberta_half.best")
    eval_all_leaves(exp_dir)
    reduce_and_save_metrics(exp_dir)
