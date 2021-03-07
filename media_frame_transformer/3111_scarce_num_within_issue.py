from collections import defaultdict
from os import mkdir
from os.path import exists, join
from pprint import pprint
from random import Random, shuffle

import numpy as np
from config import ISSUES, MODELS_DIR

from media_frame_transformer import models
from media_frame_transformer.dataset import (
    PrimaryFrameDataset,
    fold2split2samples_to_datasets,
    load_kfold_primary_frame_samples,
)
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiment_config import (
    ARCH,
    BATCHSIZE,
    FOLDS_TO_RUN,
    KFOLD,
)
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.learning import train
from media_frame_transformer.utils import (
    load_json,
    mkdir_overwrite,
    write_str_list_as_txt,
)
from media_frame_transformer.viualization import plot_series_w_labels

RNG = Random()
RNG.seed(0xDEADBEEF)

EXPERIMENT_NAME = f"3111.{ARCH}"
DATASET_SIZES = [250, 500, 1000, 1750, 2500, 4000, 6000]


def _train():
    # root/issue/fold/numsample
    path2datasets = {}

    for issue in ISSUES:
        fold2split2samples = load_kfold_primary_frame_samples([issue], KFOLD)
        num_train_sample = len(fold2split2samples[0]["train"])
        for ki in FOLDS_TO_RUN:
            split2samples = fold2split2samples[ki]
            RNG.shuffle(split2samples["train"])

            for numsample in DATASET_SIZES:
                if numsample > num_train_sample:
                    continue
                train_samples = split2samples["train"][:numsample]
                valid_samples = split2samples["valid"]
                train_dataset = PrimaryFrameDataset(train_samples)
                valid_dataset = PrimaryFrameDataset(valid_samples)

                path2datasets[
                    join(
                        MODELS_DIR, EXPERIMENT_NAME, issue, f"fold_{ki}", str(numsample)
                    )
                ] = {
                    "train": train_dataset,
                    "valid": valid_dataset,
                }

    run_experiments(ARCH, path2datasets, batchsize=BATCHSIZE)


# def _plot():
#     metrics_json_path = join(MODELS_DIR, EXPERIMENT_NAME, "mean_metrics.json")
#     metrics = load_json(metrics_json_path)
#     pprint(metrics)

#     metricname2prop2val = defaultdict(dict)
#     for prop in metrics:
#         if prop == "mean":
#             continue
#         for metricname, val in metrics[prop]["mean"].items():
#             metricname2prop2val[metricname][prop] = val
#     pprint(metricname2prop2val)
#     # acc and loss aggregated over all issues
#     acc_name2metrics = {}
#     for metricname in ["train_acc", "valid_acc"]:
#         prop2val = metricname2prop2val[metricname]
#         props = sorted(list(prop2val.keys()))
#         vals = [prop2val[prop] for prop in props]
#         acc_name2metrics[metricname] = zip(props, vals)
#     plot_series_w_labels(
#         acc_name2metrics,
#         "accuracy v proportion of training data",
#         join(MODELS_DIR, EXPERIMENT_NAME, "plot_accs.png"),
#     )
#     loss_name2metrics = {}
#     for metricname in ["train_loss", "valid_loss"]:
#         prop2val = metricname2prop2val[metricname]
#         props = sorted(list(prop2val.keys()))
#         vals = [prop2val[prop] for prop in props]
#         loss_name2metrics[metricname] = zip(props, vals)
#     plot_series_w_labels(
#         loss_name2metrics,
#         "loss v proportion of training data",
#         join(MODELS_DIR, EXPERIMENT_NAME, "plot_loss.png"),
#     )

#     # valid acc per issue
#     issue2prop2val = defaultdict(dict)
#     for prop in sorted(list(metrics.keys())):
#         if prop == "mean":
#             continue
#         for issue in metrics[prop]:
#             if issue == "mean":
#                 continue
#             issue_mean_valid_acc = metrics[prop][issue]["mean"]["valid_acc"]
#             issue2prop2val[issue][prop] = issue_mean_valid_acc
#     validacc_issue2metrics = {
#         issue: [(prop, val) for prop, val in prop2val.items()]
#         for issue, prop2val in issue2prop2val.items()
#     }
#     plot_series_w_labels(
#         validacc_issue2metrics,
#         "valid acc per issue v prop of training data",
#         join(MODELS_DIR, EXPERIMENT_NAME, "plot_issue_acc.png"),
#     )

#     # valid loss per issue
#     issue2prop2val = defaultdict(dict)
#     for prop in sorted(list(metrics.keys())):
#         if prop == "mean":
#             continue
#         for issue in metrics[prop]:
#             if issue == "mean":
#                 continue
#             issue_mean_valid_loss = metrics[prop][issue]["mean"]["valid_loss"]
#             issue2prop2val[issue][prop] = issue_mean_valid_loss
#     validloss_issue2metrics = {
#         issue: [(prop, val) for prop, val in prop2val.items()]
#         for issue, prop2val in issue2prop2val.items()
#     }
#     plot_series_w_labels(
#         validloss_issue2metrics,
#         "valid loss per issue v prop of training data",
#         join(MODELS_DIR, EXPERIMENT_NAME, "plot_issue_loss.png"),
#     )


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
    # _plot()
