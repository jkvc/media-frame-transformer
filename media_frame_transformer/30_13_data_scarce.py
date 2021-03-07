from collections import defaultdict
from os import mkdir
from os.path import exists, join
from pprint import pprint
from random import shuffle

import numpy as np
from config import ISSUES, MODELS_DIR

from media_frame_transformer import models
from media_frame_transformer.dataset import (
    PrimaryFrameDataset,
    fold2split2samples_to_datasets,
    load_all_primary_frame_samples,
    load_kfold_primary_frame_samples,
)
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiment_config import (
    ARCH,
    BATCHSIZE,
    FOLDS_TO_RUN,
    KFOLD,
)
from media_frame_transformer.learning import train
from media_frame_transformer.utils import (
    load_json,
    mkdir_overwrite,
    write_str_list_as_txt,
)
from media_frame_transformer.viualization import plot_series_w_labels

EXPERIMENT_NAME = f"3013.{ARCH}"

DATASET_SIZE_PROPS = [0.2, 0.4, 0.6, 0.8, 1.0]


def _train():
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    if not exists(save_root):
        mkdir(save_root)

    # root/prop/issue/fold
    for prop in DATASET_SIZE_PROPS:
        save_prop_path = join(save_root, str(prop))
        if not exists(save_prop_path):
            mkdir(save_prop_path)

        for holdout_issue in ISSUES:
            model_name = f"holdout_{holdout_issue}"
            print(model_name)
            save_issue_path = join(save_prop_path, model_name)
            if not exists(save_issue_path):
                mkdir(save_issue_path)

            train_issues = [iss for iss in ISSUES if iss != holdout_issue]
            fold2split2samples = load_kfold_primary_frame_samples(train_issues, KFOLD)

            holdout_issue_all_samples = load_all_primary_frame_samples([holdout_issue])
            holdout_issue_dataset = PrimaryFrameDataset(holdout_issue_all_samples)

            # subsample all folds
            subsampled_fold2split2samples = []
            for ki, split2samples in enumerate(fold2split2samples):
                num_train_samples = int(np.ceil(len(split2samples["train"]) * prop))
                print(">> prop", prop, "split", ki, "num train", num_train_samples)
                shuffle(split2samples["train"])
                subsampled_fold2split2samples.append(
                    {
                        "valid": split2samples["valid"],
                        "train": split2samples["train"][:num_train_samples],
                    }
                )
            kfold_datasets = fold2split2samples_to_datasets(
                subsampled_fold2split2samples
            )

            for ki, datasets in enumerate(kfold_datasets):
                if ki not in FOLDS_TO_RUN:
                    print(">> not running fold", ki)
                    continue

                # skip done
                save_fold_path = join(save_issue_path, f"fold_{ki}")
                if exists(join(save_fold_path, "_complete")):
                    print(">> skip", ki)
                    continue
                mkdir_overwrite(save_fold_path)

                train_dataset = datasets["train"]
                valid_dataset = datasets["valid"]

                model = models.get_model(ARCH)
                train(
                    model,
                    train_dataset,
                    valid_dataset,
                    save_fold_path,
                    max_epochs=30,
                    batchsize=BATCHSIZE,
                    additional_valid_datasets={"holdout_issue": holdout_issue_dataset},
                )

                # mark done
                write_str_list_as_txt(["."], join(save_fold_path, "_complete"))


def _plot():
    metrics_json_path = join(MODELS_DIR, EXPERIMENT_NAME, "mean_metrics.json")
    metrics = load_json(metrics_json_path)
    pprint(metrics)

    metricname2prop2val = defaultdict(dict)
    for prop in metrics:
        if prop == "mean":
            continue
        for metricname, val in metrics[prop]["mean"].items():
            metricname2prop2val[metricname][prop] = val
    pprint(metricname2prop2val)
    # acc  aggregated over all issues
    acc_name2metrics = {}
    for metricname in ["train_acc", "valid_acc"]:
        prop2val = metricname2prop2val[metricname]
        props = sorted(list(prop2val.keys()))
        vals = [prop2val[prop] for prop in props]
        acc_name2metrics[metricname] = zip(props, vals)
    plot_series_w_labels(
        acc_name2metrics,
        "accuracy v proportion of training data",
        join(MODELS_DIR, EXPERIMENT_NAME, "plot_accs.png"),
    )

    # valid acc per issue
    issue2prop2val = defaultdict(dict)
    for prop in sorted(list(metrics.keys())):
        if prop == "mean":
            continue
        for issue in metrics[prop]:
            if issue == "mean":
                continue
            issue_mean_valid_acc = metrics[prop][issue]["mean"]["valid_acc"]
            issue2prop2val[issue][prop] = issue_mean_valid_acc
    validacc_issue2metrics = {
        issue: [(prop, val) for prop, val in prop2val.items()]
        for issue, prop2val in issue2prop2val.items()
    }
    plot_series_w_labels(
        validacc_issue2metrics,
        "valid acc per issue v prop of training data",
        join(MODELS_DIR, EXPERIMENT_NAME, "plot_issue_acc.png"),
    )

    # holdout_issue acc per issue
    issue2prop2val = defaultdict(dict)
    for prop in sorted(list(metrics.keys())):
        if prop == "mean":
            continue
        for issue in metrics[prop]:
            if issue == "mean":
                continue
            issue_mean_holdout_issue_acc = metrics[prop][issue]["mean"][
                "holdout_issue_acc"
            ]
            issue2prop2val[issue][prop] = issue_mean_holdout_issue_acc
    holdout_issueacc_issue2metrics = {
        issue: [(prop, val) for prop, val in prop2val.items()]
        for issue, prop2val in issue2prop2val.items()
    }
    plot_series_w_labels(
        holdout_issueacc_issue2metrics,
        "holdout_issue acc per issue v prop of training data",
        join(MODELS_DIR, EXPERIMENT_NAME, "plot_holdout_issue_acc.png"),
    )


if __name__ == "__main__":
    # _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
    _plot()