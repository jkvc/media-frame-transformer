from os import mkdir
from os.path import exists, join
from typing import List

from config import FRAMING_DATA_DIR, ISSUES, MODELS_DIR
from tqdm import tqdm

from media_frame_transformer import models
from media_frame_transformer.dataset import (
    TextSample,
    fold2split2samples_to_datasets,
    load_kfold_primary_frame_samples,
)
from media_frame_transformer.learning import train
from media_frame_transformer.utils import (
    load_json,
    mkdir_overwrite,
    write_str_list_as_txt,
)

EXPERIMENT_NAME = "2.0.11.weight.2"
ARCH = "roberta_base_half"

AUG_WEIGHT = 0.2

# if true, only run experiement on 0th fold (i.e. a single fixed valid set)
ZEROTH_FOLD_ONLY = True
KFOLD = 8

N_EPOCH = 10
BATCHSIZE = 50


def get_kfold_span_frame_train_samples(
    issues: List[str], k: int, augment_sample_weight: float
):
    for issue in issues:
        assert exists(
            join(FRAMING_DATA_DIR, f"{issue}_{k}_folds.json")
        ), f"{issue}_{k}_folds.json does not exist, run create_kfold first"

    fold2samples = [[] for _ in range(k)]

    for issue in tqdm(issues):
        frame_span_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_frame_spans.json"))
        kfold_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_{k}_folds.json"))

        for ki, fold in enumerate(kfold_data["primary_frame"]):
            for id in fold["train"]:
                spans_in_that_article = frame_span_data.get(id, [])
                for span in spans_in_that_article:
                    fold2samples[ki].append(
                        TextSample(
                            text=span["text"],
                            code=span["code"],
                            weight=augment_sample_weight,
                        )
                    )
    return fold2samples


def augment_train_splits(base_fold2split2samples, aug_fold2samples):
    assert len(base_fold2split2samples) == len(aug_fold2samples)
    for ki in range(len(base_fold2split2samples)):
        base_fold2split2samples[ki]["train"].extend(aug_fold2samples[ki])


if __name__ == "__main__":
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    if not exists(save_root):
        mkdir(save_root)

    for issue in ISSUES:
        print(issue)
        save_issue_path = join(save_root, issue)
        if not exists(save_issue_path):
            mkdir(save_issue_path)

        fold2split2samples = load_kfold_primary_frame_samples([issue], KFOLD)
        print(">> before aug", len(fold2split2samples[0]["train"]))
        aug_fold2samples = get_kfold_span_frame_train_samples(
            [issue], KFOLD, AUG_WEIGHT
        )
        augment_train_splits(fold2split2samples, aug_fold2samples)
        print(">>  after aug", len(fold2split2samples[0]["train"]))

        augmented_datasets = fold2split2samples_to_datasets(fold2split2samples)
        for ki, datasets in enumerate(augmented_datasets):

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
                N_EPOCH,
                BATCHSIZE,
            )

            # mark done
            write_str_list_as_txt(["."], join(save_fold_path, "_complete"))

            if ZEROTH_FOLD_ONLY:
                break
