from os import mkdir
from os.path import exists, join

from config import ISSUES, MODELS_DIR

from media_frame_transformer import models
from media_frame_transformer.dataset import load_kfold
from media_frame_transformer.learning import train
from media_frame_transformer.utils import mkdir_overwrite, write_str_list_as_txt

EXPERIMENT_NAME = "1.3.a"
ARCH = "roberta_base_half"

KFOLD = 8
N_EPOCH = 8
BATCHSIZE = 50


if __name__ == "__main__":
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    if not exists(save_root):
        mkdir(save_root)

    for holdout_issue in ISSUES:
        model_name = f"holdout_{holdout_issue}"
        print(model_name)
        save_issue_path = join(save_root, model_name)
        if not exists(save_issue_path):
            mkdir(save_issue_path)

        # train on all issues other than the holdout one
        train_issues = [iss for iss in ISSUES if iss != holdout_issue]

        kfold_datasets = load_kfold(train_issues, "primary_frame", KFOLD)
        for ki, datasets in enumerate(kfold_datasets):

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
