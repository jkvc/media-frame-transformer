from os.path import exists, join

from config import ISSUES, MODELS_DIR

from media_frame_transformer.dataset import load_kfold
from media_frame_transformer.learning import train
from media_frame_transformer.utils import mkdir_overwrite

EXPERIMENT_NAME = "1.3-a"

KFOLD = 8
N_EPOCH = 8


if __name__ == "__main__":
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    assert not exists(
        save_root
    ), f"{save_root} already exists, remove existing or choose another experiment name"
    mkdir_overwrite(save_root)

    for holdout_issue in ISSUES:
        model_name = f"holdout_{holdout_issue}"
        print(model_name)
        save_dir = join(save_root, model_name)
        mkdir_overwrite(save_dir)

        # train on all issues other than the holdout one
        train_issues = [iss for iss in ISSUES if iss != holdout_issue]

        kfold_datasets = load_kfold(train_issues, "primary_frame", KFOLD)
        for ki, datasets in enumerate(kfold_datasets):
            save_fold = join(save_dir, f"fold_{ki}")
            mkdir_overwrite(save_fold)

            train_dataset = datasets["train"]
            valid_dataset = datasets["valid"]
            train(train_dataset, valid_dataset, save_fold, N_EPOCH)
