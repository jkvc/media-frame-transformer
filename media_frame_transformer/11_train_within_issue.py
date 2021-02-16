from os import mkdir, write
from os.path import exists, join

from config import ISSUES, MODELS_DIR

from media_frame_transformer import models
from media_frame_transformer.dataset import get_kfold_primary_frames_datasets
from media_frame_transformer.learning import train
from media_frame_transformer.utils import mkdir_overwrite, write_str_list_as_txt

EXPERIMENT_NAME = "1.1.test"
ARCH = "roberta_base_half"


KFOLD = 8
N_EPOCH = 10
BATCHSIZE = 50

if __name__ == "__main__":
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    if not exists(save_root):
        mkdir(save_root)

    for issue in ISSUES:
        print(issue)
        save_issue_path = join(save_root, issue)
        if not exists(save_issue_path):
            mkdir(save_issue_path)

        kfold_datasets = get_kfold_primary_frames_datasets([issue], KFOLD)
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
