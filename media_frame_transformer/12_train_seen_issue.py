from os import mkdir
from os.path import exists, join

from config import ISSUES, MODELS_DIR

from media_frame_transformer import models
from media_frame_transformer.dataset import load_kfold
from media_frame_transformer.learning import train
from media_frame_transformer.utils import mkdir_overwrite, write_str_list_as_txt

EXPERIMENT_NAME = "1.2-e"
ARCH = "roberta_highdrop_half"

KFOLD = 8
N_EPOCH = 15
BATCHSIZE = 30

if __name__ == "__main__":
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    if not exists(save_root):
        mkdir(save_root)

    kfold_datasets = load_kfold(ISSUES, "primary_frame", KFOLD)
    for ki, datasets in enumerate(kfold_datasets):
        save_fold = join(save_root, f"fold_{ki}")
        if exists(join(save_fold, "_complete")):
            print(">> skip", ki)
            continue
        mkdir_overwrite(save_fold)

        train_dataset = datasets["train"]
        valid_dataset = datasets["valid"]

        model = models.get_model(ARCH)
        train(
            model,
            train_dataset,
            valid_dataset,
            save_fold,
            N_EPOCH,
            BATCHSIZE,
        )

        write_str_list_as_txt(["."], join(save_fold, "_complete"))
