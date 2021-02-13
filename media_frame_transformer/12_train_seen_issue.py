from os.path import exists, join

from config import ISSUES, MODELS_DIR

from media_frame_transformer.dataset import load_kfold
from media_frame_transformer.learning import train
from media_frame_transformer.utils import mkdir_overwrite

EXPERIMENT_NAME = "1.2-d"

KFOLD = 8
N_EPOCH = 10

if __name__ == "__main__":
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    assert not exists(
        save_root
    ), f"{save_root} already exists, remove existing or choose another experment name"
    mkdir_overwrite(save_root)

    kfold_datasets = load_kfold(ISSUES, "primary_frame", KFOLD)
    for ki, datasets in enumerate(kfold_datasets):
        save_fold = join(save_root, f"fold_{ki}")
        mkdir_overwrite(save_fold)

        train_dataset = datasets["train"]
        valid_dataset = datasets["valid"]
        train(train_dataset, valid_dataset, save_fold, N_EPOCH)
