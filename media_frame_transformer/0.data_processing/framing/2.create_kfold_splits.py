from os import makedirs
from os.path import join
from random import Random

import numpy as np
from config import DATA_DIR, KFOLD, RANDOM_SEED
from media_frame_transformer.datadef import zoo
from media_frame_transformer.utils import load_json, save_json

ISSUES = zoo.get_datadef("framing").source_names

RNG = Random()
RNG.seed(RANDOM_SEED)

FRAMING_DATA_DIR = join(DATA_DIR, "framing_labeled")
SAVEDIR = join(FRAMING_DATA_DIR, f"{KFOLD}fold")

if __name__ == "__main__":
    makedirs(SAVEDIR, exist_ok=True)

    for issue in ISSUES:
        print(">>", issue)
        trainsets = load_json(join(FRAMING_DATA_DIR, f"{issue}_train_sets.json"))

        to_save = {}

        for task, all_ids in trainsets.items():
            all_ids = sorted(all_ids)
            RNG.shuffle(all_ids)
            numsamples = len(all_ids)
            chunksize = int(np.round(numsamples / KFOLD))

            folds = []
            for ki in range(KFOLD):
                valid_ids = set(all_ids[ki * chunksize : (ki + 1) * chunksize])
                train_ids = [id for id in all_ids if id not in valid_ids]
                valid_ids = list(valid_ids)
                folds.append({"train": train_ids, "valid": valid_ids})
            to_save[task] = folds

            print("--", task)
            for fold in folds:
                print("--", "train", len(fold["train"]), "valid", len(fold["valid"]))

        save_json(to_save, join(SAVEDIR, f"{issue}.json"))
