import random
import sys
from os.path import join

import numpy as np
from config import FRAMING_DATA_DIR, ISSUES

from media_frame_transformer.utils import load_json, save_json

random.seed(0xDEADBEEF)

K = int(sys.argv[1])

if __name__ == "__main__":

    for issue in ISSUES:
        print(">>", issue)
        trainsets = load_json(join(FRAMING_DATA_DIR, f"{issue}_train_sets.json"))

        to_save = {}

        for task, all_ids in trainsets.items():
            all_ids = sorted(all_ids)
            random.shuffle(all_ids)
            numsamples = len(all_ids)
            chunksize = int(np.round(numsamples / K))

            folds = []
            for ki in range(K):
                valid_ids = set(all_ids[ki * chunksize : (ki + 1) * chunksize])
                train_ids = [id for id in all_ids if id not in valid_ids]
                valid_ids = list(valid_ids)
                folds.append({"train": train_ids, "valid": valid_ids})
            to_save[task] = folds

            print("--", task)
            for fold in folds:
                print("--", "train", len(fold["train"]), "valid", len(fold["valid"]))

        save_json(to_save, join(FRAMING_DATA_DIR, f"{issue}_{K}_folds.json"))
