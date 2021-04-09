import sys
from os import makedirs
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt
from config import FIGURES_DIR, ISSUES, MODELS_DIR
from media_frame_transformer.experiment_config import (
    ARCH,
    BATCHSIZE,
    DATASET_SIZES,
    FOLDS_TO_RUN,
    KFOLD,
)
from media_frame_transformer.utils import (
    load_json,
    mkdir_overwrite,
    write_str_list_as_txt,
)
from media_frame_transformer.viualization import plot_series_w_labels

_arch = sys.argv[1]

if __name__ == "__main__":
    savedir = join(FIGURES_DIR, _arch)
    makedirs(savedir, exist_ok=True)

    for issue in ISSUES + ["mean"]:

        type2numsample0acc = {}

        # ood (13f holdout issue)
        ood_raw_json = load_json(join(MODELS_DIR, f"13f.{_arch}", "mean_metrics.json"))
        if issue != "mean":
            ood_acc = ood_raw_json[f"holdout_{issue}"]["mean"]["valid_f1"]
        else:
            ood_acc = ood_raw_json["mean"]["valid_f1"]
        type2numsample0acc["ood"] = [
            (numsample, ood_acc) for numsample in DATASET_SIZES
        ]

        # ood adapted (14 holdout adapt)
        adapt_raw_json = load_json(
            join(MODELS_DIR, f"14.{_arch}", "best_earlystop.json")
        )
        type2numsample0acc["ood adapted"] = []
        for numsample in DATASET_SIZES:
            if issue != "mean":
                type2numsample0acc["ood adapted"].append(
                    (
                        numsample,
                        adapt_raw_json[str(numsample)][issue]["mean"][
                            "best_earlystop_valid_f1"
                        ],
                    )
                )
            else:
                type2numsample0acc["ood adapted"].append(
                    (
                        numsample,
                        adapt_raw_json[str(numsample)]["mean"][
                            "best_earlystop_valid_f1"
                        ],
                    )
                )

        # vanilla adapted (3111 scarce num within issue)
        vanilla_raw_json = load_json(
            join(MODELS_DIR, f"3111.{_arch.replace('_dev','')}", "mean_metrics.json")
        )
        type2numsample0acc["vanilla adapted"] = []
        for numsample in DATASET_SIZES:
            if issue != "mean":
                type2numsample0acc["vanilla adapted"].append(
                    (
                        numsample,
                        vanilla_raw_json[f"{numsample:04}_samples"][issue]["mean"][
                            "valid_f1"
                        ],
                    )
                )
            else:
                type2numsample0acc["vanilla adapted"].append(
                    (
                        numsample,
                        vanilla_raw_json[f"{numsample:04}_samples"]["mean"]["valid_f1"],
                    )
                )

        plot_series_w_labels(
            type2numsample0acc,
            title=f"Finetuned {_arch} accuracy: {issue}",
            save_path=join(savedir, f"2_{issue}.png"),
        )
