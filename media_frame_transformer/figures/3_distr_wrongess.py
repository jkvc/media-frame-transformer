import sys
from os import makedirs
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt
from config import FIGURES_DIR, ISSUES, MODELS_DIR, OUTPUTS_DIR
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
# assert _arch.endswith("_dev")

if __name__ == "__main__":
    savedir = join(FIGURES_DIR, _arch)
    makedirs(savedir, exist_ok=True)
    issue2numsample2trial2f1 = load_json(
        join(OUTPUTS_DIR, _arch, "13f_distr_wrongness.json")
    )

    for issue in ISSUES:
        save_path = join(savedir, f"3_{issue}.png")
        plt.clf()

        numsample2trial2f1 = issue2numsample2trial2f1[issue]
        scatter_x, scatter_y = [], []
        for numsample, trial2f1 in numsample2trial2f1.items():
            for f1 in trial2f1.values():
                scatter_x.append(int(numsample))
                scatter_y.append(f1)
        plt.scatter(scatter_x, scatter_y, c="black", marker="x")

        name2numsample0acc = {}

        # avg f1 with distribution estimated from limited size sample
        name2numsample0acc["estimated distribution"] = []
        for numsample, trial2f1 in numsample2trial2f1.items():
            numsample = int(numsample)
            f1s = list(trial2f1.values())
            f1 = sum(f1s) / len(f1s)
            name2numsample0acc["estimated distribution"].append((numsample, f1))

        # avg f1 with gt distribution
        gtdev_metrics = load_json(join(MODELS_DIR, f"13f.{_arch}", "mean_metrics.json"))
        gt_distribution_f1 = gtdev_metrics[f"holdout_{issue}"]["mean"]["valid_f1"]
        name2numsample0acc["gt distribution"] = [
            (numsample, gt_distribution_f1) for numsample in DATASET_SIZES
        ]

        plot_series_w_labels(
            name2numsample0acc,
            title=f"estimated distribution on pre-adapted model : {issue}",
            save_path=save_path,
        )
