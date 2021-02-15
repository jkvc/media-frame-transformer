from collections import defaultdict
from os.path import exists, join

import pandas as pd
from config import ISSUES, MODELS_DIR

from media_frame_transformer.dataset import load_kfold
from media_frame_transformer.learning import get_kfold_metrics, valid

EXPERIMENT_NAME = "1.2-e"
KFOLD = 8


if __name__ == "__main__":
    experiment_root_path = join(MODELS_DIR, EXPERIMENT_NAME)
    assert exists(
        experiment_root_path
    ), f"{experiment_root_path} does not exist, choose the correct experiment name"

    metrics_save_filepath = join(experiment_root_path, "metrics.csv")
    assert not exists(metrics_save_filepath)

    metrics = get_kfold_metrics(ISSUES, "primary_frame", KFOLD, experiment_root_path)
    metrics = {"all": metrics}

    df = pd.DataFrame.from_dict(metrics, orient="index")
    print(df)
    df.to_csv(metrics_save_filepath)
