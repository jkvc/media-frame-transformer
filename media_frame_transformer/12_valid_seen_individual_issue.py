from collections import defaultdict
from os.path import exists, join

import pandas as pd
from config import ISSUES, MODELS_DIR

from media_frame_transformer.dataset import load_kfold
from media_frame_transformer.learning import get_kfold_metrics, valid

EXPERIMENT_NAME = "1.2-c"
KFOLD = 8


if __name__ == "__main__":
    experiment_root_path = join(MODELS_DIR, EXPERIMENT_NAME)
    assert exists(
        experiment_root_path
    ), f"{experiment_root_path} does not exist, choose the correct experiment name"

    metrics_save_filepath = join(experiment_root_path, "metrics_individual_issues.csv")
    assert not exists(metrics_save_filepath)

    issue2metrics = {}
    for issue in ISSUES:
        print(issue)
        metrics = get_kfold_metrics(
            [issue], "primary_frame", KFOLD, experiment_root_path
        )
        issue2metrics[issue] = metrics

    df = pd.DataFrame.from_dict(issue2metrics, orient="index")
    df.loc["mean"] = df.mean()
    print(df)
    df.to_csv(metrics_save_filepath)
