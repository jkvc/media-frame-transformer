from collections import defaultdict
from os.path import exists, join

import pandas as pd
from config import ISSUES, MODELS_DIR

from media_frame_transformer.dataset import load_kfold
from media_frame_transformer.learning import get_kfold_metrics, valid

EXPERIMENT_NAME = "1.1-h"
KFOLD = 8


if __name__ == "__main__":
    root_path = join(MODELS_DIR, EXPERIMENT_NAME)
    assert exists(
        root_path
    ), f"{root_path} does not exist, choose the correct experiment name"

    metrics_save_filepath = join(root_path, "metrics.csv")
    assert not exists(metrics_save_filepath)

    issue2metrics = {}
    for issue in ISSUES:
        print(issue)
        issue_path = join(root_path, issue)

        metrics = get_kfold_metrics([issue], "primary_frame", KFOLD, issue_path)
        issue2metrics[issue] = metrics

    df = pd.DataFrame.from_dict(issue2metrics, orient="index")
    df.loc["mean"] = df.mean()
    print(df)
    df.to_csv(metrics_save_filepath)
