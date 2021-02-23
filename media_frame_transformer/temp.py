import shutil
from glob import glob
from os import mkdir
from os.path import dirname, exists, join
from shutil import copy

import pandas as pd
import torch
import torch.nn as nn
from config import ISSUES, MODELS_DIR

from media_frame_transformer import models
from media_frame_transformer.dataset import get_kfold_primary_frames_datasets
from media_frame_transformer.learning import get_kfold_metrics, train
from media_frame_transformer.utils import mkdir_overwrite, write_str_list_as_txt

exp_dir = join(MODELS_DIR, "1.1.roberta_half.best")

metrics_json_path = glob(join(exp_dir, "**", "metrics.json"), recursive=True)
print(metrics_json_path)

for p in metrics_json_path:
    copy(p, join(dirname(p), "leaf_metrics.json"))
