import sys
from os.path import basename, dirname, join, realpath

from config import BATCHSIZE, MODELS_DIR
from media_frame_transformer.dataset.arxiv.definition import (
    ARXIV_CATEGORIES,
    ARXIV_N_CATEGORIES,
    LABELPROPS_DIR,
    YEARRANGE_N_CLASSES,
)
from media_frame_transformer.dataset.arxiv.samples import (
    load_all_arxiv_abstract_samples,
)
from media_frame_transformer.dataset.roberta_dataset import RobertaDataset
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.model.roberta_config.base import load_roberta_model_config

_N_TRAIN_EPOCH = 6

_ARCH = sys.argv[1]
_CONFIG = load_roberta_model_config(_ARCH, YEARRANGE_N_CLASSES, ARXIV_N_CATEGORIES)

_SCRIPT_PATH = realpath(__file__)
_EXPERIMENT_NAME = basename(_SCRIPT_PATH).replace(".py", "")
_DATASOURCE_NAME = basename(dirname(_SCRIPT_PATH))
_SAVE_DIR = join(MODELS_DIR, _DATASOURCE_NAME, _EXPERIMENT_NAME, _ARCH)


path2datasets = {}

for holdout_cat in ARXIV_CATEGORIES:
    print(">>", holdout_cat)
    savedir = join(_SAVE_DIR, holdout_cat)

    train_issues = [i for i in ARXIV_CATEGORIES if i != holdout_cat]
    train_samples = load_all_arxiv_abstract_samples(train_issues, "train")
    valid_samples = load_all_arxiv_abstract_samples([holdout_cat], "train")

    train_dataset = RobertaDataset(
        train_samples,
        n_classes=YEARRANGE_N_CLASSES,
        source_names=ARXIV_CATEGORIES,
        labelprop_split="train",
        labelprop_dir=LABELPROPS_DIR,
    )
    valid_dataset = RobertaDataset(
        valid_samples,
        n_classes=YEARRANGE_N_CLASSES,
        source_names=ARXIV_CATEGORIES,
        labelprop_split="train",
        labelprop_dir=LABELPROPS_DIR,
    )

    path2datasets[join(_SAVE_DIR, holdout_cat)] = {
        "train": train_dataset,
        "valid": valid_dataset,
    }

run_experiments(
    _CONFIG,
    path2datasets,
    batchsize=BATCHSIZE,
    save_model_checkpoint=True,
    # train a fixed number of epoch since we can't use holdout issue as
    # validation data to early stop
    max_epochs=_N_TRAIN_EPOCH,
    num_early_stop_non_improve_epoch=_N_TRAIN_EPOCH,
)
reduce_and_save_metrics(_SAVE_DIR)
