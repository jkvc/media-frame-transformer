# Usage: python <script_name> <dataset_name> <lexicon_arch> <roberta_arch>

import sys
from os import makedirs
from os.path import exists, join
from pprint import pprint
from random import Random

import matplotlib.pyplot as plt
import numpy as np
import torch
from config import FIGURES_DIR, LEXICON_DIR, MODELS_DIR, RANDOM_SEED
from media_frame_transformer.datadef.zoo import get_datadef
from media_frame_transformer.dataset.bow_dataset import eval_lexicon_model
from media_frame_transformer.dataset.common import calculate_labelprops
from media_frame_transformer.dataset.roberta_dataset import RobertaDataset
from media_frame_transformer.learning import valid_epoch
from media_frame_transformer.model.logreg_config.base import load_logreg_model_config
from media_frame_transformer.model.logreg_config.grid_search import (
    load_logreg_model_config_all_archs,
)
from media_frame_transformer.utils import (
    DEVICE,
    load_json,
    read_txt_as_str_list,
    save_json,
)
from media_frame_transformer.viualization import plot_series_w_labels
from torch.utils.data import DataLoader

_DATASET_NAME = sys.argv[1]
_LEXICON_ARCH = sys.argv[2]
_ROBERTA_ARCH = sys.argv[3]

_DATADEF = get_datadef(_DATASET_NAME)
_LEXICON_MODEL_ROOT = join(LEXICON_DIR, _DATASET_NAME, "holdout_source", _LEXICON_ARCH)
_ROBERTA_MODEL_ROOT = join(MODELS_DIR, _DATASET_NAME, "holdout_source", _ROBERTA_ARCH)
assert exists(join(_LEXICON_MODEL_ROOT, "mean_metrics.json"))
assert exists(join(_ROBERTA_MODEL_ROOT, "mean_metrics.json"))
_LEXICON_CONFIG = load_logreg_model_config_all_archs(
    _DATADEF.n_classes, _DATADEF.n_sources
)[_LEXICON_ARCH]

_SAVE_DIR = join(
    FIGURES_DIR,
    "holdout_source_estimated_labelprops",
    _DATASET_NAME,
)
makedirs(_SAVE_DIR, exist_ok=True)

_RNG = Random()
_RNG.seed(RANDOM_SEED)

_LABELPROPS_ESTIMATE_NSAMPLES = [50, 100, 150, 200, 250, 300]

# load samples, shuffle once, use this seeded shuffle order for all evals

source2samples = {}
for source in _DATADEF.source_names:
    samples = _DATADEF.load_splits_func([source], ["train"])["train"]
    _RNG.shuffle(samples)
    source2samples[source] = samples

# lexicon model predicting with gt & estimated labelprops

_LEXICON_MODEL_PERFORMANCE_SAVE_PATH = join(_SAVE_DIR, f"{_LEXICON_ARCH}.json")

if not exists(_LEXICON_MODEL_PERFORMANCE_SAVE_PATH):
    orig_metrics = load_json(join(_LEXICON_MODEL_ROOT, "mean_metrics.json"))
    gt_labelprops_performances = [
        orig_metrics[source]["mean"]["valid_f1"] for source in _DATADEF.source_names
    ]
    gt_labelprops_performance = np.array(gt_labelprops_performances).mean()

    nsample2estimated_labelprops_perf = {}
    for nsample in _LABELPROPS_ESTIMATE_NSAMPLES:
        labelprops_perfs = []
        for source in _DATADEF.source_names:

            samples = source2samples[source]
            selected_samples = samples[:nsample]
            estimated_labelprops = {
                "estimated": calculate_labelprops(
                    selected_samples, _DATADEF.n_classes, _DATADEF.source_names
                )
            }
            datadef = get_datadef(_DATASET_NAME)
            datadef.load_labelprops_func = lambda _split: estimated_labelprops[_split]

            model = torch.load(join(_LEXICON_MODEL_ROOT, source, "model.pth")).to(
                DEVICE
            )
            vocab = read_txt_as_str_list(join(_LEXICON_MODEL_ROOT, source, "vocab.txt"))
            metrics = eval_lexicon_model(
                model,
                datadef,
                samples,
                vocab,
                use_source_individual_norm=_LEXICON_CONFIG[
                    "use_source_individual_norm"
                ],
                labelprop_split="estimated",  # match _load_labelprops_func()
            )
            labelprops_perfs.append(metrics["valid_f1"])

        nsample2estimated_labelprops_perf[str(nsample)] = np.array(
            labelprops_perfs
        ).mean()

    lexicon_model_perf = nsample2estimated_labelprops_perf
    lexicon_model_perf["gt"] = gt_labelprops_performance
    save_json(lexicon_model_perf, _LEXICON_MODEL_PERFORMANCE_SAVE_PATH)

else:
    lexicon_model_perf = load_json(_LEXICON_MODEL_PERFORMANCE_SAVE_PATH)

# roberta model predicting with gt and estimated labelprops

_ROBERTA_MODEL_PERFORMANCE_SAVE_PATH = join(_SAVE_DIR, f"{_ROBERTA_ARCH}.json")

if not exists(_ROBERTA_MODEL_PERFORMANCE_SAVE_PATH):
    orig_metrics = load_json(join(_ROBERTA_MODEL_ROOT, "mean_metrics.json"))
    gt_labelprops_performances = [
        orig_metrics[source]["mean"]["valid_f1.best"]
        for source in _DATADEF.source_names
    ]
    gt_labelprops_performance = np.array(gt_labelprops_performances).mean()

    nsample2estimated_labelprops_perf = {}
    for nsample in _LABELPROPS_ESTIMATE_NSAMPLES:
        labelprops_perfs = []
        for source in _DATADEF.source_names:

            samples = source2samples[source]
            selected_samples = samples[:nsample]
            estimated_labelprops = calculate_labelprops(
                selected_samples, _DATADEF.n_classes, _DATADEF.source_names
            )

            model = torch.load(join(_ROBERTA_MODEL_ROOT, source, "checkpoint.pth")).to(
                DEVICE
            )

            valid_dataset = RobertaDataset(
                samples,
                _DATADEF.n_classes,
                _DATADEF.source_names,
                source2labelprops=estimated_labelprops,
            )
            valid_loader = DataLoader(
                valid_dataset, batch_size=100, shuffle=False, num_workers=4
            )
            metrics = valid_epoch(model, valid_loader)
            labelprops_perfs.append(metrics["f1"])

        nsample2estimated_labelprops_perf[str(nsample)] = np.array(
            labelprops_perfs
        ).mean()

    roberta_model_perf = nsample2estimated_labelprops_perf
    roberta_model_perf["gt"] = gt_labelprops_performance
    save_json(roberta_model_perf, _ROBERTA_MODEL_PERFORMANCE_SAVE_PATH)

else:
    roberta_model_perf = load_json(_ROBERTA_MODEL_PERFORMANCE_SAVE_PATH)

# plot them

_PLOT_SAVE_PATH = join(_SAVE_DIR, f"_plot.{_LEXICON_ARCH}.{_ROBERTA_ARCH}.png")
plt.clf()
plt.plot(
    _LABELPROPS_ESTIMATE_NSAMPLES,
    [lexicon_model_perf[str(nsample)] for nsample in _LABELPROPS_ESTIMATE_NSAMPLES],
    c="firebrick",
    label=f"{_LEXICON_ARCH} estimated",
)
plt.axhline(
    lexicon_model_perf["gt"],
    color="firebrick",
    linestyle="--",
    label=f"{_LEXICON_ARCH} ground truth",
)
plt.plot(
    _LABELPROPS_ESTIMATE_NSAMPLES,
    [roberta_model_perf[str(nsample)] for nsample in _LABELPROPS_ESTIMATE_NSAMPLES],
    c="teal",
    label=f"{_ROBERTA_ARCH} estimated",
)
plt.axhline(
    roberta_model_perf["gt"],
    color="teal",
    linestyle="--",
    label=f"{_ROBERTA_ARCH} ground truth",
)
plt.title(f"holdout source acc under estimated labelprops ({_DATASET_NAME})")
plt.legend()
plt.xlabel("# sample for labelprops est.")
plt.ylabel("holdout source acc")
plt.savefig(_PLOT_SAVE_PATH)