# Usage: python <script_name> <dataset_name> <model_arch1> <model_arch2>

import sys
from os import makedirs
from os.path import basename, exists, join, realpath

import numpy as np
import torch
from config import BATCHSIZE, LEXICON_DIR, MODELS_DIR, OUTPUT_DIR
from media_frame_transformer.datadef.zoo import get_datadef
from media_frame_transformer.dataset.bow_dataset import (
    build_bow_full_batch,
    get_all_tokens,
)
from media_frame_transformer.dataset.roberta_dataset import RobertaDataset
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.model.logreg_config.grid_search import (
    load_logreg_model_config_all_archs,
)
from media_frame_transformer.model.roberta_config.base import load_roberta_model_config
from media_frame_transformer.utils import (
    DEVICE,
    load_json,
    read_txt_as_str_list,
    save_json,
)
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_DATASET_NAME = sys.argv[1]
_ARCH1 = sys.argv[2]
_ARCH2 = sys.argv[3]

_DATADEF = get_datadef(_DATASET_NAME)


def _get_model_dir(arch):
    if arch.startswith("logreg"):
        return join(LEXICON_DIR, _DATASET_NAME, "holdout_source", arch)
    elif arch.startswith("roberta"):
        return join(MODELS_DIR, _DATASET_NAME, "holdout_source", arch)


def valid_roberta_model(arch):
    print(">>", arch)

    model_dir = join(MODELS_DIR, _DATASET_NAME, "holdout_source", arch)
    save_preds_dir = join(model_dir, "valid_preds")
    makedirs(save_preds_dir, exist_ok=True)

    for holdout_source in _DATADEF.source_names:
        print(">>>>", holdout_source)
        save_preds_path = join(model_dir, "valid_preds", f"{holdout_source}.json")
        if exists(save_preds_path):
            continue

        # valid using holdout issue all samples
        valid_samples = _DATADEF.load_splits_func([holdout_source], ["train"])["train"]
        valid_dataset = RobertaDataset(
            valid_samples,
            n_classes=_DATADEF.n_classes,
            source_names=_DATADEF.source_names,
            source2labelprops=_DATADEF.load_labelprops_func("train"),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=150,
            shuffle=True,
            num_workers=6,
        )

        checkpoint_path = join(model_dir, holdout_source, "checkpoint.pth")
        model = torch.load(checkpoint_path).to(DEVICE)
        model.eval()

        id2results = {}
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                outputs = model(batch)
                logits = outputs["logits"].detach().cpu().numpy()
                preds = np.argmax(logits, axis=1)
                labels = outputs["labels"].detach().cpu().numpy()
                ids = batch["id"]
                for id, pred, label in zip(ids, preds, labels):
                    id2results[id] = {
                        "pred": int(pred),
                        "label": int(label),
                        "correct": bool(pred == label),
                    }
        save_json(id2results, save_preds_path)


def valid_logreg_model(arch):
    print(">>", arch)
    config = load_logreg_model_config_all_archs(_DATADEF.n_classes, _DATADEF.n_sources)[
        arch
    ]

    model_dir = join(LEXICON_DIR, _DATASET_NAME, "holdout_source", arch)
    save_preds_dir = join(model_dir, "valid_preds")
    makedirs(save_preds_dir, exist_ok=True)

    for holdout_source in _DATADEF.source_names:
        print(">>>>", holdout_source)
        save_preds_path = join(model_dir, "valid_preds", f"{holdout_source}.json")
        if exists(save_preds_path):
            continue

        # valid using holdout issue all samples
        valid_samples = _DATADEF.load_splits_func([holdout_source], ["train"])["train"]
        model = torch.load(join(model_dir, holdout_source, "model.pth"))
        batch = build_bow_full_batch(
            valid_samples,
            _DATADEF,
            get_all_tokens(valid_samples),
            read_txt_as_str_list(join(model_dir, holdout_source, "vocab.txt")),
            use_source_individual_norm=config["use_source_individual_norm"],
            labelprop_split="train",
        )

        model.eval()
        with torch.no_grad():
            outputs = model(batch)

        logits = outputs["logits"].detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        labels = outputs["labels"].detach().cpu().numpy()
        ids = [s.id for s in valid_samples]

        id2results = {}
        for id, pred, label in zip(ids, preds, labels):
            id2results[id] = {
                "pred": int(pred),
                "label": int(label),
                "correct": bool(pred == label),
            }
        save_json(id2results, save_preds_path)


# run prediction on respective valid sets
for arch in [_ARCH1, _ARCH2]:
    if arch.startswith("roberta"):
        valid_roberta_model(arch)
    if arch.startswith("logreg"):
        valid_logreg_model(arch)

_OUTPUT_SAVE_DIR = join(OUTPUT_DIR, "mcnemar_holdout_source_roberta")
makedirs(_OUTPUT_SAVE_DIR, exist_ok=True)

results = {}
fulltable = np.zeros((2, 2))
for holdout_source in _DATADEF.source_names:
    table = np.zeros((2, 2))
    arch1_preds = load_json(
        join(
            _get_model_dir(_ARCH1),
            "valid_preds",
            f"{holdout_source}.json",
        )
    )
    arch2_preds = load_json(
        join(
            _get_model_dir(_ARCH2),
            "valid_preds",
            f"{holdout_source}.json",
        )
    )
    ids = list(arch1_preds.keys())
    for id in ids:
        arch1_correct = arch1_preds[id]["correct"]
        arch2_correct = arch2_preds[id]["correct"]
        if arch1_correct and arch2_correct:
            table[0][0] += 1
            fulltable[0][0] += 1
        if arch1_correct and not arch2_correct:
            table[1][0] += 1
            fulltable[1][0] += 1
        if not arch1_correct and arch2_correct:
            table[0][1] += 1
            fulltable[0][1] += 1
        if not arch1_correct and not arch2_correct:
            table[1][1] += 1
            fulltable[1][1] += 1

    result = mcnemar(table)
    results[holdout_source] = {
        "pvalue": result.pvalue,
        "statistic": result.statistic,
    }
all_result = mcnemar(fulltable)
results["all"] = {
    "pvalue": all_result.pvalue,
    "statistic": all_result.statistic,
}
results["fulltable"] = fulltable.tolist()

save_json(results, join(_OUTPUT_SAVE_DIR, f"{_DATASET_NAME}.{_ARCH1}@{_ARCH2}.json"))
print(results["all"]["pvalue"])
