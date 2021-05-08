from os import makedirs
from os.path import exists, join
from pprint import pprint

import torch

from media_frame_transformer.learning import train
from media_frame_transformer.model import get_model
from media_frame_transformer.utils import mkdir_overwrite, write_str_list_as_txt


def run_experiments(
    config,
    logdir2datasets,
    logdir2checkpointpath=None,
    model_transform=None,
    **kwargs,
):
    logdir2datasets = {
        k: logdir2datasets[k] for k in sorted(list(logdir2datasets.keys()))
    }
    pprint(list(logdir2datasets.keys()))

    for logdir, datasets in logdir2datasets.items():
        makedirs(logdir, exist_ok=True)
        if exists(join(logdir, "_complete")):
            print(">> skip", logdir)
            continue

        mkdir_overwrite(logdir)
        print(">>", logdir)

        if logdir2checkpointpath is None:
            print(">> fresh model")
            model = get_model(config)
        else:
            checkpoint_path = logdir2checkpointpath[logdir]
            print(">> load checkpoint from", checkpoint_path)
            model = torch.load(checkpoint_path)

        if model_transform is not None:
            model = model_transform(model)

        train(
            model=model,
            train_dataset=datasets["train"],
            valid_dataset=datasets["valid"],
            logdir=logdir,
            additional_valid_datasets=(
                datasets["additional_valid_datasets"]
                if "additional_valid_datasets" in datasets
                else None
            ),
            **kwargs,
        )

        # mark done
        write_str_list_as_txt(["."], join(logdir, "_complete"))
