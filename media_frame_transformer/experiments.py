from os import makedirs
from os.path import exists, join

from media_frame_transformer import models
from media_frame_transformer.learning import train
from media_frame_transformer.utils import mkdir_overwrite, write_str_list_as_txt


def run_experiments(arch, path2datasets, **kwargs):
    from pprint import pprint

    pprint(list(path2datasets.keys()))

    for path, datasets in path2datasets.items():
        makedirs(path, exist_ok=True)
        if exists(join(path, "_complete")):
            print(">> skip", path)
            continue

        mkdir_overwrite(path)
        print(">>", path)

        model = models.get_model(arch)

        train(
            model=model,
            train_dataset=datasets["train"],
            valid_dataset=datasets["valid"],
            logdir=path,
            additional_valid_datasets=(
                datasets["additional_valid_datasets"]
                if "additional_valid_datasets" in datasets
                else None
            ),
            **kwargs,
        )

        # mark done
        write_str_list_as_txt(["."], join(path, "_complete"))