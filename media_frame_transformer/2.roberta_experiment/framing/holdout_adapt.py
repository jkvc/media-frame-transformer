# import sys
# from os.path import basename, dirname, join, realpath
# from random import Random

# from config import BATCHSIZE, MODELS_DIR, RANDOM_SEED, ROBERTA_ADAPT_N_SAMPLES
# from media_frame_transformer.dataset.framing.definition import (
#     ISSUES,
#     LABELPROPS_DIR,
#     N_ISSUES,
#     PRIMARY_FRAME_N_CLASSES,
#     PRIMARY_FRAME_NAMES,
# )
# from media_frame_transformer.dataset.framing.samples import (
#     load_all_framing_samples,
#     load_kfold_framing_samples,
# )
# from media_frame_transformer.dataset.roberta_dataset import RobertaDataset
# from media_frame_transformer.eval import reduce_and_save_metrics
# from media_frame_transformer.experiments import run_experiments
# from media_frame_transformer.model.roberta_config.base import load_roberta_model_config

# _N_TRAIN_EPOCH = 10

# _ARCH = sys.argv[1]
# _CONFIG = load_roberta_model_config(_ARCH, PRIMARY_FRAME_N_CLASSES, N_ISSUES)

# _SCRIPT_PATH = realpath(__file__)
# _EXPERIMENT_NAME = basename(_SCRIPT_PATH).replace(".py", "")
# _DATASET_NAME = basename(dirname(_SCRIPT_PATH))
# _SAVE_DIR = join(MODELS_DIR, _DATASET_NAME, _EXPERIMENT_NAME, _ARCH)
# _LOAD_CHECKPOINT_DIR = join(MODELS_DIR, _DATASET_NAME, "holdout_source", _ARCH)

# _RNG = Random()

# logdir2datasets, logdir2checkpointpath = {}, {}

# for adapt_source in ISSUES:
#     # use 0th only (single valid set)
#     split2samples = load_kfold_framing_samples([adapt_source], "primary_frame")[0]
#     train_samples = split2samples["train"]
#     valid_samples = split2samples["valid"]

#     _RNG.seed(RANDOM_SEED)
#     _RNG.shuffle(train_samples)

#     for nsample in ROBERTA_ADAPT_N_SAMPLES:
#         selected_train_samples = train_samples[:nsample]
#         train_dataset = RobertaDataset(
#             selected_train_samples,
#             n_classes=PRIMARY_FRAME_N_CLASSES,
#             source_names=ISSUES,
#             labelprop_split="train",
#             labelprop_dir=LABELPROPS_DIR,
#         )
#         valid_dataset = RobertaDataset(
#             valid_samples,
#             n_classes=PRIMARY_FRAME_N_CLASSES,
#             source_names=ISSUES,
#             labelprop_split="train",
#             labelprop_dir=LABELPROPS_DIR,
#         )

#         logdir = join(_SAVE_DIR, f"{nsample:04}_samples", adapt_source)
#         logdir2datasets[logdir] = {
#             "train": train_dataset,
#             "valid": valid_dataset,
#         }
#         logdir2checkpointpath[logdir] = join(
#             _LOAD_CHECKPOINT_DIR, adapt_source, "checkpoint.pth"
#         )

# run_experiments(
#     _CONFIG,
#     logdir2datasets=logdir2datasets,
#     logdir2checkpointpath=logdir2checkpointpath,
#     save_model_checkpoint=True,
#     keep_latest=True,
#     batchsize=BATCHSIZE,
#     max_epochs=_N_TRAIN_EPOCH,
#     num_early_stop_non_improve_epoch=_N_TRAIN_EPOCH,
#     skip_train_zeroth_epoch=True,
# )

# reduce_and_save_metrics(_SAVE_DIR)
# for e in range(_N_TRAIN_EPOCH):
#     reduce_and_save_metrics(_SAVE_DIR, f"leaf_epoch_{e}.json", f"mean_epoch_{e}.json")
