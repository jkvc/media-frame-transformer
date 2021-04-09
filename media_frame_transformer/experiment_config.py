import torch

KFOLD = 8
# FOLDS_TO_RUN = [0, 1, 2, 3, 4, 5, 6, 7]
FOLDS_TO_RUN = [0, 1, 2, 3]

ARCH = "roberta_meddrop"
BATCHSIZE = 25

DATASET_SIZES = [125, 250, 500, 1000]


# ARCH = "roberta_meddrop_half"
# BATCHSIZE = 50
