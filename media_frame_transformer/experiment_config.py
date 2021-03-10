import torch

G_CUDA_MEM = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024

KFOLD = 8
FOLDS_TO_RUN = [0, 1, 2]

# ARCH = "roberta_meddrop"
# BATCHSIZE = 25

ARCH = "roberta_meddrop_half"
BATCHSIZE = 50
