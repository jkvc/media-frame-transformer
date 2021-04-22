MULTICLASS_STRATEGY = ["multinomial", "ovr"]

import torch
import torch.nn.functional as F
from config import N_CLASSES

from media_frame_transformer.utils import DEVICE


def calc_multiclass_loss(logits, labels, multiclass_strategy):
    if multiclass_strategy == "multinomial":
        loss = F.cross_entropy(logits, labels, reduction="none")
    elif multiclass_strategy == "ovr":
        # convert label to one-hot
        labels = torch.eye(N_CLASSES).to(DEVICE)[labels].to(torch.float)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = loss.mean(dim=-1)
    else:
        raise NotImplementedError()
    return loss
