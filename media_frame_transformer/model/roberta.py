from typing import Any, Dict

import torch
import torch.nn as nn
from media_frame_transformer.model.common import (
    MULTICLASS_STRATEGY,
    calc_multiclass_loss,
)
from media_frame_transformer.model.model_utils import ReversalLayer
from media_frame_transformer.model.zoo import register_model
from media_frame_transformer.utils import DEVICE
from transformers import RobertaModel

ROBERAT_EMB_SIZE = 768


@register_model
class RobertaClassifier(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config

        multiclass_strategy = config["multiclass_strategy"]
        assert multiclass_strategy in MULTICLASS_STRATEGY
        self.multiclass_strategy = multiclass_strategy

        self.dropout_p = config["dropout_p"]

        self.roberta = RobertaModel.from_pretrained(
            "roberta-base", hidden_dropout_prob=self.dropout_p
        )

        self.n_classes = config["n_classes"]
        self.yff = nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            nn.Linear(ROBERAT_EMB_SIZE, ROBERAT_EMB_SIZE),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(ROBERAT_EMB_SIZE, self.n_classes),
        )

        self.use_learned_residual = config["use_learned_residual"]
        if self.use_learned_residual:
            self.n_sources = config["n_sources"]
            self.cff = nn.Sequential(
                nn.Linear(self.n_sources, 64),
                nn.Tanh(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(64, self.n_classes),
            )

        self.use_log_labelprop_bias = config["use_log_labelprop_bias"]

    def forward(self, batch):
        x = batch["x"].to(DEVICE)
        labels = batch["y"].to(DEVICE)

        x = self.roberta(x)[0]

        # huggingface robertaclassifier applies dropout before this, we apply dropout after this
        # shouldnt make a big difference
        e = x[:, 0, :]  # the <s> tokens, i.e. <CLS>

        logits = self.yff(e)

        if hasattr(self, "use_log_labelprop_bias") and self.use_log_labelprop_bias:
            labelprops = (
                batch["labelprops"].to(DEVICE).to(torch.float)
            )  # nsample, nclass
            logits = logits + torch.log(labelprops)

        if hasattr(self, "use_learned_residual") and self.use_learned_residual:
            if self.training:
                batchsize = len(labels)
                source_idx = batch["source_idx"].to(DEVICE)
                source_onehot = torch.FloatTensor(batchsize, self.n_sources).to(DEVICE)
                source_onehot.zero_()
                source_onehot.scatter_(1, source_idx.unsqueeze(-1), 1)
                c = self.cff(source_onehot)
                logits = logits + c

        loss, labels = calc_multiclass_loss(logits, labels, self.multiclass_strategy)

        loss = loss.mean()
        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
        }
