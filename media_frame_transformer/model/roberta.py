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

        self.use_log_labelprop_bias = config["use_log_labelprop_bias"]
        self.n_classes = config["n_classes"]
        yff_use_bias = not self.use_log_labelprop_bias
        self.yff = nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            nn.Linear(ROBERAT_EMB_SIZE, ROBERAT_EMB_SIZE),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(ROBERAT_EMB_SIZE, self.n_classes, bias=yff_use_bias),
        )

        self.use_gradient_reversal = config["use_gradient_reversal"]
        n_sources = config["n_sources"]
        if self.use_gradient_reversal:
            self.gradient_reversal_strength = config["gradient_reversal_strength"]
            self.cff = nn.Sequential(
                ReversalLayer(),
                nn.Linear(ROBERAT_EMB_SIZE, n_sources),
            )

    def forward(self, batch):
        x = batch["x"].to(DEVICE)
        labels = batch["y"].to(DEVICE)

        x = self.roberta(x)[0]

        # huggingface robertaclassifier applies dropout before this, we apply dropout after this
        # shouldnt make a big difference
        e = x[:, 0, :]  # the <s> tokens, i.e. <CLS>

        logits = self.yff(e)

        if self.use_log_labelprop_bias:
            labelprops = (
                batch["labelprops"].to(DEVICE).to(torch.float)
            )  # nsample, nclass
            logits = logits + torch.log(labelprops)

        loss, labels = calc_multiclass_loss(logits, labels, self.multiclass_strategy)

        if self.use_gradient_reversal and self.training:
            confound_logits = self.cff(e)
            confound_loss, _ = calc_multiclass_loss(
                confound_logits, batch["source_idx"].to(DEVICE), "multinomial"
            )
            loss = loss + confound_loss * self.gradient_reversal_strength

        loss = loss.mean()
        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
        }
