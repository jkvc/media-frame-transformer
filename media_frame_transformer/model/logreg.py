from typing import Any, Dict, List

import pandas as pd
import torch
import torch.nn as nn
from media_frame_transformer.model.common import (
    MULTICLASS_STRATEGY,
    calc_multiclass_loss,
)
from media_frame_transformer.model.zoo import register_model
from media_frame_transformer.utils import DEVICE


@register_model
class LogRegModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config

        multiclass_strategy = config["multiclass_strategy"]
        assert multiclass_strategy in MULTICLASS_STRATEGY
        self.multiclass_strategy = multiclass_strategy

        use_log_labelprop_bias = config["use_log_labelprop_bias"]
        self.use_log_labelprop_bias = use_log_labelprop_bias

        vocab_size = config["vocab_size"]
        n_classes = config["n_classes"]
        ff_use_bias = not use_log_labelprop_bias  # FIXME
        self.ff = nn.Linear(vocab_size, n_classes, bias=ff_use_bias)

        self.reg = config["reg"]

    def forward(self, batch):
        x = batch["x"].to(DEVICE).to(torch.float)  # nsample, vocabsize

        nsample, vocabsize = x.shape
        assert vocabsize == self.config["vocab_size"]

        logits = self.ff(x)  # nsample, nclass

        if self.use_log_labelprop_bias:
            labelprops = (
                batch["labelprops"].to(DEVICE).to(torch.float)
            )  # nsample, nclass
            logits = logits + torch.log(labelprops)

        labels = batch["y"].to(DEVICE)
        loss, labels = calc_multiclass_loss(logits, labels, self.multiclass_strategy)
        loss = loss.mean()

        # l1 regularization
        loss = loss + torch.abs(self.ff.weight).sum() * self.reg

        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
        }

    def get_weighted_lexicon(
        self, vocab: List[str], colnames: List[str]
    ) -> pd.DataFrame:
        weights = self.ff.weight.data.detach().cpu().numpy()
        nclass, vocabsize = weights.shape
        assert len(colnames) == nclass

        df = pd.DataFrame()
        df["word"] = vocab
        for c in range(nclass):
            df[colnames[c]] = weights[c]
        return df


@register_model
class LogRegLearnedResidualization(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config

        multiclass_strategy = config["multiclass_strategy"]
        assert multiclass_strategy in MULTICLASS_STRATEGY
        self.multiclass_strategy = multiclass_strategy

        vocab_size = config["vocab_size"]
        n_classes = config["n_classes"]
        self.tff = nn.Linear(vocab_size, n_classes, bias=False)

        self.confound_input_size = config["confound_input_size"]
        self.cff = nn.Linear(self.confound_input_size, n_classes)

        self.reg = config["reg"]

    def forward(self, batch):
        x = batch["x"].to(DEVICE).to(torch.float)  # nsample, vocabsize

        nsample, vocabsize = x.shape
        assert vocabsize == self.config["vocab_size"]

        if self.training:
            issues_onehot = (
                torch.eye(self.confound_input_size)[batch["source_idx"]]
                .to(DEVICE)
                .to(torch.float)
            )
            yhat = self.cin(issues_onehot)
            e = self.tin(x)
            logits = self.cout(yhat) + self.tout(e)
        else:
            e = self.tin(x)
            logits = self.tout(e)

        labels = batch["y"].to(DEVICE)
        loss, labels = calc_multiclass_loss(logits, labels, self.multiclass_strategy)
        loss = loss.mean()

        # l1 reg on t weights only
        loss = loss + torch.abs(self.tff).sum() * self.reg

        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
        }

    def get_weighted_lexicon(
        self, vocab: List[str], colnames: List[str]
    ) -> pd.DataFrame:
        weights = self.tff.weight.data.detach().cpu().numpy()
        nclass, vocabsize = weights.shape
        assert len(colnames) == nclass

        df = pd.DataFrame()
        df["word"] = vocab
        for c in range(nclass):
            df[colnames[c]] = weights[c]
        return df
