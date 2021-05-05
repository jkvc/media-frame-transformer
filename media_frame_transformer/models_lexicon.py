# models for lexicon induction
# deprecated, because torch optimizer does not handle shallow model well

from typing import List

import pandas as pd
import torch
import torch.nn as nn
from config import N_CLASSES, VOCAB_SIZE

from media_frame_transformer.models import register_model
from media_frame_transformer.models_common import (
    MULTICLASS_STRATEGY,
    calc_multiclass_loss,
)
from media_frame_transformer.utils import DEVICE


class LexiconModel(nn.Module):
    def __init__(
        self,
        multiclass_strategy,
        use_label_distribution_deviation=False,
    ):
        super().__init__()

        assert multiclass_strategy in MULTICLASS_STRATEGY
        self.multiclass_strategy = multiclass_strategy

        # use_bias = not use_label_distribution_deviation
        self.ff = nn.Linear(VOCAB_SIZE, N_CLASSES)
        self.use_label_distribution_deviation = use_label_distribution_deviation

    def forward(self, batch):
        x = batch["x"].to(DEVICE).to(torch.float)  # nsample, vocabsize

        nsample, vocabsize = x.shape
        assert vocabsize == VOCAB_SIZE

        logits = self.ff(x)  # nsample, nclass

        if self.use_label_distribution_deviation:
            label_distribution = batch["label_distribution"]  # nsample, nclass
            logits = logits + torch.log(label_distribution.to(DEVICE).to(torch.float))

        labels = batch["y"].to(DEVICE)
        loss, labels = calc_multiclass_loss(logits, labels, self.multiclass_strategy)
        loss = loss.mean()

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


class DirectResidulizationLogReg(nn.Module):
    def __init__(
        self,
        multiclass_strategy,
        use_label_distribution_deviation=False,
    ):
        super().__init__()
        assert multiclass_strategy in MULTICLASS_STRATEGY
        self.multiclass_strategy = multiclass_strategy

        hidden_size = 64
        self.hidden_size = hidden_size

        self.tin = nn.Linear(VOCAB_SIZE, hidden_size, bias=False)
        self.cin = nn.Linear(6, hidden_size, bias=False)
        self.tout = nn.Linear(hidden_size, N_CLASSES, bias=False)
        self.cout = nn.Linear(hidden_size, N_CLASSES, bias=False)

        self.use_label_distribution_deviation = use_label_distribution_deviation

    def forward(self, batch):
        x = batch["x"].to(DEVICE).to(torch.float)  # nsample, vocabsize
        nsample, vocabsize = x.shape
        assert vocabsize == VOCAB_SIZE

        if self.training:
            issues_onehot = torch.eye(6)[batch["issue_idx"]].to(DEVICE).to(torch.float)
            yhat = self.cin(issues_onehot)
            e = self.tin(x)
            logits = self.cout(yhat) + self.tout(e)
        else:
            e = self.tin(x)
            logits = self.tout(e)

        if self.use_label_distribution_deviation:
            label_distribution = batch["label_distribution"]  # nsample, nclass
            logits = logits + torch.log(label_distribution.to(DEVICE).to(torch.float))

        labels = batch["y"].to(DEVICE)
        loss, labels = calc_multiclass_loss(logits, labels, self.multiclass_strategy)
        loss = loss.mean()

        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
        }

    def get_weighted_lexicon(
        self, vocab: List[str], colnames: List[str]
    ) -> pd.DataFrame:
        tin_weights = self.tin.weight.data.detach().cpu().numpy()  # hiddendim, vocab
        tout_weights = self.tout.weight.data.detach().cpu().numpy()  # nclass, hiddendim
        weights = tout_weights @ tin_weights

        nclass, vocabsize = weights.shape
        assert len(colnames) == nclass

        df = pd.DataFrame()
        df["word"] = vocab
        for c in range(nclass):
            df[colnames[c]] = weights[c]
        return df


def create_models_closure(multiclass_strategy):
    @register_model(f"lexicon.dr.{multiclass_strategy}")
    def _():
        return DirectResidulizationLogReg(
            multiclass_strategy=multiclass_strategy,
            use_label_distribution_deviation=False,
        )

    @register_model(f"lexicon.dr.{multiclass_strategy}+dev")
    def _():
        return DirectResidulizationLogReg(
            multiclass_strategy=multiclass_strategy,
            use_label_distribution_deviation=True,
        )

    @register_model(f"lexicon.{multiclass_strategy}")
    def _():
        return LexiconModel(
            multiclass_strategy=multiclass_strategy,
            use_label_distribution_deviation=False,
        )

    @register_model(f"lexicon.{multiclass_strategy}+dev")
    def _():
        return LexiconModel(
            multiclass_strategy=multiclass_strategy,
            use_label_distribution_deviation=True,
        )


for multiclass_strategy in MULTICLASS_STRATEGY:
    create_models_closure(multiclass_strategy)
