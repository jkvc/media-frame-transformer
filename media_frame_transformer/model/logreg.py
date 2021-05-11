from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from media_frame_transformer.model.common import (
    MULTICLASS_STRATEGY,
    calc_multiclass_loss,
)
from media_frame_transformer.model.model_utils import ReversalLayer
from media_frame_transformer.model.zoo import register_model
from media_frame_transformer.utils import DEVICE


def elicit_lexicon(
    weights: np.ndarray, vocab: List[str], colnames: List[str]
) -> pd.DataFrame:
    nclass, vocabsize = weights.shape
    assert len(colnames) == nclass

    df = pd.DataFrame()
    df["word"] = vocab
    for c in range(nclass):
        df[colnames[c]] = weights[c]
    return df


@register_model
class LogisticRegressionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config

        self.multiclass_strategy = config["multiclass_strategy"]
        assert self.multiclass_strategy in MULTICLASS_STRATEGY

        self.vocab_size = config["vocab_size"]
        self.n_classes = config["n_classes"]
        self.n_sources = config["n_sources"]
        self.use_log_labelprop_bias = config["use_log_labelprop_bias"]
        self.use_learned_residualization = config["use_learned_residualization"]
        self.use_gradient_reversal = config["use_gradient_reversal"]
        self.hidden_size = config["hidden_size"]
        self.reg = config["reg"]

        self.tff = nn.Linear(self.vocab_size, self.hidden_size, bias=False)
        self.yout = nn.Linear(self.hidden_size, self.n_classes, bias=False)
        self.cff = nn.Sequential(
            nn.Linear(self.n_classes, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.n_classes),
        )
        self.cout = nn.Sequential(
            ReversalLayer(),
            nn.Linear(self.hidden_size, self.n_sources),
        )

    def forward(self, batch):
        x = batch["x"].to(DEVICE).to(torch.float)  # nsample, vocabsize
        nsample, vocabsize = x.shape
        assert vocabsize == self.vocab_size

        e_t = self.tff(x)
        logits = self.yout(e_t)

        if self.use_log_labelprop_bias:
            labelprops = (
                batch["labelprops"].to(DEVICE).to(torch.float)
            )  # nsample, nclass
            logits = logits + torch.log(labelprops)

        if self.use_learned_residualization:
            if self.training:
                source_onehot = (
                    torch.eye(self.n_sources)[batch["source_idx"]]
                    .to(DEVICE)
                    .to(torch.float)
                )
                clogits = self.cff(source_onehot)
                logits = clogits + logits

        labels = batch["y"].to(DEVICE)
        loss, labels = calc_multiclass_loss(logits, labels, self.multiclass_strategy)

        if self.use_gradient_reversal:
            if self.training:
                confound_logits = self.cout(e)
                confound_loss, _ = calc_multiclass_loss(
                    confound_logits, batch["source_idx"].to(DEVICE), "multinomial"
                )
                loss = loss + confound_loss

        # l1 reg on t weights only
        loss = loss + torch.abs(self.yout.weight @ self.tff.weight).sum() * self.reg
        loss = loss.mean()

        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
        }

    def get_weighted_lexicon(
        self, vocab: List[str], colnames: List[str]
    ) -> pd.DataFrame:
        weights = (
            self.yout.weight.data.detach().cpu().numpy()
            @ self.tff.weight.data.detach().cpu().numpy()
        )
        return elicit_lexicon(weights, vocab, colnames)


# @register_model
# class LogRegModel(nn.Module):
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__()

#         self.config = config

#         multiclass_strategy = config["multiclass_strategy"]
#         assert multiclass_strategy in MULTICLASS_STRATEGY
#         self.multiclass_strategy = multiclass_strategy

#         use_log_labelprop_bias = config["use_log_labelprop_bias"]
#         self.use_log_labelprop_bias = use_log_labelprop_bias

#         vocab_size = config["vocab_size"]
#         n_classes = config["n_classes"]
#         ff_use_bias = not use_log_labelprop_bias  # FIXME
#         self.ff = nn.Linear(vocab_size, n_classes, bias=ff_use_bias)

#         self.reg = config["reg"]

#     def forward(self, batch):
#         x = batch["x"].to(DEVICE).to(torch.float)  # nsample, vocabsize
#         nsample, vocabsize = x.shape
#         assert vocabsize == self.config["vocab_size"]

#         logits = self.ff(x)  # nsample, nclass

#         if self.use_log_labelprop_bias:
#             labelprops = (
#                 batch["labelprops"].to(DEVICE).to(torch.float)
#             )  # nsample, nclass
#             logits = logits + torch.log(labelprops)

#         labels = batch["y"].to(DEVICE)
#         loss, labels = calc_multiclass_loss(logits, labels, self.multiclass_strategy)
#         loss = loss.mean()

#         # l1 regularization
#         loss = loss + torch.abs(self.ff.weight).sum() * self.reg

#         return {
#             "logits": logits,
#             "loss": loss,
#             "labels": labels,
#         }

#     def get_weighted_lexicon(
#         self, vocab: List[str], colnames: List[str]
#     ) -> pd.DataFrame:
#         weights = self.ff.weight.data.detach().cpu().numpy()
#         return elicit_lexicon(weights, vocab, colnames)


# @register_model
# class LogRegLearnedResidualization(nn.Module):
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__()

#         self.config = config

#         multiclass_strategy = config["multiclass_strategy"]
#         assert multiclass_strategy in MULTICLASS_STRATEGY
#         self.multiclass_strategy = multiclass_strategy

#         vocab_size = config["vocab_size"]
#         n_classes = config["n_classes"]
#         self.tff = nn.Linear(vocab_size, n_classes, bias=False)

#         self.n_sources = config["n_sources"]
#         self.cff = nn.Linear(self.n_sources, n_classes)

#         self.reg = config["reg"]

#     def forward(self, batch):
#         x = batch["x"].to(DEVICE).to(torch.float)  # nsample, vocabsize

#         nsample, vocabsize = x.shape
#         assert vocabsize == self.config["vocab_size"]

#         if self.training:
#             source_onehot = (
#                 torch.eye(self.n_sources)[batch["source_idx"]]
#                 .to(DEVICE)
#                 .to(torch.float)
#             )
#             clogits = self.cff(source_onehot)
#             tlogits = self.tff(x)
#             logits = clogits + tlogits
#         else:
#             logits = self.tff(x)

#         labels = batch["y"].to(DEVICE)
#         loss, labels = calc_multiclass_loss(logits, labels, self.multiclass_strategy)
#         loss = loss.mean()

#         # l1 reg on t weights only
#         loss = loss + torch.abs(self.tff.weight).sum() * self.reg

#         return {
#             "logits": logits,
#             "loss": loss,
#             "labels": labels,
#         }

#     def get_weighted_lexicon(
#         self, vocab: List[str], colnames: List[str]
#     ) -> pd.DataFrame:
#         weights = self.tff.weight.data.detach().cpu().numpy()
#         return elicit_lexicon(weights, vocab, colnames)


# @register_model
# class LogRegGradientReversal(nn.Module):
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__()

#         self.config = config

#         multiclass_strategy = config["multiclass_strategy"]
#         assert multiclass_strategy in MULTICLASS_STRATEGY
#         self.multiclass_strategy = multiclass_strategy

#         use_log_labelprop_bias = config["use_log_labelprop_bias"]
#         self.use_log_labelprop_bias = use_log_labelprop_bias

#         vocab_size = config["vocab_size"]
#         n_classes = config["n_classes"]
#         hidden_size = config["hidden_size"]
#         n_sources = config["n_sources"]

#         self.tff = nn.Linear(vocab_size, hidden_size, bias=False)
#         self.yout = nn.Linear(hidden_size, n_classes, bias=False)
#         self.cout = nn.Sequential(
#             ReversalLayer(),
#             nn.Linear(hidden_size, n_sources),
#         )

#         self.reg = config["reg"]

#     def forward(self, batch):
#         x = batch["x"].to(DEVICE).to(torch.float)  # nsample, vocabsize

#         nsample, vocabsize = x.shape
#         assert vocabsize == self.config["vocab_size"]

#         labels = batch["y"].to(DEVICE)

#         e = self.tff(x)
#         logits = self.yout(e)

#         if self.use_log_labelprop_bias:
#             labelprops = (
#                 batch["labelprops"].to(DEVICE).to(torch.float)
#             )  # nsample, nclass
#             logits = logits + torch.log(labelprops)
#         loss, labels = calc_multiclass_loss(logits, labels, self.multiclass_strategy)

#         if self.training:
#             confound_logits = self.cout(e)
#             confound_loss, _ = calc_multiclass_loss(
#                 confound_logits, batch["source_idx"].to(DEVICE), "multinomial"
#             )
#             loss = loss + confound_loss

#         # l1 reg on t weights only
#         loss = loss + torch.abs(self.yout.weight @ self.tff.weight).sum() * self.reg
#         loss = loss.mean()

#         return {
#             "logits": logits,
#             "loss": loss,
#             "labels": labels,
#         }

# def get_weighted_lexicon(
#     self, vocab: List[str], colnames: List[str]
# ) -> pd.DataFrame:
#     weights = (
#         self.yout.weight.data.detach().cpu().numpy()
#         @ self.tff.weight.data.detach().cpu().numpy()
#     )
#     return elicit_lexicon(weights, vocab, colnames)
