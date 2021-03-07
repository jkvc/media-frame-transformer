import torch
import torch.nn as nn
from transformers import (
    DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
    RobertaModel,
    RobertaTokenizer,
)

from media_frame_transformer.utils import DEVICE

_MODELS = {}


def get_model(arch: str):
    return _MODELS[arch]()


def register_model(arch: str):
    def _register(f):
        assert arch not in _MODELS
        _MODELS[arch] = f
        return f

    return _register


@register_model("roberta_base")
def roberta_base():
    return RobertaSimpleClassifier(dropout=0.1)


@register_model("roberta_meddrop")
def roberta_meddrop():
    return RobertaSimpleClassifier(dropout=0.15)


@register_model("roberta_meddrop_half")
def roberta_meddrop_half():
    return _freeze_roberta_top_n_layers(roberta_meddrop(), 6)


def _freeze_roberta_top_n_layers(model, n):
    # pretrained roberta = embeddings -> encoder.laysers -> classfier
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    for i, module in enumerate(model.roberta.encoder.layer):
        if i < n:
            for param in module.parameters():
                param.requires_grad = False
    return model


class RobertaSimpleClassifier(nn.Module):
    def __init__(self, dropout=0.1, n_class=15):
        super(RobertaSimpleClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(768, 768)
        self.out_proj = nn.Linear(768, n_class)
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, batch):
        x = batch["x"].to(DEVICE)
        x = self.roberta(x)
        x = x[0]
        x = self.dropout(x)
        x = x[:, 0, :]  # the <s> tokens, i.e. <CLS>
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        y = batch["y"].to(DEVICE)
        loss = self.loss(x, y)
        loss_weight = batch["weight"].to(DEVICE)
        loss = (loss * loss_weight).mean()

        return {
            "logits": x,
            "loss": loss,
            "is_correct": torch.argmax(x, dim=-1) == y,
        }
