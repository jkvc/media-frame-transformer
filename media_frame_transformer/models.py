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


def _freeze_roberta_top_n_layers(model, n):
    # pretrained roberta = embeddings -> encoder.laysers -> classfier
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    for i, module in enumerate(model.roberta.encoder.layer):
        if i < n:
            for param in module.parameters():
                param.requires_grad = False
    return model


@register_model("roberta_base")
def roberta_base():
    return RobertaSimpleClassifier(dropout=0.1)


@register_model("roberta_meddrop")
def roberta_meddrop():
    return RobertaSimpleClassifier(dropout=0.15)


@register_model("roberta_meddrop_half")
def roberta_meddrop_half():
    return _freeze_roberta_top_n_layers(roberta_meddrop(), 6)


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


@register_model("roberta_meddrop_issuesup")
def roberta_meddrop_issuesup():
    model = RobertaWithIssueSupervision(dropout=0.15)
    return model


@register_model("roberta_meddrop_half_issuesup")
def roberta_meddrop_half_issuesup():
    model = RobertaWithIssueSupervision(dropout=0.15)
    model = _freeze_roberta_top_n_layers(model, 6)
    return model


class RobertaWithIssueSupervision(nn.Module):
    def __init__(self, dropout=0.1, n_class=15):
        super(RobertaWithIssueSupervision, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(768, 768)
        self.out_proj = nn.Linear(768, n_class)
        self.issue_proj = nn.Linear(768, 6)
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, batch):
        x = batch["x"].to(DEVICE)
        x = self.roberta(x)
        x = x[0]
        x = self.dropout(x)
        cls_emb = x[:, 0, :]  # the <s> tokens, i.e. <CLS>
        cls_emb = self.dropout(cls_emb)

        frame_out = self.dense(cls_emb)
        frame_out = torch.tanh(frame_out)
        frame_out = self.dropout(frame_out)
        frame_out = self.out_proj(frame_out)

        issue_out = self.issue_proj(cls_emb)

        y = batch["y"].to(DEVICE)

        frame_loss = self.loss(frame_out, y)
        loss_weight = batch["weight"].to(DEVICE)
        frame_loss = (frame_loss * loss_weight).mean()

        issue_idx = batch["issue_idx"].to(DEVICE)
        issue_loss = self.loss(issue_out, issue_idx)
        issue_loss = (issue_loss * loss_weight).mean()

        loss = frame_loss + issue_loss

        return {
            "logits": x,
            "loss": loss,
            "is_correct": torch.argmax(frame_out, dim=-1) == y,
        }


@register_model("roberta_meddrop_labelprops")
def roberta_meddrop_labelprops():
    return RobertaWithLabelProps(dropout=0.15)


@register_model("roberta_meddrop_half_labelprops")
def roberta_meddrop_half_labelprops():
    return _freeze_roberta_top_n_layers(RobertaWithLabelProps(dropout=0.15), 6)


class RobertaWithLabelProps(nn.Module):
    def __init__(self, dropout=0.1, n_class=15):
        super(RobertaWithLabelProps, self).__init__()
        self.label_prop_intake = nn.Linear(n_class, n_class)
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(768 + n_class, 768)
        self.out_proj = nn.Linear(768, n_class)
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, batch):
        x = batch["x"].to(DEVICE)
        x = self.roberta(x)
        x = x[0]
        x = self.dropout(x)
        x = x[:, 0, :]  # the <s> tokens, i.e. <CLS>
        x = self.dropout(x)  # (b, 768)

        label_props = batch["label_props"].to(DEVICE).to(torch.float)
        label_props = self.label_prop_intake(label_props)
        label_props = torch.relu(label_props)  # (b, nclass)

        x = torch.cat([x, label_props], dim=1)
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
