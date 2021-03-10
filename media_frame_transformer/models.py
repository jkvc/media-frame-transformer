import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

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


class RobertaFrameClassifier(nn.Module):
    def __init__(
        self,
        dropout=0.1,
        n_class=15,
        task="classification",
        issue_supervision=False,
        subframe_supervision=False,
    ):
        super(RobertaFrameClassifier, self).__init__()
        self.task = task
        assert task in ["classification", "retrieval"]

        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(p=dropout)
        self.frame_ff = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(768, n_class),
        )
        self.issue_supervision = issue_supervision
        if issue_supervision:
            self.issue_ff = nn.Sequential(
                nn.Linear(768, 768),
                nn.Tanh(),
                nn.Dropout(p=dropout),
                nn.Linear(768, 6),
            )
        self.subframe_supervision = subframe_supervision
        if subframe_supervision:
            self.subframe_ff = nn.Sequential(
                nn.Linear(768, 768),
                nn.Tanh(),
                nn.Dropout(p=dropout),
                nn.Linear(768, n_class),
            )

    def forward(self, batch):
        x = batch["x"].to(DEVICE)
        x = self.roberta(x)
        x = x[0]
        x = self.dropout(x)
        cls_emb = x[:, 0, :]  # the <s> tokens, i.e. <CLS>
        cls_emb = self.dropout(cls_emb)

        frame_out = self.frame_ff(cls_emb)
        if self.task == "classification":
            labels = batch["primary_frame"].to(DEVICE)
            frame_loss = F.cross_entropy(frame_out, labels, reduction="none")
            loss = frame_loss
        elif self.task == "retrieval":
            labels = batch["retrieval"].to(DEVICE)
            frame_loss = F.binary_cross_entropy_with_logits(
                frame_out, labels, reduction="none"
            )
            frame_loss = frame_loss.mean(dim=-1)
            loss = frame_loss

        if self.issue_supervision:
            issue_out = self.issue_ff(cls_emb)
            issue_idx = batch["issue_idx"].to(DEVICE)
            issue_loss = F.cross_entropy(issue_out, issue_idx, reduction="none")
            loss = loss + issue_loss
        if self.subframe_supervision:
            subframe_out = self.subframe_ff(cls_emb)
            subframes = batch["subframes"].to(DEVICE).to(torch.float)
            subframe_loss = F.binary_cross_entropy_with_logits(
                subframe_out, subframes, reduction="none"
            )  # (b, 15)
            subframe_loss = subframe_loss.mean(dim=-1)  # (b,)
            loss = loss + subframe_loss

        loss_weight = batch["weight"].to(DEVICE)
        loss = (loss * loss_weight).mean()

        return {
            "logits": frame_out,
            "loss": loss,
            "labels": labels,
        }


@register_model("roberta_base")
def roberta_base():
    return RobertaFrameClassifier(dropout=0.1)


@register_model("roberta_meddrop")
def roberta_meddrop():
    return RobertaFrameClassifier(dropout=0.15)


@register_model("roberta_meddrop_half")
def roberta_meddrop_half():
    return _freeze_roberta_top_n_layers(roberta_meddrop(), 6)


@register_model("roberta_meddrop_retrieval")
def roberta_meddrop_retrieval():
    return RobertaFrameClassifier(dropout=0.15, task="retrieval")


@register_model("roberta_meddrop_half_retrieval")
def roberta_meddrop_half_retrieval():
    return _freeze_roberta_top_n_layers(
        RobertaFrameClassifier(dropout=0.15, task="retrieval"), 6
    )


@register_model("roberta_meddrop_issuesup")
def roberta_meddrop_issuesup():
    model = RobertaFrameClassifier(dropout=0.15, issue_supervision=True)
    return model


@register_model("roberta_meddrop_half_issuesup")
def roberta_meddrop_half_issuesup():
    model = RobertaFrameClassifier(dropout=0.15, issue_supervision=True)
    model = _freeze_roberta_top_n_layers(model, 6)
    return model


@register_model("roberta_meddrop_subframesup")
def roberta_meddrop_subframesup():
    model = RobertaFrameClassifier(dropout=0.15, subframe_supervision=True)
    return model


@register_model("roberta_meddrop_half_subframesup")
def roberta_meddrop_half_subframesup():
    model = RobertaFrameClassifier(dropout=0.15, subframe_supervision=True)
    model = _freeze_roberta_top_n_layers(model, 6)
    return model


@register_model("roberta_meddrop_issuesup_subframesup")
def roberta_meddrop_issuesup_subframesup():
    model = RobertaFrameClassifier(
        dropout=0.15, issue_supervision=True, subframe_supervision=True
    )
    return model


@register_model("roberta_meddrop_half_issuesup_subframesup")
def roberta_meddrop_half_issuesup_subframesup():
    model = RobertaFrameClassifier(
        dropout=0.15, issue_supervision=True, subframe_supervision=True
    )
    model = _freeze_roberta_top_n_layers(model, 6)
    return model


class RobertaWithLabelProps(nn.Module):
    def __init__(self, dropout=0.1, n_class=15, labelprop_hidden=50):
        super(RobertaWithLabelProps, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(p=dropout)

        self.dense1 = nn.Linear(768, 768)
        self.label_prop_intake = nn.Linear(n_class, labelprop_hidden)
        self.dense2 = nn.Linear(768 + labelprop_hidden, 768)
        self.out_proj = nn.Linear(768, n_class)
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, batch):
        x = batch["x"].to(DEVICE)
        x = self.roberta(x)
        x = x[0]
        x = x[:, 0, :]  # the <s> tokens, i.e. <CLS>
        x = self.dropout(x)  # (b, 768)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        label_props = batch["label_props"].to(DEVICE).to(torch.float)
        label_props = self.label_prop_intake(label_props)
        label_props = torch.tanh(label_props)  # (b, labelprop_hidden)

        x = torch.cat([x, label_props], dim=1)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        y = batch["y"].to(DEVICE)
        loss = self.loss(x, y)
        loss_weight = batch["weight"].to(DEVICE)
        loss = (loss * loss_weight).mean()

        return {
            "logits": x,
            "loss_to_backward": loss,
            "loss": loss,
            "is_correct": torch.argmax(x, dim=-1) == y,
        }


@register_model("roberta_meddrop_labelprops")
def roberta_meddrop_labelprops():
    return RobertaWithLabelProps(dropout=0.15)


@register_model("roberta_meddrop_half_labelprops")
def roberta_meddrop_half_labelprops():
    return _freeze_roberta_top_n_layers(RobertaWithLabelProps(dropout=0.15), 6)
