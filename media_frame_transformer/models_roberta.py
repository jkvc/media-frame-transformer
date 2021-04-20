import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ROBERTA_CLASSFIER_N_CLASSES, VOCAB_SIZE
from transformers import RobertaModel

from media_frame_transformer.models import register_model
from media_frame_transformer.utils import DEVICE

MULTICLASS_STRATEGY = ["multinomial", "ovr"]


class RobertaFrameClassifier(nn.Module):
    def __init__(
        self,
        multiclass_strategy,
        dropout=0.1,
        n_class=15,
        use_label_distribution_deviation=False,
    ):
        super(RobertaFrameClassifier, self).__init__()
        self.multiclass_strategy = multiclass_strategy
        assert multiclass_strategy in MULTICLASS_STRATEGY

        self.roberta = RobertaModel.from_pretrained(
            "roberta-base", hidden_dropout_prob=dropout
        )
        self.dropout = nn.Dropout(p=dropout)
        self.roberta_emb_size = 768

        self.frame_ff = nn.Sequential(
            nn.Linear(self.roberta_emb_size, 768),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(768, n_class),
        )

        self.use_label_distribution_deviation = use_label_distribution_deviation

    def forward(self, batch):
        x = batch["x"].to(DEVICE)
        x = self.roberta(x)
        x = x[0]
        x = self.dropout(x)
        cls_emb = x[:, 0, :]  # the <s> tokens, i.e. <CLS>
        cls_emb = self.dropout(cls_emb)

        logits = self.frame_ff(cls_emb)  # (b, nclass)

        if self.use_label_distribution_deviation:
            label_distribution = batch["label_distribution"]
            logits = logits + torch.log(label_distribution.to(DEVICE).to(torch.float))

        labels = batch["y"].to(DEVICE)

        # calculate loss
        if self.multiclass_strategy == "multinomial":
            loss = F.cross_entropy(logits, labels, reduction="none")
        elif self.multiclass_strategy == "ovr":
            # convert label to one-hot
            labels = (
                torch.eye(ROBERTA_CLASSFIER_N_CLASSES)
                .to(DEVICE)[labels]
                .to(torch.float)
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = loss.mean(dim=-1)

        loss = (loss).mean()

        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
        }


def freeze_roberta_top_n_layers(model, n):
    # pretrained roberta = embeddings -> encoder.laysers -> classfier
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    for i, module in enumerate(model.roberta.encoder.layer):
        if i < n:
            for param in module.parameters():
                param.requires_grad = False
    return model


def freeze_roberta_all_transformer(model):
    model = freeze_roberta_top_n_layers(model, 12)
    return model


def freeze_roberta_module(model):
    for param in model.roberta.parameters():
        param.requires_grad = False
    return model


def create_models(name, dropout, multiclass_strategy):
    @register_model(f"roberta_{name}.{multiclass_strategy}")
    def _():
        return RobertaFrameClassifier(
            multiclass_strategy=multiclass_strategy,
            dropout=dropout,
        )

    @register_model(f"roberta_{name}.{multiclass_strategy}+dev")
    def _():
        return RobertaFrameClassifier(
            multiclass_strategy=multiclass_strategy,
            dropout=dropout,
            use_label_distribution_deviation=True,
        )


for name, dropout in [("md", 0.15), ("hd", 0.2)]:
    for multiclass_strategy in MULTICLASS_STRATEGY:
        create_models(name, dropout, multiclass_strategy)
