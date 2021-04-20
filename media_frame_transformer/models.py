import torch
import torch.nn as nn
import torch.nn.functional as F
from config import VOCAB_SIZE
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


class LinearRegressionModel(nn.Module):
    def __init__(
        self,
        task="c",
        use_label_distribution_deviation=False,
    ):
        super().__init__()
        self.task = task
        assert task in ["c", "rs"]

        self.use_label_distribution_deviation = use_label_distribution_deviation
        print(use_label_distribution_deviation)

        self.ff = nn.Linear(
            VOCAB_SIZE,
            15,
            # bias=not use_label_distribution_deviation,
        )
        # nn.init.constant_(self.ff.weight, 0)

    def forward(self, batch):

        bow = batch["bow"].to(DEVICE).to(torch.float)
        logits = self.ff(bow)

        if self.task == "c":
            labels = batch["primary_frame_idx"].to(DEVICE)
            frame_loss = F.cross_entropy(logits, labels, reduction="none")
            loss = frame_loss
        # elif self.task == "rm":
        #     labels = batch["both_frame_vec"].to(DEVICE)
        #     frame_loss = F.binary_cross_entropy_with_logits(
        #         logits, labels, reduction="none"
        #     )
        #     frame_loss = frame_loss.mean(dim=-1)
        #     loss = frame_loss
        elif self.task == "rs":
            labels = batch["primary_frame_vec"].to(DEVICE)
            frame_loss = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )
            frame_loss = frame_loss.mean(dim=-1)
            loss = frame_loss
        else:
            raise NotImplementedError()

        if (
            hasattr(self, "use_label_distribution_deviation")
            and self.use_label_distribution_deviation
        ):
            if self.task in ["c", "rs"]:
                label_distribution = batch["primary_frame_distr"]
            elif self.task == "rm":
                label_distribution = batch["both_frame_distr"]
            else:
                raise NotImplementedError()
            logits = logits + torch.log(label_distribution.to(DEVICE).to(torch.float))

        loss = loss.sum()
        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
        }


@register_model("linreg.c")
def _():
    return LinearRegressionModel(task="c")


@register_model("linreg.rs")
def _():
    return LinearRegressionModel(task="rs")


@register_model("linreg.c_dev")
def _():
    return LinearRegressionModel(task="c", use_label_distribution_deviation=True)
