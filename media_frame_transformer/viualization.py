from os.path import exists, join

import matplotlib.pyplot as plt

from media_frame_transformer.utils import load_json


def plot_series_w_labels(name2xys, title, save_path=None):
    plt.clf()
    for name, xys in name2xys.items():
        xs, ys = list(zip(*xys))
        plt.plot(xs, ys, label=name)
        for x, y in zip(xs, ys):
            label = label = "{:.3f}".format(y)
            plt.annotate(
                label,  # this is the text
                (x, y),  # this is the point to label
                ha="center",
            )  # horizontal alignment can be left, right or center
    plt.legend()
    plt.title(title)
    if not save_path:
        return
    plt.savefig(save_path)
    plt.clf()


def visualize_num_sample_num_epoch(
    model_root,
    numsamples,
    numepochs,
    title,
    filename="learning.png",
    legend_title="",
    xlabel="",
    ylabel="",
):
    val_before_pretrain = None
    if exists(join(model_root, "mean_metrics_before_adapt.json")):
        val_before_pretrain = load_json(
            join(model_root, "mean_metrics_before_adapt.json")
        )["mean"]["holdout_issue_f1"]

    # expect root/numsample/issue/fold, and populated epoch metrics
    numepoch2metrics = {
        epoch: load_json(join(model_root, f"mean_epoch_{epoch}.json"))
        for epoch in numepochs
    }
    numsample2xys = {}
    for numsample in numsamples:
        xys = []
        numsample2xys[numsample] = xys
        if val_before_pretrain is not None:
            xys.append((0, val_before_pretrain))
        for numepoch in numepochs:
            xys.append(
                (
                    len(xys),
                    numepoch2metrics[numepoch][f"{numsample:04}_samples"]["mean"][
                        "valid_f1"
                    ],
                )
            )
    plot_series_w_labels(numsample2xys, title)
    plt.legend(title=legend_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(join(model_root, filename))
    plt.clf()
