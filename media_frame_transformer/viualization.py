import matplotlib.pyplot as plt


def plot_series_w_labels(name2xys, title, save_path):
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
    plt.savefig(save_path)
    plt.clf()
