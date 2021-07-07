from os import makedirs
from os.path import exists, join

import torch
from torch.optim import SGD
from tqdm import trange

from media_frame_transformer.dataset.bow_dataset import (
    build_bow_full_batch,
    build_vocab,
    get_all_tokens,
)
from media_frame_transformer.learning import calc_f1, print_metrics
from media_frame_transformer.model import get_model
from media_frame_transformer.utils import (
    DEVICE,
    read_txt_as_str_list,
    save_json,
    write_str_list_as_txt,
)


def train_lexicon_model(
    model,
    datadef,
    train_samples,
    vocab_size,
    use_source_individual_norm,
    labelprop_split,
):
    vocab, all_tokens = build_vocab(train_samples, vocab_size)
    batch = build_bow_full_batch(
        train_samples,
        datadef,
        all_tokens,
        vocab,
        use_source_individual_norm,
        labelprop_split,
    )

    optimizer = SGD(model.parameters(), lr=1e-1, weight_decay=0)

    model.train()
    for e in trange(5000):
        optimizer.zero_grad()
        outputs = model(batch)

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

    train_outputs = model(batch)
    f1, precision, recall = calc_f1(
        train_outputs["logits"].detach().cpu().numpy(),
        train_outputs["labels"].detach().cpu().numpy(),
    )
    metrics = {"train_f1": f1, "train_precision": precision, "train_recall": recall}

    return vocab, model, metrics


def eval_lexicon_model(
    model,
    datadef,
    valid_samples,
    vocab,
    use_source_individual_norm,
    labelprop_split,
):
    batch = build_bow_full_batch(
        valid_samples,
        datadef,
        get_all_tokens(valid_samples),
        vocab,
        use_source_individual_norm,
        labelprop_split,
    )

    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    f1, precision, recall = calc_f1(
        outputs["logits"].detach().cpu().numpy(),
        outputs["labels"].detach().cpu().numpy(),
    )

    metrics = {"valid_f1": f1, "valid_precision": precision, "valid_recall": recall}
    return metrics


def run_lexicon_experiment(
    config,
    datadef,
    train_samples,
    valid_samples,
    vocab_size,
    logdir,
    train_labelprop_split,
    valid_labelprop_split,
):
    complete_marker_path = join(logdir, "_complete")
    if exists(complete_marker_path):
        return

    model = get_model(config).to(DEVICE)
    makedirs(logdir, exist_ok=True)

    use_source_individual_norm = config["use_source_individual_norm"]

    vocab, model, train_metrics = train_lexicon_model(
        model,
        datadef,
        train_samples,
        vocab_size,
        use_source_individual_norm,
        train_labelprop_split,
    )
    valid_metrics = eval_lexicon_model(
        model,
        datadef,
        valid_samples,
        vocab,
        use_source_individual_norm,
        valid_labelprop_split,
    )

    write_str_list_as_txt(vocab, join(logdir, "vocab.txt"))
    torch.save(model, join(logdir, "model.pth"))

    metrics = {}
    metrics.update(train_metrics)
    metrics.update(valid_metrics)
    print_metrics(metrics)
    save_json(metrics, join(logdir, "leaf_metrics.json"))

    df = model.get_weighted_lexicon(vocab, datadef.label_names)
    df.to_csv(join(logdir, "lexicon.csv"), index=False)

    # return vocab, model, metrics, df
    write_str_list_as_txt(["yay"], complete_marker_path)

    return model, vocab
