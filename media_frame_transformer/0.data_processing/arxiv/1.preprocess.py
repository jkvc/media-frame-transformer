import glob
import gzip
import json
from collections import defaultdict
from os import makedirs
from os.path import basename, exists, join
from pprint import pprint
from random import randint, shuffle

import matplotlib.pyplot as plt
import numpy as np
from config import DATA_DIR
from media_frame_transformer.dataset.arxiv.definition import ARXIV_CATEGORIES
from media_frame_transformer.utils import (
    ParallelHandler,
    load_json,
    load_pkl,
    save_json,
    save_pkl,
)
from tqdm import tqdm

_TRAIN_PROP, _VALID_PROP, _TEST_PROP = [0.8, 0.1, 0.1]

_SRC_DATA_PATH = join(DATA_DIR, "arxiv", "arxiv-metadata-oai-snapshot.json")
_DST_DATA_DIR = join(DATA_DIR, "arxiv")
_SPLITS_DIR = join(_DST_DATA_DIR, "splits")


# filter to keep only cs.* categories
_FILTERED_DATA_PATH = join(DATA_DIR, "arxiv", "arxiv-cs-only.pkl")
if not exists(_FILTERED_DATA_PATH):
    csdata = []
    with open(_SRC_DATA_PATH) as f:
        for l in tqdm(f.readlines()):
            s = json.loads(l)
            cats = s["categories"].split(" ")
            if any(c in ARXIV_CATEGORIES for c in cats):
                csdata.append(s)
    save_pkl(csdata, _FILTERED_DATA_PATH)
else:
    csdata = load_pkl(_FILTERED_DATA_PATH)
print(len(csdata))

# {
#     "id": "0704.0010",
#     "submitter": "Sergei Ovchinnikov",
#     "authors": "Sergei Ovchinnikov",
#     "title": "Partial cubes: structures, characterizations, and constructions",
#     "comments": "36 pages, 17 figures",
#     "journal-ref": null,
#     "doi": null,
#     "report-no": null,
#     "categories": "math.CO",
#     "license": null,
#     "abstract": "  Partial cubes are isometric subgraphs of hypercubes. Structures on a graph\ndefined by means of semicubes, and Djokovi\\'{c}'s and Winkler's relations play\nan important role in the theory of partial cubes. These structures are employed\nin the paper to characterize bipartite graphs and partial cubes of arbitrary\ndimension. New characterizations are established and new proofs of some known\nresults are given.\n  The operations of Cartesian product and pasting, and expansion and\ncontraction processes are utilized in the paper to construct new partial cubes\nfrom old ones. In particular, the isometric and lattice dimensions of finite\npartial cubes obtained by means of these operations are calculated.\n",
#     "versions": [
#         {
#             "version": "v1",
#             "created": "Sat, 31 Mar 2007 05:10:16 GMT"
#         }
#     ],
#     "update_date": "2007-05-23",
#     "authors_parsed": [
#         [
#             "Ovchinnikov",
#             "Sergei",
#             ""
#         ]
#     ]
# }

cat2id2sample = defaultdict(dict)
for sample in tqdm(csdata):
    if "abstract" not in sample or len(sample["abstract"]) == 0:
        continue

    for cat in sample["categories"].split(" "):
        if cat in ARXIV_CATEGORIES:
            break
    id = sample["id"]
    year = int(sample["update_date"][:4])
    cat2id2sample[cat][id] = {
        "id": id,
        "year": year,
        "abstract": sample["abstract"],
    }


# count per years per category
cat2years = defaultdict(list)
for cat, id2sample in cat2id2sample.items():
    for sample in id2sample.values():
        cat2years[cat].append(sample["year"])
fig, axs = plt.subplots(nrows=1, ncols=len(cat2years), figsize=(5 * len(cat2years), 5))
for ax, cat in zip(axs, cat2years):
    ax.hist(cat2years[cat])
    ax.set_title(cat)
plt.savefig(join(_DST_DATA_DIR, "years.png"))

makedirs(_SPLITS_DIR, exist_ok=True)
for cat, id2sample in cat2id2sample.items():
    save_json(id2sample, join(_DST_DATA_DIR, f"{cat}.json"))

    all_ids = list(id2sample.keys())
    shuffle(all_ids)
    n_train = int(len(all_ids) * _TRAIN_PROP)
    n_valid = int(len(all_ids) * _VALID_PROP)
    n_test = len(all_ids) - n_train - n_valid
    save_json(all_ids[:n_train], join(_SPLITS_DIR, f"{cat}.train.json"))
    save_json(
        all_ids[n_train : n_train + n_valid],
        join(_SPLITS_DIR, f"{cat}.valid.json"),
    )
    save_json(all_ids[-n_test:], join(_SPLITS_DIR, f"{cat}.test.json"))
