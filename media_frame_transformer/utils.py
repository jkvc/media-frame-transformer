import json
import pickle
import shutil
from multiprocessing import Pool, cpu_count
from os import mkdir
from os.path import join
from typing import Iterable, List, Tuple

import torch
from genericpath import exists
from tqdm import tqdm

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def read_txt_as_str_list(filepath: str) -> List[str]:
    with open(filepath) as f:
        ls = f.readlines()
    ls = [l.strip() for l in ls]
    return ls


def write_str_list_as_txt(lst: List[str], filepath: str):
    with open(filepath, "w") as f:
        f.writelines([f"{s}\n" for s in lst])


def save_pkl(obj, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(save_path: str):
    with open(save_path, "rb") as f:
        return pickle.load(f)


def save_json(obj, save_path: str):
    with open(save_path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(save_path: str):
    with open(save_path, "r") as f:
        return json.load(f)


def load_yaml(save_path: str):
    from yaml import load

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    with open(save_path, "r") as f:
        return load(f, Loader=Loader)


def mkdir_overwrite(path: str):
    if exists(path):
        shutil.rmtree(path)
    mkdir(path)


class ParallelHandler:
    def __init__(self, f):
        self.f = f

    def f_wrapper(self, param):
        if isinstance(param, tuple) or isinstance(param, list):
            return self.f(*param)
        else:
            return self.f(param)

    def run(self, params, num_procs=(cpu_count()), desc=None, quiet=False):
        pool = Pool(
            processes=num_procs,
        )
        rets = list(
            tqdm(
                pool.imap_unordered(self.f_wrapper, params),
                total=len(params),
                desc=desc,
                disable=quiet,
            )
        )
        pool.close()
        return rets


def stylize_model_arch_for_figures(arch: str) -> str:
    toks = arch.split("+")
    for i in range(1, len(toks)):
        toks[i] = toks[i].upper()
    name = "+".join(toks)
    name = name.replace("roberta", "RoBERTa")
    name = name.replace("logreg", "Lexicon")
    return name
