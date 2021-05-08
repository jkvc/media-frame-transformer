# a dataset zoo, for quick definition / retrieval of dataset definition by name

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
from media_frame_transformer.dataset.data_sample import DataSample


@dataclass
class DatasetDefinition:
    source_names: List[str]
    label_names: List[str]
    load_splits_func: Callable[[List[str], List[str]], Dict[str, List[DataSample]]]
    load_labelprops_func: Callable[[str], Dict[str, np.array]]

    n_sources: int = field(init=False)
    n_classes: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_sources = len(self.source_names)
        self.n_classes = len(self.label_names)


_DATADEFS: Dict[str, DatasetDefinition] = {}


def get_datadef(name: str) -> DatasetDefinition:
    return _DATADEFS[name]


def get_datadef_names() -> List[str]:
    return sorted(list(_DATADEFS.keys()))


def register_datadef(name: str, datadef: DatasetDefinition):
    assert name not in _DATADEFS
    _DATADEFS[name] = datadef
