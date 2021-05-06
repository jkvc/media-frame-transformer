from dataclasses import dataclass


@dataclass
class DataSample:
    id: str
    text: str
    y_idx: int
    source_name: str
    source_idx: int
