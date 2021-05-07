from os.path import join

from config import DATA_DIR
from media_frame_transformer.dataset.amazon.definition import LABELPROPS_DIR

ARXIV_CATEGORIES = [
    "cs.AI",
    "cs.CL",  # computation and language
    "cs.CV",
    "cs.LG",  # machine learning
    "cs.NE",  # neural
    "cs.SI",  # social and information network
]
ARXIV_CATEGORY2IDX = {c: i for i, c in enumerate(ARXIV_CATEGORIES)}
ARXIV_N_CATEGORIES = len(ARXIV_CATEGORIES)

YEARRANGE2BOUNDS = {
    "upto2008": (0, 2008),
    "2009-2014": (2009, 2014),
    "2015-2018": (2015, 2018),
    "2019after": (2019, 6969),
}
YEARRANGE_NAMES = list(YEARRANGE2BOUNDS.keys())
YEARRANGE_N_CLASSES = len(YEARRANGE2BOUNDS)


def year2yidx(year: int) -> int:
    for i, yearrange_name in enumerate(YEARRANGE_NAMES):
        lb, ub = YEARRANGE2BOUNDS[yearrange_name]
        if year >= lb and year <= ub:
            return i
    raise ValueError()


LABELPROPS_DIR = join(DATA_DIR, "arxiv", "labelprops")
