from os.path import join

from config import DATA_DIR

CATEGORIES = [
    "Clothing_Shoes_and_Jewelry",
    "Electronics",
    "Home_and_Kitchen",
    "Kindle_Store",
    "Movies_and_TV",
]
CATEGORY2CIDX = {issue: i for i, issue in enumerate(CATEGORIES)}
N_CATEGORIES = len(CATEGORIES)

RATING_NAMES = ["low", "medium", "high"]
RATING_N_CLASSES = len(RATING_NAMES)


def rating_to_ridx(rating: float) -> int:
    # 1:low
    # 2-4:medium
    # 5:high

    assert int(rating) == rating
    assert rating >= 1.0 and rating <= 5.0
    if rating == 1.0:
        return 0
    elif rating in [2.0, 3.0, 4.0]:
        return 1
    elif rating == 5.0:
        return 2
    else:
        raise NotImplementedError()


LABELPROPS_DIR = join(DATA_DIR, "amazon_subsampled", "labelprops")
