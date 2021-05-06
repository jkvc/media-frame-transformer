from os.path import join

from config import DATA_DIR

CATEGORIES = [
    "Automotive",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Electronics",
    "Grocery_and_Gourmet_Food",
    "Home_and_Kitchen",
    "Kindle_Store",
    "Movies_and_TV",
    "Pet_Supplies",
    "Sports_and_Outdoors",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
CATEGORY2CIDX = {issue: i for i, issue in enumerate(CATEGORIES)}
N_CATEGORIES = len(CATEGORIES)

RATING_NAMES = ["1.0", "2.0", "3.0", "4.0", "5.0"]
RATING_N_CLASSES = len(RATING_NAMES)


def rating_to_ridx(rating: float) -> int:
    # rating [1.0, 5.0]
    assert int(rating) == rating
    assert rating >= 1.0 and rating <= 5.0
    return int(rating) - 1


LABELPROPS_DIR = join(DATA_DIR, "amazon_subsampled", "labelprops")
