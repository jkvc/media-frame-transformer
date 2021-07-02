from os.path import join

import pandas as pd
from config import DATA_DIR
from media_frame_transformer.datadef import zoo
from media_frame_transformer.utils import load_json, save_json

FRAMING_DATA_DIR = join(DATA_DIR, "framing_labeled")
ISSUES = zoo.get_datadef("framing").source_names


if __name__ == "__main__":
    stats = []

    for issue in ISSUES:
        print(">>", issue)
        data = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))
        ids = list(data.keys())

        testsets = load_json(join(FRAMING_DATA_DIR, f"{issue}_test_sets.json"))
        testsets = {setname: set(ids) for setname, ids in testsets.items()}

        trainsets = {}
        # relevance train set: any sample not in test set relevance
        trainsets["relevance"] = list(
            {id for id in data if (id in ids and id not in testsets["relevance"])}
        )

        # primary frame trainset: any sample not in testset primary frame, and has non null primary fram
        trainsets["primary_frame"] = list(
            {
                id
                for id, item in data.items()
                if (
                    id in ids
                    and id not in testsets["primary_frame"]
                    and item["primary_frame"] != 0
                    and item["primary_frame"] != None
                )
            }
        )

        # primary tone trainset: any sample not in testset primary tone, and has none null primary tone
        trainsets["primary_tone"] = list(
            {
                id
                for id, item in data.items()
                if (
                    id in ids
                    and id not in testsets["primary_tone"]
                    and item["primary_tone"] != 0
                    and item["primary_tone"] != None
                )
            }
        )
        save_json(trainsets, join(FRAMING_DATA_DIR, f"{issue}_train_sets.json"))

        stat = {
            "raw": len(data),
        }
        stat.update(
            {f"train_{setname}": len(ids) for setname, ids in trainsets.items()}
        )
        stat.update({f"test_{setname}": len(ids) for setname, ids in testsets.items()})
        stats.append(stat)

        for k, v in stat.items():
            print("--", k, v)

    df = pd.DataFrame(stats)
    df.to_csv(join(FRAMING_DATA_DIR, "stats.csv"))
