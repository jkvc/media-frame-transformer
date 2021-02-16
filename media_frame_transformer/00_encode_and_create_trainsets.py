from os.path import join

import pandas as pd
from config import FRAMING_DATA_DIR, ISSUES
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from media_frame_transformer.utils import load_json, save_json

TOKENIZER = RobertaTokenizerFast.from_pretrained("roberta-base")


def encode_data(data):
    id2tokens = {}
    for k, v in tqdm(data.items()):

        # preprocess text
        text = v["text"]
        lines = text.split("\n\n")
        lines = lines[3:]  # first 3 lines are id, "PRIMARY", title
        text = "\n".join(lines)
        tokens = TOKENIZER.encode(text, add_special_tokens=True, truncation=True)
        assert len(tokens) <= 512
        id2tokens[k] = tokens

    return id2tokens


if __name__ == "__main__":
    stats = []

    for issue in ISSUES:
        print(">>", issue)
        data = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))

        id2tokens = encode_data(data)
        save_json(
            id2tokens,
            join(FRAMING_DATA_DIR, f"{issue}_encoded.json"),
        )

        testsets = load_json(join(FRAMING_DATA_DIR, f"{issue}_test_sets.json"))
        testsets = {setname: set(ids) for setname, ids in testsets.items()}

        trainsets = {}
        # relevance train set: any sample not in test set relevance, and has tokenized
        trainsets["relevance"] = list(
            {id for id in data if (id in id2tokens and id not in testsets["relevance"])}
        )

        # primary frame trainset: any sample not in testset primary frame, and has tokenized, and has non null primary fram
        trainsets["primary_frame"] = list(
            {
                id
                for id, item in data.items()
                if (
                    id in id2tokens
                    and id not in testsets["primary_frame"]
                    and item["primary_frame"] != 0
                    and item["primary_frame"] != None
                )
            }
        )

        # primary tone trainset: any sample not in testset primary tone, and has tokenized, and has none null primary tone
        trainsets["primary_tone"] = list(
            {
                id
                for id, item in data.items()
                if (
                    id in id2tokens
                    and id not in testsets["primary_tone"]
                    and item["primary_tone"] != 0
                    and item["primary_tone"] != None
                )
            }
        )
        save_json(trainsets, join(FRAMING_DATA_DIR, f"{issue}_train_sets.json"))

        stat = {
            "raw": len(data),
            "tokens": len(id2tokens),
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
