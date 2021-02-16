from collections import defaultdict
from os.path import join

import pandas as pd
from config import FRAMING_DATA_DIR, ISSUES
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from media_frame_transformer.utils import load_json, save_json

TOKENIZER = RobertaTokenizerFast.from_pretrained("roberta-base")

MIN_SPAN_NUM_CHAR = 150

if __name__ == "__main__":

    stats = {}

    for issue in ISSUES:
        print(">>", issue)
        data = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))
        codes = load_json(join(FRAMING_DATA_DIR, "codes.json"))

        labeled_span_data = defaultdict(list)

        for articleid, article in tqdm(data.items()):
            text = article["text"]
            annotations = article["annotations"]
            framing_annotations = annotations["framing"]

            for annotator, spans in framing_annotations.items():
                for span in spans:
                    start = span["start"]
                    end = span["end"]
                    if end - start < MIN_SPAN_NUM_CHAR:
                        continue
                    code = span["code"]
                    text_segment = text[start:end]
                    encoded = TOKENIZER.encode(
                        text, add_special_tokens=True, truncation=True
                    )
                    labeled_span_data[articleid].append(
                        {
                            "text": text_segment,
                            "code": code,
                            "encoded": encoded,
                        }
                    )
        save_json(labeled_span_data, join(FRAMING_DATA_DIR, f"{issue}_spans.json"))

        num_spans = sum(len(l) for l in labeled_span_data.values())
        stats[issue] = {
            "num_spans": num_spans,
            "num_articles": len(labeled_span_data),
        }
        print(len(labeled_span_data), num_spans)

    df = pd.DataFrame.from_dict(stats, orient="index")
    df["ratio"] = df["num_spans"] / df["num_articles"]
    df.to_csv(join(FRAMING_DATA_DIR, "stats_spans.csv"))
