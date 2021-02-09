from os.path import join

from config import FRAMING_DATA_DIR, ISSUES
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, RobertaTokenizer

from media_frame_transformer.utils import load_json, save_json

MODEL_NAME = "roberta-base"


if __name__ == "__main__":

    for issue in ISSUES:

        data = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))

        tokenizer: PreTrainedTokenizerBase = RobertaTokenizer.from_pretrained(
            MODEL_NAME
        )

        id2tokens = {}
        for k, v in tqdm(data.items(), desc=issue):

            # preprocess text
            text = v["text"]
            lines = text.split("\n\n")
            lines = lines[3:]  # first 3 lines are id, "PRIMARY", title
            text = "\n".join(lines)

            # filter out long text
            if len(tokenizer.tokenize(text)) > 500:
                continue

            tokens = tokenizer.encode(
                text, add_special_tokens=True, padding="max_length"
            )
            assert len(tokens) == 512
            id2tokens[k] = tokens

        save_json(id2tokens, join(FRAMING_DATA_DIR, f"{issue}_tokenized.json"))
