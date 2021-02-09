from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertConfig, RobertaForSequenceClassification

from media_frame_transformer.dataset import PrimaryFrameDataset

ISSUES = ["climate"]
BATCH_SIZE = 10
NUM_DATALOADER_WORKER = 2

if __name__ == "__main__":
    for issue in ISSUES:
        train_set = PrimaryFrameDataset([issue], "train")
        test_set = PrimaryFrameDataset([issue], "test")
        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_DATALOADER_WORKER,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_DATALOADER_WORKER,
        )

        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=14,
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        model = model.cuda()
        print(model)
