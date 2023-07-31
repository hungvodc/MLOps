import datasets
import torch
from datasets import load_dataset
from extract_config import extract_config
from transformers import AutoTokenizer

config = extract_config()


class DataModule:
    def __init__(
        self, model_name=config["model_name"], batch_size=config["batch_size"]
    ):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=int(config["model_layer"]["max_token_length"]),
        )

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )
