import math
from typing import List
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import IPython.display as display


class AttentionHeadImportance:
    def __init__(self, model_name="t5-base", device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        # BEGIN CODE : ahi.__init__
        # >>> {segment:ahi.__init__} <<<
        # END CODE

    def get_dataloader(self, path, name=None, split="validation", batch_size=8, shuffle=False):
        dataset = load_dataset(path, name, split=split)
        dataset = self._preprocess_dataset(path, dataset)
        dataset.set_format(type="torch", columns=["input_ids", "labels"])
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _preprocess_dataset(self, path, dataset):
        if path == "glue":
            return self._preprocess_sst2(dataset)
        elif path == "squad":
            return self._preprocess_squad(dataset)
        elif path == "cnn_dailymail":
            return self._preprocess_cnn_dailymail(dataset)
        elif path == "wmt16":
            return self._preprocess_wmt16(dataset)
        else:
            raise ValueError(f"Preprocessing for dataset {path} is not implemented.")

    def _preprocess_sst2(self, dataset):
        def preprocess(batch):
            source_texts = ["sst2 sentence: " + ex for ex in batch["sentence"]]
            target_texts = ["positive" if label == 1 else "negative" for label in batch["label"]]
            inputs = self.tokenizer(source_texts, padding="max_length", truncation=True, max_length=128)
            labels = self.tokenizer(target_texts, padding="max_length", truncation=True, max_length=10)
            return {"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}

        dataset = dataset.map(preprocess, batched=True, remove_columns=["sentence", "label", "idx"])
        return dataset

    def _preprocess_squad(self, dataset):
        def preprocess(batch):
            source_texts = ["question: " + q + " context: " + c for q, c in zip(batch["question"], batch["context"])]
            target_texts = [ans["text"][0] if len(ans["text"]) > 0 else "" for ans in batch["answers"]]
            inputs = self.tokenizer(source_texts, padding="max_length", truncation=True, max_length=128)
            labels = self.tokenizer(target_texts, padding="max_length", truncation=True, max_length=32)
            return {"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}

        dataset = dataset.map(preprocess, batched=True, remove_columns=["id", "title", "context", "question", "answers"])
        return dataset

    def _preprocess_cnn_dailymail(self, dataset):
        def preprocess(batch):
            source_texts = ["summarize: " + text for text in batch["article"]]
            target_texts = batch["highlights"]
            inputs = self.tokenizer(source_texts, padding="max_length", truncation=True, max_length=512)
            labels = self.tokenizer(target_texts, padding="max_length", truncation=True, max_length=128)
            return {"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}

        dataset = dataset.map(preprocess, batched=True, remove_columns=["article", "highlights", "id"])
        return dataset

    def _preprocess_wmt16(self, dataset):
        def preprocess(batch):
            source_texts = ["translate English to German: " + ex["en"] for ex in batch["translation"]]
            target_texts = [ex["de"] for ex in batch["translation"]]
            inputs = self.tokenizer(source_texts, padding="max_length", truncation=True, max_length=128)
            labels = self.tokenizer(target_texts, padding="max_length", truncation=True, max_length=128)
            return {"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}

        dataset = dataset.map(preprocess, batched=True, remove_columns=["translation"])
        return dataset

    def compute_head_importance(self, dataloader, encoder_head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None):
        
        # BEGIN CODE : ahi.compute_head_importance
        # ADD YOUR CODE HERE
        # >>> {segment:ahi.compute_head_importance} <<<
        # END CODE
        pass
    
    # BEGIN CODE : ahi.additional_methods
    # ADD YOUR CODE HERE
    # >>> {segment:ahi.additional_methods} <<<
    # END CODE

        


 


