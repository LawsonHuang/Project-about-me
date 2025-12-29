# main.py
'''
HW3: Sentiment Analysis with Deep Learning
Using DeBERTa V3 Large for multi-class sentiment classification
'''

import os
import gc
import json
import random
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, PretrainedConfig, PreTrainedModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset
class SentimentDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: AutoTokenizer, max_length: int):
        df = pd.read_csv(csv_path)
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist() if "label" in df.columns else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Model Config
class SentimentConfig(PretrainedConfig):
    model_type = "sentiment-deberta"
    def __init__(self, model_name="microsoft/deberta-v3-large", num_labels=3, head="mlp", max_length=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_labels = num_labels
        self.head = head
        self.max_length = max_length
        self.dropout = dropout


# Model
class SentimentClassifier(PreTrainedModel):
    config_class = SentimentConfig

    def __init__(self, config: Optional[SentimentConfig] = None):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(self.hidden_size)
        self.head = nn.Linear(self.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.dropout)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        feat = outputs.last_hidden_state[:, 0, :]
        feat = self.dropout(self.norm(feat))
        logits = self.head(feat)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)
        return result


# Evaluation
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_y, all_pred = [], []
    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE) if "labels" in batch else None

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.argmax(outputs["logits"], dim=-1)

            all_y.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(all_y, all_pred)
    return acc, np.array(all_y), np.array(all_pred)


# Training
def train(
    model_name: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    out_dir: str,
    epochs: int,
    batch_size: int,
    max_length: int,
    lr_encoder: float = 1e-5,
    lr_head: float = 1e-4,
    warmup_ratio: float = 0.1,
    dropout: float = 0.1,
    seed: int = 42
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds_train = SentimentDataset(train_csv, tokenizer, max_length)
    ds_val   = SentimentDataset(val_csv, tokenizer, max_length)
    ds_test  = SentimentDataset(test_csv, tokenizer, max_length)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_test  = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    config = SentimentConfig(model_name=model_name, num_labels=3, dropout=dropout)
    model = SentimentClassifier(config).to(DEVICE)

    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr_encoder},
        {"params": model.head.parameters(), "lr": lr_head}
    ])
    total_steps = len(dl_train) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*warmup_ratio), num_training_steps=total_steps)

    best_val = -1.0
    ckpt_dir = os.path.join(out_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    tokenizer.save_pretrained(ckpt_dir)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE) if "labels" in batch else None

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss/(pbar.n or 1):.4f}")

        val_acc, _, _ = evaluate(model, dl_val)
        print(f"Epoch {epoch}: Val Acc = {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            model.save_pretrained(ckpt_dir)

    best = SentimentClassifier.from_pretrained(ckpt_dir).to(DEVICE)

    def eval(split, dl):
        acc, y, yhat = evaluate(best, dl)
        cm = confusion_matrix(y, yhat, labels=[0,1,2])
        pd.DataFrame(cm).to_csv(os.path.join(ckpt_dir, f"{split}_cm.csv"))
        rpt = classification_report(y, yhat, digits=4, labels=[0,1,2])
        with open(os.path.join(ckpt_dir, f"{split}_report.txt"), "w") as f:
            f.write(rpt)
        return float(acc)

    train_acc = eval("train", dl_train)
    val_acc   = eval("val", dl_val)
    test_acc  = eval("test", dl_test)

    summary = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "params_trainable": int(sum(p.numel() for p in best.parameters() if p.requires_grad))
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Cleanup
    try:
        best.to("cpu"); model.to("cpu")
    except:
        pass
    del best, model, tokenizer, optimizer, scheduler, dl_train, dl_val, dl_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="./dataset/train.csv")
    parser.add_argument("--test_csv", type=str, default="./dataset/test.csv")
    parser.add_argument("--out_dir", type=str, default="./saved_models/")

    # Use DeBERTa V3 Large by default
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)

    parser.add_argument("--head", type=str, choices=["mlp"], default="mlp")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    full = pd.read_csv(args.train_csv)
    train_df, val_df = train_test_split(full, test_size=0.1, random_state=args.seed, stratify=full["label"])

    os.makedirs(args.out_dir, exist_ok=True)
    train_split = os.path.join(args.out_dir, "train_split.csv")
    val_split   = os.path.join(args.out_dir, "val_split.csv")
    train_df.to_csv(train_split, index=False)
    val_df.to_csv(val_split, index=False)

    train(
        model_name=args.model_name,
        train_csv=train_split,
        val_csv=val_split,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
