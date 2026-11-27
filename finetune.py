"""Fine-tune DistilBERT on the tickets, head-only or full."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from tickets_data import LABELS

MODEL_NAME = "distilbert/distilbert-base-uncased"
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


class TicketDataset(Dataset):
    """Tokenized tickets ready for the Trainer."""

    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=64
        )
        self.labels = [LABEL2ID[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(values[index]) for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index])
        return item


def compute_metrics(eval_pred):
    """Accuracy of the predictions."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


def build_model(variant):
    """Load DistilBERT with a fresh 4-label head. Freeze the body for head-only."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if variant == "head":
        for param in model.distilbert.parameters():
            param.requires_grad = False
    return model


def epoch_history(trainer):
    """Pull training loss and validation accuracy for each epoch."""
    train_loss = {}
    val_accuracy = {}
    for entry in trainer.state.log_history:
        epoch = round(entry["epoch"])
        if "loss" in entry:
            train_loss[epoch] = entry["loss"]
        if "eval_accuracy" in entry:
            val_accuracy[epoch] = entry["eval_accuracy"]
    history = []
    for epoch in sorted(val_accuracy):
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss.get(epoch),
                "val_accuracy": val_accuracy[epoch],
            }
        )
    return history


def train_model(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    variant="full",
    epochs=3,
    learning_rate=2e-5,
    seed=42,
    output_dir="model_tmp",
):
    """Fine-tune the model and return the trainer, tokenizer, and history."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(variant)
    train_data = TicketDataset(train_texts, train_labels, tokenizer)
    val_data = TicketDataset(val_texts, val_labels, tokenizer)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer, tokenizer, epoch_history(trainer)


def predict_labels(model, tokenizer, texts):
    """Predict a label for each text."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=64
            )
            logits = model(**inputs).logits
            predictions.append(ID2LABEL[int(torch.argmax(logits, dim=1))])
    return predictions
