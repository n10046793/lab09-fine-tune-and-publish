"""Tests for finetune.py: tokenization, freezing, training, and saving."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetune import (
    LABEL2ID,
    MODEL_NAME,
    TicketDataset,
    build_model,
    predict_labels,
    train_model,
)
from share import CARD

# A tiny set of messages, two per queue, for fast training tests.
TRAIN_TEXTS = [
    "I was charged twice this month",
    "My invoice looks wrong",
    "The bike chain is broken",
    "The front tire is flat",
    "The station at Queen Street is full",
    "There are no bikes at the dock",
    "I cannot log into my account",
    "Please reset my password",
]
TRAIN_LABELS = [
    "billing",
    "billing",
    "bike_fault",
    "bike_fault",
    "station",
    "station",
    "account",
    "account",
]
VAL_TEXTS = ["My card was charged again", "The seat is loose"]
VAL_LABELS = ["billing", "bike_fault"]


def test_tokenization_truncates_and_keeps_labels():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    long_text = " ".join(["billing"] * 200)
    dataset = TicketDataset([long_text], ["billing"], tokenizer)
    item = dataset[0]
    # The long message is cut to the max length of 64 tokens.
    assert len(item["input_ids"]) == 64
    # The label is kept, as its number.
    assert dataset.labels == [LABEL2ID["billing"]]


def test_head_only_freezes_the_body():
    model = build_model("head")
    # No body weight is trainable.
    body_trainable = [p for p in model.distilbert.parameters() if p.requires_grad]
    assert body_trainable == []
    # The head is still trainable.
    head_trainable = [p for p in model.classifier.parameters() if p.requires_grad]
    assert len(head_trainable) > 0


def test_head_only_training_keeps_body_weights(tmp_path):
    # A body weight before training, the pretrained starting point.
    before = build_model("head")
    before_weight = before.distilbert.embeddings.word_embeddings.weight.clone()

    trainer, _, _ = train_model(
        TRAIN_TEXTS,
        TRAIN_LABELS,
        VAL_TEXTS,
        VAL_LABELS,
        variant="head",
        epochs=1,
        output_dir=str(tmp_path),
    )
    after_weight = trainer.model.distilbert.embeddings.word_embeddings.weight
    # Head-only training must not change the body.
    assert torch.equal(before_weight, after_weight)


def test_full_finetuning_changes_body_weights(tmp_path):
    before = build_model("full")
    before_weight = before.distilbert.embeddings.word_embeddings.weight.clone()

    trainer, _, _ = train_model(
        TRAIN_TEXTS,
        TRAIN_LABELS,
        VAL_TEXTS,
        VAL_LABELS,
        variant="full",
        epochs=1,
        output_dir=str(tmp_path),
    )
    after_weight = trainer.model.distilbert.embeddings.word_embeddings.weight
    # Full fine-tuning must change the body.
    assert not torch.equal(before_weight, after_weight)


def test_same_seed_repeats_a_run(tmp_path):
    torch.manual_seed(42)
    trainer1, tokenizer1, _ = train_model(
        TRAIN_TEXTS,
        TRAIN_LABELS,
        VAL_TEXTS,
        VAL_LABELS,
        variant="full",
        epochs=1,
        seed=42,
        output_dir=str(tmp_path),
    )
    first = predict_labels(trainer1.model, tokenizer1, VAL_TEXTS)

    torch.manual_seed(42)
    trainer2, tokenizer2, _ = train_model(
        TRAIN_TEXTS,
        TRAIN_LABELS,
        VAL_TEXTS,
        VAL_LABELS,
        variant="full",
        epochs=1,
        seed=42,
        output_dir=str(tmp_path),
    )
    second = predict_labels(trainer2.model, tokenizer2, VAL_TEXTS)

    assert first == second


def test_model_can_memorize_a_small_batch(tmp_path):
    trainer, tokenizer, _ = train_model(
        TRAIN_TEXTS,
        TRAIN_LABELS,
        VAL_TEXTS,
        VAL_LABELS,
        variant="full",
        epochs=30,
        learning_rate=1e-4,
        output_dir=str(tmp_path),
    )
    predictions = predict_labels(trainer.model, tokenizer, TRAIN_TEXTS)
    correct = sum(p == t for p, t in zip(predictions, TRAIN_LABELS, strict=True))
    # After many epochs on 8 messages, the model fits them all.
    assert correct == len(TRAIN_LABELS)


def test_saved_model_reloads_with_same_prediction(tmp_path):
    model = build_model("full")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    text = "I was charged twice this month"
    before = predict_labels(model, tokenizer, [text])

    save_dir = tmp_path / "saved_model"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    reloaded = AutoModelForSequenceClassification.from_pretrained(save_dir)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)
    after = predict_labels(reloaded, reloaded_tokenizer, [text])

    assert before == after


def test_model_card_has_required_headings():
    for heading in [
        "## Intended use",
        "## Training data",
        "## Results",
        "## Limitations",
    ]:
        assert heading in CARD
