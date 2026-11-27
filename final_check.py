"""Final check: train the chosen model, score test and own messages once."""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from baselines import most_common_label, tfidf_model
from finetune import predict_labels, train_model
from prepare import as_lists, make_split
from tickets_data import LABELS

CHOSEN_LR = 5e-05
EPOCHS = 3


def score(y_true, y_pred):
    """Accuracy and macro F1 for a set of predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def main():
    os.makedirs("reports", exist_ok=True)
    train_df, val_df, test_df = make_split()
    train_texts, train_labels = as_lists(train_df)
    val_texts, val_labels = as_lists(val_df)
    test_texts, test_labels = as_lists(test_df)

    # Train the chosen method (full fine-tuning) on all training rows.
    trainer, tokenizer, history = train_model(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        variant="full",
        epochs=EPOCHS,
        learning_rate=CHOSEN_LR,
        seed=42,
        output_dir="model_tmp",
    )
    model = trainer.model
    os.makedirs("model", exist_ok=True)
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")

    # Score the chosen model on the test rows, once.
    test_pred = predict_labels(model, tokenizer, test_texts)
    chosen_scores = score(test_labels, test_pred)
    matrix = confusion_matrix(test_labels, test_pred, labels=LABELS).tolist()
    # Save a picture of the confusion matrix.
    fig, ax = plt.subplots()
    ax.imshow(matrix)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, matrix[i][j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig("reports/confusion_matrix.png")
    plt.close(fig)
    # Baselines trained on all training rows, scored on test.
    common = most_common_label(train_labels)
    most_pred = [common] * len(test_texts)
    tfidf = tfidf_model()
    tfidf.fit(train_texts, train_labels)
    tfidf_pred = list(tfidf.predict(test_texts))

    # Own 40 messages, scored once.
    own = pd.read_csv("data/own_messages.csv")
    own_pred = predict_labels(model, tokenizer, own["text"].tolist())
    own_acc = accuracy_score(own["label"].tolist(), own_pred)

    final = {
        "chosen_method": "full",
        "learning_rate": CHOSEN_LR,
        "epochs": EPOCHS,
        "test": {
            "full": chosen_scores,
            "most_common": score(test_labels, most_pred),
            "tfidf": score(test_labels, tfidf_pred),
        },
        "confusion_matrix": {"labels": LABELS, "matrix": matrix},
        "own_messages_accuracy": own_acc,
    }
    with open("reports/final.json", "w") as handle:
        json.dump(final, handle, indent=2)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
