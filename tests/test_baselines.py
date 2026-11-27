"""Tests for baselines.py: the most common label and TF-IDF."""

from sklearn.metrics import accuracy_score

from baselines import most_common_label, tfidf_model
from prepare import as_lists, make_split


def test_most_common_label_picks_the_frequent_one():
    labels = ["billing", "station", "billing", "account", "billing"]
    assert most_common_label(labels) == "billing"


def test_tfidf_beats_most_common_on_validation():
    train_df, val_df, _ = make_split()
    train_texts, train_labels = as_lists(train_df)
    val_texts, val_labels = as_lists(val_df)

    # Most common label: predict one label for every message.
    common = most_common_label(train_labels)
    common_accuracy = accuracy_score(val_labels, [common] * len(val_texts))

    # TF-IDF: fit on training rows only, score on validation rows.
    model = tfidf_model()
    model.fit(train_texts, train_labels)
    tfidf_accuracy = accuracy_score(val_labels, model.predict(val_texts))

    assert tfidf_accuracy > common_accuracy
