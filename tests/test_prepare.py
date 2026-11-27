"""Tests for prepare.py: the split and the small-sample function."""

import pytest

from prepare import make_split, small_sample
from tickets_data import LABELS


@pytest.mark.parametrize("seed", [1, 42, 7])
def test_split_sizes_and_no_overlap(seed):
    train_df, val_df, test_df = make_split(seed=seed)
    # 60/20/20 of 1200 rows.
    assert len(train_df) == 720
    assert len(val_df) == 240
    assert len(test_df) == 240
    # The three parts must not share any row.
    train_ids = set(train_df.index)
    val_ids = set(val_df.index)
    test_ids = set(test_df.index)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_small_sample_count_and_labels():
    train_df, _, _ = make_split()
    sample = small_sample(train_df, 100, 1)
    assert len(sample) == 100
    # All four labels are still there.
    assert set(sample["label"]) == set(LABELS)


def test_small_sample_only_from_train():
    train_df, val_df, test_df = make_split()
    sample = small_sample(train_df, 100, 1)
    # Every sampled row comes from the training part only.
    assert set(sample.index).issubset(set(train_df.index))
    assert set(sample.index).isdisjoint(set(val_df.index))
    assert set(sample.index).isdisjoint(set(test_df.index))
