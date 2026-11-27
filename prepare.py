"""Split the tickets and draw small training samples."""

from sklearn.model_selection import train_test_split

from tickets_data import make_tickets


def make_split(seed=42):
    """Make one stratified 60/20/20 split of all the tickets."""
    df = make_tickets()
    # First cut: 60 percent train, 40 percent held out.
    train_df, held_out = train_test_split(
        df, test_size=0.4, stratify=df["label"], random_state=seed
    )
    # Second cut: split the held out part in half.
    val_df, test_df = train_test_split(
        held_out, test_size=0.5, stratify=held_out["label"], random_state=seed
    )
    return train_df, val_df, test_df


def small_sample(train_df, n, seed):
    """Draw a stratified sample of n rows from the training rows."""
    sample, _ = train_test_split(
        train_df, train_size=n, stratify=train_df["label"], random_state=seed
    )
    return sample


def as_lists(df):
    """Return the texts and labels of a frame as two lists."""
    return df["text"].tolist(), df["label"].tolist()
