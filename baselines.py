"""Two simple baselines: most common label, and TF-IDF with logistic regression."""

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def most_common_label(labels):
    """Return the label that appears most often."""
    return Counter(labels).most_common(1)[0][0]


def tfidf_model():
    """A TF-IDF vectorizer followed by logistic regression."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
