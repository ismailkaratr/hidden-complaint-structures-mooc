"""
Factory helpers so every step builds BERTopic identically (only the bits
that *should* vary across steps actually do).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from config import HDBSCAN_PARAMS, UMAP_PARAMS


def build_topic_model(*, seed: int, min_topic_size: int,
                       calculate_probabilities: bool = False,
                       low_memory: bool = True,
                       verbose: bool = False):
    """Return a fresh BERTopic instance with deterministic UMAP + HDBSCAN."""
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(random_state=seed, **UMAP_PARAMS)
    hdb_params = dict(HDBSCAN_PARAMS)
    hdb_params["min_cluster_size"] = max(min_topic_size, hdb_params["min_cluster_size"])
    hdb_model = HDBSCAN(prediction_data=True, **hdb_params)
    vectorizer = CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2))
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdb_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        min_topic_size=min_topic_size,
        calculate_probabilities=calculate_probabilities,
        low_memory=low_memory,
        verbose=verbose,
    )


def topic_diversity(topics_words: Iterable[list[str]], top_k: int = 10) -> float:
    """Fraction of unique words across the top-k words of every topic."""
    seen, total = set(), 0
    for words in topics_words:
        for w in words[:top_k]:
            seen.add(w)
            total += 1
    return len(seen) / total if total else 0.0


def outlier_ratio(labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    return float((labels == -1).sum() / len(labels)) if len(labels) else 0.0
