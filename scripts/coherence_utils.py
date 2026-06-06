"""
Coherence (c_v, c_npmi) via gensim, computed from BERTopic's top-k words.

We deliberately recompute the dictionary/corpus from the *same* tokenization
BERTopic's CountVectorizer used, so coherence is comparable across runs.
"""
from __future__ import annotations

from typing import Sequence

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def _tokenize(texts: Sequence[str], vectorizer) -> list[list[str]]:
    analyzer = vectorizer.build_analyzer()
    return [analyzer(t) for t in texts]


def compute_coherence(*, topic_model, docs: Sequence[str], top_k: int = 10) -> dict:
    """Return c_v and c_npmi for the current topic model (excludes -1)."""
    topics = topic_model.get_topics()
    topic_words: list[list[str]] = []
    for tid, word_scores in topics.items():
        if tid == -1:
            continue
        words = [w for w, _ in word_scores][:top_k]
        if len(words) >= 2:
            topic_words.append(words)

    if not topic_words:
        return {"c_v": float("nan"), "c_npmi": float("nan"), "n_topics_scored": 0}

    tokenized = _tokenize(docs, topic_model.vectorizer_model)
    dictionary = Dictionary(tokenized)
    corpus = [dictionary.doc2bow(t) for t in tokenized]

    out = {}
    for measure in ("c_v", "c_npmi"):
        cm = CoherenceModel(
            topics=topic_words, texts=tokenized, corpus=corpus,
            dictionary=dictionary, coherence=measure, topn=top_k,
        )
        out[measure] = float(cm.get_coherence())
    out["n_topics_scored"] = len(topic_words)
    return out
