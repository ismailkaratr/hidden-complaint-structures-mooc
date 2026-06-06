"""
Step 4 — Final BERTopic model.

After step 03 (stability) and step 04 (grid) you decide a min_topic_size; this
script fits that single configuration, persists the model, and writes the
artifacts needed for both validity reporting and downstream analysis:

  - outputs/tables/05_final_coherence.json   (c_v, c_npmi)
  - outputs/tables/05_topic_info.csv         (id, count, name, top-10 words)
  - outputs/tables/05_topic_keywords.csv     (long form: topic_id, rank, word, weight)
  - outputs/tables/05_topic_representative_docs.csv  (top-N exemplar docs per topic)
  - outputs/tables/05_doc_topics.parquet     (doc_id, topic, probability)
  - outputs/models/final_bertopic/           (BERTopic.save)

Usage:
    python scripts/05_final_model.py --min-topic-size 50 --seed 42 --top-docs 5
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from bertopic_utils import build_topic_model, outlier_ratio, topic_diversity
from coherence_utils import compute_coherence
from config import (
    EMBEDDINGS_NPY, GLOBAL_SEED, MODEL_DIR, OUTPUT_DIR,
    PREPROCESSED_PARQUET, RunRecord, TABLE_DIR, dump_env,
    set_global_seed, setup_logger,
)

SCRIPT_NAME = "05_final_model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-topic-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--top-docs", type=int, default=5)
    args = parser.parse_args()

    set_global_seed(args.seed)
    log = setup_logger(SCRIPT_NAME)
    dump_env(log)
    log.info("Final fit: min_topic_size=%d seed=%d", args.min_topic_size, args.seed)

    df = pd.read_parquet(PREPROCESSED_PARQUET, columns=["doc_id", "reviews", "rating"])
    embeddings = np.load(EMBEDDINGS_NPY, mmap_mode="r")
    docs = df["reviews"].tolist()

    model = build_topic_model(
        seed=args.seed, min_topic_size=args.min_topic_size,
        calculate_probabilities=True, verbose=True,
    )
    topics, probs = model.fit_transform(docs, embeddings=np.asarray(embeddings))
    labels = np.asarray(topics, dtype=np.int32)
    log.info("Fit done. n_topics=%d outliers=%.2f%%",
             len(set(labels)) - (1 if -1 in labels else 0),
             100 * outlier_ratio(labels))

    # ---- Save model
    model_path = MODEL_DIR / "final_bertopic"
    model.save(str(model_path), serialization="safetensors", save_ctfidf=True,
               save_embedding_model=False)
    log.info("Saved model to %s", model_path)

    # ---- Coherence (validity)
    coh = compute_coherence(topic_model=model, docs=docs, top_k=10)
    coh.update({"min_topic_size": args.min_topic_size, "seed": args.seed})
    (TABLE_DIR / "05_final_coherence.json").write_text(json.dumps(coh, indent=2), encoding="utf-8")
    log.info("Coherence: %s", coh)

    # ---- topic_info + diversity
    info_df = model.get_topic_info()
    info_df.to_csv(TABLE_DIR / "05_topic_info.csv", index=False)

    # ---- Long-form topic keywords (top-10)
    kw_rows = []
    for tid, word_scores in model.get_topics().items():
        if tid == -1:
            continue
        for rank, (word, weight) in enumerate(word_scores[:10], start=1):
            kw_rows.append({"topic_id": tid, "rank": rank, "word": word, "weight": float(weight)})
    pd.DataFrame(kw_rows).to_csv(TABLE_DIR / "05_topic_keywords.csv", index=False)

    diversity = topic_diversity(
        [[w for w, _ in model.get_topics()[t]] for t in model.get_topics() if t != -1],
        top_k=10,
    )
    log.info("Topic diversity: %.4f", diversity)

    # ---- Representative documents
    rep_rows = []
    rep_docs = model.get_representative_docs() or {}
    for tid in sorted(t for t in model.get_topics() if t != -1):
        for rank, doc in enumerate(rep_docs.get(tid, [])[:args.top_docs], start=1):
            rep_rows.append({"topic_id": tid, "rank": rank, "doc": doc})
    pd.DataFrame(rep_rows).to_csv(TABLE_DIR / "05_topic_representative_docs.csv", index=False)

    # ---- Doc-level assignment for downstream steps
    doc_topics = pd.DataFrame({
        "doc_id": df["doc_id"].values,
        "topic": labels,
        "max_prob": (probs.max(axis=1) if probs is not None and probs.ndim == 2 else np.full(len(labels), np.nan)),
        "rating": df["rating"].values,
    })
    doc_topics.to_parquet(TABLE_DIR / "05_doc_topics.parquet", index=False)

    RunRecord(
        script=SCRIPT_NAME, seed=args.seed,
        params={"min_topic_size": args.min_topic_size, "top_docs": args.top_docs},
        n_inputs=len(docs), n_outputs=len(info_df),
        extra={"coherence": coh, "diversity": diversity,
               "outlier_ratio": round(outlier_ratio(labels), 4)},
    ).save(OUTPUT_DIR / "05_final_model.run.json")


if __name__ == "__main__":
    main()
