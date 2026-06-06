"""
Step 9 — Transparency about how BERTopic outliers (-1) are handled.

Loads the final model, reports:
  - outlier_share of the as-fit assignment
  - reassigns outliers via BERTopic.reduce_outliers (distribution-based)
  - re-runs the same descriptive analysis (per-topic mean/median mismatch)
    on (a) original labels, (b) labels with outliers dropped, (c) labels with
    outliers reassigned — so the methods section can state whether findings
    move when the outlier choice changes.

Usage:
    python scripts/10_outlier_handling.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    EMBEDDINGS_NPY, MODEL_DIR, PREPROCESSED_PARQUET, TABLE_DIR,
    dump_env, setup_logger,
)

SCRIPT_NAME = "10_outlier_handling"
SCORE_COL = "mismatch_raw"


def per_topic_summary(topics, scores) -> pd.DataFrame:
    tmp = pd.DataFrame({"topic": np.asarray(topics), SCORE_COL: np.asarray(scores)})
    g = tmp.groupby("topic", sort=True)[SCORE_COL]
    return pd.DataFrame({
        "topic": g.size().index,
        "n": g.size().values,
        "mean": g.mean().values,
        "median": g.median().values,
    })


def main() -> None:
    log = setup_logger(SCRIPT_NAME)
    dump_env(log)

    from bertopic import BERTopic
    model_path = MODEL_DIR / "final_bertopic"
    log.info("Loading %s", model_path)
    model = BERTopic.load(str(model_path))

    docs_df = pd.read_parquet(PREPROCESSED_PARQUET, columns=["doc_id", "reviews"])
    embeddings = np.load(EMBEDDINGS_NPY, mmap_mode="r")
    docs = docs_df["reviews"].tolist()

    doc_topics = pd.read_parquet(TABLE_DIR / "05_doc_topics.parquet")
    scores = pd.read_parquet(TABLE_DIR / "08_mismatch_scores.parquet",
                              columns=["doc_id", SCORE_COL])

    base = doc_topics.merge(scores, on="doc_id", how="left").dropna(subset=[SCORE_COL])
    base["topic_original"] = base["topic"]

    n_total = len(base)
    n_outlier = int((base["topic_original"] == -1).sum())
    outlier_share = n_outlier / n_total
    log.info("Outliers in final model: %d / %d (%.2f%%)", n_outlier, n_total, 100 * outlier_share)

    # ---- (a) original labels (outliers kept as their own group)
    summ_orig = per_topic_summary(base["topic_original"].values, base[SCORE_COL].values)
    summ_orig.to_csv(TABLE_DIR / "10_topic_summary_original.csv", index=False)

    # ---- (b) drop outliers
    mask = base["topic_original"].values != -1
    summ_dropped = per_topic_summary(
        base["topic_original"].values[mask], base[SCORE_COL].values[mask]
    )
    summ_dropped.to_csv(TABLE_DIR / "10_topic_summary_outliers_dropped.csv", index=False)

    # ---- (c) reassign outliers via reduce_outliers (embedding-cosine; cheapest)
    log.info("Reassigning outliers with strategy='embeddings' ...")
    new_topics = model.reduce_outliers(
        docs, list(base["topic_original"].values),
        strategy="embeddings", embeddings=np.asarray(embeddings),
    )
    base["topic_reassigned"] = new_topics
    summ_reassigned = per_topic_summary(base["topic_reassigned"].values, base[SCORE_COL].values)
    summ_reassigned.to_csv(TABLE_DIR / "10_topic_summary_outliers_reassigned.csv", index=False)

    # ---- Rank-correlation of per-topic medians across the three views
    # (only on topics present in all three)
    common = (set(summ_orig["topic"]) & set(summ_dropped["topic"]) & set(summ_reassigned["topic"]))
    common = sorted(t for t in common if t != -1)
    def medians(df):
        m = df.set_index("topic")["median"]
        return m.reindex(common).values
    rho_drop, p_drop = spearmanr(medians(summ_orig), medians(summ_dropped))
    rho_reas, p_reas = spearmanr(medians(summ_orig), medians(summ_reassigned))
    log.info("Spearman rho(original, dropped)    = %.4f (p=%.4g)", rho_drop, p_drop)
    log.info("Spearman rho(original, reassigned) = %.4f (p=%.4g)", rho_reas, p_reas)

    summary = {
        "n_docs": int(n_total),
        "n_outliers": n_outlier,
        "outlier_share": round(outlier_share, 4),
        "spearman_original_vs_dropped": {"rho": float(rho_drop), "p": float(p_drop)},
        "spearman_original_vs_reassigned": {"rho": float(rho_reas), "p": float(p_reas)},
        "policy": ("Outliers (-1) are reported as a separate group in step 09 by default. "
                   "Use --include-outliers to test inclusion; this script provides the "
                   "reassignment alternative for sensitivity reporting."),
    }
    (TABLE_DIR / "10_outlier_handling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
