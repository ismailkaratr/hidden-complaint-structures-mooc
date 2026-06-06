"""
Step 7 — Robustness of the rating–sentiment mismatch flag.

We compute, for every review, a continuous mismatch score and sweep:
  - 4 normalization strategies for the rating side
  - a grid of decision thresholds
Then we report the share of reviews flagged 'mismatched' under each
combination — to show the finding is not an artifact of one arbitrary cutoff.

Mismatch score definitions (all in [0,1]):
  raw       : |rating_norm_raw - sent_pos_prob|
              rating_norm_raw = (rating - 1) / 4
  rank      : |rating_norm_rank - sent_pos_prob|
              rating_norm_rank = empirical CDF of rating
  zscore    : sigmoid(|z(rating) - z(sent_pos_prob)|) -> [0,1]
  binary    : rating>=4 -> pos(1); <=2 -> neg(0); ==3 dropped; abs(.. - sent_pos_prob)

Usage:
    python scripts/08_mismatch_robustness.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, zscore

from config import (
    PREPROCESSED_PARQUET, SENT_DIR, TABLE_DIR, dump_env, setup_logger,
)

SCRIPT_NAME = "08_mismatch_robustness"

THRESHOLD_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
NORMALIZATIONS = ("raw", "rank", "zscore", "binary")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_mismatch(rating: np.ndarray, pos_prob: np.ndarray, method: str) -> np.ndarray:
    rating = rating.astype(float)
    pos_prob = pos_prob.astype(float)
    if method == "raw":
        return np.abs((rating - 1.0) / 4.0 - pos_prob)
    if method == "rank":
        r_rank = rankdata(rating, method="average") / len(rating)
        return np.abs(r_rank - pos_prob)
    if method == "zscore":
        zr = zscore(rating, nan_policy="omit")
        zp = zscore(pos_prob, nan_policy="omit")
        return _sigmoid(np.abs(zr - zp)) * 2.0 - 1.0  # scale so 0 diff -> 0
    if method == "binary":
        r_bin = np.where(rating >= 4, 1.0, np.where(rating <= 2, 0.0, np.nan))
        return np.abs(r_bin - pos_prob)
    raise ValueError(method)


def main() -> None:
    log = setup_logger(SCRIPT_NAME)
    dump_env(log)

    reviews = pd.read_parquet(PREPROCESSED_PARQUET, columns=["doc_id", "rating"])
    sent = pd.read_parquet(SENT_DIR / "07_sentiment_predictions.parquet")
    df = reviews.merge(sent, on="doc_id", how="inner", validate="1:1")
    log.info("Joined rows: %d", len(df))

    rows = []
    score_table = {"doc_id": df["doc_id"].values, "rating": df["rating"].values,
                   "sent_pos_prob": df["sent_pos_prob"].values}

    for method in NORMALIZATIONS:
        score = compute_mismatch(df["rating"].values, df["sent_pos_prob"].values, method)
        score_table[f"mismatch_{method}"] = score
        valid = ~np.isnan(score)
        for thr in THRESHOLD_GRID:
            flagged = (score[valid] >= thr).mean()
            rows.append({
                "normalization": method,
                "threshold": thr,
                "n_valid": int(valid.sum()),
                "flagged_share": round(float(flagged), 4),
                "mean_score": round(float(np.nanmean(score)), 4),
                "median_score": round(float(np.nanmedian(score)), 4),
            })

    pd.DataFrame(score_table).to_parquet(TABLE_DIR / "08_mismatch_scores.parquet", index=False)
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(TABLE_DIR / "08_mismatch_robustness.csv", index=False)
    log.info("Robustness grid:\n%s", grid_df.to_string(index=False))

    summary = {
        "normalizations": list(NORMALIZATIONS),
        "thresholds": list(THRESHOLD_GRID),
        "n_reviews": int(len(df)),
        "flagged_share_range_per_norm": {
            m: [round(float(grid_df[grid_df["normalization"] == m]["flagged_share"].min()), 4),
                round(float(grid_df[grid_df["normalization"] == m]["flagged_share"].max()), 4)]
            for m in NORMALIZATIONS
        },
    }
    (TABLE_DIR / "08_mismatch_robustness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
