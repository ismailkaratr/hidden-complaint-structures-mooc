"""
Step 6b — Build the 300-row validation template for the sentiment classifier.

Stratified sample over rating so each star count is represented. For each row
we pre-fill a *rating proxy* (rating>=4 -> positive, <=2 -> negative, ==3
left blank) the human can accept or overwrite. The 'gold_label' column is
the column reviewers should populate (or leave the proxy in place).

Usage:
    python scripts/07b_sentiment_validation_template.py [--n 300] [--seed 42]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from config import (
    GLOBAL_SEED, PREPROCESSED_PARQUET, SENT_DIR, set_global_seed, setup_logger,
)

SCRIPT_NAME = "07b_sentiment_validation_template"


def proxy_label(rating: int) -> str:
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return ""  # rating == 3 — coder must decide


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = parser.parse_args()

    set_global_seed(args.seed)
    log = setup_logger(SCRIPT_NAME)

    df = pd.read_parquet(PREPROCESSED_PARQUET, columns=["doc_id", "reviews", "rating"])
    # Stratified per rating; near-equal allocation. Sample indices per group, then
    # reindex from the original frame so the grouping column is preserved
    # (pandas 3 drops it from apply(...) output otherwise).
    per_rating = max(1, args.n // df["rating"].nunique())
    rng = np.random.default_rng(args.seed)
    picked = []
    for _, g in df.groupby("rating", sort=True):
        take = min(len(g), per_rating)
        picked.extend(rng.choice(g.index.values, size=take, replace=False).tolist())
    sample = df.loc[picked].reset_index(drop=True)

    if len(sample) > args.n:
        sample = sample.sample(args.n, random_state=args.seed).reset_index(drop=True)

    sample["proxy_label"] = sample["rating"].map(proxy_label)
    sample["gold_label"] = sample["proxy_label"]  # coder overwrites where needed
    sample["coder_notes"] = ""
    sample = sample[["doc_id", "rating", "reviews", "proxy_label", "gold_label", "coder_notes"]]

    out = SENT_DIR / "07b_sentiment_validation_template.csv"
    sample.to_csv(out, index=False)
    log.info("Wrote %d rows -> %s", len(sample), out)
    log.info("Coder instructions: open the CSV, review 'gold_label', overwrite where needed "
             "with one of {'positive','negative'}, leave rating==3 rows blank if neutral or "
             "delete those rows. Save the file, then run 07c_sentiment_validation_metrics.py.")

    summary = {
        "n_sample": int(len(sample)),
        "per_rating_target": per_rating,
        "rating_distribution": sample["rating"].value_counts().sort_index().to_dict(),
        "seed": args.seed,
    }
    (SENT_DIR / "07b_template_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
