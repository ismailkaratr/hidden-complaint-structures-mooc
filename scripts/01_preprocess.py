"""
Step 1 — Preprocess Coursera reviews.

Reads the raw CSVs, applies filtering rules, and logs every drop with a row
count so the methods section can report exactly what was removed and why.
The cleaned table is written to Parquet for fast downstream loading.

Usage:
    python scripts/01_preprocess.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from config import (
    COURSES_CSV, GLOBAL_SEED, OUTPUT_DIR, PREPROCESSED_PARQUET, REVIEWS_CSV,
    RunRecord, TABLE_DIR, dump_env, set_global_seed, setup_logger,
)

SCRIPT_NAME = "01_preprocess"

# Filtering thresholds — keep here so the methods section can quote them.
MIN_TOKEN_COUNT = 5         # discard near-empty reviews
MAX_TOKEN_COUNT = 512       # cut extreme outliers (also fits MiniLM-L6-v2 context)
VALID_RATINGS = {1, 2, 3, 4, 5}

# Language ID (fastText lid.176). DistilBERT SST-2 is English-only; mixing other
# languages produces artefactual mismatches (see methods discussion).
LANGID_MODEL_PATH = "outputs/models/lid.176.bin"
LANGID_TARGET = "en"
LANGID_MIN_CONF = 0.65       # below this we treat language as unknown -> drop


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def token_count(text: str) -> int:
    return len(text.split())


def main() -> None:
    set_global_seed(GLOBAL_SEED)
    log = setup_logger(SCRIPT_NAME)
    versions = dump_env(log)

    log.info("Reading %s", REVIEWS_CSV)
    reviews = pd.read_csv(REVIEWS_CSV, dtype=str, keep_default_na=False, na_values=[""])
    courses = pd.read_csv(COURSES_CSV, dtype=str, keep_default_na=False, na_values=[""])
    log.info("Raw shape: reviews=%s courses=%s", reviews.shape, courses.shape)

    drop_log: list[dict] = []
    n0 = len(reviews)

    # ---- 1. Required columns must be non-null
    required = ["reviews", "rating", "course_id"]
    mask_required = reviews[required].notna().all(axis=1)
    n_missing_required = int((~mask_required).sum())
    reviews = reviews[mask_required].copy()
    drop_log.append({"step": "missing_required_fields", "removed": n_missing_required, "kept": len(reviews)})
    log.info("Dropped %d rows missing required fields", n_missing_required)

    # ---- 2. Rating must parse to {1..5}
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    mask_rating = reviews["rating"].isin(VALID_RATINGS)
    n_bad_rating = int((~mask_rating).sum())
    reviews = reviews[mask_rating].copy()
    reviews["rating"] = reviews["rating"].astype(int)
    drop_log.append({"step": "invalid_rating", "removed": n_bad_rating, "kept": len(reviews)})
    log.info("Dropped %d rows with invalid rating", n_bad_rating)

    # ---- 3. Whitespace + token-count filter
    reviews["reviews"] = reviews["reviews"].astype(str).map(normalize_whitespace)
    reviews["n_tokens"] = reviews["reviews"].map(token_count)
    mask_len = reviews["n_tokens"].between(MIN_TOKEN_COUNT, MAX_TOKEN_COUNT)
    n_bad_len = int((~mask_len).sum())
    reviews = reviews[mask_len].copy()
    drop_log.append({
        "step": f"token_count_outside[{MIN_TOKEN_COUNT},{MAX_TOKEN_COUNT}]",
        "removed": n_bad_len, "kept": len(reviews),
    })
    log.info("Dropped %d rows outside token range", n_bad_len)

    # ---- 4. Exact-duplicate review text within the same course
    before = len(reviews)
    reviews = reviews.drop_duplicates(subset=["course_id", "reviews"]).copy()
    n_dups = before - len(reviews)
    drop_log.append({"step": "duplicate_review_text_per_course", "removed": n_dups, "kept": len(reviews)})
    log.info("Dropped %d duplicate (course_id, review) pairs", n_dups)

    # ---- 4b. Language ID — keep only high-confidence English reviews.
    # Reason: DistilBERT SST-2 is English-only; non-English reviews produce
    # systematic false negatives in step 07 and would inflate "mismatch".
    log.info("Running fastText language ID (target=%s, min_conf=%.2f) on %d rows ...",
             LANGID_TARGET, LANGID_MIN_CONF, len(reviews))
    import fasttext
    fasttext.FastText.eprint = lambda x: None
    ft = fasttext.load_model(LANGID_MODEL_PATH)
    texts = reviews["reviews"].str.replace("\n", " ", regex=False).tolist()
    langs = []
    confs = []
    BATCH = 5000
    for i in range(0, len(texts), BATCH):
        labs, probs = ft.predict(texts[i:i + BATCH], k=1)
        langs.extend([lab[0].replace("__label__", "") for lab in labs])
        confs.extend([float(p[0]) for p in probs])
    reviews["lang"] = langs
    reviews["lang_conf"] = confs
    mask_lang = (reviews["lang"] == LANGID_TARGET) & (reviews["lang_conf"] >= LANGID_MIN_CONF)
    n_non_en = int((~mask_lang).sum())
    lang_breakdown = (reviews.loc[~mask_lang, "lang"].value_counts().head(10).to_dict())
    reviews = reviews[mask_lang].copy()
    drop_log.append({
        "step": f"language_filter (keep {LANGID_TARGET}, conf>={LANGID_MIN_CONF})",
        "removed": n_non_en, "kept": len(reviews),
        "note": f"top-10 dropped languages: {lang_breakdown}",
    })
    log.info("Dropped %d non-English reviews; top dropped languages: %s",
             n_non_en, lang_breakdown)

    # ---- 5. Parse date (best-effort; keep rows even if unparseable)
    reviews["date_reviews"] = pd.to_datetime(reviews["date_reviews"], errors="coerce", format="%b %d, %Y")
    n_bad_date = int(reviews["date_reviews"].isna().sum())
    log.info("Unparseable dates retained: %d (date kept as NaT)", n_bad_date)

    # ---- 6. Join with course metadata (left, keep all reviews even if metadata missing)
    reviews = reviews.merge(courses, on="course_id", how="left", validate="m:1")
    n_no_course = int(reviews["name"].isna().sum())
    log.info("Rows with no matching course metadata: %d", n_no_course)
    drop_log.append({"step": "no_course_metadata (kept, flagged)", "removed": 0, "kept": len(reviews),
                     "note": f"{n_no_course} rows have NaN course name"})

    # ---- 7. Stable doc_id so embeddings can be re-aligned by index
    reviews = reviews.reset_index(drop=True)
    reviews["doc_id"] = reviews.index.astype(int)

    n1 = len(reviews)
    log.info("Final shape: %s (kept %d of %d, %.2f%%)", reviews.shape, n1, n0, 100 * n1 / n0)

    # ---- Write outputs
    reviews.to_parquet(PREPROCESSED_PARQUET, index=False)
    log.info("Wrote %s", PREPROCESSED_PARQUET)

    drop_df = pd.DataFrame(drop_log)
    drop_df.to_csv(TABLE_DIR / "01_preprocess_drop_log.csv", index=False)
    log.info("Wrote drop log to outputs/tables/01_preprocess_drop_log.csv")

    # Summary stats for methods section
    summary = {
        "raw_reviews": int(n0),
        "final_reviews": int(n1),
        "kept_pct": round(100 * n1 / n0, 4),
        "rating_distribution": reviews["rating"].value_counts().sort_index().to_dict(),
        "unique_courses": int(reviews["course_id"].nunique()),
        "date_range": [
            str(reviews["date_reviews"].min()),
            str(reviews["date_reviews"].max()),
        ],
        "seed": GLOBAL_SEED,
        "filters": {
            "min_tokens": MIN_TOKEN_COUNT,
            "max_tokens": MAX_TOKEN_COUNT,
            "valid_ratings": sorted(VALID_RATINGS),
        },
        "versions": versions,
    }
    (TABLE_DIR / "01_preprocess_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    log.info("Summary: %s", json.dumps(summary, indent=2, default=str))

    RunRecord(
        script=SCRIPT_NAME, seed=GLOBAL_SEED,
        params={"min_tokens": MIN_TOKEN_COUNT, "max_tokens": MAX_TOKEN_COUNT},
        n_inputs=n0, n_outputs=n1,
    ).save(OUTPUT_DIR / "01_preprocess.run.json")


if __name__ == "__main__":
    main()
