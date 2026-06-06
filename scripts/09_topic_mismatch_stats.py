"""
Step 8 — Statistical inference for topic x mismatch.

Replaces 'topic A looks higher in the bar chart' with proper tests:
  - Kruskal-Wallis omnibus over topics
  - Dunn's post-hoc with Bonferroni correction on pairwise comparisons
  - Per-pair effect size (rank-biserial r from Mann-Whitney U)
  - 95% bootstrap CIs for each topic's median mismatch

Continuous mismatch score is the 'raw' variant from step 08 by default;
override with --score-col.

Usage:
    python scripts/09_topic_mismatch_stats.py [--score-col mismatch_raw]
        [--min-topic-n 30] [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from config import GLOBAL_SEED, TABLE_DIR, dump_env, set_global_seed, setup_logger

SCRIPT_NAME = "09_topic_mismatch_stats"


def rank_biserial(u: float, n1: int, n2: int) -> float:
    """Effect size for Mann-Whitney U; range [-1, 1]."""
    return 1.0 - (2.0 * u) / (n1 * n2)


def bootstrap_median_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator,
                         alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    medians = np.median(values[idx], axis=1)
    return float(np.quantile(medians, alpha / 2)), float(np.quantile(medians, 1 - alpha / 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-col", default="mismatch_raw",
                        choices=["mismatch_raw", "mismatch_rank", "mismatch_zscore", "mismatch_binary"])
    parser.add_argument("--min-topic-n", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--include-outliers", action="store_true",
                        help="Include topic == -1 in the omnibus test.")
    args = parser.parse_args()

    set_global_seed(GLOBAL_SEED)
    rng = np.random.default_rng(GLOBAL_SEED)
    log = setup_logger(SCRIPT_NAME)
    dump_env(log)

    scores = pd.read_parquet(TABLE_DIR / "08_mismatch_scores.parquet",
                              columns=["doc_id", args.score_col])
    topics = pd.read_parquet(TABLE_DIR / "05_doc_topics.parquet",
                              columns=["doc_id", "topic"])
    df = scores.merge(topics, on="doc_id", how="inner", validate="1:1")
    df = df.dropna(subset=[args.score_col]).copy()
    if not args.include_outliers:
        df = df[df["topic"] != -1].copy()
    log.info("Rows after join/filter: %d", len(df))

    counts = df.groupby("topic").size()
    kept_topics = counts[counts >= args.min_topic_n].index.tolist()
    df = df[df["topic"].isin(kept_topics)].copy()
    log.info("Topics with >= %d docs: %d", args.min_topic_n, len(kept_topics))

    # ---- Per-topic descriptives + bootstrap CI on the median
    desc_rows = []
    groups = {}
    for tid, g in df.groupby("topic"):
        vals = g[args.score_col].values
        groups[tid] = vals
        lo, hi = bootstrap_median_ci(vals, args.n_boot, rng)
        desc_rows.append({
            "topic_id": int(tid),
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "median_ci_lo": lo,
            "median_ci_hi": hi,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        })
    desc_df = pd.DataFrame(desc_rows).sort_values("median", ascending=False)
    desc_df.to_csv(TABLE_DIR / "09_topic_mismatch_descriptives.csv", index=False)

    # ---- Kruskal-Wallis omnibus
    H, p = kruskal(*groups.values())
    log.info("Kruskal-Wallis: H=%.4f, p=%.4g, k=%d groups", H, p, len(groups))

    # ---- Dunn's post-hoc with Bonferroni correction
    try:
        import scikit_posthocs as sp
        dunn = sp.posthoc_dunn(df, val_col=args.score_col, group_col="topic", p_adjust="bonferroni")
        dunn.to_csv(TABLE_DIR / "09_dunn_posthoc_pvalues.csv")
        log.info("Wrote Dunn post-hoc matrix")
    except ImportError:
        log.warning("scikit-posthocs not installed; skipping Dunn")
        dunn = None

    # ---- Pairwise effect sizes (rank-biserial r) — capped to keep file small
    tids = sorted(groups.keys())
    pair_rows = []
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            u_stat, p_val = mannwhitneyu(groups[a], groups[b], alternative="two-sided")
            pair_rows.append({
                "topic_a": int(a), "topic_b": int(b),
                "n_a": int(len(groups[a])), "n_b": int(len(groups[b])),
                "U": float(u_stat),
                "p_uncorrected": float(p_val),
                "rank_biserial_r": round(rank_biserial(u_stat, len(groups[a]), len(groups[b])), 4),
            })
    pd.DataFrame(pair_rows).to_csv(TABLE_DIR / "09_pairwise_effect_sizes.csv", index=False)

    summary = {
        "score_col": args.score_col,
        "include_outliers": args.include_outliers,
        "n_topics_tested": len(groups),
        "n_obs_total": int(sum(len(v) for v in groups.values())),
        "kruskal_H": float(H),
        "kruskal_p": float(p),
        "min_topic_n": args.min_topic_n,
        "n_boot": args.n_boot,
        "seed": GLOBAL_SEED,
    }
    (TABLE_DIR / "09_topic_mismatch_stats_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
