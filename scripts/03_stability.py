"""
Step 2 (validity/reliability) — BERTopic stability across 5 seeds.

Fits BERTopic with identical hyperparameters but different random_state values,
then quantifies how much topic assignments shift between runs using ARI and NMI
on every pair. Outputs the per-run metrics and the pairwise similarity matrix.

Usage:
    python scripts/03_stability.py [--min-topic-size 50]
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from bertopic_utils import build_topic_model, outlier_ratio
from config import (
    EMBEDDINGS_NPY, OUTPUT_DIR, PREPROCESSED_PARQUET, RunRecord,
    STABILITY_SEEDS, TABLE_DIR, dump_env, set_global_seed, setup_logger,
)

SCRIPT_NAME = "03_stability"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-topic-size", type=int, default=50,
                        help="Held constant across seeds — varying only random_state.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(STABILITY_SEEDS))
    args = parser.parse_args()

    log = setup_logger(SCRIPT_NAME)
    dump_env(log)
    log.info("Seeds: %s | min_topic_size=%d", args.seeds, args.min_topic_size)

    log.info("Loading docs + embeddings")
    df = pd.read_parquet(PREPROCESSED_PARQUET, columns=["doc_id", "reviews"])
    embeddings = np.load(EMBEDDINGS_NPY, mmap_mode="r")
    assert len(df) == len(embeddings), f"length mismatch {len(df)} vs {len(embeddings)}"
    docs = df["reviews"].tolist()

    per_run: list[dict] = []
    label_matrix = np.full((len(args.seeds), len(docs)), fill_value=-99, dtype=np.int32)

    for i, seed in enumerate(args.seeds):
        set_global_seed(seed)
        log.info("[run %d/%d] seed=%d fitting...", i + 1, len(args.seeds), seed)
        t0 = time.time()
        model = build_topic_model(seed=seed, min_topic_size=args.min_topic_size, verbose=False)
        topics, _ = model.fit_transform(docs, embeddings=np.asarray(embeddings))
        dt = time.time() - t0
        labels = np.asarray(topics, dtype=np.int32)
        label_matrix[i] = labels
        n_topics = int(len(set(labels)) - (1 if -1 in labels else 0))
        per_run.append({
            "seed": seed,
            "n_topics": n_topics,
            "outlier_ratio": round(outlier_ratio(labels), 4),
            "fit_seconds": round(dt, 1),
        })
        log.info("  -> n_topics=%d outliers=%.2f%% in %.1fs",
                 n_topics, 100 * outlier_ratio(labels), dt)
        # Persist raw labels so analysis can be re-run without re-fitting.
        np.save(OUTPUT_DIR / f"stability_labels_seed{seed}.npy", labels)

    runs_df = pd.DataFrame(per_run)
    runs_df.to_csv(TABLE_DIR / "03_stability_runs.csv", index=False)
    log.info("Per-run summary:\n%s", runs_df.to_string(index=False))

    # ---- Pairwise ARI / NMI
    pair_rows = []
    for (i, sa), (j, sb) in combinations(enumerate(args.seeds), 2):
        a, b = label_matrix[i], label_matrix[j]
        pair_rows.append({
            "seed_a": sa, "seed_b": sb,
            "ARI": round(adjusted_rand_score(a, b), 4),
            "NMI": round(normalized_mutual_info_score(a, b), 4),
        })
    pairs_df = pd.DataFrame(pair_rows)
    pairs_df.to_csv(TABLE_DIR / "03_stability_pairwise.csv", index=False)
    log.info("Pairwise ARI/NMI:\n%s", pairs_df.to_string(index=False))

    summary = {
        "seeds": args.seeds,
        "min_topic_size": args.min_topic_size,
        "n_topics_mean": float(runs_df["n_topics"].mean()),
        "n_topics_std": float(runs_df["n_topics"].std(ddof=1)) if len(runs_df) > 1 else 0.0,
        "outlier_ratio_mean": float(runs_df["outlier_ratio"].mean()),
        "outlier_ratio_std": float(runs_df["outlier_ratio"].std(ddof=1)) if len(runs_df) > 1 else 0.0,
        "ARI_mean": float(pairs_df["ARI"].mean()),
        "ARI_std": float(pairs_df["ARI"].std(ddof=1)) if len(pairs_df) > 1 else 0.0,
        "NMI_mean": float(pairs_df["NMI"].mean()),
        "NMI_std": float(pairs_df["NMI"].std(ddof=1)) if len(pairs_df) > 1 else 0.0,
    }
    (TABLE_DIR / "03_stability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("Summary: %s", json.dumps(summary, indent=2))

    RunRecord(script=SCRIPT_NAME, seed=args.seeds[0],
              params={"min_topic_size": args.min_topic_size, "seeds": args.seeds},
              n_inputs=len(docs), n_outputs=len(args.seeds), extra=summary,
              ).save(OUTPUT_DIR / "03_stability.run.json")


if __name__ == "__main__":
    main()
