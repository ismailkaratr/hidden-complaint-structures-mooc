"""
Step 3 — Parameter grid for min_topic_size.

Sweeps min_topic_size in {15, 25, 50, 100, 150, 200} with seed fixed, and for
each value records: number of topics, outlier ratio, c_v, c_npmi, topic
diversity, and fit time. The CSV is the input the human uses to make an
*informed* choice — this script never picks a 'winner' on its own.

Usage:
    python scripts/04_param_grid.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from bertopic_utils import build_topic_model, outlier_ratio, topic_diversity
from coherence_utils import compute_coherence
from config import (
    EMBEDDINGS_NPY, GLOBAL_SEED, MIN_TOPIC_SIZE_GRID, OUTPUT_DIR,
    PREPROCESSED_PARQUET, RunRecord, TABLE_DIR, dump_env,
    set_global_seed, setup_logger,
)

SCRIPT_NAME = "04_param_grid"


def main() -> None:
    log = setup_logger(SCRIPT_NAME)
    dump_env(log)

    log.info("Loading docs + embeddings")
    df = pd.read_parquet(PREPROCESSED_PARQUET, columns=["doc_id", "reviews"])
    embeddings = np.load(EMBEDDINGS_NPY, mmap_mode="r")
    docs = df["reviews"].tolist()

    rows = []
    for mts in MIN_TOPIC_SIZE_GRID:
        set_global_seed(GLOBAL_SEED)  # identical UMAP init across grid points
        log.info("--- min_topic_size=%d ---", mts)
        t0 = time.time()
        model = build_topic_model(seed=GLOBAL_SEED, min_topic_size=mts, verbose=False)
        topics, _ = model.fit_transform(docs, embeddings=np.asarray(embeddings))
        dt = time.time() - t0
        labels = np.asarray(topics, dtype=np.int32)

        n_topics = int(len(set(labels)) - (1 if -1 in labels else 0))
        outlier = outlier_ratio(labels)

        topic_words = [
            [w for w, _ in model.get_topics()[tid]][:10]
            for tid in model.get_topics() if tid != -1
        ]
        diversity = topic_diversity(topic_words, top_k=10)

        coh = compute_coherence(topic_model=model, docs=docs, top_k=10)

        row = {
            "min_topic_size": mts,
            "seed": GLOBAL_SEED,
            "n_topics": n_topics,
            "outlier_ratio": round(outlier, 4),
            "c_v": round(coh["c_v"], 4),
            "c_npmi": round(coh["c_npmi"], 4),
            "topic_diversity": round(diversity, 4),
            "fit_seconds": round(dt, 1),
        }
        log.info("Result: %s", row)
        rows.append(row)

    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(TABLE_DIR / "04_param_grid.csv", index=False)
    log.info("Grid:\n%s", grid_df.to_string(index=False))

    (TABLE_DIR / "04_param_grid_meta.json").write_text(
        json.dumps({"seed": GLOBAL_SEED, "grid": list(MIN_TOPIC_SIZE_GRID)}, indent=2),
        encoding="utf-8",
    )
    RunRecord(script=SCRIPT_NAME, seed=GLOBAL_SEED,
              params={"grid": list(MIN_TOPIC_SIZE_GRID)},
              n_inputs=len(docs), n_outputs=len(rows)
              ).save(OUTPUT_DIR / "04_param_grid.run.json")
    log.info("DONE. Open outputs/tables/04_param_grid.csv and pick a value by hand.")


if __name__ == "__main__":
    main()
