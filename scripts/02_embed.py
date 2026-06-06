"""
Step 2 — Sentence embeddings (computed ONCE on MPS/CUDA/CPU, cached to disk).

Why this is its own script: the stability sweep (5 seeds) and the parameter
grid (6 sizes) each re-fit BERTopic, but the *embeddings* never change. Caching
once turns ~30 fits from 'unfeasible' into 'overnight'.

Output:
    outputs/embeddings/all-MiniLM-L6-v2.npy        (float32 NxD matrix)
    outputs/embeddings/embedding_index.parquet     (doc_id alignment)

Usage:
    python scripts/02_embed.py [--batch-size 256]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    EMBEDDINGS_NPY, EMBEDDING_INDEX_PARQUET, EMBEDDING_MODEL_NAME,
    GLOBAL_SEED, OUTPUT_DIR, PREPROCESSED_PARQUET, RunRecord, TABLE_DIR,
    dump_env, get_torch_device, set_global_seed, setup_logger,
)

SCRIPT_NAME = "02_embed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    args = parser.parse_args()

    set_global_seed(GLOBAL_SEED)
    log = setup_logger(SCRIPT_NAME)
    dump_env(log)

    if EMBEDDINGS_NPY.exists() and not args.force:
        emb = np.load(EMBEDDINGS_NPY, mmap_mode="r")
        log.info("Cache HIT: %s shape=%s — pass --force to recompute", EMBEDDINGS_NPY, emb.shape)
        return

    log.info("Loading preprocessed reviews from %s", PREPROCESSED_PARQUET)
    df = pd.read_parquet(PREPROCESSED_PARQUET, columns=["doc_id", "reviews"])
    log.info("Encoding %d documents", len(df))

    device = get_torch_device()
    log.info("Embedding model: %s | device: %s | batch_size=%d",
             EMBEDDING_MODEL_NAME, device, args.batch_size)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    # MiniLM-L6-v2 already truncates to 256 by default; explicit is better.
    model.max_seq_length = 256

    t0 = time.time()
    embeddings = model.encode(
        df["reviews"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # cosine handled later inside UMAP
    ).astype(np.float32)
    dt = time.time() - t0
    log.info("Encoded in %.1fs (%.1f docs/s)", dt, len(df) / dt)

    np.save(EMBEDDINGS_NPY, embeddings)
    df[["doc_id"]].to_parquet(EMBEDDING_INDEX_PARQUET, index=False)
    log.info("Saved embeddings %s shape=%s", EMBEDDINGS_NPY, embeddings.shape)

    meta = {
        "model": EMBEDDING_MODEL_NAME,
        "device": device,
        "n_docs": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
        "batch_size": args.batch_size,
        "duration_seconds": round(dt, 1),
        "seed": GLOBAL_SEED,
    }
    (TABLE_DIR / "02_embed_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    RunRecord(script=SCRIPT_NAME, seed=GLOBAL_SEED, params=meta,
              n_inputs=len(df), n_outputs=len(df)).save(OUTPUT_DIR / "02_embed.run.json")


if __name__ == "__main__":
    main()
