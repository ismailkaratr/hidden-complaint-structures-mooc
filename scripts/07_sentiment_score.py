"""
Step 6a — Run the DistilBERT SST-2 sentiment classifier on every review.

We store both the predicted label ('positive' / 'negative') and the raw model
probability for positive, so step 08 can replay different thresholds without
recomputing.

Output:
    outputs/sentiment/07_sentiment_predictions.parquet
        columns = doc_id, sent_label, sent_pos_prob

Usage:
    python scripts/07_sentiment_score.py [--batch-size 64]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    GLOBAL_SEED, OUTPUT_DIR, PREPROCESSED_PARQUET, RunRecord, SENT_DIR,
    SENTIMENT_MODEL_NAME, dump_env, get_torch_device, set_global_seed,
    setup_logger,
)

SCRIPT_NAME = "07_sentiment_score"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    set_global_seed(GLOBAL_SEED)
    log = setup_logger(SCRIPT_NAME)
    dump_env(log)

    device = get_torch_device()
    log.info("Sentiment model: %s | device: %s", SENTIMENT_MODEL_NAME, device)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME).to(device)
    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    pos_id = next(k for k, v in id2label.items() if v.startswith("pos"))
    log.info("Label map: %s | positive id=%d", id2label, pos_id)

    df = pd.read_parquet(PREPROCESSED_PARQUET, columns=["doc_id", "reviews"])
    texts = df["reviews"].tolist()
    log.info("Scoring %d reviews", len(texts))

    pos_probs = np.empty(len(texts), dtype=np.float32)
    labels = np.empty(len(texts), dtype=object)

    t0 = time.time()
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), args.batch_size)):
            batch = texts[start:start + args.batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=args.max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            p_pos = probs[:, pos_id]
            pos_probs[start:start + len(batch)] = p_pos
            preds = np.argmax(probs, axis=-1)
            labels[start:start + len(batch)] = ["positive" if i == pos_id else "negative" for i in preds]
    dt = time.time() - t0
    log.info("Scored in %.1fs (%.1f docs/s)", dt, len(texts) / dt)

    out = pd.DataFrame({
        "doc_id": df["doc_id"].values,
        "sent_label": labels,
        "sent_pos_prob": pos_probs,
    })
    out.to_parquet(SENT_DIR / "07_sentiment_predictions.parquet", index=False)
    log.info("Saved -> outputs/sentiment/07_sentiment_predictions.parquet")

    meta = {
        "model": SENTIMENT_MODEL_NAME, "device": device,
        "n_docs": int(len(texts)), "batch_size": args.batch_size,
        "max_length": args.max_length, "duration_seconds": round(dt, 1),
        "seed": GLOBAL_SEED,
    }
    (SENT_DIR / "07_sentiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    RunRecord(script=SCRIPT_NAME, seed=GLOBAL_SEED, params=meta,
              n_inputs=len(texts), n_outputs=len(texts)
              ).save(OUTPUT_DIR / "07_sentiment_score.run.json")


if __name__ == "__main__":
    main()
