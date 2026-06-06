"""
Step 6c — Validate the sentiment classifier against the human gold labels.

Reads the filled validation template and the model predictions, joins by
doc_id, then reports accuracy, precision/recall/F1 per class, macro average,
and the confusion matrix. If the model is too weak, mismatch findings would
be invalid — this is the gate.

Usage:
    python scripts/07c_sentiment_validation_metrics.py
        [--template outputs/sentiment/07b_sentiment_validation_template.csv]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)

from config import SENT_DIR, TABLE_DIR, dump_env, setup_logger

SCRIPT_NAME = "07c_sentiment_validation_metrics"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default=str(SENT_DIR / "07b_sentiment_validation_template.csv"))
    parser.add_argument("--predictions",
                        default=str(SENT_DIR / "07_sentiment_predictions.parquet"))
    args = parser.parse_args()

    log = setup_logger(SCRIPT_NAME)
    dump_env(log)

    gold = pd.read_csv(args.template)
    preds = pd.read_parquet(args.predictions)

    gold["gold_label"] = gold["gold_label"].astype(str).str.strip().str.lower()
    gold = gold[gold["gold_label"].isin({"positive", "negative"})].copy()
    log.info("Gold-labeled rows after filtering: %d", len(gold))
    if len(gold) == 0:
        raise SystemExit("No usable gold labels. Fill 'gold_label' as positive/negative.")

    merged = gold.merge(preds, on="doc_id", how="left", validate="1:1")
    if merged["sent_label"].isna().any():
        missing = int(merged["sent_label"].isna().sum())
        log.warning("%d rows missing predictions — did you run 07_sentiment_score.py?", missing)
        merged = merged.dropna(subset=["sent_label"])

    y_true = merged["gold_label"].values
    y_pred = merged["sent_label"].astype(str).str.lower().values

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=["positive", "negative"], zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=["positive", "negative"])

    cm_df = pd.DataFrame(cm,
                         index=["true_positive", "true_negative"],
                         columns=["pred_positive", "pred_negative"])
    cm_df.to_csv(TABLE_DIR / "07_sentiment_confusion_matrix.csv")

    rep_dict = classification_report(
        y_true, y_pred, labels=["positive", "negative"],
        digits=4, output_dict=True, zero_division=0,
    )
    pd.DataFrame(rep_dict).T.to_csv(TABLE_DIR / "07_sentiment_classification_report.csv")

    summary = {
        "n_gold": int(len(merged)),
        "accuracy": round(acc, 4),
        "per_class": {
            "positive": {"precision": float(p[0]), "recall": float(r[0]),
                         "f1": float(f1[0]), "support": int(support[0])},
            "negative": {"precision": float(p[1]), "recall": float(r[1]),
                         "f1": float(f1[1]), "support": int(support[1])},
        },
        "macro_f1": float(rep_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(rep_dict["weighted avg"]["f1-score"]),
    }
    (TABLE_DIR / "07_sentiment_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("Summary: %s", json.dumps(summary, indent=2))
    if summary["macro_f1"] < 0.75:
        log.warning("macro_F1 < 0.75 — mismatch analysis findings should be reported with caution.")


if __name__ == "__main__":
    main()
