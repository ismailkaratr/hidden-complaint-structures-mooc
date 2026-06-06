"""
Step 5b — Inter-coder agreement.

Reads the filled template (06_coder_template.csv with label_coder_A and
label_coder_B populated) and reports:
  - Cohen's kappa on the raw labels (treated as nominal categories)
  - Exact match rate
  - Confusion matrix written to CSV

If coders used free-text labels rather than a fixed code list, kappa on raw
strings is meaningful only if the label set is small/agreed in advance. We
also dump the disagreement rows so the team can adjudicate.

Usage:
    python scripts/06b_coder_agreement.py [--template path]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from config import CODER_DIR, TABLE_DIR, dump_env, setup_logger

SCRIPT_NAME = "06b_coder_agreement"


def _normalize(label: str) -> str:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return ""
    return str(label).strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default=str(CODER_DIR / "06_coder_template.csv"))
    args = parser.parse_args()

    log = setup_logger(SCRIPT_NAME)
    dump_env(log)

    df = pd.read_csv(args.template)
    for col in ("label_coder_A", "label_coder_B"):
        if col not in df.columns:
            raise SystemExit(f"Template is missing column: {col}")

    df["A"] = df["label_coder_A"].map(_normalize)
    df["B"] = df["label_coder_B"].map(_normalize)

    filled = df[(df["A"] != "") & (df["B"] != "")].copy()
    n_total, n_filled = len(df), len(filled)
    log.info("Template rows: %d | both-coders-filled: %d", n_total, n_filled)
    if n_filled == 0:
        raise SystemExit("No rows have both coders filled in. Nothing to score.")

    labels = sorted(set(filled["A"]) | set(filled["B"]))
    kappa = cohen_kappa_score(filled["A"], filled["B"], labels=labels)
    match_rate = float((filled["A"] == filled["B"]).mean())
    cm = confusion_matrix(filled["A"], filled["B"], labels=labels)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(TABLE_DIR / "06_coder_confusion_matrix.csv")

    disagree = filled.loc[filled["A"] != filled["B"],
                          ["topic_id", "top_keywords", "label_coder_A", "label_coder_B"]]
    disagree.to_csv(TABLE_DIR / "06_coder_disagreements.csv", index=False)

    summary = {
        "n_topics": int(n_total),
        "n_double_coded": int(n_filled),
        "cohen_kappa": round(float(kappa), 4),
        "exact_match_rate": round(match_rate, 4),
        "n_unique_labels": len(labels),
        "n_disagreements": int(len(disagree)),
    }
    (TABLE_DIR / "06_coder_agreement_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
