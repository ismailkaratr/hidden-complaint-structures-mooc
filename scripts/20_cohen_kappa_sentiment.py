"""
Cohen's kappa for Coder A (first author) vs Coder B (independent
relabelling of the 60 three-star validation rows by Claude Opus 4).

Two perspectives:
  (a) Strict — three categories {positive, negative, blank/abstain},
      so abstention also counts as a label decision.
  (b) Two-category — exclude rows where either coder abstained;
      compute kappa on the strict positive/negative subset, which is
      what the downstream classification metrics use.

Emit a small JSON file the manuscript can cite verbatim.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

SRC = Path("outputs/sentiment/07b_sentiment_validation_template.csv")
OUT = Path("outputs/tables/cohen_kappa_sentiment.json")


def main() -> None:
    df = pd.read_csv(SRC)
    three = df[df.rating == 3].copy()
    A = three["gold_label"].fillna("blank")
    B = three["coder_B"].fillna("blank")
    n_total = len(three)

    # Strict three-class kappa — abstention counts as a label decision.
    # We deliberately do not report a two-class subset kappa: restricting to
    # rows where both coders committed would exclude exactly the ambiguous
    # cases where coder judgement matters and would overstate agreement.
    k_strict = cohen_kappa_score(A, B,
                                   labels=["positive", "negative", "blank"])
    n_agree_strict = int((A == B).sum())
    cm = confusion_matrix(A, B,
                           labels=["positive", "negative", "blank"])

    summary = {
        "design_note": (
            "60 three-star reviews from the validation template were "
            "labelled independently by Coder A (first author) and Coder B "
            "(Claude Opus 4, blind to Coder A's choices). The reliability "
            "figure reported in the manuscript is the strict three-class "
            "Cohen's kappa, which counts abstention as a label decision. "
            "We deliberately do not report the two-class subset kappa: "
            "restricting to rows where both coders committed to a polarity "
            "would exclude precisely the ambiguous cases where coder "
            "judgement matters most, and would therefore overstate "
            "agreement."),
        "n_three_star": n_total,
        "labels": ["positive", "negative", "blank"],
        "strict_three_class": {
            "n": n_total,
            "agreement_n": n_agree_strict,
            "agreement_pct": round(n_agree_strict / n_total * 100, 2),
            "cohen_kappa": round(k_strict, 3),
        },
        "confusion_matrix_strict": {
            "rows_coder_A": ["positive", "negative", "blank"],
            "cols_coder_B": ["positive", "negative", "blank"],
            "matrix": cm.tolist(),
        },
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
