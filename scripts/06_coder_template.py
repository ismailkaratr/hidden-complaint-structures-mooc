"""
Step 5a — Build the inter-coder template.

Produces a single CSV that two independent human coders fill in. Each row =
one topic with: id, top-10 keywords, 5 representative docs, and two empty
columns ('label_coder_A', 'label_coder_B'). Coders edit the same file (or two
copies); step 06b reads it back and computes agreement.

Usage:
    python scripts/06_coder_template.py [--top-docs 5]
"""
from __future__ import annotations

import argparse

import pandas as pd

from config import CODER_DIR, TABLE_DIR, dump_env, setup_logger

SCRIPT_NAME = "06_coder_template"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-docs", type=int, default=5)
    args = parser.parse_args()

    log = setup_logger(SCRIPT_NAME)
    dump_env(log)

    kw = pd.read_csv(TABLE_DIR / "05_topic_keywords.csv")
    reps = pd.read_csv(TABLE_DIR / "05_topic_representative_docs.csv")
    info = pd.read_csv(TABLE_DIR / "05_topic_info.csv")

    kw_top = (kw.sort_values(["topic_id", "rank"])
                .groupby("topic_id")["word"].apply(lambda s: ", ".join(s.head(10))))

    reps_top = (reps.sort_values(["topic_id", "rank"])
                    .groupby("topic_id")["doc"]
                    .apply(lambda s: list(s.head(args.top_docs))))

    rows = []
    for tid in sorted(kw_top.index):
        rep_list = reps_top.get(tid, [])
        row = {
            "topic_id": int(tid),
            "count": int(info.loc[info["Topic"] == tid, "Count"].values[0]) if (info["Topic"] == tid).any() else None,
            "top_keywords": kw_top[tid],
        }
        for i in range(args.top_docs):
            row[f"rep_doc_{i+1}"] = rep_list[i] if i < len(rep_list) else ""
        row["label_coder_A"] = ""
        row["label_coder_B"] = ""
        row["notes"] = ""
        rows.append(row)

    out = CODER_DIR / "06_coder_template.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("Wrote template with %d topics -> %s", len(rows), out)
    log.info("Coders: open in a spreadsheet, fill label_coder_A and label_coder_B, save, "
             "then run 06b_coder_agreement.py.")


if __name__ == "__main__":
    main()
