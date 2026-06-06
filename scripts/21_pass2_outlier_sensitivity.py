"""
Reviewer comment 2e — sensitivity of the Pass-2 (hidden complaint tree)
*typology composition* to the outlier-handling policy.

We already report Spearman ρ = 0.986 for outlier sensitivity on the
mismatch ranking (Methods §4.7). The reviewer asks whether the
4-macro / 12-meso typology itself is sensitive to the policy.

Procedure
---------
1. Load the existing Pass-2 BERTopic model (saved by step 13).
2. Three policies for the 8,358 Pass-2 outlier reviews:
       (a) keep        — current paper (n = 16,916 clustered)
       (b) drop        — same numbers; report for completeness
       (c) reassign    — reduce_outliers(strategy="embeddings")
3. For (c), recompute macro-level totals: how do the four macros'
   counts shift once outliers are assigned to their nearest non-outlier
   micro topic?
4. Emit a CSV + JSON the manuscript can cite verbatim.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bertopic import BERTopic

TBL = Path("outputs/tables")
EMB = Path("outputs/embeddings/all-MiniLM-L6-v2.npy")
MODEL_DIR = Path("outputs/models/low_rating_bertopic")
DOCS_PARQUET = Path("outputs/reviews_clean.parquet")
OUT_CSV = TBL / "21_pass2_typology_after_reassign.csv"
OUT_JSON = TBL / "21_pass2_typology_sensitivity.json"


def main() -> None:
    print("Loading docs + Pass-2 model + level labels ...")
    docs_df = pd.read_parquet(DOCS_PARQUET, columns=["doc_id", "reviews", "rating"])
    sub_df = docs_df[docs_df.rating <= 3].reset_index(drop=False)
    emb_full = np.load(EMB, mmap_mode="r")
    sub_emb = np.asarray(emb_full)[sub_df["index"].values]
    sub_docs = sub_df["reviews"].tolist()

    bt = BERTopic.load(str(MODEL_DIR))
    topics = bt.topics_
    lvl = pd.read_csv(TBL / "13_low_rating_level_labels.csv")
    tl = pd.read_csv(TBL / "13_low_rating_topic_to_levels.csv")
    macro = lvl[lvl.level == "makro"][["cluster_id", "label"]].rename(
        columns={"cluster_id": "makro_id", "label": "Macro_label"})
    meso = lvl[lvl.level == "mezo"][["cluster_id", "label"]].rename(
        columns={"cluster_id": "mezo_id", "label": "Meso_label"})

    # Baseline composition: outliers kept as their own group
    base = pd.DataFrame({"topic": np.asarray(topics)})
    base["doc_id"] = sub_df["doc_id"].values

    base_clust = base[base.topic != -1].merge(
        tl[["topic_id", "mezo_id", "makro_id"]],
        left_on="topic", right_on="topic_id")
    base_macro = base_clust.groupby("makro_id").size().rename("baseline_n")
    base_meso = base_clust.groupby(["makro_id", "mezo_id"]).size().rename("baseline_n_meso")

    n_outliers = int((base.topic == -1).sum())
    print(f"Pass-2 outliers (-1): {n_outliers}")
    print("Reassigning with strategy='embeddings' …")

    new_topics = bt.reduce_outliers(
        sub_docs, list(topics),
        strategy="embeddings", embeddings=sub_emb,
    )
    print("Reassignment done.")

    after = pd.DataFrame({"topic": np.asarray(new_topics, dtype=int),
                            "doc_id": sub_df["doc_id"].values})
    n_still_out = int((after.topic == -1).sum())
    print(f"Still outlier after reassign: {n_still_out}")

    after_clust = after[after.topic != -1].merge(
        tl[["topic_id", "mezo_id", "makro_id"]],
        left_on="topic", right_on="topic_id")
    after_macro = after_clust.groupby("makro_id").size().rename("after_n")
    after_meso = after_clust.groupby(["makro_id", "mezo_id"]).size().rename("after_n_meso")

    macro_cmp = (pd.concat([base_macro, after_macro], axis=1).fillna(0).astype(int)
                   .reset_index()
                   .merge(macro, on="makro_id"))
    macro_cmp["delta_n"] = macro_cmp["after_n"] - macro_cmp["baseline_n"]
    macro_cmp["baseline_pct_of_clustered"] = (
        100 * macro_cmp["baseline_n"] / macro_cmp["baseline_n"].sum()).round(2)
    macro_cmp["after_pct_of_clustered"] = (
        100 * macro_cmp["after_n"] / macro_cmp["after_n"].sum()).round(2)
    macro_cmp = macro_cmp.sort_values("after_n", ascending=False)
    macro_cmp.to_csv(OUT_CSV, index=False)

    # Spearman rho on per-macro size ranking — but with only 4 macros this is
    # less informative than the raw % shift. We compute both.
    from scipy.stats import spearmanr
    rho_macro, p_macro = spearmanr(macro_cmp["baseline_n"], macro_cmp["after_n"])

    meso_cmp = pd.concat([base_meso, after_meso], axis=1).fillna(0).astype(int)
    rho_meso, p_meso = spearmanr(meso_cmp["baseline_n_meso"], meso_cmp["after_n_meso"])

    summary = {
        "design_note": (
            "Sensitivity of the Pass-2 macro and meso composition to the "
            "HDBSCAN outlier policy. The baseline figures (which the "
            "manuscript uses) keep the 8,358 Pass-2 outliers as their own "
            "group; here we reassign every outlier to its nearest non-"
            "outlier topic via the embedding centroid and recompute the "
            "macro and meso counts."),
        "n_low_rating_total": int(len(sub_df)),
        "n_outliers_pass2": n_outliers,
        "n_still_outlier_after_reassign": n_still_out,
        "macro_level": {
            "spearman_rho_size_ranking": round(rho_macro, 3),
            "spearman_p": float(p_macro),
            "macro_table_csv": str(OUT_CSV),
        },
        "meso_level": {
            "spearman_rho_size_ranking": round(rho_meso, 3),
            "spearman_p": float(p_meso),
        },
        "interpretation_macro": macro_cmp.to_dict("records"),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print()
    print("Macro composition before vs after reassign:")
    print(macro_cmp[["makro_id", "Macro_label",
                       "baseline_n", "after_n", "delta_n",
                       "baseline_pct_of_clustered", "after_pct_of_clustered"]
                      ].to_string(index=False))
    print()
    print(f"Spearman ρ (macro size ranking): {rho_macro:.3f}")
    print(f"Spearman ρ (meso size ranking):  {rho_meso:.3f}")


if __name__ == "__main__":
    main()
