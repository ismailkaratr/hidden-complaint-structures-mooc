"""
Step 12 — Hierarchical (mikro / mezo / makro) topic structure.

Uses BERTopic's hierarchical_topics() to build a Ward-linkage dendrogram over
the 250 mikro topics, then cuts the tree at two thresholds to produce a
*mezo* level (~20-30 groups) and a *makro* level (~5-8 groups). Each level is
labelled by an LLM with a single prompt call per cluster.

The cluster geometry of the original 250 topics is NOT touched — we are only
*grouping* them. Each document keeps its mikro topic; mezo/makro are derived.

Outputs:
    outputs/tables/12_hierarchy_tree.csv          (BERTopic raw linkage table)
    outputs/tables/12_topic_to_levels.csv         (topic_id, mezo_id, makro_id)
    outputs/tables/12_level_labels.csv            (level, cluster_id, label, n_docs, n_topics)
    outputs/tables/12_doc_topics_levels.parquet   (doc_id, mikro, mezo, makro, rating, mismatch_raw)
    outputs/figures/12_dendrogram.html            (interactive)

Usage:
    .venv/bin/python scripts/12_hierarchical_topics.py
        [--n-mezo 25] [--n-makro 6]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

from config import (
    EMBEDDINGS_NPY, FIG_DIR, MODEL_DIR, PREPROCESSED_PARQUET,
    TABLE_DIR, dump_env, setup_logger,
)

SCRIPT_NAME = "12_hierarchical_topics"

MEZO_PROMPT = """You are reviewing several Coursera-review topics that were grouped together
because they are semantically related.

The topics in this group (with their existing labels) are:
[CHILD_LABELS]

Give a short, specific English label (max 8 words) for the *common student-experience theme*
of this group. Examples of good mezo-level themes: "course assessment system complaints",
"instructor and teaching praise", "infrastructure and platform issues",
"language-learning courses praise", "data science skill courses praise".

Reply with ONLY the label, no quotes, no explanation."""

MAKRO_PROMPT = """You are reviewing several Coursera-review topic groups that were merged together
into one broad theme.

The mezo-level groups in this macro cluster are:
[CHILD_LABELS]

Give a short, broad English label (max 6 words) that captures the umbrella theme.
Examples of good makro themes: "content and learning experience",
"platform and assessment infrastructure", "instructor and teaching quality",
"specific subject domains".

Reply with ONLY the label, no quotes, no explanation."""


def _load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def ask_llm(client, model: str, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=64,
            )
            txt = (r.choices[0].message.content or "").strip()
            if txt:
                return txt.splitlines()[0][:200]
        except Exception:
            time.sleep(2 ** attempt)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mezo", type=int, default=25,
                        help="Target number of mezo clusters (cut height auto-picked).")
    parser.add_argument("--n-makro", type=int, default=6,
                        help="Target number of makro clusters.")
    args = parser.parse_args()

    _load_env()
    log = setup_logger(SCRIPT_NAME)
    dump_env(log)

    from bertopic import BERTopic
    bt_path = MODEL_DIR / "final_bertopic_llm"
    if not bt_path.exists():
        bt_path = MODEL_DIR / "final_bertopic"
    log.info("Loading BERTopic from %s", bt_path)
    topic_model = BERTopic.load(str(bt_path))

    df = pd.read_parquet(PREPROCESSED_PARQUET, columns=["doc_id", "reviews", "rating"])
    embeddings = np.load(EMBEDDINGS_NPY, mmap_mode="r")
    docs = df["reviews"].tolist()

    # ---- 1. Hierarchical topics via BERTopic
    log.info("Computing hierarchical_topics ...")
    hier = topic_model.hierarchical_topics(docs)
    hier.to_csv(TABLE_DIR / "12_hierarchy_tree.csv", index=False)
    log.info("Hierarchy rows: %d", len(hier))

    # BERTopic returns a Pandas DataFrame with parent/child relations, plus
    # 'Distance' for each merge. We build a scipy linkage matrix to use fcluster.
    # The simplest route: use BERTopic's own helper to get linkage_matrix.
    from bertopic._utils import (  # type: ignore
        check_documents_type, check_is_fitted,
    )
    # Internal API location for the linkage matrix differs across versions.
    # We build it ourselves from the hier table to avoid version skew.

    # ---- Build a leaf-by-leaf distance matrix using BERTopic's topic embeddings,
    # then re-run Ward linkage. This is what bertopic uses internally.
    log.info("Computing leaf topic embeddings (topic_embeddings_) ...")
    topic_embeddings = topic_model.topic_embeddings_
    # topic_embeddings includes index 0 for outlier; align ids carefully
    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1].sort_values("Topic").reset_index(drop=True)
    leaf_ids = info["Topic"].tolist()
    # Extract embedding rows in topic id order (BERTopic stores them in topic order)
    emb_map = {tid: topic_embeddings[i] for i, tid in enumerate(sorted(topic_model.get_topics().keys()))}
    leaf_emb = np.array([emb_map[t] for t in leaf_ids])
    log.info("Leaf matrix: %s", leaf_emb.shape)

    from scipy.cluster.hierarchy import linkage
    # Ward linkage requires Euclidean. L2-normalize embeddings so Euclidean
    # distance is monotonic with cosine distance — gives Ward access to
    # balanced merges that 'average + cosine' tends to defeat (single mega-cluster).
    norms = np.linalg.norm(leaf_emb, axis=1, keepdims=True)
    leaf_emb_n = leaf_emb / np.clip(norms, 1e-12, None)
    Z = linkage(leaf_emb_n, method="ward", metric="euclidean")
    log.info("Linkage shape: %s (method=ward, normalized euclidean)", Z.shape)

    # ---- 2. Cut tree at n_mezo and n_makro
    mezo_labels = fcluster(Z, t=args.n_mezo, criterion="maxclust")
    makro_labels = fcluster(Z, t=args.n_makro, criterion="maxclust")
    levels = pd.DataFrame({
        "topic_id": leaf_ids,
        "mezo_id": mezo_labels.astype(int),
        "makro_id": makro_labels.astype(int),
    })
    levels.to_csv(TABLE_DIR / "12_topic_to_levels.csv", index=False)
    log.info("Mezo clusters: %d | Makro clusters: %d",
             levels["mezo_id"].nunique(), levels["makro_id"].nunique())

    # ---- 3. Attach mikro LLM labels (from step 11)
    mikro_lbl_path = TABLE_DIR / "11_llm_topic_labels.csv"
    if mikro_lbl_path.exists():
        mikro_lbls = pd.read_csv(mikro_lbl_path)[["topic_id", "llm_label", "count"]]
        mikro_lbls = mikro_lbls.rename(columns={"llm_label": "mikro_label",
                                                  "count": "mikro_count"})
    else:
        mikro_lbls = pd.DataFrame({"topic_id": leaf_ids,
                                     "mikro_label": [str(t) for t in leaf_ids],
                                     "mikro_count": 0})
    grouped = levels.merge(mikro_lbls, on="topic_id", how="left")

    # ---- 4. LLM-label the mezo and makro clusters
    _load_env()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    if not api_key or api_key.startswith("sk-..."):
        log.warning("No API key — skipping LLM labels; using placeholder names.")
        client = None
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

    rows = []
    # mezo
    for mz, sub in grouped.groupby("mezo_id"):
        children = sub.sort_values("mikro_count", ascending=False)
        bullets = "\n".join(f"- {r.mikro_label} (n={int(r.mikro_count)})"
                            for _, r in children.head(10).iterrows())
        if client is None:
            label = f"mezo-{mz}"
        else:
            prompt = MEZO_PROMPT.replace("[CHILD_LABELS]", bullets)
            label = ask_llm(client, model_name, prompt) or f"mezo-{mz}"
        rows.append({"level": "mezo", "cluster_id": int(mz), "label": label,
                     "n_mikro_topics": len(sub), "n_docs": int(sub["mikro_count"].sum())})
        log.info("mezo %2d (k=%2d, n=%6d): %s", mz, len(sub), sub["mikro_count"].sum(), label)
    # makro
    mezo_labels_df = pd.DataFrame(rows)
    mz_label_map = dict(zip(mezo_labels_df.cluster_id, mezo_labels_df.label))
    for mk, sub in grouped.groupby("makro_id"):
        mezo_in = sub["mezo_id"].unique()
        bullets = "\n".join(f"- {mz_label_map.get(int(mz), str(mz))}" for mz in mezo_in)
        if client is None:
            label = f"makro-{mk}"
        else:
            prompt = MAKRO_PROMPT.replace("[CHILD_LABELS]", bullets)
            label = ask_llm(client, model_name, prompt) or f"makro-{mk}"
        rows.append({"level": "makro", "cluster_id": int(mk), "label": label,
                     "n_mikro_topics": len(sub), "n_docs": int(sub["mikro_count"].sum())})
        log.info("makro %d (n=%d): %s", mk, sub["mikro_count"].sum(), label)

    level_df = pd.DataFrame(rows)
    level_df.to_csv(TABLE_DIR / "12_level_labels.csv", index=False)

    # ---- 5. Attach mezo/makro ids + labels to every document
    doc_topics = pd.read_parquet(TABLE_DIR / "05_doc_topics.parquet")
    scores = pd.read_parquet(TABLE_DIR / "08_mismatch_scores.parquet",
                              columns=["doc_id", "mismatch_raw"])
    doc_level = doc_topics.merge(levels, left_on="topic", right_on="topic_id", how="left")
    doc_level = doc_level.merge(scores, on="doc_id", how="left")
    doc_level = doc_level.rename(columns={"topic": "mikro_id"})
    doc_level = doc_level[["doc_id", "mikro_id", "mezo_id", "makro_id",
                              "rating", "mismatch_raw"]]
    doc_level.to_parquet(TABLE_DIR / "12_doc_topics_levels.parquet", index=False)

    # ---- 6. Try saving the BERTopic interactive dendrogram (best-effort)
    try:
        fig = topic_model.visualize_hierarchy(hierarchical_topics=hier)
        fig.write_html(str(FIG_DIR / "12_dendrogram.html"))
        log.info("Saved interactive dendrogram -> outputs/figures/12_dendrogram.html")
    except Exception as e:
        log.warning("visualize_hierarchy failed: %s", e)

    # ---- 7. Mismatch summary at each level
    summary = {}
    for level_name, col in [("mikro", "mikro_id"), ("mezo", "mezo_id"), ("makro", "makro_id")]:
        sub = doc_level.dropna(subset=[col, "mismatch_raw"])
        sub = sub[sub["mikro_id"] != -1]  # exclude outliers
        g = sub.groupby(col)["mismatch_raw"]
        summary[level_name] = {
            "n_clusters": int(sub[col].nunique()),
            "n_docs": int(len(sub)),
            "median_mismatch_range": [float(g.median().min()), float(g.median().max())],
        }
    (TABLE_DIR / "12_levels_summary.json").write_text(json.dumps(summary, indent=2),
                                                       encoding="utf-8")
    log.info("Levels summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
