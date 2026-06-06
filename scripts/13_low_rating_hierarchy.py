"""
Step 13 — "Hidden complaint tree": hierarchical topic model fit only on
low-rating reviews (rating ≤ 3).

The main pipeline shows that 85%+ of reviews are praise-driven. To surface
the *structure of dissatisfaction*, we re-fit BERTopic on the subset of
reviews with rating ≤ 3 (~30K documents), then build a 3-level hierarchy
exactly as in step 12. Documents are not re-embedded — we reuse the cached
all-MiniLM-L6-v2 vectors.

Outputs:
    outputs/models/low_rating_bertopic/
    outputs/tables/13_low_rating_topic_info.csv
    outputs/tables/13_low_rating_topic_to_levels.csv
    outputs/tables/13_low_rating_level_labels.csv
    outputs/tables/13_low_rating_makro_alignment.csv   (which main-makro do these map to?)

Usage:
    .venv/bin/python scripts/13_low_rating_hierarchy.py
        [--min-topic-size 50] [--n-mezo 12] [--n-makro 4]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from bertopic_utils import build_topic_model
from config import (
    EMBEDDINGS_NPY, GLOBAL_SEED, MODEL_DIR, PREPROCESSED_PARQUET, TABLE_DIR,
    dump_env, set_global_seed, setup_logger,
)

SCRIPT_NAME = "13_low_rating_hierarchy"

MEZO_PROMPT = """You are reviewing Coursera-review topics that were grouped together.
All these reviews come from students who gave LOW ratings (1-3 stars).

The topics in this group are:
[CHILD_LABELS]

Give a short, specific English label (max 8 words) for the *common complaint theme*
of this group. Examples: "auto-grader and assignment problems",
"video quality and platform issues", "course outdated and irrelevant content",
"peer-review unfairness", "course difficulty and prerequisites issues".

Reply with ONLY the label, no quotes, no explanation."""

MAKRO_PROMPT = """You are reviewing groups of low-rating Coursera-review topics.

The mezo-level groups are:
[CHILD_LABELS]

Give a short, broad English label (max 6 words) for the umbrella complaint area.
Examples: "assessment and grading complaints", "content and pedagogy complaints",
"platform and access complaints", "course-design complaints".

Reply with ONLY the label, no quotes, no explanation."""

LABEL_PROMPT = """I have a topic that contains the following Coursera student reviews,
all from low-rating (1-3 stars) feedback:

[DOCUMENTS]

The topic keywords are: [KEYWORDS]

Give a short, specific English label (max 8 words) describing the *complaint theme*.
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


def ask(client, model, prompt, retries=3):
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=64,
            )
            txt = (r.choices[0].message.content or "").strip()
            if txt:
                return txt.splitlines()[0][:200]
        except Exception:
            time.sleep(2 ** i)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-topic-size", type=int, default=50,
                        help="Smaller default — fewer docs available")
    parser.add_argument("--n-mezo", type=int, default=12)
    parser.add_argument("--n-makro", type=int, default=4)
    args = parser.parse_args()

    set_global_seed(GLOBAL_SEED)
    _load_env()
    log = setup_logger(SCRIPT_NAME)
    dump_env(log)

    # ---- Subset to rating ≤ 3
    df = pd.read_parquet(PREPROCESSED_PARQUET, columns=["doc_id", "reviews", "rating"])
    full_embeddings = np.load(EMBEDDINGS_NPY, mmap_mode="r")
    mask = df["rating"] <= 3
    sub_df = df[mask].reset_index(drop=False)
    sub_emb = np.asarray(full_embeddings)[sub_df["index"].values]
    sub_docs = sub_df["reviews"].tolist()
    log.info("Low-rating subset: %d / %d (%.1f%%)",
             len(sub_docs), len(df), 100 * len(sub_docs) / len(df))
    log.info("Subset embedding shape: %s", sub_emb.shape)

    # ---- Fit a fresh BERTopic
    log.info("Fitting BERTopic (min_topic_size=%d, seed=%d) ...",
             args.min_topic_size, GLOBAL_SEED)
    model = build_topic_model(seed=GLOBAL_SEED, min_topic_size=args.min_topic_size,
                                calculate_probabilities=False, verbose=False)
    t0 = time.time()
    topics, _ = model.fit_transform(sub_docs, embeddings=sub_emb)
    dt = time.time() - t0
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    outl = float((np.asarray(topics) == -1).mean())
    log.info("Fit done in %.1fs | n_topics=%d | outlier=%.2f%%", dt, n_topics, 100 * outl)

    model_path = MODEL_DIR / "low_rating_bertopic"
    model.save(str(model_path), serialization="safetensors", save_ctfidf=True,
                save_embedding_model=False)
    log.info("Saved -> %s", model_path)

    info = model.get_topic_info()
    info.to_csv(TABLE_DIR / "13_low_rating_topic_info.csv", index=False)

    # ---- LLM label each leaf topic (small budget; ~50-80 topics expected)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    client = None
    if api_key and not api_key.startswith("sk-..."):
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

    import tiktoken
    try:
        tok = tiktoken.encoding_for_model(model_name)
    except KeyError:
        tok = tiktoken.get_encoding("cl100k_base")

    rep_docs = model.get_representative_docs() or {}
    leaf_labels = []
    for tid in sorted(t for t in model.get_topics() if t != -1):
        words = model.get_topic(tid) or []
        kw = ", ".join(w for w, _ in words[:10])
        reps = rep_docs.get(tid, [])[:5]
        block = "\n".join(f"- {tok.decode(tok.encode(d)[:120])}" for d in reps) or "(no docs)"
        if client is None:
            label = words[0][0] if words else f"topic-{tid}"
        else:
            label = ask(client, model_name,
                        LABEL_PROMPT.replace("[DOCUMENTS]", block).replace("[KEYWORDS]", kw)) \
                    or (words[0][0] if words else f"topic-{tid}")
        cnt = int((np.asarray(topics) == tid).sum())
        leaf_labels.append({"topic_id": int(tid), "count": cnt, "llm_label": label})
        log.info("  leaf %3d (n=%5d): %s", tid, cnt, label)
    leaves_df = pd.DataFrame(leaf_labels)

    # ---- Build 3-level hierarchy on topic embeddings (Ward + L2-normed)
    topic_embeddings = model.topic_embeddings_
    ids_in_order = sorted(model.get_topics().keys())
    emb_map = {tid: topic_embeddings[i] for i, tid in enumerate(ids_in_order)}
    leaf_ids = sorted(t for t in model.get_topics() if t != -1)
    leaf_emb = np.array([emb_map[t] for t in leaf_ids])
    leaf_emb_n = leaf_emb / np.clip(np.linalg.norm(leaf_emb, axis=1, keepdims=True), 1e-12, None)
    Z = linkage(leaf_emb_n, method="ward", metric="euclidean")

    n_mezo = min(args.n_mezo, max(2, len(leaf_ids) - 1))
    n_makro = min(args.n_makro, max(2, n_mezo - 1))
    mezo = fcluster(Z, t=n_mezo, criterion="maxclust")
    makro = fcluster(Z, t=n_makro, criterion="maxclust")
    levels = pd.DataFrame({"topic_id": leaf_ids, "mezo_id": mezo.astype(int),
                            "makro_id": makro.astype(int)})
    levels = levels.merge(leaves_df, on="topic_id")
    levels.to_csv(TABLE_DIR / "13_low_rating_topic_to_levels.csv", index=False)
    log.info("Mezo=%d  Makro=%d", levels.mezo_id.nunique(), levels.makro_id.nunique())

    # ---- Label mezo + makro
    rows = []
    mz_label = {}
    for mz, sub in levels.groupby("mezo_id"):
        children = sub.sort_values("count", ascending=False).head(12)
        bullets = "\n".join(f"- {r.llm_label} (n={int(r['count'])})" for _, r in children.iterrows())
        lbl = (ask(client, model_name, MEZO_PROMPT.replace("[CHILD_LABELS]", bullets))
               if client else f"mezo-{mz}") or f"mezo-{mz}"
        mz_label[int(mz)] = lbl
        rows.append({"level": "mezo", "cluster_id": int(mz), "label": lbl,
                     "n_mikro_topics": len(sub), "n_docs": int(sub["count"].sum())})
        log.info("mezo %2d (k=%2d, n=%5d): %s", mz, len(sub), sub["count"].sum(), lbl)
    for mk, sub in levels.groupby("makro_id"):
        bullets = "\n".join(f"- {mz_label.get(int(mz), str(mz))}" for mz in sub.mezo_id.unique())
        lbl = (ask(client, model_name, MAKRO_PROMPT.replace("[CHILD_LABELS]", bullets))
               if client else f"makro-{mk}") or f"makro-{mk}"
        rows.append({"level": "makro", "cluster_id": int(mk), "label": lbl,
                     "n_mikro_topics": len(sub), "n_docs": int(sub["count"].sum())})
        log.info("makro %d (n=%d): %s", mk, sub["count"].sum(), lbl)
    pd.DataFrame(rows).to_csv(TABLE_DIR / "13_low_rating_level_labels.csv", index=False)

    # ---- Align low-rating mikro topics to main-makro clusters (which main-makro
    # does each low-rating document mostly fall into?). This shows where the
    # 'hidden complaints' live in the main hierarchy.
    main_levels = pd.read_csv(TABLE_DIR / "12_topic_to_levels.csv")
    main_doc = pd.read_parquet(TABLE_DIR / "12_doc_topics_levels.parquet")
    sub_doc = main_doc[main_doc["doc_id"].isin(sub_df["doc_id"])].copy()
    align = (sub_doc.groupby("makro_id").size().reset_index(name="n_low_rating_docs")
             .sort_values("n_low_rating_docs", ascending=False))
    main_makro_lbl = pd.read_csv(TABLE_DIR / "12_level_labels.csv")
    main_makro_lbl = main_makro_lbl[main_makro_lbl.level == "makro"][["cluster_id", "label"]] \
        .rename(columns={"cluster_id": "makro_id", "label": "main_makro_label"})
    align = align.merge(main_makro_lbl, on="makro_id", how="left")
    align["pct_of_low_rating"] = (100 * align["n_low_rating_docs"] / align["n_low_rating_docs"].sum()).round(2)
    align.to_csv(TABLE_DIR / "13_low_rating_makro_alignment.csv", index=False)
    log.info("Low-rating doc distribution across main makro clusters:\n%s",
             align.to_string(index=False))

    summary = {
        "n_low_rating_docs": int(len(sub_docs)),
        "n_topics": int(n_topics), "outlier_share": round(outl, 4),
        "n_mezo": int(levels.mezo_id.nunique()),
        "n_makro": int(levels.makro_id.nunique()),
        "seed": GLOBAL_SEED, "min_topic_size": args.min_topic_size,
    }
    (TABLE_DIR / "13_low_rating_summary.json").write_text(json.dumps(summary, indent=2),
                                                            encoding="utf-8")


if __name__ == "__main__":
    main()
