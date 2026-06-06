"""
Stage 1 — diagnose the Pass-2 hierarchy.

We rebuild the Ward linkage on the Pass-2 topic embeddings and answer
three questions empirically:

  Q1  At what merge height do the four current macros sit?
      Are Macro 1 and Macro 4 merged at a much smaller height than
      they are merged with Macro 2 or Macro 3 — i.e. are they nearly
      the same cluster the algorithm artificially split?

  Q2  What is the silhouette score for k = 2, 3, …, 10?
      Where is the natural cut?

  Q3  How similar are the four current macros at the c-TF-IDF
      vocabulary level (Jaccard overlap of top-10 words per macro)?

The script writes:
  outputs/figures/23_pass2_dendrogram.png      annotated dendrogram
  outputs/tables/23_pass2_silhouette.csv       per-k silhouette
  outputs/tables/23_pass2_macro_distances.csv  symmetric distance matrix
  outputs/tables/23_pass2_macro_vocab_jaccard.csv  top-10 word overlap
  outputs/tables/23_pass2_diagnosis.json       headline numbers
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

from bertopic import BERTopic

TBL = Path("outputs/tables")
FIG = Path("outputs/figures")
MODEL_DIR = Path("outputs/models/low_rating_bertopic")


def main() -> None:
    print("Loading Pass-2 model ...")
    bt = BERTopic.load(str(MODEL_DIR))

    info = bt.get_topic_info()
    info = info[info.Topic != -1].sort_values("Topic").reset_index(drop=True)
    leaf_ids = info["Topic"].tolist()
    topic_embeddings = bt.topic_embeddings_  # rows aligned to sorted topic ids
    emb_map = {tid: topic_embeddings[i]
                for i, tid in enumerate(sorted(bt.get_topics().keys()))}
    leaf_emb = np.array([emb_map[t] for t in leaf_ids])
    norms = np.linalg.norm(leaf_emb, axis=1, keepdims=True)
    leaf_emb_n = leaf_emb / np.clip(norms, 1e-12, None)
    print(f"Leaf topic embeddings: {leaf_emb_n.shape}")

    # ---- Linkage
    Z = linkage(leaf_emb_n, method="ward", metric="euclidean")
    print(f"Linkage matrix: {Z.shape}")

    # Current labels (4 macros)
    current_macros = fcluster(Z, t=4, criterion="maxclust")
    macro_lookup = dict(zip(leaf_ids, current_macros))

    # ---- Q1: at what heights do macros merge?
    # Walk up the linkage tree; once two clusters that started in
    # different macros merge, we know the inter-macro height.
    n = len(leaf_emb_n)
    # cluster id -> set of leaf indices
    cluster_members = {i: {i} for i in range(n)}
    macro_of_leaf = {i: macro_lookup[leaf_ids[i]] for i in range(n)}
    inter_macro_heights = {}  # frozenset({macro_a, macro_b}) -> first merge height
    for step, (a, b, h, _) in enumerate(Z):
        a, b = int(a), int(b)
        merged_id = n + step
        members_a = cluster_members[a]
        members_b = cluster_members[b]
        macros_in_a = {macro_of_leaf[i] for i in members_a}
        macros_in_b = {macro_of_leaf[i] for i in members_b}
        # Did this merge connect two different macros for the first time?
        for ma in macros_in_a:
            for mb in macros_in_b:
                if ma != mb:
                    key = frozenset((int(ma), int(mb)))
                    if key not in inter_macro_heights:
                        inter_macro_heights[key] = float(h)
        cluster_members[merged_id] = members_a | members_b

    inter_macro_df = pd.DataFrame(
        [{"pair": " - ".join(map(str, sorted(k))),
           "first_merge_height": v} for k, v in inter_macro_heights.items()]
    ).sort_values("first_merge_height").reset_index(drop=True)
    print()
    print("First merge height between each macro pair:")
    print(inter_macro_df.to_string(index=False))

    # ---- Q2: silhouette score over k = 2..10
    sil_rows = []
    for k in range(2, 11):
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(leaf_emb_n, labels, metric="euclidean")
        sil_rows.append({"k": k, "silhouette": round(float(score), 4),
                         "n_singletons": int(sum(1 for c in set(labels)
                                                  if (labels == c).sum() == 1))})
    sil_df = pd.DataFrame(sil_rows)
    sil_df.to_csv(TBL / "23_pass2_silhouette.csv", index=False)
    print()
    print("Silhouette by k:")
    print(sil_df.to_string(index=False))
    best_k = int(sil_df.loc[sil_df.silhouette.idxmax(), "k"])
    print(f"Best k by silhouette: {best_k}")

    # ---- Q3: c-TF-IDF Jaccard overlap of top-10 words per current macro
    # Collect top-10 keywords per topic, aggregate per macro, dedupe.
    macro_topic = {}
    for tid in leaf_ids:
        macro = macro_lookup[tid]
        topic_words = [w for w, _ in bt.get_topic(int(tid))[:10]]
        macro_topic.setdefault(int(macro), []).extend(topic_words)
    # Aggregate per macro
    macro_vocab = {m: set(words) for m, words in macro_topic.items()}
    macros = sorted(macro_vocab.keys())
    K = len(macros)
    jac = np.zeros((K, K))
    for i, mi in enumerate(macros):
        for j, mj in enumerate(macros):
            inter = len(macro_vocab[mi] & macro_vocab[mj])
            union = len(macro_vocab[mi] | macro_vocab[mj])
            jac[i, j] = inter / union if union else 0.0
    jac_df = pd.DataFrame(jac,
                            index=[f"Macro {m}" for m in macros],
                            columns=[f"Macro {m}" for m in macros])
    jac_df.to_csv(TBL / "23_pass2_macro_vocab_jaccard.csv")
    print()
    print("c-TF-IDF Jaccard overlap (top-10 words per topic, aggregated per macro):")
    print(jac_df.round(3).to_string())

    # ---- Q1 supplementary: macro centroid distances
    centroid = {m: leaf_emb_n[[i for i in range(n)
                                  if macro_of_leaf[i] == m]].mean(axis=0)
                for m in macros}
    centroid_arr = np.vstack([centroid[m] for m in macros])
    dist = squareform(pdist(centroid_arr, metric="euclidean"))
    dist_df = pd.DataFrame(dist,
                              index=[f"Macro {m}" for m in macros],
                              columns=[f"Macro {m}" for m in macros])
    dist_df.to_csv(TBL / "23_pass2_macro_distances.csv")
    print()
    print("Macro centroid Euclidean distances:")
    print(dist_df.round(3).to_string())

    # ---- Dendrogram figure (annotated with current macros as colours)
    fig, ax = plt.subplots(figsize=(14, 7))
    color_palette = {1: "#3A7CA5", 2: "#2E8B57", 3: "#C0392B", 4: "#8E44AD"}
    leaf_colors = {i: color_palette[int(macro_lookup[leaf_ids[i]])]
                    for i in range(n)}

    # link_color_func returns colour for each non-leaf node
    def link_color(node_id):
        node_id = int(node_id)
        if node_id < n:
            return leaf_colors[node_id]
        members = cluster_members.get(node_id, set())
        macros_inside = {macro_lookup[leaf_ids[i]] for i in members}
        if len(macros_inside) == 1:
            return color_palette[int(next(iter(macros_inside)))]
        return "#888888"

    dendrogram(
        Z, ax=ax,
        labels=[f"T{i+1}" for i in range(n)],
        leaf_font_size=6,
        link_color_func=link_color,
    )
    ax.set_title("Pass-2 Ward dendrogram (256 micro topics, L2-normalised)\n"
                  "Colours follow the current 4-macro cut",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("Merge height (Ward linkage)")
    # Reference lines for current and alternative cuts
    for k in (3, 4, 5, 6, 7, 8):
        # Approximate cut height that produces k clusters
        flat = fcluster(Z, t=k, criterion="maxclust")
        if len(set(flat)) == k:
            # Find min height that gives k clusters by walking up
            heights = sorted(Z[:, 2])
            for h in heights:
                if len(set(fcluster(Z, t=h, criterion="distance"))) == k:
                    ax.axhline(h, color="#aaa", linestyle="--", linewidth=0.7)
                    ax.text(ax.get_xlim()[1], h, f"  k={k}", fontsize=8,
                             color="#666", va="center")
                    break
    plt.tight_layout()
    fig.savefig(FIG / "23_pass2_dendrogram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print()
    print(f"Wrote {FIG / '23_pass2_dendrogram.png'}")

    # ---- Headline diagnosis JSON
    summary = {
        "current_k": 4,
        "n_leaf_micros": int(n),
        "best_k_by_silhouette": best_k,
        "silhouette_at_current_k": float(
            sil_df.loc[sil_df.k == 4, "silhouette"].iloc[0]),
        "silhouette_at_best_k": float(
            sil_df.loc[sil_df.k == best_k, "silhouette"].iloc[0]),
        "inter_macro_first_merge_heights": [
            {"pair": r.pair, "height": r.first_merge_height}
            for r in inter_macro_df.itertuples()
        ],
        "lowest_inter_macro_merge_pair": inter_macro_df.iloc[0]["pair"],
        "lowest_inter_macro_merge_height": float(
            inter_macro_df.iloc[0]["first_merge_height"]),
        "max_jaccard_off_diagonal": float(
            jac_df.values[np.triu_indices(K, k=1)].max()),
        "max_jaccard_pair": (lambda idx: f"{jac_df.index[idx[0]]} vs {jac_df.index[idx[1]]}")(
            np.unravel_index(np.argmax(np.triu(jac, k=1)), jac.shape)),
    }
    (TBL / "23_pass2_diagnosis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
