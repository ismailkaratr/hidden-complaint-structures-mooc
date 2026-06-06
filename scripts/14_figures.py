"""
Step 14 — Publication-quality figures for the paper.

Produces:
  outputs/figures/14_overall_sunburst.html        (interactive makro→mezo→mikro)
  outputs/figures/14_lowrating_sunburst.html      (only low-rating subset)
  outputs/figures/14_mismatch_heatmap.png         (top-30 mikro × norm/threshold)
  outputs/figures/14_topic_size_distribution.png  (log-scale topic size hist)
  outputs/figures/14_mismatch_violin_makro.png    (mismatch dist per makro)
  outputs/figures/14_top_complaint_topics.png     (horizontal bar, top-20)
  outputs/figures/14_param_grid.png               (coherence vs min_topic_size)
  outputs/figures/14_stability_pairwise.png       (ARI/NMI per seed pair)
  outputs/figures/14_lowrating_treemap.html       (interactive makro→mezo→mikro)

Usage:
    .venv/bin/python scripts/14_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIG_DIR, TABLE_DIR, setup_logger

SCRIPT_NAME = "14_figures"

# Make all figures look consistent.
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 250,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _short(s: str, n: int = 50) -> str:
    s = str(s) if pd.notna(s) else ""
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------- sunburst
def figure_overall_sunburst(log) -> None:
    import plotly.express as px

    levels = pd.read_csv(TABLE_DIR / "12_topic_to_levels.csv")
    labels = pd.read_csv(TABLE_DIR / "12_level_labels.csv")
    mikro = pd.read_csv(TABLE_DIR / "11_llm_topic_labels.csv")[["topic_id", "llm_label", "count"]]
    df = (levels.merge(mikro, on="topic_id")
                .merge(labels[labels.level == "mezo"][["cluster_id", "label"]]
                       .rename(columns={"cluster_id": "mezo_id", "label": "mezo_label"}),
                       on="mezo_id")
                .merge(labels[labels.level == "makro"][["cluster_id", "label"]]
                       .rename(columns={"cluster_id": "makro_id", "label": "makro_label"}),
                       on="makro_id"))
    df["makro_label"] = "[M" + df.makro_id.astype(str) + "] " + df.makro_label.map(_short)
    df["mezo_label"] = "[m" + df.mezo_id.astype(str) + "] " + df.mezo_label.map(_short)
    df["mikro_label"] = df.llm_label.map(_short)

    fig = px.sunburst(df, path=["makro_label", "mezo_label", "mikro_label"],
                       values="count", color="makro_label",
                       title="Coursera Reviews — Hierarchical Topic Structure (250 → 25 → 6)")
    fig.update_traces(insidetextorientation="radial")
    fig.write_html(str(FIG_DIR / "14_overall_sunburst.html"))
    log.info("Wrote overall sunburst")


def figure_lowrating_sunburst(log) -> None:
    import plotly.express as px

    levels = pd.read_csv(TABLE_DIR / "13_low_rating_topic_to_levels.csv")
    labels = pd.read_csv(TABLE_DIR / "13_low_rating_level_labels.csv")
    df = (levels
          .merge(labels[labels.level == "mezo"][["cluster_id", "label"]]
                 .rename(columns={"cluster_id": "mezo_id", "label": "mezo_label"}),
                 on="mezo_id")
          .merge(labels[labels.level == "makro"][["cluster_id", "label"]]
                 .rename(columns={"cluster_id": "makro_id", "label": "makro_label"}),
                 on="makro_id"))
    df["makro_label"] = "[CM" + df.makro_id.astype(str) + "] " + df.makro_label.map(_short)
    df["mezo_label"] = "[cm" + df.mezo_id.astype(str) + "] " + df.mezo_label.map(_short)
    df["mikro_label"] = df.llm_label.map(_short)

    fig = px.sunburst(df, path=["makro_label", "mezo_label", "mikro_label"],
                       values="count", color="makro_label",
                       title="Hidden Complaint Tree — Reviews with rating ≤ 3 only")
    fig.write_html(str(FIG_DIR / "14_lowrating_sunburst.html"))
    log.info("Wrote low-rating sunburst")


def figure_lowrating_treemap(log) -> None:
    import plotly.express as px

    levels = pd.read_csv(TABLE_DIR / "13_low_rating_topic_to_levels.csv")
    labels = pd.read_csv(TABLE_DIR / "13_low_rating_level_labels.csv")
    df = (levels
          .merge(labels[labels.level == "mezo"][["cluster_id", "label"]]
                 .rename(columns={"cluster_id": "mezo_id", "label": "mezo_label"}),
                 on="mezo_id")
          .merge(labels[labels.level == "makro"][["cluster_id", "label"]]
                 .rename(columns={"cluster_id": "makro_id", "label": "makro_label"}),
                 on="makro_id"))
    df["makro_label"] = "[CM" + df.makro_id.astype(str) + "] " + df.makro_label.map(_short)
    df["mezo_label"] = "[cm" + df.mezo_id.astype(str) + "] " + df.mezo_label.map(_short)
    df["mikro_label"] = df.llm_label.map(_short, n=40)

    fig = px.treemap(df, path=[px.Constant("All complaints"),
                                "makro_label", "mezo_label", "mikro_label"],
                      values="count", color="count",
                      color_continuous_scale="Reds",
                      title="Hidden Complaint Tree (rating ≤ 3): area = #reviews")
    fig.write_html(str(FIG_DIR / "14_lowrating_treemap.html"))
    log.info("Wrote low-rating treemap")


# ------------------------------------------------------------- mismatch heatmap
def figure_mismatch_heatmap(log) -> None:
    robust = pd.read_csv(TABLE_DIR / "08_mismatch_robustness.csv")
    pv = robust.pivot(index="normalization", columns="threshold", values="flagged_share")
    pv = pv.loc[["raw", "rank", "zscore", "binary"]]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    sns.heatmap(pv * 100, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={"label": "Flagged reviews (%)"}, ax=ax, vmin=0, vmax=90)
    ax.set_title("Mismatch flagging robustness across normalization × threshold")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Normalization method")
    fig.savefig(FIG_DIR / "14_mismatch_heatmap.png")
    plt.close(fig)
    log.info("Wrote mismatch heatmap")


# ------------------------------------------------------ topic-size distribution
def figure_topic_size_distribution(log) -> None:
    ti = pd.read_csv(TABLE_DIR / "05_topic_info.csv")
    ti = ti[ti.Topic != -1].copy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ti["Count"], bins=np.logspace(np.log10(ti["Count"].min()),
                                             np.log10(ti["Count"].max()), 30),
             color="#4c72b0", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Topic size (#reviews, log scale)")
    ax.set_ylabel("Number of topics")
    ax.set_title(f"Topic size distribution — {len(ti)} topics, median = {int(ti.Count.median())}, "
                  f"max = {int(ti.Count.max()):,}")
    fig.savefig(FIG_DIR / "14_topic_size_distribution.png")
    plt.close(fig)
    log.info("Wrote topic-size histogram")


# ----------------------------------------------------- mismatch violin per makro
def figure_mismatch_violin_makro(log) -> None:
    doc = pd.read_parquet(TABLE_DIR / "12_doc_topics_levels.parquet")
    labels = pd.read_csv(TABLE_DIR / "12_level_labels.csv")
    mak = labels[labels.level == "makro"][["cluster_id", "label"]]\
        .rename(columns={"cluster_id": "makro_id", "label": "makro_label"})
    df = doc.dropna(subset=["mismatch_raw", "makro_id"]).copy()
    df = df[df["mikro_id"] != -1]
    df = df.merge(mak, on="makro_id")
    df["short"] = "M" + df.makro_id.astype(int).astype(str) + ": " + df.makro_label.map(lambda s: _short(s, 35))

    order = df.groupby("short")["mismatch_raw"].median().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.violinplot(data=df, x="mismatch_raw", y="short", order=order,
                    cut=0, inner="quartile", ax=ax, color="#4c72b0")
    ax.set_xlabel("Per-review rating–sentiment mismatch (raw, 0–1)")
    ax.set_ylabel("")
    ax.set_title("Mismatch distribution within each makro cluster")
    fig.savefig(FIG_DIR / "14_mismatch_violin_makro.png")
    plt.close(fig)
    log.info("Wrote mismatch violin (makro)")


# ----------------------------------------------------- top complaint mikro bar
def figure_top_complaint_topics(log) -> None:
    desc = pd.read_csv(TABLE_DIR / "09_topic_mismatch_descriptives.csv")
    lbl = pd.read_csv(TABLE_DIR / "11_llm_topic_labels.csv")[["topic_id", "llm_label"]]
    m = desc.merge(lbl, on="topic_id").sort_values("median", ascending=False).head(20)
    m["disp"] = m["llm_label"].map(lambda s: _short(s, 50)) + f"  (n="\
        + m["n"].astype(int).astype(str) + ")"

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(len(m)), m["median"], color="#d62728", alpha=0.85)
    ax.errorbar(m["median"], range(len(m)),
                 xerr=[m["median"] - m["median_ci_lo"],
                       m["median_ci_hi"] - m["median"]],
                 fmt="none", color="black", capsize=2, linewidth=0.8)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m["disp"][::-1].tolist()[::-1], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Median rating–sentiment mismatch (95 % bootstrap CI)")
    ax.set_title("Top-20 topics by mismatch — surfaced structural complaints")
    ax.set_xlim(0, max(m["median_ci_hi"]) * 1.05)
    fig.savefig(FIG_DIR / "14_top_complaint_topics.png")
    plt.close(fig)
    log.info("Wrote top-complaint bar")


# ----------------------------------------------------------- param grid line
def figure_param_grid(log) -> None:
    g = pd.read_csv(TABLE_DIR / "04_param_grid.csv")
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(g["min_topic_size"], g["c_v"], "o-", color="#4c72b0", label="c_v")
    ax1.plot(g["min_topic_size"], g["c_npmi"], "s--", color="#55a868", label="c_npmi")
    ax1.plot(g["min_topic_size"], g["topic_diversity"], "^:", color="#c44e52", label="diversity")
    ax1.set_xlabel("min_topic_size")
    ax1.set_ylabel("Coherence / Diversity")
    ax1.legend(loc="upper left", frameon=False)
    ax2 = ax1.twinx()
    ax2.plot(g["min_topic_size"], g["n_topics"], "d-.", color="gray", alpha=0.7, label="n_topics")
    ax2.set_ylabel("Number of topics")
    ax2.legend(loc="upper right", frameon=False)
    fig.suptitle("Hyperparameter sweep over min_topic_size")
    fig.savefig(FIG_DIR / "14_param_grid.png")
    plt.close(fig)
    log.info("Wrote param-grid figure")


# ----------------------------------------------------- stability pairwise heat
def figure_stability_pairwise(log) -> None:
    p = pd.read_csv(TABLE_DIR / "03_stability_pairwise.csv")
    seeds = sorted(set(p.seed_a) | set(p.seed_b))
    ari = pd.DataFrame(index=seeds, columns=seeds, dtype=float)
    nmi = pd.DataFrame(index=seeds, columns=seeds, dtype=float)
    for _, r in p.iterrows():
        ari.loc[r.seed_a, r.seed_b] = ari.loc[r.seed_b, r.seed_a] = r.ARI
        nmi.loc[r.seed_a, r.seed_b] = nmi.loc[r.seed_b, r.seed_a] = r.NMI
    ari_arr = ari.astype(float).to_numpy().copy()
    nmi_arr = nmi.astype(float).to_numpy().copy()
    np.fill_diagonal(ari_arr, 1.0)
    np.fill_diagonal(nmi_arr, 1.0)
    ari = pd.DataFrame(ari_arr, index=ari.index, columns=ari.columns)
    nmi = pd.DataFrame(nmi_arr, index=nmi.index, columns=nmi.columns)

    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
    sns.heatmap(ari, annot=True, fmt=".3f", cmap="Blues",
                ax=a, vmin=0.5, vmax=1.0, cbar_kws={"label": "ARI"})
    a.set_title("Adjusted Rand Index across seeds")
    sns.heatmap(nmi, annot=True, fmt=".3f", cmap="Greens",
                ax=b, vmin=0.5, vmax=1.0, cbar_kws={"label": "NMI"})
    b.set_title("Normalized Mutual Information across seeds")
    fig.savefig(FIG_DIR / "14_stability_pairwise.png")
    plt.close(fig)
    log.info("Wrote stability heatmap")


def main() -> None:
    log = setup_logger(SCRIPT_NAME)

    # Static figures
    figure_param_grid(log)
    figure_stability_pairwise(log)
    figure_topic_size_distribution(log)
    figure_mismatch_heatmap(log)
    figure_top_complaint_topics(log)
    figure_mismatch_violin_makro(log)

    # Interactive figures (plotly)
    figure_overall_sunburst(log)
    figure_lowrating_sunburst(log)
    figure_lowrating_treemap(log)

    # Aggregate index
    fig_files = sorted(p.name for p in FIG_DIR.glob("14_*"))
    (FIG_DIR / "14_index.json").write_text(json.dumps(fig_files, indent=2), encoding="utf-8")
    log.info("DONE — %d figures in %s", len(fig_files), FIG_DIR)


if __name__ == "__main__":
    main()
