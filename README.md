# Hidden Complaint Structures in MOOC Reviews

Reproducible code and intermediate artefacts for the IRRODL submission *"Hidden Complaint Structures in MOOC Reviews: A Three-Level Hierarchical Topic Analysis with BERTopic and LLM-Augmented Labelling"*.

The repository implements a two-pass hierarchical topic-modelling pipeline on the publicly-available Coursera review corpus, with LLM-augmented labelling at three levels and a topic-stratified rating–sentiment mismatch index.

## Pipeline overview

| # | Script | What it does |
|---|---|---|
| 01 | `scripts/01_preprocess.py` | Filter, deduplicate and language-detect raw reviews (1.45M → 365,697) |
| 02 | `scripts/02_embed.py` | Sentence-transformer embeddings on MPS, cached to disk |
| 03 | `scripts/03_stability.py` | Five-seed BERTopic stability with pairwise ARI/NMI |
| 04 | `scripts/04_param_grid.py` | Sweep `min_topic_size` ∈ {15, 25, 50, 100, 150, 200} |
| 05 | `scripts/05_final_model.py` | Final BERTopic fit at `min_topic_size = 200`, seed 42 |
| 06–06b | `scripts/06_coder_template.py`, `scripts/06b_coder_agreement.py` | Two-coder topic-label template and Cohen κ |
| 07 | `scripts/07_sentiment_score.py` | DistilBERT-SST2 sentiment per review |
| 07b–07c | `scripts/07b_sentiment_validation_template.py`, `scripts/07c_sentiment_validation_metrics.py` | 300-review validation template + metrics |
| 08 | `scripts/08_mismatch_robustness.py` | Four normalisations × six thresholds of mismatch |
| 09 | `scripts/09_topic_mismatch_stats.py` | Kruskal–Wallis + Dunn post-hoc + rank-biserial |
| 10 | `scripts/10_outlier_handling.py` | Outlier sensitivity analysis (Spearman *ρ*) |
| 11 | `scripts/11_llm_topic_labels.py` | LLM topic labels (gpt-4o-mini via OpenAI API) |
| 12 | `scripts/12_hierarchical_topics.py` | Ward hierarchy 250 → 25 → 6 |
| 13 | `scripts/13_low_rating_hierarchy.py` | Second-pass BERTopic on rating ≤ 3 subset |
| 14 | `scripts/14_figures.py` | All publication figures |
| 17 | `scripts/17_patch_fallback_labels.py` | Patch Pass 1 fallback meso label |
| 18 | `scripts/18_baseline_filter_test.py` | Filter-and-rank baseline vs. Pass 2 |
| 19 | `scripts/19_coder_B_sentiment.py` | Second-coder sentiment labels (60 reviews) |
| 20 | `scripts/20_cohen_kappa_sentiment.py` | Cohen κ for sentiment (3-class) |
| 21 | `scripts/21_pass2_outlier_sensitivity.py` | Pass 2 outlier sensitivity (Spearman *ρ*) |
| 22 | `scripts/22_coder_B_topic_labels.py` | Second-coder topic-label evaluation (50 micro topics) |
| 23 | `scripts/23_pass2_dendrogram_diagnosis.py` | Pass 2 dendrogram diagnostics (silhouette, inter-macro distances, Jaccard) |
| 25 | `scripts/25_pass2_final_k3_hierarchy.py` | Final k=3 macro / k=7 meso hand-labelled Pass 2 hierarchy |
| 99 | `scripts/99_summary_table.py` | Methods-section summary table |

## Quickstart

```bash
# Python 3.11 recommended (tested on 3.11.15)
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt   # pinned versions

# Acquire the dataset (CC0)
# https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera
# place Coursera_reviews.csv and Coursera_courses.csv at the repo root

# OpenAI API key for LLM labelling steps (11, 12, 13)
cp .env.example .env   # edit OPENAI_API_KEY and OPENAI_MODEL

# Reproduce the full pipeline
bash run_all_v2.sh
```

Wall-clock time on an Apple M2 with MPS acceleration: approximately three hours for the full pipeline excluding LLM labelling, plus approximately five minutes and ~US$0.10–0.30 in API fees for LLM labelling.

## Repository layout

```
.
├── README.md                this file
├── LICENSE                  CC BY 4.0 (code: MIT)
├── requirements.txt         loose dependency pins
├── requirements.lock.txt    exact versions used in the paper
├── .env.example             OpenAI API key template
├── run_all_v2.sh            end-to-end driver
├── scripts/                 numbered Python scripts (01–25)
└── outputs/                 reproducible artefacts (released for review)
    ├── tables/              CSV/JSON/Parquet/Excel tables referenced in the paper
    ├── figures/             intermediate publication figures (PNG/HTML)
    └── sentiment/           sentiment validation template and metadata
```

## Headline numbers (reproducible)

| Quantity | Value |
|---|---|
| Reviews analysed | 365,697 (out of 1,454,711 raw) |
| Unique courses | 597 |
| Micro / meso / macro topics (Pass 1) | 250 / 25 / 6 |
| Hidden complaint tree (Pass 2) | 68 / 7 / 3 on rating ≤ 3 (n = 25,274) |
| Pass 2 macro composition | M1 *Course-specific* (n = 8,824) · M2 *Generic pedagogical* (n = 7,563) · M3 *Administrative / enrolment* (n = 529) |
| Topic-model stability (5 seeds) | ARI = 0.741 ± 0.024; NMI = 0.880 ± 0.006 |
| Coherence | c_v = 0.592, c_NPMI = 0.090 |
| Topic diversity | 0.888 |
| Mismatch rate (raw / binary normalisation) | 7–8 % |
| Topic-level mismatch heterogeneity | Kruskal–Wallis *H* = 39,357, *p* < .001, *k* = 250 |
| Outlier-sensitivity (Pass 1) | Spearman *ρ* = 0.986 |
| Outlier-sensitivity (Pass 2, k = 3 / k = 7) | Spearman *ρ* = 1.000 |
| Sentiment-classifier validation | accuracy = 0.859, macro F1 = 0.857, n = 284 |
| Inter-coder agreement, sentiment (3-class, 60 reviews) | Cohen κ = 0.725 |
| Inter-coder agreement, topic labels (50 of 250 Pass 1 micros) | 41 accept / 6 reject / 3 abstain (82 %) |

## Citation

If you use this code or build on this work, please cite the paper (citation will be added once published). Until then:

```bibtex
@unpublished{hiddenmooc2026,
  title  = {Hidden Complaint Structures in MOOC Reviews: A Three-Level Hierarchical
            Topic Analysis with BERTopic and LLM-Augmented Labelling},
  author = {Author},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## Data and ethics

The Coursera review corpus used here is the publicly-released *Coursera Course Reviews* dataset (Rana, 2020) on Kaggle, distributed under CC0 1.0. We do not redistribute the raw corpus; users must download it directly from Kaggle. All downstream outputs released here are derived statistics, topic models and labels — none contain raw learner text identifiable to an individual.

## AI usage statement

`gpt-4o-mini` (OpenAI) was used as a topic-labelling tool, as documented in `scripts/11_llm_topic_labels.py` and `scripts/13_low_rating_hierarchy.py`. The model and prompt versions are pinned in code; the LLM output is logged verbatim in `outputs/tables/11_llm_topic_labels.csv` and `outputs/tables/13_low_rating_level_labels.csv`. No part of the manuscript text itself was generated by a large language model.

## License

Code: MIT (see `LICENSE`).
Data and figures in `outputs/`: CC BY 4.0.
