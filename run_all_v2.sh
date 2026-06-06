#!/usr/bin/env bash
# v2: with fastText English-only language filter in step 01.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
ts() { date "+%Y-%m-%d %H:%M:%S"; }
run() {
    local name="$1"; shift
    echo
    echo "==================== [$(ts)] BEGIN $name ===================="
    "$PY" "$@"
    echo "==================== [$(ts)] END   $name ===================="
}

MIN_TOPIC_SIZE=200
FINAL_SEED=42

run "01_preprocess"               scripts/01_preprocess.py
run "02_embed"                    scripts/02_embed.py
run "03_stability"                scripts/03_stability.py --min-topic-size "$MIN_TOPIC_SIZE"
run "04_param_grid"               scripts/04_param_grid.py
run "05_final_model"              scripts/05_final_model.py --min-topic-size "$MIN_TOPIC_SIZE" --seed "$FINAL_SEED"
run "06_coder_template"           scripts/06_coder_template.py
run "07_sentiment_score"          scripts/07_sentiment_score.py
run "07b_sent_val_template"       scripts/07b_sentiment_validation_template.py --n 300
run "08_mismatch_robustness"      scripts/08_mismatch_robustness.py
run "09_topic_mismatch_stats"     scripts/09_topic_mismatch_stats.py
run "10_outlier_handling"         scripts/10_outlier_handling.py
run "11_llm_topic_labels"         scripts/11_llm_topic_labels.py
run "12_hierarchical_topics"      scripts/12_hierarchical_topics.py
run "13_low_rating_hierarchy"     scripts/13_low_rating_hierarchy.py
run "14_figures"                  scripts/14_figures.py
run "17_patch_fallback_labels"    scripts/17_patch_fallback_labels.py
run "18_baseline_filter_test"     scripts/18_baseline_filter_test.py
run "20_cohen_kappa_sentiment"    scripts/20_cohen_kappa_sentiment.py
run "21_pass2_outlier_sensitivity" scripts/21_pass2_outlier_sensitivity.py
run "22_coder_B_topic_labels"     scripts/22_coder_B_topic_labels.py
run "23_pass2_dendrogram_diagnosis" scripts/23_pass2_dendrogram_diagnosis.py
run "25_pass2_final_k3_hierarchy" scripts/25_pass2_final_k3_hierarchy.py
run "99_summary_table"            scripts/99_summary_table.py

echo
echo "==================== [$(ts)] PIPELINE DONE ===================="
