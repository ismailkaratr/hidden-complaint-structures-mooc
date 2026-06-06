"""
Collects the one-number-per-validity-check artifacts that the earlier scripts
produced into a single table for the methods section.

Usage:
    python scripts/99_summary_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import TABLE_DIR, setup_logger

SCRIPT_NAME = "99_summary_table"


def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    log = setup_logger(SCRIPT_NAME)

    pre = _load_json(TABLE_DIR / "01_preprocess_summary.json")
    emb = _load_json(TABLE_DIR / "02_embed_meta.json")
    stab = _load_json(TABLE_DIR / "03_stability_summary.json")
    coh = _load_json(TABLE_DIR / "05_final_coherence.json")
    coder = _load_json(TABLE_DIR / "06_coder_agreement_summary.json")
    sent = _load_json(TABLE_DIR / "07_sentiment_validation_summary.json")
    robust = _load_json(TABLE_DIR / "08_mismatch_robustness_summary.json")
    stats = _load_json(TABLE_DIR / "09_topic_mismatch_stats_summary.json")
    outl = _load_json(TABLE_DIR / "10_outlier_handling_summary.json")

    rows = [
        ("Adim 1 — Veri",
         f"{pre.get('final_reviews', '?')} yorum, {pre.get('unique_courses', '?')} kurs "
         f"(ham: {pre.get('raw_reviews', '?')}, korunan: %{pre.get('kept_pct', '?')})"),
        ("Adim 1 — Seed", f"global_seed = {pre.get('seed', '?')}"),
        ("Adim 2 — Embedding",
         f"{emb.get('model', '?')} on {emb.get('device', '?')}, dim={emb.get('embedding_dim', '?')}"),
        ("Adim 2 — Karalilik (5 seed)",
         f"ARI = {stab.get('ARI_mean', '?')} ± {stab.get('ARI_std', '?')}; "
         f"NMI = {stab.get('NMI_mean', '?')} ± {stab.get('NMI_std', '?')}; "
         f"konu sayisi = {stab.get('n_topics_mean', '?')} ± {stab.get('n_topics_std', '?')}"),
        ("Adim 3 — Parametre taramasi",
         "outputs/tables/04_param_grid.csv (min_topic_size x coherence/diversity/outlier)"),
        ("Adim 4 — Final tutarlilik",
         f"c_v = {coh.get('c_v', '?')}, c_npmi = {coh.get('c_npmi', '?')} "
         f"(min_topic_size = {coh.get('min_topic_size', '?')}, seed = {coh.get('seed', '?')})"),
        ("Adim 5 — Kodlayici uyumu",
         f"Cohen's kappa = {coder.get('cohen_kappa', '? (henuz etiketlenmedi)')}, "
         f"tam eslesme = {coder.get('exact_match_rate', '?')} "
         f"(n = {coder.get('n_double_coded', '?')})"),
        ("Adim 6 — Duygu siniflandirici dogrulamasi",
         f"accuracy = {sent.get('accuracy', '? (henuz etiketlenmedi)')}, "
         f"macro F1 = {sent.get('macro_f1', '?')}, n = {sent.get('n_gold', '?')}"),
        ("Adim 7 — Uyumsuzluk saglamlik",
         f"normalizasyonlar = {robust.get('normalizations', '?')}, "
         f"esikler = {robust.get('thresholds', '?')}"),
        ("Adim 8 — Topic x mismatch testi",
         f"Kruskal-Wallis H = {stats.get('kruskal_H', '?')}, "
         f"p = {stats.get('kruskal_p', '?')}, k = {stats.get('n_topics_tested', '?')} konu"),
        ("Adim 9 — Outlier ele alinisi",
         f"outlier_share = {outl.get('outlier_share', '?')}, "
         f"rho(original, reassigned) = "
         f"{outl.get('spearman_original_vs_reassigned', {}).get('rho', '?')}"),
    ]

    df = pd.DataFrame(rows, columns=["Madde", "Rapor edilecek deger"])
    out = TABLE_DIR / "99_methods_summary.csv"
    df.to_csv(out, index=False)
    log.info("Wrote summary -> %s", out)
    log.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
