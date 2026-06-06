"""
Retry only the rows in 11_llm_topic_labels.csv where the LLM call failed
(label is empty / single keyword). Re-asks the LLM and patches the CSV in place.
Cheap (only ~75 calls) and safe to re-run.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import tiktoken
from openai import OpenAI

from config import MODEL_DIR, PREPROCESSED_PARQUET, TABLE_DIR, setup_logger
import sys, importlib
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
m11 = import_module("11_llm_topic_labels")  # reuse PROMPT


def _load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def is_fallback(label: str) -> bool:
    if not isinstance(label, str):
        return True
    return len(label.split()) <= 2


def main() -> None:
    _load_env()
    log = setup_logger("11b_llm_labels_retry")
    api_key = os.environ["OPENAI_API_KEY"].strip()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    log.info("Retrying with model: %s", model)

    df = pd.read_csv(TABLE_DIR / "11_llm_topic_labels.csv")
    df = df[df.topic_id != -1].copy()
    mask = df["llm_label"].fillna("").astype(str).map(is_fallback)
    targets = df[mask].copy()
    log.info("Targets needing retry: %d", len(targets))
    if not len(targets):
        return

    from bertopic import BERTopic
    bt = BERTopic.load(str(MODEL_DIR / "final_bertopic_llm"))
    docs_df = pd.read_parquet(PREPROCESSED_PARQUET, columns=["doc_id", "reviews"])
    docs = docs_df["reviews"].tolist()

    # Pull representative docs from the BERTopic model for the failed topics
    rep_docs = bt.get_representative_docs() or {}

    client = OpenAI(api_key=api_key)
    try:
        tok = tiktoken.encoding_for_model(model)
    except KeyError:
        tok = tiktoken.get_encoding("cl100k_base")

    full = pd.read_csv(TABLE_DIR / "11_llm_topic_labels.csv")
    fixed = 0
    for tid in targets["topic_id"]:
        words = bt.get_topic(int(tid)) or []
        kw = ", ".join(w for w, _ in words[:10])
        reps = rep_docs.get(int(tid), [])[:5]
        docs_block = "\n".join(f"- {tok.decode(tok.encode(d)[:120])}" for d in reps) or "(no documents)"
        prompt = m11.PROMPT.replace("[DOCUMENTS]", docs_block).replace("[KEYWORDS]", kw)
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=64,
                )
                label = (r.choices[0].message.content or "").strip().splitlines()[0][:200] if r.choices[0].message.content else ""
                if label:
                    full.loc[full.topic_id == tid, "llm_label"] = label
                    fixed += 1
                    log.info("  topic %4d -> %s", tid, label)
                    break
            except Exception as e:
                wait = 2 ** attempt
                log.warning("  topic %d attempt %d failed: %s (sleep %ds)", tid, attempt + 1, e, wait)
                time.sleep(wait)
        time.sleep(0.5)

    full.to_csv(TABLE_DIR / "11_llm_topic_labels.csv", index=False)
    log.info("Patched %d / %d labels; CSV updated.", fixed, len(targets))


if __name__ == "__main__":
    main()
