"""
Step 11 — LLM-based topic labels via OpenAI representation model.

Loads the already-fit BERTopic model (step 05), attaches a new representation
model that asks an LLM for a short human label per topic, then re-extracts the
representations *without* re-clustering. Cluster assignments stay identical —
only the topic 'name' changes.

Outputs:
    outputs/tables/11_llm_topic_labels.csv
        topic_id, count, keywords_baseline, llm_label, llm_description (optional)
    outputs/models/final_bertopic_llm/   (BERTopic.save)

Cost note: ~250 topics × ~1 chat call ≈ $0.10-0.30 with gpt-4o-mini.
Cluster geometry is NOT re-computed, so this step takes minutes, not hours.

Usage:
    # .env: OPENAI_API_KEY=sk-..., OPENAI_MODEL=gpt-4o-mini
    .venv/bin/python scripts/11_llm_topic_labels.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    EMBEDDINGS_NPY, MODEL_DIR, OUTPUT_DIR, PREPROCESSED_PARQUET, RunRecord,
    TABLE_DIR, dump_env, setup_logger,
)

SCRIPT_NAME = "11_llm_topic_labels"

# Prompt — single short label per topic. We ask the LLM to focus on student
# *experience* themes (assignments, pacing, instructors, infra) rather than
# course names, because the latter are already obvious from keywords.
PROMPT = """I have a topic that contains the following documents (Coursera student reviews):
[DOCUMENTS]

The topic is described by the following keywords: [KEYWORDS]

Based on the documents and keywords, give a short, specific English label (max 8 words)
that describes the *student-experience theme* of this topic. Prefer themes like:
"complaints about peer-grading", "praise for instructor X", "request for updated content",
"video pacing issues", over generic course names like "Python course".

Reply with ONLY the label, no quotes, no explanation."""


try:
    from bertopic.representation._base import BaseRepresentation as _BaseRepr
except Exception:
    _BaseRepr = object


class ChatLabelRepresentation(_BaseRepr):
    """Minimal BERTopic representation: ask an LLM for one short label per topic.

    Compatible with both classic chat-completion models (gpt-4o*) and the
    gpt-5.x family that rejects 'stop' / 'max_tokens'. BERTopic 0.17's bundled
    OpenAI wrapper still passes those parameters, so we roll our own.
    """

    def __init__(self, *, client, model: str, prompt: str,
                  nr_docs: int = 5, doc_length: int = 120,
                  tokenizer=None, log=None):
        self.client = client
        self.model = model
        self.prompt = prompt
        self.nr_docs = nr_docs
        self.doc_length = doc_length
        self.tokenizer = tokenizer
        self.log = log

    def _truncate(self, text: str) -> str:
        if self.tokenizer is None:
            return text[: self.doc_length * 5]
        ids = self.tokenizer.encode(text)
        return self.tokenizer.decode(ids[: self.doc_length])

    def _ask(self, prompt_text: str) -> str:
        # Use max_completion_tokens (new style); supported by all current models.
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_text}],
                max_completion_tokens=64,
            )
        except Exception as e:
            if self.log:
                self.log.warning("LLM call failed: %s", e)
            return ""
        msg = r.choices[0].message.content or ""
        return msg.strip().splitlines()[0][:200] if msg else ""

    # BERTopic representation API
    def extract_topics(self, topic_model, documents, c_tf_idf, topics):
        from tqdm import tqdm

        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, nr_repr_docs=self.nr_docs
        )
        updated = {}
        for topic_id, words in tqdm(topics.items(), desc="LLM labels"):
            kw = ", ".join(w for w, _ in words[:10])
            if topic_id == -1:
                # keep outlier topic's existing words; no LLM call needed
                updated[topic_id] = words
                continue
            reps = repr_docs_mappings.get(topic_id, [])[: self.nr_docs]
            docs_block = "\n".join(f"- {self._truncate(d)}" for d in reps) or "(no documents)"
            prompt_text = self.prompt.replace("[DOCUMENTS]", docs_block).replace("[KEYWORDS]", kw)
            label = self._ask(prompt_text) or words[0][0]
            # BERTopic expects a list of (word, weight) tuples. Put the LLM label
            # first so it surfaces as Topic.Name; keep the c-tf-idf words after.
            updated[topic_id] = [(label, 1.0)] + list(words[:9])
        return updated


def _load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    _load_env()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-..."):
        raise SystemExit("OPENAI_API_KEY missing or unset in .env")
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()

    log = setup_logger(SCRIPT_NAME)
    dump_env(log)
    log.info("LLM model: %s", model_name)

    import openai
    import tiktoken
    from bertopic import BERTopic

    # ---- Load BERTopic model + docs + embeddings
    bt_path = MODEL_DIR / "final_bertopic"
    log.info("Loading BERTopic model from %s", bt_path)
    topic_model = BERTopic.load(str(bt_path))

    df = pd.read_parquet(PREPROCESSED_PARQUET, columns=["doc_id", "reviews"])
    embeddings = np.load(EMBEDDINGS_NPY, mmap_mode="r")
    docs = df["reviews"].tolist()
    log.info("Docs: %d | Embeddings: %s", len(docs), embeddings.shape)

    # ---- Build OpenAI representation
    client = openai.OpenAI(api_key=api_key)
    try:
        tokenizer = tiktoken.encoding_for_model(model_name)
    except KeyError:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    rep_model = ChatLabelRepresentation(
        client=client, model=model_name, prompt=PROMPT,
        nr_docs=5, doc_length=120, tokenizer=tokenizer, log=log,
    )

    # ---- Re-extract topic *labels* without re-clustering.
    # update_topics keeps the cluster assignments fixed and only recomputes
    # the representation (top words + LLM label).
    log.info("Generating LLM labels for %d topics ...",
             len([t for t in topic_model.get_topics() if t != -1]))
    t0 = time.time()
    topic_model.update_topics(
        docs,
        representation_model=rep_model,
        # keep the same n-gram / stopword vectorizer
        vectorizer_model=topic_model.vectorizer_model,
    )
    dt = time.time() - t0
    log.info("LLM labelling done in %.1fs", dt)

    # ---- Persist
    out_model_path = MODEL_DIR / "final_bertopic_llm"
    topic_model.save(str(out_model_path), serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=False)
    log.info("Saved LLM-labeled model to %s", out_model_path)

    # ---- Build comparison table
    baseline = pd.read_csv(TABLE_DIR / "05_topic_info.csv")
    baseline = baseline.rename(columns={"Topic": "topic_id", "Count": "count",
                                          "Name": "keywords_baseline"})
    baseline = baseline[["topic_id", "count", "keywords_baseline"]]

    rows = []
    new_info = topic_model.get_topic_info()
    for _, r in new_info.iterrows():
        tid = int(r["Topic"])
        if tid == -1:
            continue
        # BERTopic concatenates the LLM output into Name with the keywords
        # appended; the Representation column holds the raw LLM label.
        llm_label = r.get("Representation", r["Name"])
        if isinstance(llm_label, list):
            llm_label = llm_label[0] if llm_label else ""
        rows.append({"topic_id": tid, "llm_label": str(llm_label).strip()})

    llm_df = pd.DataFrame(rows)
    merged = baseline.merge(llm_df, on="topic_id", how="left")
    merged = merged.sort_values("count", ascending=False)
    out_csv = TABLE_DIR / "11_llm_topic_labels.csv"
    merged.to_csv(out_csv, index=False)
    log.info("Wrote comparison table -> %s", out_csv)

    # ---- Tiny summary
    summary = {
        "model": model_name,
        "n_topics_labelled": int(len(rows)),
        "duration_seconds": round(dt, 1),
        "approx_cost_usd_note": "gpt-4o-mini ~$0.10-0.30; verify on dashboard",
    }
    (TABLE_DIR / "11_llm_topic_labels_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    RunRecord(script=SCRIPT_NAME, seed=0, params=summary,
              n_inputs=len(rows), n_outputs=len(rows)
              ).save(OUTPUT_DIR / "11_llm_topic_labels.run.json")

    # Preview
    log.info("Preview (top 15 by count):")
    log.info("\n%s", merged[["topic_id", "count", "llm_label"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
