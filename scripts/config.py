"""
Shared configuration: paths, seeds, logging.
All scripts import from here so seeds and paths stay consistent.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
EMB_DIR = OUTPUT_DIR / "embeddings"
MODEL_DIR = OUTPUT_DIR / "models"
CODER_DIR = OUTPUT_DIR / "coder_templates"
SENT_DIR = OUTPUT_DIR / "sentiment"

for _d in (LOG_DIR, TABLE_DIR, FIG_DIR, EMB_DIR, MODEL_DIR, CODER_DIR, SENT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

COURSES_CSV = DATA_DIR / "Coursera_courses.csv"
REVIEWS_CSV = DATA_DIR / "Coursera_reviews.csv"

PREPROCESSED_PARQUET = OUTPUT_DIR / "reviews_clean.parquet"
EMBEDDINGS_NPY = EMB_DIR / "all-MiniLM-L6-v2.npy"
EMBEDDING_INDEX_PARQUET = EMB_DIR / "embedding_index.parquet"

# ---------- Reproducibility ----------
GLOBAL_SEED = 42
STABILITY_SEEDS: tuple[int, ...] = (13, 42, 123, 777, 2024)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# BERTopic / UMAP / HDBSCAN defaults — overridden by grid where relevant
UMAP_PARAMS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
HDBSCAN_PARAMS = dict(min_cluster_size=50, metric="euclidean", cluster_selection_method="eom")
MIN_TOPIC_SIZE_GRID = (15, 25, 50, 100, 150, 200)


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Seed all RNGs we control. Call once at script entry."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_torch_device() -> str:
    """Return 'mps' on Apple Silicon, else 'cuda' / 'cpu'."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def setup_logger(name: str) -> logging.Logger:
    """File + stdout logger. One log file per script in outputs/logs/."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / f"{name}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def dump_env(logger: logging.Logger) -> dict:
    """Record key package versions into the log — for methods section."""
    import importlib

    pkgs = [
        "pandas", "numpy", "scipy", "sklearn", "torch", "sentence_transformers",
        "transformers", "bertopic", "umap", "hdbscan", "gensim", "nltk",
        "statsmodels", "scikit_posthocs",
    ]
    versions = {}
    for p in pkgs:
        try:
            mod = importlib.import_module(p)
            versions[p] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[p] = "NOT INSTALLED"
    versions["python"] = sys.version.split()[0]
    versions["device"] = get_torch_device()
    logger.info("Environment: %s", json.dumps(versions, indent=2))
    return versions


@dataclass
class RunRecord:
    """Serialized to JSON next to each artifact for traceability."""
    script: str
    seed: int
    params: dict
    n_inputs: int | None = None
    n_outputs: int | None = None
    extra: dict | None = None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")


def iter_chunks(seq: Iterable, size: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
