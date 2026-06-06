"""
One-off patch: replace fallback labels (e.g. 'mezo-11') in the
hierarchy tables with a descriptive label derived by the authors
from the constituent micro-topics.

This is a *manual* edit, not an LLM call — the LLM step left one
meso label as the raw fallback `mezo-11`; the eight micro-topics
under it (corporate finance, financial accounting, marketing
analytics, people analytics, economics, Robert Shiller, business
metrics, forensic accounting) jointly read as a clear business-
and-finance theme. We apply the patch in the same loader paths
that the figure / docx pipeline reads from.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TBL = Path("outputs/tables")

# Replacements: (level_label, old_text) -> new_text
REPLACEMENTS = [
    ("mezo", "mezo-11", "business and finance course praise"),
]


def patch_hierarchy_level_labels(path: Path) -> int:
    df = pd.read_csv(path)
    n_changed = 0
    for level, old, new in REPLACEMENTS:
        mask = (df.level == level) & (df.label == old)
        n_changed += int(mask.sum())
        df.loc[mask, "label"] = new
    df.to_csv(path, index=False)
    return n_changed


def patch_topic_to_levels(path: Path) -> int:
    """No label columns in topic_to_levels; nothing to patch directly,
    but we keep this function for symmetry / future-proofing."""
    return 0


def main() -> None:
    n12 = patch_hierarchy_level_labels(TBL / "12_level_labels.csv")
    print(f"12_level_labels.csv → patched {n12} row(s)")

    n13 = patch_hierarchy_level_labels(TBL / "13_low_rating_level_labels.csv")
    print(f"13_low_rating_level_labels.csv → patched {n13} row(s)")


if __name__ == "__main__":
    main()
