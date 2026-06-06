"""
Stage 2 finalisation — rebuild the Pass-2 hierarchy as k=3 macros with
manually-written, mechanism-focused labels.

After Stage 1 diagnosis (script 23) and Stage 2 cut exploration
(script 24) we settled on a three-macro structure: the four delivery-
quality micros (former Macro 2) and the two course-clarity micros
(former Macro 4 / Meso 7) are absorbed into the two large content
clusters, leaving:

    Macro 1 — Course-specific content and assignment problems
              (former Macro 1 + former delivery-quality cluster)
    Macro 2 — Generic pedagogical-quality complaints
              (former Macro 3 / Meso 8, our largest single complaint
              cluster)
    Macro 3 — Administrative and enrolment frictions
              (former Macro 3 / Meso 6, a small but operationally
              distinct cluster)

Mesos are then re-cut at k=6 inside Macros 1 and 2 to retain a
readable sub-structure. All labels are written by hand here by the
authors (cross-checked against the constituent micros). The script
overwrites:

    outputs/tables/13_low_rating_level_labels.csv
    outputs/tables/13_low_rating_topic_to_levels.csv

so that every downstream consumer (the bar chart, the manuscript
tables, Appendix B2) sees the new labels.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TBL = Path("outputs/tables")
OLD_LEVELS = TBL / "13_low_rating_level_labels.csv"
OLD_TOPIC2LEVEL = TBL / "13_low_rating_topic_to_levels.csv"


# ---------------------------------------------------------------------------
# Manual macro/meso assignment
# ---------------------------------------------------------------------------
#
# topic_id -> (new_macro_id, new_meso_id)
#
# Macro 1 — Course-specific content and assignment problems
#   M1 / m1 — Assignment design and lecture–assessment alignment
#   M1 / m2 — Outdated or thin course content (course-specific)
#   M1 / m3 — Lab, tooling and platform-specific frictions
#   M1 / m4 — Audio, slide and delivery-quality issues
#
# Macro 2 — Generic pedagogical-quality complaints
#   M2 / m5 — Lack of depth and engagement
#   M2 / m6 — Outdated content and assessment errors (generic)
#
# Macro 3 — Administrative and enrolment frictions
#   M3 / m7 — Certificate, payment and enrolment problems
#
TOPIC_MAP: dict[int, tuple[int, int]] = {
    # ---- Macro 1 / Meso 1 — Assignment design and lecture–assessment alignment
    2:  (1, 1),   # Lack of clarity and support for assignments
    3:  (1, 1),   # Discrepancy between lectures and assignments
    11: (1, 1),   # Inadequate alignment of assignments and lectures
    6:  (1, 1),   # Course organization and unclear assignments
    7:  (1, 1),   # Lack of depth and engagement in content (assignment-flavoured)
    23: (1, 1),   # Insufficient mathematical depth and support
    33: (1, 1),   # Lack of clarity in video assignments and instructions
    50: (1, 1),   # Overwhelming course content and unclear instructions
    51: (1, 1),   # Mismatch between lectures and assignments

    # ---- Macro 1 / Meso 2 — Outdated or thin course content (course-specific)
    13: (1, 2),   # Outdated course materials and service issues
    16: (1, 2),   # Outdated content and lack of relevant emphasis
    21: (1, 2),   # Outdated tutorial videos and interface issues
    44: (1, 2),   # Course content is outdated and lacks updates
    29: (1, 2),   # Outdated content and poor JavaScript instruction
    19: (1, 2),   # Course content misalignment with expectations and difficulty
    32: (1, 2),   # Course difficulty and content coverage issues
    34: (1, 2),   # Lack of depth and specificity in content
    18: (1, 2),   # Lack of depth and clarity in content
    20: (1, 2),   # Lack of depth in nutrition content
    47: (1, 2),   # Content bias and lack of diverse perspectives
    17: (1, 2),   # Insufficient explanations and course complexity issues

    # ---- Macro 1 / Meso 3 — Lab, tooling and platform-specific frictions
    5:  (1, 3),   # Lab specifications and clarity issues
    12: (1, 3),   # Course content lacks depth and practical value
    37: (1, 3),   # Lack of practical machine learning training
    38: (1, 3),   # Lab functionality and performance issues
    52: (1, 3),   # Inadequate Java learning resources and tools
    54: (1, 3),   # Inadequate instruction for beginners in commands
    55: (1, 3),   # Autograder issues hinder coding learning experience
    65: (1, 3),   # Lack of technical depth and practical examples
    59: (1, 3),   # Lack of practical application in networking training
    46: (1, 3),   # Limited use of open-source tools in course
    40: (1, 3),   # Programming assignments and course clarity issues
    36: (1, 3),   # Poor course structure and inadequate support
    67: (1, 3),   # Insufficient depth and explanation in course content
    4:  (1, 3),   # Video clarity and course materials [sanitised]
    30: (1, 3),   # Course content and accessibility issues
    31: (1, 3),   # Lack of clarity in math instruction

    # ---- Macro 1 / Meso 4 — Audio, slide and delivery-quality issues
    25: (1, 4),   # Poor audio quality and delivery issues
    27: (1, 4),   # Excessive video length and poor engagement
    26: (1, 4),   # Presentation quality and slide accessibility issues
    28: (1, 4),   # Inadequate lecture quality and structure
    61: (1, 4),   # Lack of logical structure and engaging materials
    49: (1, 4),   # Outdated course content needs urgent update (very small)

    # ---- Macro 2 / Meso 5 — Lack of depth and engagement
    0:  (2, 5),   # Lack of depth and engagement in course material (BIG)
    10: (2, 5),   # Ineffective and monotonous teaching style
    22: (2, 5),   # Poor exercise design and lack of support
    24: (2, 5),   # Course cohesion and clarity issues
    14: (2, 5),   # Insufficient video content and reliance on book
    15: (2, 5),   # Language proficiency and accent difficulties
    35: (2, 5),   # Lack of clarity and difficulty in material
    39: (2, 5),   # Course quality and content issues
    42: (2, 5),   # Misleading course content and format issues
    43: (2, 5),   # Course difficulty and unclear instructions
    45: (2, 5),   # Insufficient structure and engagement in capstone
    48: (2, 5),   # Lack of practical application and clarity
    53: (2, 5),   # Course workload and time expectations
    56: (2, 5),   # Course structure and assignment challenges
    57: (2, 5),   # Course content and organization issues
    58: (2, 5),   # Lack of practical content and disorganization
    60: (2, 5),   # Course content quality and assignment issues
    64: (2, 5),   # Course content focus and structure issues
    66: (2, 5),   # Inadequate course content and structure
    62: (2, 5),   # Lack of support in discussion forums

    # ---- Macro 2 / Meso 6 — Outdated content and assessment errors (generic)
    1:  (2, 6),   # Quiz errors and outdated content complaints (BIG)
    8:  (2, 6),   # Inadequate peer review and grading process

    # ---- Macro 3 / Meso 7 — Certificate, payment and enrolment problems
    9:  (3, 7),   # Certificate issues and poor support experience
    41: (3, 7),   # Unenrollment and course cancellation requests
    63: (3, 7),   # Payment issues and financial aid problems
}


# Manually written, mechanism-focused labels.
MACRO_LABELS = {
    1: "Course-specific content and assignment problems",
    2: "Generic pedagogical-quality complaints",
    3: "Administrative and enrolment frictions",
}

MESO_LABELS = {
    1: "Assignment design and lecture-assessment alignment",
    2: "Outdated or thin course content (course-specific)",
    3: "Lab, tooling and platform-specific frictions",
    4: "Audio, slide and delivery-quality issues",
    5: "Lack of depth, engagement and clarity",
    6: "Outdated content and assessment errors",
    7: "Certificate, payment and enrolment problems",
}


def main() -> None:
    # ---- Sanity check: every Pass-2 micro is assigned exactly once
    old_t2l = pd.read_csv(OLD_TOPIC2LEVEL)
    all_micros = set(old_t2l["topic_id"].astype(int))
    mapped = set(TOPIC_MAP.keys())
    missing = sorted(all_micros - mapped)
    extra = sorted(mapped - all_micros)
    print(f"Pass-2 micros total: {len(all_micros)}, mapped: {len(mapped)}")
    if missing:
        print(f"MISSING from manual map: {missing}")
        raise SystemExit("Update TOPIC_MAP to cover every micro.")
    if extra:
        print(f"EXTRA (not real micros): {extra}")
        raise SystemExit("Remove invalid ids from TOPIC_MAP.")

    # ---- Re-write 13_low_rating_topic_to_levels.csv
    new_t2l = old_t2l[["topic_id"]].copy()
    new_t2l["makro_id"] = new_t2l["topic_id"].map(lambda t: TOPIC_MAP[int(t)][0])
    new_t2l["mezo_id"] = new_t2l["topic_id"].map(lambda t: TOPIC_MAP[int(t)][1])
    # carry over count + label
    new_t2l = new_t2l.merge(old_t2l[["topic_id", "count", "llm_label"]],
                              on="topic_id", how="left")
    new_t2l.to_csv(OLD_TOPIC2LEVEL, index=False)
    print(f"Wrote {OLD_TOPIC2LEVEL}  ({len(new_t2l)} micros)")

    # ---- Re-write 13_low_rating_level_labels.csv
    rows = []
    for mid, lbl in MACRO_LABELS.items():
        sub = new_t2l[new_t2l.makro_id == mid]
        rows.append({"level": "makro", "cluster_id": mid, "label": lbl,
                     "n_mikro_topics": int(len(sub)),
                     "n_docs": int(sub["count"].sum())})
    for sid, lbl in MESO_LABELS.items():
        sub = new_t2l[new_t2l.mezo_id == sid]
        if sub.empty:
            continue
        rows.append({"level": "mezo", "cluster_id": sid, "label": lbl,
                     "n_mikro_topics": int(len(sub)),
                     "n_docs": int(sub["count"].sum())})
    new_levels = pd.DataFrame(rows)
    new_levels.to_csv(OLD_LEVELS, index=False)
    print(f"Wrote {OLD_LEVELS}  ({len(new_levels)} level rows)")

    # ---- Pretty-print composition summary
    total = int(new_t2l["count"].sum())
    print()
    print(f"Total clustered docs (Pass-2, after manual re-cut): {total:,}")
    print()
    print(f"{'Macro':<60} {'docs':>7} {'%':>6} {'mesos':>6} {'micros':>7}")
    for mid in sorted(MACRO_LABELS):
        sub = new_t2l[new_t2l.makro_id == mid]
        n = int(sub["count"].sum())
        print(f"{f'M{mid} — {MACRO_LABELS[mid]}':<60} "
              f"{n:>7,} {n/total*100:>5.1f}% "
              f"{sub.mezo_id.nunique():>6d} {len(sub):>7d}")

    print()
    print(f"{'Macro':<6} {'Meso':<55} {'docs':>7} {'%':>6} {'micros':>7}")
    for mid in sorted(MACRO_LABELS):
        msub = new_t2l[new_t2l.makro_id == mid]
        for sid in sorted(msub.mezo_id.unique()):
            ssub = msub[msub.mezo_id == sid]
            n = int(ssub["count"].sum())
            label = MESO_LABELS.get(sid, "??")
            print(f"{mid:<6} {f'm{sid} — {label}':<55} "
                  f"{n:>7,} {n/total*100:>5.1f}% {len(ssub):>7d}")


if __name__ == "__main__":
    main()
