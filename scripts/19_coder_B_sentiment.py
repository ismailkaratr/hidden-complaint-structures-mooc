"""
Coder B sentiment labels for the 60 three-star validation rows.
Labels assigned independently by Claude Opus 4, without reference to
the original (Coder A) labels stored in the validation template.
Rule: 'positive' if dominant tone is praise; 'negative' if dominant
tone is complaint / disappointment; blank if genuinely mixed or
descriptive without clear tone.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# doc_id  →  Coder B label  ('positive' | 'negative' | None)
CODER_B = {
    183821: "negative",   # great materials BUT forums not attended; complaint dominates
    128560: "negative",   # 'good content but ... need updating' — complaint dominates
    99206:  "negative",   # extended unease about TensorFlow teaching depth
    186393: "negative",   # 'quality of the material is quite poor'; multi-complaint
    88636:  "negative",   # 'great material' + work needs to be done; constructive but complaint dominant
    323112: None,         # genuinely ambivalent: 'good, but...' + speculative
    139178: "negative",   # would be better if... complaint
    119992: "negative",   # assignments difficult for intro
    208221: "positive",   # 'Ok as intro' — mild positive
    42924:  "positive",   # 'Good, This course covers...'
    356302: "negative",   # 'didn't teach you how' — complaint dominant despite advice tone
    167526: "positive",   # 'Great for what it is'
    161196: "negative",   # narrator too fast, could not engage
    282044: "negative",   # 'Not as much hands on as I hoped'
    168573: "negative",   # 'too basic', 'videos could be much shorter' — complaint
    161150: "negative",   # 'OK' but extended complaint about glossed-over step
    52885:  "positive",   # 'Good introduction'
    97445:  "negative",   # 'needs an update', 'confused'
    333501: "positive",   # 'Good course if...'
    178200: "positive",   # mixed but ends positive — 'great start for moving into the NLP field'
    229241: "negative",   # 'too much on the easy side', 'not rigorous enough'
    193659: "positive",   # 'nice course at least to start with'
    251861: "negative",   # 'Sometimes too technical', 'completely useless for me'
    17972:  "positive",   # 'enjoyed the rest'
    97512:  "negative",   # 'material is chaotic'
    270958: None,         # explanations not thorough BUT 'learned the information, course did what it needed'
    183760: None,         # mixed: wished more resources BUT 'really liked how...'
    331689: None,         # mixed: 'very well thought out' BUT week on Islam dull, peer-assign issues
    243231: "negative",   # 'assessments don't seem to align', 'more strategies needed'
    52756:  None,         # mixed: appreciates BUT not as much scientific info as hoped
    302233: "negative",   # 'feel like you are being read to', 'spacing out'
    93499:  None,         # purely descriptive: 'basics about TensorFlow', 'simple'
    289467: None,         # purely descriptive: 'theoretical knowledge about photography'
    4902:   None,         # 'thank you' + 'make it more engaging' — too short to pick
    128486: "positive",   # 'just perfect' for need, 'great speaker'
    284018: "negative",   # 'a total bust', 'not worth it for the money'
    210400: "negative",   # 'tests only compatible with firefox' — complaint
    247263: "positive",   # 'I learnt a lot from this course, thank you Coursera'
    30988:  None,         # mixed: 'simplistic' BUT 'very well organized structure'
    181205: "negative",   # 'less assigment or practice test'
    119865: None,         # mixed: 'rushed and poorly explained' BUT 'explained the bare bones'
    222903: "positive",   # 'very interesting and usefull course. Learned a lot! 5 star.'
    201346: "negative",   # 'doesn't focus much on helping to improve'
    282849: "negative",   # 'teaching was very poor', 'very confused'
    97573:  "negative",   # 'contents should be updated', 'confused'
    95609:  "negative",   # 'instructions ... abyssmal'
    202630: "positive",   # 'Very good overview'
    214592: "negative",   # 'lacked interactive material', 'no answers given'
    239640: None,         # purely a suggestion, no tone
    179660: None,         # explicitly 'why three stars?' — mixed by author's own statement
    171191: "negative",   # 'data is outdated', 'looking for new course'
    139140: "negative",   # 'Not nearly as valuable', 'more subjective'
    42946:  "negative",   # 'pace is quite slow'
    144872: "negative",   # numerous technical problems
    205512: "negative",   # 'no prior material shown'
    133611: None,         # request for more content; not really tone-bearing
    325760: "positive",   # 'Fun lectures', looking forward to more
    213750: None,         # mixed: good course BUT could have more programming assigns; jupyter buggy
    228812: "negative",   # 'Sorry' if want to learn coding
    89530:  "positive",   # 'so great to know that much in Korean'
}

assert len(CODER_B) == 60, len(CODER_B)


def main() -> None:
    src = Path("outputs/sentiment/07b_sentiment_validation_template.csv")
    df = pd.read_csv(src)
    df["coder_B"] = df["doc_id"].map(CODER_B)
    df["coder_B"] = df["coder_B"].fillna("").replace({"": pd.NA})
    df.to_csv(src, index=False)
    print(f"Wrote Coder B labels into {src}")

    three = df[df.rating == 3].copy()
    A = three["gold_label"]
    B = three["coder_B"]
    print()
    print("Three-star joint distribution (rows = Coder A, cols = Coder B):")
    print(pd.crosstab(A.fillna("(blank)"), B.fillna("(blank)"),
                       margins=True).to_string())


if __name__ == "__main__":
    main()
