import os
import re
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
# from bertopic import BERTopic
# from bertopic.representation import KeyBERTInspired
# from umap import UMAP
from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

from config import SENT_TRANSFORMER

ZERO_SHOT_TOPICS = [
    "data center colocation capacity megawatt power infrastructure",
    "artificial intelligence machine learning automation model",
    "revenue earnings growth margin net income EBITDA",
    "guidance outlook forecast expectation quarter",
    "competition market share competitor pricing pressure",
    "cost efficiency expense inflation reduction",
    "acquisition merger joint venture divestiture",
    "regulation compliance regulatory legal ESG sustainability",
    "product innovation launch roadmap development",
    "customer bookings renewals churn retention backlog",
    "partnership alliance collaboration ecosystem",
    "hiring talent employee leadership retention",
]
#
# # Boilerplate patterns to strip from earnings call transcripts
BOILERPLATE_PATTERNS = re.compile(
    r"(?:"
    r"(?:i'?m|my name is|this is)\s+.{0,50}(?:president|ceo|cfo|chief|officer|director|controller)"
    r"|(?:good\s+(?:day|morning|afternoon|evening))"
    r"|(?:thank\s+(?:you|all|everyone)\s+(?:for\s+)?(?:joining|being|attending))"
    r"|(?:let\s+(?:me|us)\s+(?:turn|hand)\s+it\s+over\s+to)"
    r"|(?:without\s+further\s+adieu)"
    r"|(?:i'?d\s+like\s+to\s+(?:welcome|introduce))"
    r")",
    re.IGNORECASE,
)
#
MIN_CHUNK_LEN = 120
#
def clean_paragraph(text: str) -> str|None:
    # clean paragraph; return None if boilerplate is too short
    text = text.strip().replace("\n", " ")
    if len(text) < MIN_CHUNK_LEN:
        return None
    if BOILERPLATE_PATTERNS.search(text):
        return None
    return text

def get_cached_embeddings(docs: list, model_name: str, cache_dir="data/cache"):
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(
        (model_name + "|" + "|".join(docs)).encode("utf-8")
    ).hexdigest()[:16]
    path = os.path.join(cache_dir, f"emb_{key}.npy")

    if os.path.exists(path):
        print(f"Loading cached embeddings: {path}")
        return np.load(path)

    emb_model = SentenceTransformer(model_name)
    vecs = emb_model.encode(
        docs, batch_size=128, show_progress_bar=True,
        normalize_embeddings=True,          # makes cosine_similarity a dot product
    )
    np.save(path, vecs)
    print(f"Cached {vecs.shape} -> {path}")
    return vecs

def build_chunks(df: pd.DataFrame) -> pd.DataFrame:
    col = next((c for c in ("prepared_remarks", "content")
                if c in df.columns), None)
    if col is None:
        raise KeyError(f"No text column. Columns: {list(df.columns)}")
    print(f"Using text column: '{col}'")

    records = []
    for idx, row in df.iterrows():                          # level 1
        text = str(row.get(col, "")).replace("\n", " ")
        if len(text) < MIN_CHUNK_LEN:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", text)
        buf, size = [], 0
        for s in sentences:                                 # level 2
            buf.append(s)
            size += len(s)
            if size >= 700:                                 # level 3
                window = " ".join(buf)
                if not BOILERPLATE_PATTERNS.search(window):
                    records.append({"original_index": idx, "text": window})
                buf, size = [], 0                           # level 3, still inside if

        if buf and size >= MIN_CHUNK_LEN:                   # level 2 - OUTSIDE inner loop
            window = " ".join(buf)
            if not BOILERPLATE_PATTERNS.search(window):
                records.append({"original_index": idx, "text": window})

    out = pd.DataFrame(records)                             # level 0 - outside both loops
    print(f"Built {len(out)} chunks from {len(df)} transcripts "
          f"(avg {len(out) / max(len(df), 1):.1f} per transcript)")
    return out                                              # level 0


def get_topic_name(topic_id: int) -> str:
    # map a zero-shot topic ID back to its category name
    if topic_id == -1:
        return "Unclassified"
    try:
        return ZERO_SHOT_TOPICS[topic_id]
    except IndexError:
        return f"Topic {topic_id}"

# def classify_chunks(docs: list, labels: list, model_name: str, top_k: int = 2):
#     emb_model = SentenceTransformer(model_name)
#
#     # Chunk texts are long; truncate so the encoder isn't diluted by boilerplate tail
#     doc_vecs = emb_model.encode(docs, batch_size=32, show_progress_bar=True,
#                                  truncate_dim=None)
#     label_vecs = emb_model.encode(labels, batch_size=32)
#
#     sims = cosine_similarity(doc_vecs, label_vecs)          # (n_docs, n_labels)
#     best = sims.argmax(axis=1)
#     margin = np.sort(sims, axis=1)[:, -1] - np.sort(sims, axis=1)[:, -2]
#
#     return best, sims, margin

def classify_chunks(docs, labels, model_name):
    doc_vecs = get_cached_embeddings(docs, model_name)
    label_vecs = get_cached_embeddings(labels, model_name)
    # already normalized -> cosine similarity (dot product matrix multiplication btwn a matrix of document vectors (doc_vecs)
    # and a transpose of a matrix of label vectors (label_vecs)
    sims = doc_vecs @ label_vecs.T
    best = sims.argmax(axis=1)
    margin = np.sort(sims, axis=1)[:, -1] - np.sort(sims, axis=1)[:, -2]
    return best, sims, margin

# # ------------ DIAGNOSE EMBEDDINGS --------------------
# def diagnose_embeddings():
#     """Print diagnostic info about embeddings & zero-shot label separation."""
#     from sentence_transformers import SentenceTransformer
#     from sklearn.metrics.pairwise import cosine_similarity
#     import numpy as np
#
#     print("\n" + "=" * 60)
#     print("DIAGNOSTIC: embedding model + zero-shot labels")
#     print("=" * 60)
#
#     emb = SentenceTransformer(SENT_TRANSFORMER)
#     print("MODEL:", type(emb).__name__, "| dim =", emb.get_sentence_embedding_dimension())
#
#     # 1. Do unrelated sentences actually separate?
#     probe = [
#         "Revenue grew 12 percent year over year.",
#         "We are hiring 400 engineers this quarter.",
#         "Our data center campus adds 64 megawatts of capacity.",
#     ]
#     v = emb.encode(probe)
#     print("\nPROBE SIMILARITY (want ~0.3-0.7 off-diagonal, NOT 1.000):")
#     print(np.round(cosine_similarity(v), 3))
#
#     # 2. Are the 13 topic labels distinguishable?
#     tv = emb.encode(ZERO_SHOT_TOPICS)
#     print("\nLABEL SIMILARITY MATRIX (watch for ~1.0 everywhere):")
#     print(np.round(cosine_similarity(tv), 3))
#
#     # 3. What does each probe get assigned to? This mimics BERTopic's argmax.
#     print("\nASSIGNMENT CHECK:")
#     for p in probe:
#         s = cosine_similarity(emb.encode([p]), tv)[0]
#         best = int(s.argmax())
#         print(f"{best:>2}  {ZERO_SHOT_TOPICS[best]:<50} {np.round(s, 3)}")
#
#     print("=" * 60 + "\n")
#


def main():
    parser = argparse.ArgumentParser(description="Classify earnings calls into predefined topics")
    parser.parse_args()

    sent_path = "data/raw/with_sentiment.parquet"
    if not os.path.exists(sent_path):
        raise FileNotFoundError(f"{sent_path} does not exist. Run ml_sentiment.py")

    df = pd.read_parquet(sent_path)

    print("Generating paragraph chunks...")
    chunk_df = build_chunks(df)

    #---------- diagnostic ------------------
    print(f"Transcripts: {len(df)}, Chunks: {len(chunk_df)}")
    print("Chunks per transcript:", chunk_df["original_index"].value_counts().describe())
    print("Non-empty remarks:", (df["prepared_remarks"].astype(str).str.len() > 200).sum())
    # -----------------------------------------

    if len(chunk_df) < 15:
        print("[WARNING] Too few chunks.")
        df["topic_id"] = 0
        df["topic_name"] = ZERO_SHOT_TOPICS[0]
    else:
        docs = chunk_df["text"].tolist()
        print(f"Classifying {len(docs)} chunks against {len(ZERO_SHOT_TOPICS)} categories...")

        best, sims, margin = classify_chunks(docs, ZERO_SHOT_TOPICS, SENT_TRANSFORMER)

        chunk_df["topic_id"] = best
        chunk_df["confidence"] = sims.max(axis=1)
        chunk_df["margin"] = margin

        # Optional: drop weak assignments so they don't pollute the vote
        # MIN_CONF = 0.15
        MIN_CONF = 0.05
        strong = chunk_df[chunk_df["confidence"] >= MIN_CONF].copy()
        print(f"\nChunks kept after confidence filter: {len(strong)}/{len(chunk_df)}")
        print("Confidence stats:\n", chunk_df["confidence"].describe())

        print("\nCHUNK-LEVEL DISTRIBUTION:")
        print(strong["topic_id"].value_counts().to_string())

        # Split into two separate aggregations instead of tuples
        def first_topic(series):
            c = series.value_counts()
            return c.index[0] if len(c) >= 1 else -1

        def second_topic(series):
            c = series.value_counts()
            # return c.index[1] if len(c) >= 2 else -1
            if len(c) >= 2:
                return c.index[1]
            if len(c) == 1:
                return c.index[0]
            return -1

        if len(strong) == 0:
            print("[WARNING] No strong chunks found. Assigning -1 to all.")
            df["topic_id"] = -1
            df["topic_id_2"] = -1
        else:
            g = strong.groupby("original_index")["topic_id"]
            df["topic_id"]   = g.agg(first_topic).reindex(df.index).fillna(-1).astype(int)
            df["topic_id_2"] = g.agg(second_topic).reindex(df.index).fillna(-1).astype(int)

        df["topic_name"]   = df["topic_id"].apply(get_topic_name)
        df["topic_name_2"] = df["topic_id_2"].apply(get_topic_name)

    print("\n[OK] Topic Distribution (primary):")
    print(df["topic_name"].value_counts().to_string())
    if "topic_name_2" in df.columns:
        print("\n[OK] Topic Distribution (secondary):")
        print(df["topic_name_2"].value_counts().to_string())

    df.to_parquet("data/raw/with_topics.parquet")
    df.to_csv("data/raw/with_topics.csv", index=False)

if __name__ == "__main__":
    # diagnose_embeddings()
    main()
