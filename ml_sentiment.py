# ml_sentiment.py
import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
from transformers import pipeline

# ROOT = Path(__file__).resolve().parent
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

from config import NLP_MODEL

# DATA_DIR = ROOT / "data"
# RAW_DIR = DATA_DIR / "raw"

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x


def score_text(text, model):
    """Splits transcripts into clean sentences, samples proportionally, and scores via FinBERT."""
    if not text or len(str(text)) < 50:
        return 0.0, "neutral"

    text = str(text)
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)   # lookbehind

    clean_sentences = []
    for s in raw_sentences:
        s_clean = s.strip()
        if len(s_clean) < 15:
            continue
        if any(w in s_clean.lower() for w in ["good day", "welcome to", "standing by", "press star"]):
            continue
        clean_sentences.append(s_clean)

    if not clean_sentences:
        return 0.0, "neutral"

    # Proportional sampling to capture data across the entire call duration safely on CPU
    max_sentences = 40
    total_found = len(clean_sentences)
    if total_found > max_sentences:
        sampled_sentences = [
            clean_sentences[int(i * (total_found - 1) / (max_sentences - 1))]
            for i in range(max_sentences)
        ]
    else:
        sampled_sentences = clean_sentences

    try:
        # batch_size=16 heavily optimizes performance on multi-core local CPUs
        results = model(sampled_sentences, batch_size=16)
    except Exception as e:
        print(f"\n[Warning] Inference error: {e}")
        return 0.0, "neutral"

    # Official ProsusAI/finbert mapping configuration
    label_map = {
        "POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0,
        "LABEL_0": 1.0,  # Positive
        "LABEL_1": -1.0,  # Negative
        "LABEL_2": 0.0,  # Neutral
    }

    raw_scores = []
    for item in results:
        label = str(item.get("label", "")).upper()
        score = float(item.get("score", 0.0))
        raw_scores.append(label_map.get(label, 0.0) * score)

    avg = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
    label = "positive" if avg > 0.15 else "negative" if avg < -0.15 else "neutral"
    return float(avg), label


def main():
    parser = argparse.ArgumentParser(description="Run FinBERT sentiment analysis on earnings calls.")
    parser.parse_args()

    proc_path = "data/raw/processed.parquet"
    if not os.path.exists(proc_path):
        raise FileNotFoundError(f"'{proc_path}' not found. Run preprocess.py first.")

    print(f"Loading '{proc_path}'...")
    df = pd.read_parquet(proc_path)

    print(f"Loading FinBERT model ({NLP_MODEL}) on CPU...")
    finbert = pipeline("sentiment-analysis", model=NLP_MODEL, device=-1)

    print("Scoring prepared remarks...")
    prep_results = [score_text(t, finbert) for t in tqdm(df["prepared_remarks"], desc="Prepared Remarks")]
    df["sentiment_prepared"] = [r[0] for r in prep_results]
    df["sentiment_label_prepared"] = [r[1] for r in prep_results]

    print("Scoring Q&A sections...")
    qa_results = [score_text(t, finbert) for t in tqdm(df["qa_section"], desc="Q&A Section")]
    df["sentiment_qa"] = [r[0] for r in qa_results]
    df["sentiment_label_qa"] = [r[1] for r in qa_results]

    df["sentiment_composite"] = 0.7 * df["sentiment_prepared"] + 0.3 * df["sentiment_qa"]

    print("\n[OK] Sentiment distribution (prepared remarks):")
    print(df["sentiment_label_prepared"].value_counts())

    out_path = "data/raw/with_sentiment.parquet"
    df.to_parquet(out_path)
    print(f"[OK] Saved to '{out_path}'")


if __name__ == "__main__":
    main()
