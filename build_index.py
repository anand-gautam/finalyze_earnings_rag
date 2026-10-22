# build_index.py
import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import SENT_TRANSFORMER


# ROOT = Path(__file__).resolve().parent
# DATA_DIR = ROOT / "data"
# RAW_DIR = DATA_DIR / "raw"
# INDEX_DIR = DATA_DIR / "index"

def build_index():
    if pd is None:
        raise RuntimeError("pandas is required to build the index.")
    if RecursiveCharacterTextSplitter is None or FAISS is None or BM25Retriever is None or HuggingFaceEmbeddings is None:
        raise RuntimeError("LangChain dependencies (langchain-text-splitters, langchain-community, langchain-huggingface) are required to build the index.")

    candidates = [
        Path("data/raw/with_topics.parquet"),
        Path("data/raw/with_topics.csv"),
        Path("data/raw/with_sentiment.parquet")]
    input_path = next((p for p in candidates if p.exists()), None)

    if input_path is None:
        raise FileNotFoundError("No processed dataset with topics/sentiment found in data/raw/. Run the pipeline first.")

    print(f"Loading '{input_path}'...")
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    # Fill missing metadata defaults
    if "topic_id" not in df.columns:
        df["topic_id"] = 0
    if "topic_name" not in df.columns:
        df["topic_name"] = "Topic 0: General Financial & Strategy"
    if "sentiment_composite" not in df.columns:
        df["sentiment_composite"] = 0.0
    if "sentiment_label_prepared" not in df.columns:
        df["sentiment_label_prepared"] = "neutral"
    if "company_name" not in df.columns:
        df["company_name"] = df.get("symbol", "N/A")
    if "quarter" not in df.columns:
        df["quarter"] = 1
    if "date" not in df.columns:
        df["date"] = "N/A"

    os.makedirs("data/index", exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )

    print("Splitting prepared remarks into text chunks...")
    docs_data = []
    for idx, row in df.iterrows():
        text = str(row.get("prepared_remarks", "") or "")
        if len(text) < 100:
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs_data.append({
                "text": chunk,
                "uid": f"{idx}_i",   # globally unique id (for cache impl)
                "doc_id": idx,
                "chunk_id": i,
                "symbol": str(row.get("symbol", "N/A")),
                "company_name": str(row.get("company_name", "N/A")),
                "year": int(row.get("year", 2024)),
                "quarter": int(row.get("quarter", 1)),
                "date": str(row.get("date", "N/A")),
                "sentiment": float(row.get("sentiment_composite", 0.0)),
                "sentiment_label": str(row.get("sentiment_label_prepared", "neutral")),
                "topic_id": int(row.get("topic_id", 0)),
                "topic_name": str(row.get("topic_name", "General")),
            })

    chunks_df = pd.DataFrame(docs_data)
    chunks_df.to_parquet( "data/index/chunks.parquet", index=False)
    chunks_df.to_csv( "data/index/chunks.csv", index=False)
    print(f"[OK] Total chunks generated: {len(chunks_df)}")

    # print("Building HuggingFace Embeddings ('all-MiniLM-L6-v2')...")
    print(f"Building HuggingFace Embeddings ({SENT_TRANSFORMER})...")
    embeddings = HuggingFaceEmbeddings(model_name=SENT_TRANSFORMER)

    documents = [
        Document(
            page_content=r["text"],
            metadata={
                "uid": r["uid"],
                "doc_id": r["doc_id"],
                "chunk_id": r["chunk_id"],
                "symbol": r["symbol"],
                "company_name": r["company_name"],
                "year": r["year"],
                "quarter": r["quarter"],
                "date": r["date"],
                "sentiment": r["sentiment"],
                "sentiment_label": r["sentiment_label"],
                "topic_id": r["topic_id"],
                "topic_name": r["topic_name"],
            }
        )
        for r in docs_data
    ]

    print("Creating FAISS vector index...")
    faiss_store = FAISS.from_documents(documents, embeddings)
    faiss_store.save_local(str("data/index/faiss"))
    print("[OK] Saved FAISS vector store -> data/index/faiss/")

    print("Creating BM25 sparse index...")
    bm25 = BM25Retriever.from_documents(documents)
    with open("data/index/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("[OK] Saved BM25 retriever -> data/index/bm25.pkl")


if __name__ == "__main__":
    build_index()