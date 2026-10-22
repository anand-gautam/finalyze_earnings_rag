import os
import pickle
import sys
from pathlib import Path

import pandas as pd

# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

from config import SENT_TRANSFORMER, CROSS_ENCODER

# --- Import Dependencies ---
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_core.documents.compressor import BaseDocumentCompressor
    from sentence_transformers import CrossEncoder
except ImportError as e:
    print(f"[ERROR] Dependency missing during core initialization: {e}")
    FAISS = None
    BM25Retriever = None
    EnsembleRetriever = None
    ContextualCompressionRetriever = None
    HuggingFaceEmbeddings = None
    Document = None
    BaseDocumentCompressor = None
    CrossEncoder = None

# print(f"[DEBUG] Loading retrieve.py from: {__file__}")
class CrossEncoderCompressor(BaseDocumentCompressor):
    """Reranks documents using a lightweight Cross-Encoder model."""
    top_n: int = 5
    model_name: str = CROSS_ENCODER
    model: object = None

    def __init__(self, model_name=CROSS_ENCODER, top_n=5, **kwargs):
        super().__init__(model_name=model_name, top_n=top_n, **kwargs)
        try:
            self.model = CrossEncoder(model_name)
        except Exception:
            self.model = None

    def compress_documents(self, documents, query, callbacks=None):
        if not documents:
            return []
        if self.model is None:
            return documents[: self.top_n]

        pairs = [[query, doc.page_content] for doc in documents]
        try:
            scores = self.model.predict(pairs)
            for doc, score in zip(documents, scores):
                doc.metadata["score"] = round(float(score), 4)
            return sorted(documents, key=lambda d: d.metadata.get("score", 0.0), reverse=True)[: self.top_n]
        except Exception:
            return documents[: self.top_n]


class EarningsRetriever:
    def __init__(self, index_dir=None):
        if index_dir is None:
            index_dir = str("data/index")
        self.faiss = None
        self.bm25 = None
        self.compressor = None
        self.chunks = pd.DataFrame()
        self.embeddings = None

        faiss_path = os.path.join(index_dir, "faiss")
        bm25_path = os.path.join(index_dir, "bm25.pkl")
        chunks_parquet = os.path.join(index_dir, "chunks.parquet")

        # Load metadata for filtering
        if os.path.exists(chunks_parquet):
            self.chunks = pd.read_parquet(chunks_parquet)
        else:
            print(f"[WARNING] Chunks parquet not found at {chunks_parquet}. Filtering will be disabled.")

        # Load FAISS and BM25 indices
        if os.path.exists(faiss_path) and os.path.exists(bm25_path):
            self.embeddings = HuggingFaceEmbeddings(model_name=SENT_TRANSFORMER)
            self.faiss = FAISS.load_local(
                faiss_path, embeddings=self.embeddings, allow_dangerous_deserialization=True
            )
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            self.compressor = CrossEncoderCompressor(top_n=5)
        else:
            print("[ERROR] FAISS or BM25 index not found. Run build_index.py first.")

    def _build_filter_mask(self, symbol, year, topic_id, sentiment_min, sentiment_max):
        """Create a boolean mask for filtering chunks based on metadata."""
        if self.chunks.empty:
            return pd.Series([True])  # Allow all if no metadata

        mask = pd.Series(True, index=self.chunks.index)
        if symbol:
            mask &= self.chunks["symbol"] == symbol
        if year:
            mask &= self.chunks["year"] == int(year)
        if topic_id is not None:
            mask &= self.chunks["topic_id"] == int(topic_id)
        if sentiment_min is not None:
            mask &= self.chunks["sentiment"] >= float(sentiment_min)
        if sentiment_max is not None:
            mask &= self.chunks["sentiment"] <= float(sentiment_max)
        print(f"[DEBUG] symbol={symbol}, year={year}, topic_id={topic_id}, "
              f"sentiment_min={sentiment_min}, sentiment_max={sentiment_max}")
        print(f"[DEBUG] mask.sum() = {mask.sum()} / {len(mask)}")
        if not mask.any():
            print("[DEBUG] WARNING: All chunks filtered out! Check column types.")
            print(f"[DEBUG] sample year values: {self.chunks['year'].head().tolist()}")
            print(f"[DEBUG] sample sentiment values: {self.chunks['sentiment'].head().tolist()}")
        return mask

    def retrieve(self, query, top_k=5, symbol=None, year=None,
                 topic_id=None, sentiment_min=None, sentiment_max=None):
        """
        Retrieve relevant chunks using hybrid search (FAISS + BM25) and reranking.
        """
        print(f"[DEBUG] retrieve() called. faiss={self.faiss is not None}, "
              f"bm25={self.bm25 is not None}, chunks_empty={self.chunks.empty}")

        if self.faiss is None or self.bm25 is None or self.chunks.empty:
            print("[DEBUG] EARLY EXIT: indices/chunks not loaded")
            return []

        # 1. Pre-filter using Pandas to get the set of allowed UIDs
        mask = self._build_filter_mask(symbol, year, topic_id, sentiment_min, sentiment_max)
        allowed_uids = set(self.chunks[mask]["uid"].astype(str).tolist())
        print(f"[DEBUG] allowed_uids count: {len(allowed_uids)}")

        if not allowed_uids:
            return []

        # 2. Retrieve candidates from the GLOBAL pre-built FAISS index
        fetch_k = min(max(top_k * 10, 50), len(self.chunks))
        faiss_docs = self.faiss.similarity_search(query, k=fetch_k)

        # Filter FAISS results by allowed UIDs
        faiss_kept = [d for d in faiss_docs if str(d.metadata.get("uid")) in allowed_uids]
        print(f"[DEBUG] FAISS: fetched {len(faiss_docs)}, kept {len(faiss_kept)}")

        # 3. Retrieve candidates from the GLOBAL pre-built BM25 index
        bm25_docs = self.bm25.invoke(query)
        bm25_kept = [d for d in bm25_docs if str(d.metadata.get("uid")) in allowed_uids]
        print(f"[DEBUG] BM25: fetched {len(bm25_docs)}, kept {len(bm25_kept)}")

        def _same_symbol_bonus(doc):
            if not symbol:
                return 0
            doc_symbol = str(doc.metadata.get("symbol", "")).upper()
            return 1 if doc_symbol == str(symbol).upper() else 0

        # 4. Reciprocal Rank Fusion (RRF) to combine FAISS and BM25 results
        fused_scores = {}

        for rank, doc in enumerate(faiss_kept):
            uid = str(doc.metadata["uid"])
            fused_scores[uid] = fused_scores.get(uid, 0) + 1 / (rank + 60) + 0.05 * _same_symbol_bonus(doc)

        for rank, doc in enumerate(bm25_kept):
            uid = str(doc.metadata["uid"])
            fused_scores[uid] = fused_scores.get(uid, 0) + 1 / (rank + 60) + 0.05 * _same_symbol_bonus(doc)

        ranked_uids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:20]

        # Reconstruct Document objects from the two lists for reranking
        # Create a lookup dict for quick access
        doc_lookup = {}
        for d in faiss_kept + bm25_kept:
            uid = str(d.metadata["uid"])
            if uid not in doc_lookup:
                doc_lookup[uid] = d

        candidates = [doc_lookup[uid] for uid in ranked_uids if uid in doc_lookup]

        if not candidates:
            return []

        # 5. Rerank the small candidate set with Cross-Encoder
        self.compressor.top_n = top_k
        final_docs = self.compressor.compress_documents(candidates, query)

        # 6. Format results for Streamlit
        results = []
        for doc in final_docs[:top_k]:
            m = doc.metadata
            results.append({
                "text": doc.page_content,
                "symbol": m.get("symbol", "N/A"),
                "company": m.get("company_name", "N/A"),
                "year": m.get("year", "N/A"),
                "quarter": m.get("quarter", "N/A"),
                "sentiment": round(float(m.get("sentiment", 0.0)), 3),
                "sentiment_label": m.get("sentiment_label", "neutral"),
                "topic": m.get("topic_name", "N/A"),
                "score": round(float(m.get("score", 0.0)), 4),
                "uid": m.get("uid", "N/A")
            })
        return results