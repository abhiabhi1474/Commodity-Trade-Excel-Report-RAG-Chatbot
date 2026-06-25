"""
In-memory FAISS vector store for the Schema RAG and Historical RAG layers.
Chosen over PostgreSQL+pgvector for this POC build to keep the HF Space
self-contained with no external DB dependency, per the design doc's note
that pgvector is needed "if scale requires it" -- this Space is a POC.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np
import faiss

from embeddings import embed


class ChunkStore:
    """Holds one FAISS index plus the original chunk dicts, keyed by chunk_type
    (e.g. 'schema' or 'historical') so schema and history can be queried
    independently or together, with metadata filtering."""

    def __init__(self, name: str):
        self.name = name
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []

    def build(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        if not chunks:
            self.index = None
            return
        vecs = embed([c["text"] for c in chunks])
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine sim via normalized vectors
        self.index.add(vecs)

    def search(self, query: str, k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self.index is None or not self.chunks:
            return []
        qvec = embed([query])
        # over-fetch then filter, since FAISS doesn't support arbitrary metadata filters
        fetch_k = min(len(self.chunks), max(k * 5, k))
        scores, idxs = self.index.search(qvec, fetch_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if metadata_filter:
                if not all(str(chunk.get(mk, "")).lower() == str(mv).lower() for mk, mv in metadata_filter.items()):
                    continue
            results.append({**chunk, "score": float(score)})
            if len(results) >= k:
                break
        return results


class XcelerVectorStore:
    """Bundles the Schema RAG store and Historical RAG store together."""

    def __init__(self):
        self.schema_store = ChunkStore("schema")
        self.historical_store = ChunkStore("historical")

    def build(self, schema_chunks: List[Dict[str, Any]], historical_chunks: List[Dict[str, Any]]):
        self.schema_store.build(schema_chunks)
        self.historical_store.build(historical_chunks)

    def retrieve_schema(self, query: str, k: int = 5):
        return self.schema_store.search(query, k=k)

    def retrieve_history(self, query: str, k: int = 5, metadata_filter=None):
        return self.historical_store.search(query, k=k, metadata_filter=metadata_filter)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ingestion import ingest
    from semantic_registry import schema_chunks
    from historical_rag import generate_all_historical_chunks

    df = ingest("data/LBL_and_GBL_Summary_Report.xlsx")
    store = XcelerVectorStore()
    store.build(schema_chunks(), generate_all_historical_chunks(df))

    print("Schema hits for 'closed position':")
    for r in store.retrieve_schema("closed position quantity", k=3):
        print(" -", r["text"][:120], "score=", round(r["score"], 3))

    print("\nHistory hits for 'crude palm oil':")
    for r in store.retrieve_history("which counterparty shipped the most crude palm oil", k=3):
        print(" -", r["text"][:140], "score=", round(r["score"], 3))
