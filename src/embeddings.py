"""
Layer 5: Embedding Layer.

Wraps IBM Granite embeddings (ibm-granite/granite-embedding-125m-english)
for turning schema/historical chunks into vectors for retrieval.

Two backends, selected by XCELER_EMBED_BACKEND env var:
  - "granite" (default): loads the real model via sentence-transformers /
    transformers. Requires network access to huggingface.co and is meant
    to run inside the deployed HF Space.
  - "hash": a deterministic, dependency-light fallback used for local
    pipeline testing when there's no network access to the Hub (e.g. in a
    sandboxed dev environment). NOT semantically meaningful -- swap to
    "granite" before relying on retrieval quality.
"""
from __future__ import annotations

import os
import hashlib
import numpy as np

EMBED_BACKEND = os.environ.get("XCELER_EMBED_BACKEND", "granite")
GRANITE_EMBED_MODEL = os.environ.get("XCELER_EMBED_MODEL", "ibm-granite/granite-embedding-125m-english")

_model = None


def _hash_embed(texts, dim: int = 384) -> np.ndarray:
    """Deterministic bag-of-tokens hashing embedding. Dev/offline fallback only."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in str(t).lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[i, h % dim] += 1.0
        norm = np.linalg.norm(out[i])
        if norm > 0:
            out[i] /= norm
    return out


def _load_granite_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(GRANITE_EMBED_MODEL)
    return _model


def embed(texts) -> np.ndarray:
    """Embed a list of strings. Returns an (n, dim) float32 array, L2-normalized."""
    if isinstance(texts, str):
        texts = [texts]
    if EMBED_BACKEND == "hash":
        return _hash_embed(texts)
    model = _load_granite_model()
    vecs = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype(np.float32)


if __name__ == "__main__":
    print("backend:", EMBED_BACKEND)
    v = embed(["closed position volume", "counterparty shipment ranking"])
    print(v.shape, v.dtype)
