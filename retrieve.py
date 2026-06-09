"""Milestone 4, step 2: the retrieval function.

Loads the persistent ChromaDB collection built by embed_store.py and exposes
`retrieve(query, k=5)`, which embeds the query with the same all-MiniLM-L6-v2
model and returns the top-k most similar chunks with their source metadata and
cosine distance scores.

Lower distance = more similar. With normalized embeddings and cosine space,
scores fall in [0, 2]; in practice good matches sit well below 0.5.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "reservations"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def _collection():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "chroma_db/ not found. Run `python embed_store.py` first."
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Return the top-k chunks most relevant to ``query``.

    Each result::

        {
            "rank": 1,
            "text": "...",
            "source": "01_resy_toughest_reservations_nyc.txt",
            "chunk_index": 5,
            "distance": 0.18,          # lower = closer
        }
    """
    query_emb = _model().encode([query], normalize_embeddings=True).tolist()
    res = _collection().query(
        query_embeddings=query_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    results: list[dict] = []
    for rank, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
    ):
        results.append(
            {
                "rank": rank,
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index"),
                "distance": round(float(dist), 4),
            }
        )
    return results


if __name__ == "__main__":
    # Quick manual check: `python retrieve.py "your question here"`
    import sys

    q = " ".join(sys.argv[1:]) or "How far in advance do Tatiana reservations drop?"
    print(f"Query: {q}\n")
    for r in retrieve(q, k=5):
        print(f"[{r['rank']}] distance={r['distance']:.4f}  "
              f"source={r['source']} #{r['chunk_index']}")
        print("    " + r["text"][:200].replace("\n", " ") + "...\n")
