"""Milestone 4, step 1: embed chunks and load them into ChromaDB.

Run:  python embed_store.py

Reads chunks.json (produced by build_chunks.py in Milestone 3), embeds every
chunk locally with all-MiniLM-L6-v2, and stores the vectors in a persistent
ChromaDB collection together with each chunk's source metadata (filename +
position). This is a one-time build step; rerun it whenever chunks.json changes.

Why these choices (see planning.md > Retrieval Approach):
  - all-MiniLM-L6-v2: runs locally, no API key, no rate limits, 384-dim, fast,
    and accurate enough on short English passages.
  - ChromaDB persistent client: stores vectors on disk under chroma_db/ so we
    don't re-embed on every query. We use cosine distance, which is the right
    metric for normalized sentence-transformer embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = ROOT / "chunks.json"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "reservations"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run `python build_chunks.py` first (Milestone 3)."
        )
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if not chunks:
        raise RuntimeError("chunks.json is empty.")
    return chunks


def build_store(reset: bool = True) -> chromadb.api.models.Collection.Collection:
    """Embed all chunks and (re)build the ChromaDB collection. Returns it."""
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH.name}")

    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    # ids must be strings and unique; chunk 'id' is the global index from Milestone 3.
    ids = [str(c["id"]) for c in chunks]
    metadatas = [
        {"source": c["source"], "chunk_index": c["chunk_index"], "char_len": c["char_len"]}
        for c in chunks
    ]

    print("Embedding chunks (this runs locally on CPU)...")
    # normalize_embeddings=True pairs with cosine distance for clean [0, 2] scores.
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True
    ).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        # Start clean so reruns don't duplicate or stack stale vectors.
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance
    )

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Stored {collection.count()} vectors in ChromaDB collection "
          f"'{COLLECTION_NAME}' at {CHROMA_DIR.name}/ (cosine distance).")
    return collection


if __name__ == "__main__":
    build_store()
