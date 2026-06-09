"""Document pipeline for The Unofficial Guide (Milestone 3).

Two jobs:
  1. ingest  -> load documents from disk and clean them
  2. chunker -> split cleaned text into retrievable chunks

See planning.md for the chunking spec these modules implement.
"""

from .ingest import load_documents, clean_text
from .chunker import chunk_text

__all__ = ["load_documents", "clean_text", "chunk_text"]
