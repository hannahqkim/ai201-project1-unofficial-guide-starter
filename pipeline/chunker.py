"""Chunking.

Implements the strategy specified in planning.md:

    Chunk size: ~225 tokens (about 900 characters)
    Overlap:    ~40 tokens (about 150 characters, ~15%)
    Method:     paragraph-aware. Pack whole paragraphs together up to the size
                limit; only hard-split a single paragraph when it alone exceeds
                the limit. Carry a short overlap tail into the next chunk so a
                restaurant name is never orphaned from its timing detail. Short
                documents stay whole rather than being padded or fragmented.

    Note (revised during Milestone 3): an initial 2,000-char size merged ~7
    separate restaurant entries into a single chunk, diluting retrieval — a
    query for one restaurant matched a chunk crowded with six others. Because
    this corpus is entry-structured (short per-restaurant / per-tip blocks),
    the size was reduced to ~900 chars so each chunk holds only 2-3 related
    entries. planning.md was updated to record this change.

We measure size in characters (planning.md gives the ~500 token -> ~2,000 char
mapping). This keeps chunking reproducible without having to load the embedding
model's tokenizer just to split text.
"""

from __future__ import annotations

import re

TARGET_SIZE = 900    # ~225 tokens
OVERLAP = 150        # ~40 tokens (~15%)
MIN_CHUNK = 150      # drop fragments shorter than this (unless a doc is tiny)

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _overlap_tail(text: str, overlap: int) -> str:
    """Return up to ``overlap`` characters from the end of ``text``, snapped to
    a sentence boundary where possible so the carried context reads cleanly."""
    if overlap <= 0 or len(text) <= overlap:
        return text if len(text) <= overlap else text[-overlap:]
    tail = text[-overlap:]
    # Try to start the tail at a sentence boundary for readability.
    parts = _SENTENCE_END_RE.split(tail, maxsplit=1)
    if len(parts) == 2 and parts[1]:
        return parts[1]
    return tail


def _hard_split(paragraph: str, target_size: int, overlap: int) -> list[str]:
    """Fixed-size sliding window for a single oversized paragraph."""
    pieces: list[str] = []
    start = 0
    step = max(1, target_size - overlap)
    while start < len(paragraph):
        pieces.append(paragraph[start : start + target_size].strip())
        start += step
    return [p for p in pieces if p]


def chunk_text(
    text: str,
    source: str = "unknown",
    target_size: int = TARGET_SIZE,
    overlap: int = OVERLAP,
    min_chunk: int = MIN_CHUNK,
) -> list[dict]:
    """Split ``text`` into overlapping, paragraph-aware chunks.

    Each returned chunk is a dict::

        {
            "source": source,        # filename, for attribution
            "chunk_index": 0,        # position within this document
            "text": "...",
            "char_len": 1234,
            "word_len": 210,
        }
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]

    # Tiny document: keep it whole rather than fragmenting it.
    if len(text) <= target_size:
        return [_make_chunk(text, source, 0)]

    raw_chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # A single paragraph bigger than the target gets hard-split.
        if len(para) > target_size:
            if current:
                raw_chunks.append(current)
                current = ""
            raw_chunks.extend(_hard_split(para, target_size, overlap))
            continue

        candidate = para if not current else f"{current}\n\n{para}"
        if len(candidate) <= target_size:
            current = candidate
        else:
            # Emit the current chunk and start a new one, carrying overlap.
            raw_chunks.append(current)
            tail = _overlap_tail(current, overlap)
            current = f"{tail}\n\n{para}" if tail else para

    if current:
        raw_chunks.append(current)

    # Clean up: strip, drop empties, and merge stray tiny fragments forward.
    chunks: list[dict] = []
    for piece in raw_chunks:
        piece = piece.strip()
        if not piece:                      # empty-chunk guard
            continue
        if len(piece) < min_chunk and chunks:
            # Fold a too-small fragment into the previous chunk.
            prev = chunks[-1]
            merged = f"{prev['text']}\n\n{piece}"
            chunks[-1] = _make_chunk(merged, source, prev["chunk_index"])
            continue
        chunks.append(_make_chunk(piece, source, len(chunks)))

    return chunks


def _make_chunk(text: str, source: str, index: int) -> dict:
    text = text.strip()
    return {
        "source": source,
        "chunk_index": index,
        "text": text,
        "char_len": len(text),
        "word_len": len(text.split()),
    }
