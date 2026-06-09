"""Milestone 3 entry point: build and inspect the chunk set.

Run:  python build_chunks.py

What it does, in order:
  1. Load every document from documents/ and clean it.
  2. Print one cleaned document so you can eyeball the cleaning.
  3. Chunk every document with the planning.md strategy, attaching the source
     filename as metadata to each chunk.
  4. Inspect: print 5 representative chunks (spread across sources) and 5 random
     chunks (the checkpoint), plus size statistics and the 50-2,000 sanity check.
  5. Save all chunks to chunks.json for Milestone 4 (embedding + retrieval).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pipeline import load_documents, chunk_text

OUTPUT_PATH = Path(__file__).resolve().parent / "chunks.json"
RANDOM_SEED = 42  # fixed so the "random" checkpoint is reproducible


def _rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _show_chunk(chunk: dict) -> None:
    print(
        f"\n[{chunk['source']} #{chunk['chunk_index']}] "
        f"({chunk['char_len']} chars, {chunk['word_len']} words)"
    )
    print("-" * 70)
    print(chunk["text"])


def main() -> None:
    # 1. Load + clean -----------------------------------------------------
    _rule("1. LOADING + CLEANING DOCUMENTS")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from documents/\n")
    print(f"{'source':<48} {'raw':>7} {'clean':>7}")
    print("-" * 64)
    for d in docs:
        print(f"{d['source']:<48} {d['raw_len']:>7} {d['clean_len']:>7}")

    # 2. Eyeball one cleaned document ------------------------------------
    _rule("2. ONE CLEANED DOCUMENT (read it — any nav text / HTML left?)")
    sample_doc = docs[0]
    print(f"source: {sample_doc['source']}\n")
    preview = sample_doc["text"]
    print(preview[:1500] + ("\n... [truncated for preview]" if len(preview) > 1500 else ""))

    # 3. Chunk ------------------------------------------------------------
    _rule("3. CHUNKING (target ~900 chars / ~150 overlap, paragraph-aware)")
    all_chunks: list[dict] = []
    for d in docs:
        doc_chunks = chunk_text(d["text"], source=d["source"])
        all_chunks.extend(doc_chunks)
        print(f"  {d['source']:<48} -> {len(doc_chunks):>3} chunks")

    # Re-id globally so every chunk has a unique, stable id for Milestone 4.
    for i, c in enumerate(all_chunks):
        c["id"] = i

    # 4a. Representative chunks (spread across the corpus) ----------------
    _rule("4a. FIVE REPRESENTATIVE CHUNKS (spread across sources)")
    n = len(all_chunks)
    rep_idx = sorted({round(i * (n - 1) / 4) for i in range(5)}) if n >= 5 else range(n)
    for i in rep_idx:
        _show_chunk(all_chunks[i])

    # 4b. Random checkpoint ----------------------------------------------
    _rule("4b. CHECKPOINT — FIVE RANDOM CHUNKS")
    print("Each should be readable, substantive, and self-contained.")
    rng = random.Random(RANDOM_SEED)
    for c in rng.sample(all_chunks, min(5, n)):
        _show_chunk(c)

    # 4c. Stats + sanity checks ------------------------------------------
    _rule("4c. CHUNK STATISTICS")
    char_lens = [c["char_len"] for c in all_chunks]
    empties = [c for c in all_chunks if c["char_len"] == 0]
    print(f"Total documents : {len(docs)}")
    print(f"Total chunks    : {n}")
    print(f"Empty chunks    : {len(empties)}  (must be 0)")
    print(f"Chunk chars     : min={min(char_lens)}  "
          f"avg={sum(char_lens) // n}  max={max(char_lens)}")

    if n < 50:
        print("\n[!] Fewer than 50 chunks — chunks may be too large.")
    elif n > 2000:
        print("\n[!] More than 2,000 chunks — chunks may be too small.")
    else:
        print(f"\n[ok] {n} chunks is within the healthy 50-2,000 range.")

    # 5. Persist ----------------------------------------------------------
    OUTPUT_PATH.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    _rule("5. SAVED")
    print(f"Wrote {n} chunks -> {OUTPUT_PATH.name} (ready for Milestone 4).")


if __name__ == "__main__":
    main()
