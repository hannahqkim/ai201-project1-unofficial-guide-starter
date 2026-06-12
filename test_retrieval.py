"""Milestone 4 checkpoint: test retrieval on the evaluation-plan queries.

Run:  python test_retrieval.py      (after embed_store.py has built chroma_db/)

For each test question from planning.md, prints the top-k retrieved chunks with
their source and cosine distance, then asserts:
  1. The best distance clears the milestone checkpoint (< 0.5).
  2. A known-good chunk (source + chunk_index derived from a real run) appears
     somewhere in the top-k results.

These assertions catch regressions automatically when you change chunking
parameters, swap embedding models, or re-ingest documents — you don't have to
eyeball output to know whether retrieval got better or worse.
"""

from __future__ import annotations

import textwrap

from retrieve import retrieve

# ---------------------------------------------------------------------------
# Test query definitions
# ---------------------------------------------------------------------------
# Each entry:
#   query            - the natural-language question
#   expected_source  - the document that must appear somewhere in the top-k
#   expected_chunk   - the chunk_index within that document that must be
#                      retrieved (derived from the actual run recorded in
#                      test_retrieval_output.txt)
#   note             - brief description of what we're asserting and why
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "query": "How far in advance do reservations at Tatiana drop, and at what time of day?",
        "expected_source": "01_resy_toughest_reservations_nyc.txt",
        "expected_chunk": 15,
        "note": (
            "Chunk #15 contains the Tatiana entry ('28 days in advance at 12 noon'). "
            "In the baseline run this chunk ranked 3rd (distance 0.46); if chunking "
            "or embeddings change, we want to know immediately if it drops out of top-k."
        ),
    },
    {
        "query": "What is Resy Notify and how does it help me get a sold-out table?",
        "expected_source": "03_resy_notify_how_it_works.txt",
        "expected_chunk": 0,
        "note": (
            "Chunk #0 of the Notify help doc is the primary answer source and "
            "ranked 1st (distance 0.31) in the baseline run."
        ),
    },
    {
        "query": "What walk-in strategy is recommended for Ha's Snack Bar?",
        "expected_source": "01_resy_toughest_reservations_nyc.txt",
        "expected_chunk": 12,
        "note": (
            "Chunk #12 contains the Ha's Snack Bar entry. This is a known hard case: "
            "the chunk blends Ha's with neighboring restaurant entries, so it only "
            "ranked 4th (distance 0.48) in the baseline. Asserting it appears in "
            "top-k is the safety net that would catch a chunking fix making it better "
            "OR a regression making it disappear entirely."
        ),
    },
    {
        "query": "Is it legal to buy a restaurant reservation on Appointment Trader in New York?",
        "expected_source": "08_ny_reservation_resale_law.txt",
        "expected_chunk": 0,
        "note": (
            "Chunk #0 of the NY law article ranked 2nd (distance 0.29) in the "
            "baseline. The legal-status answer depends on this chunk being retrieved."
        ),
    },
    {
        "query": "If I can't get a table reservation, what's one tactic to still eat at a hard-to-book spot?",
        "expected_source": "09_reddit_foodnyc_tips.txt",
        "expected_chunk": 0,
        "note": (
            "Chunk #0 of the r/FoodNYC tips doc ranked 1st (distance 0.27) in the "
            "baseline and anchors the walk-in / bar-seat answer."
        ),
    },
]

K = 5
CHECKPOINT_THRESHOLD = 0.5


def main() -> None:
    failures: list[str] = []

    for i, tc in enumerate(TEST_CASES, start=1):
        query = tc["query"]
        print("=" * 78)
        print(f"Q{i}: {query}")
        print("=" * 78)
        results = retrieve(query, k=K)

        for r in results:
            print(
                f"\n[{r['rank']}] distance={r['distance']:.4f}  "
                f"source={r['source']} (chunk #{r['chunk_index']})"
            )
            print(textwrap.indent(textwrap.fill(r["text"][:320], width=72), "    "))

        best = results[0]["distance"]

        # --- Assertion 1: best distance clears the checkpoint ---
        if best >= CHECKPOINT_THRESHOLD:
            msg = (
                f"Q{i} FAIL — top distance {best:.4f} >= threshold {CHECKPOINT_THRESHOLD}: "
                f"retrieval quality regression detected."
            )
            failures.append(msg)
            print(f"\n  ✗ {msg}")
        else:
            print(f"\n  ✓ distance check: best={best:.4f} < {CHECKPOINT_THRESHOLD}")

        # --- Assertion 2: expected chunk appears somewhere in top-k ---
        retrieved_pairs = {(r["source"], r["chunk_index"]) for r in results}
        expected_pair = (tc["expected_source"], tc["expected_chunk"])
        if expected_pair not in retrieved_pairs:
            msg = (
                f"Q{i} FAIL — expected chunk not in top-{K}: "
                f"source={tc['expected_source']} chunk#{tc['expected_chunk']}. "
                f"Note: {tc['note']}"
            )
            failures.append(msg)
            print(f"  ✗ expected chunk missing: {expected_pair}")
        else:
            rank_found = next(
                r["rank"] for r in results
                if r["source"] == tc["expected_source"]
                and r["chunk_index"] == tc["expected_chunk"]
            )
            print(f"  ✓ expected chunk found at rank {rank_found}: {expected_pair}")

        print()

    # --- Summary ---
    total = len(TEST_CASES)
    passed = total - len(failures)
    print("#" * 78)
    print(f"Assertion summary: {passed}/{total} test cases passed all assertions.")
    if failures:
        print("\nFailed assertions:")
        for f in failures:
            print(f"  - {f}")
        raise AssertionError(
            f"{len(failures)} assertion(s) failed. See details above."
        )
    else:
        print("All assertions passed — retrieval is consistent with the baseline.")
    print("#" * 78)


if __name__ == "__main__":
    main()
