"""Milestone 4 checkpoint: test retrieval on the evaluation-plan queries.

Run:  python test_retrieval.py      (after embed_store.py has built chroma_db/)

For each test question from planning.md, prints the top-k retrieved chunks with
their source and cosine distance, then flags whether the best result clears the
milestone's checkpoint (top distance < 0.5). Eyeball the chunks: you should be
able to point at each one and say why it answers the question.
"""

from __future__ import annotations

import textwrap

from retrieve import retrieve

# The 5 evaluation questions from planning.md (the milestone requires >= 3).
TEST_QUERIES = [
    "How far in advance do reservations at Tatiana drop, and at what time of day?",
    "What is Resy Notify and how does it help me get a sold-out table?",
    "What walk-in strategy is recommended for Ha's Snack Bar?",
    "Is it legal to buy a restaurant reservation on Appointment Trader in New York?",
    "If I can't get a table reservation, what's one tactic to still eat at a hard-to-book spot?",
]

K = 5
CHECKPOINT_THRESHOLD = 0.5


def main() -> None:
    passes = 0
    for i, query in enumerate(TEST_QUERIES, start=1):
        print("=" * 78)
        print(f"Q{i}: {query}")
        print("=" * 78)
        results = retrieve(query, k=K)

        for r in results:
            print(f"\n[{r['rank']}] distance={r['distance']:.4f}  "
                  f"source={r['source']} (chunk #{r['chunk_index']})")
            print(textwrap.indent(textwrap.fill(r["text"][:320], width=72), "    "))

        best = results[0]["distance"]
        ok = best < CHECKPOINT_THRESHOLD
        passes += ok
        verdict = "PASS" if ok else "REVIEW"
        print(f"\n  -> best distance {best:.4f}  [{verdict} "
              f"(checkpoint: top result < {CHECKPOINT_THRESHOLD})]\n")

    print("#" * 78)
    print(f"Checkpoint summary: {passes}/{len(TEST_QUERIES)} queries have a top "
          f"result below {CHECKPOINT_THRESHOLD}.")
    print("Each query above should return chunks that visibly relate to it, from "
          "the expected source.")
    print("#" * 78)


if __name__ == "__main__":
    main()
