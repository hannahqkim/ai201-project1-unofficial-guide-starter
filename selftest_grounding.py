"""Grounding logic self-test (no API key, no vector store, no network).

Injects fake retrieval + a stub LLM into `ask()` to prove the grounding
guarantees hold independently of the model:
  A. In-domain query -> answers, with context actually passed to the LLM and
     sources derived programmatically.
  B. Out-of-domain query (far distance) -> refuses WITHOUT calling the LLM.
  C. LLM emits the refusal -> we surface no sources / grounded=False.

Run:  python selftest_grounding.py
"""

from __future__ import annotations

from query import ask, REFUSAL, SYSTEM_PROMPT

# A fake "retrieved chunk" about Tatiana.
TATIANA_CHUNK = {
    "rank": 1,
    "text": "Tatiana, by Kwame Onwuachi (Lincoln Center)\n"
            "Reservations Drop: 28 days in advance at 12 noon.",
    "source": "01_resy_toughest_reservations_nyc.txt",
    "chunk_index": 15,
    "distance": 0.21,
}


def fake_retrieve_close(query, k):
    # Duplicate source to also test source de-duplication.
    return [TATIANA_CHUNK, {**TATIANA_CHUNK, "rank": 2, "distance": 0.33}]


def fake_retrieve_far(query, k):
    return [{**TATIANA_CHUNK, "distance": 0.82}]


def make_recording_generator(output):
    calls = []

    def gen(system, user):
        calls.append({"system": system, "user": user})
        return output

    return gen, calls


def main() -> None:
    passed = 0
    total = 0

    def check(name, cond):
        nonlocal passed, total
        total += 1
        passed += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # --- A. In-domain ----------------------------------------------------
    print("A. In-domain query is answered and grounded:")
    gen, calls = make_recording_generator(
        "Tatiana releases reservations 28 days in advance at 12 noon "
        "(source: 01_resy_toughest_reservations_nyc.txt)."
    )
    res = ask("How far ahead does Tatiana drop?",
              generate=gen, retrieve_fn=fake_retrieve_close)
    check("LLM was called exactly once", len(calls) == 1)
    check("context (the chunk text) was passed to the LLM",
          "28 days in advance at 12 noon" in calls[0]["user"])
    check("strict grounding instruction is in the system prompt",
          REFUSAL in SYSTEM_PROMPT and "ONLY" in SYSTEM_PROMPT)
    check("grounded flag is True", res["grounded"] is True)
    check("sources derived programmatically (deduped to 1)",
          res["sources"] == ["01_resy_toughest_reservations_nyc.txt"])

    # --- B. Out-of-domain: structural refusal, no LLM call ---------------
    print("\nB. Out-of-domain query refuses WITHOUT calling the LLM:")
    gen, calls = make_recording_generator("(should never be returned)")
    res = ask("What's the best ski resort in Colorado?",
              generate=gen, retrieve_fn=fake_retrieve_far)
    check("LLM was NOT called", len(calls) == 0)
    check("answer is the exact refusal", res["answer"] == REFUSAL)
    check("grounded flag is False", res["grounded"] is False)
    check("no sources claimed", res["sources"] == [])

    # --- C. LLM itself refuses ------------------------------------------
    print("\nC. When the LLM returns the refusal, no sources are claimed:")
    gen, _ = make_recording_generator(REFUSAL)
    res = ask("Does the guide list a vegan tasting menu price?",
              generate=gen, retrieve_fn=fake_retrieve_close)
    check("answer is the refusal", res["answer"] == REFUSAL)
    check("grounded flag is False", res["grounded"] is False)
    check("no sources claimed", res["sources"] == [])

    print(f"\n{passed}/{total} checks passed.")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
