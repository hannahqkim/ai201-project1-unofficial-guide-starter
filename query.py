"""Milestone 5: grounded generation.

Ties retrieval (Milestone 4) to an LLM and returns an answer that is grounded in
the retrieved chunks only, with source attribution that is guaranteed in code
rather than left to the model.

End-to-end entry point: `ask(question)` ->

    {
        "answer":  "...",                       # the generated (or refusal) text
        "sources": ["01_resy_...txt", ...],      # unique sources, derived in code
        "chunks":  [ {retrieval result}, ... ],  # what was actually retrieved
        "grounded": True/False,                  # did we have usable context?
    }

Grounding is enforced three ways:
  1. A strict system prompt: answer ONLY from the provided context; if it isn't
     enough, return a fixed refusal sentence; never use outside knowledge.
  2. A structural relevance gate: if the best retrieved chunk is too far from the
     query (distance > MAX_DISTANCE), we refuse *without calling the LLM* — so an
     out-of-domain question can't be answered from training knowledge.
  3. Programmatic source attribution: the sources come from retrieval metadata,
     so attribution doesn't depend on the model remembering to cite.
"""

from __future__ import annotations

import os
from typing import Callable

try:  # optional at import time; only needed when actually calling the LLM
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _default_retrieve(query: str, k: int) -> list[dict]:
    """Lazy import of the real retriever so this module imports without the ML
    stack present (handy for testing the grounding logic in isolation)."""
    from retrieve import retrieve

    return retrieve(query, k=k)

MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
# Cosine distance above which we treat retrieval as "no real match" and refuse.
# Good matches sit well under 0.5; 0.65 leaves headroom while still catching
# genuinely off-topic questions.
MAX_DISTANCE = 0.65
REFUSAL = "I don't have enough information on that."

SYSTEM_PROMPT = (
    "You are a careful assistant for an unofficial guide to getting hard-to-get "
    "restaurant reservations in New York City. You must answer using ONLY the "
    "information in the provided context documents. Follow these rules strictly:\n"
    "1. Use only facts found in the context below. Do not use any outside or prior "
    "knowledge, and do not guess or infer beyond what the text states.\n"
    f"2. If the context does not contain enough information to answer, reply with "
    f"exactly: \"{REFUSAL}\"\n"
    "3. When you do answer, cite the source filename(s) you used inline, e.g. "
    "(source: 01_resy_toughest_reservations_nyc.txt).\n"
    "4. Be concise and specific. Quote concrete details (dates, times, tactics) "
    "from the context rather than speaking generally."
)


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered, source-labeled context block."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[Document {i} | source: {c['source']} | chunk #{c['chunk_index']}]\n"
            f"{c['text']}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    return (
        f"Context documents:\n\n{context}\n\n"
        f"-----\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above, following the system rules. "
        f"If the context is insufficient, reply exactly: \"{REFUSAL}\""
    )


def _groq_generate(system: str, user: str) -> str:
    """Default generator: call Groq's llama-3.3-70b-versatile at temperature 0."""
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com"
        )
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,  # deterministic, reduces drift away from the context
    )
    return resp.choices[0].message.content.strip()


def ask(
    question: str,
    k: int = TOP_K,
    max_distance: float = MAX_DISTANCE,
    generate: Callable[[str, str], str] = _groq_generate,
    retrieve_fn: Callable[[str, int], list] | None = None,
) -> dict:
    """Retrieve, ground, and generate an answer for ``question``.

    ``generate`` and ``retrieve_fn`` are injectable so the grounding logic can be
    unit-tested without calling the real LLM or vector store.
    """
    retrieve_fn = retrieve_fn or _default_retrieve

    question = (question or "").strip()
    if not question:
        return {"answer": "Please enter a question.", "sources": [], "chunks": [],
                "grounded": False}

    chunks = retrieve_fn(question, k)

    # Structural grounding gate: no sufficiently-close chunk => refuse, no LLM call.
    if not chunks or chunks[0]["distance"] > max_distance:
        return {"answer": REFUSAL, "sources": [], "chunks": chunks, "grounded": False}

    context = build_context(chunks)
    user_prompt = build_user_prompt(question, context)
    answer = generate(SYSTEM_PROMPT, user_prompt)

    # Programmatic attribution: unique sources, in retrieval order.
    seen, sources = set(), []
    for c in chunks:
        if c["source"] not in seen:
            seen.add(c["source"])
            sources.append(c["source"])

    # If the model itself refused, don't claim sources.
    if answer.strip() == REFUSAL:
        return {"answer": REFUSAL, "sources": [], "chunks": chunks, "grounded": False}

    return {"answer": answer, "sources": sources, "chunks": chunks, "grounded": True}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How far in advance do Tatiana reservations drop?"
    result = ask(q)
    print(f"Q: {q}\n")
    print(result["answer"])
    if result["sources"]:
        print("\nRetrieved from:")
        for s in result["sources"]:
            print(f"  - {s}")
