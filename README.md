# The Unofficial Guide — Project 1

A retrieval-augmented (RAG) system that answers questions about getting hard-to-get
restaurant reservations in New York City, grounded only in a corpus of collected
source documents.

**Pipeline:** Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Grounded Generation
**Stack:** Python · sentence-transformers (`all-MiniLM-L6-v2`) · ChromaDB · Groq (`llama-3.3-70b-versatile`) · Gradio

Run order:
```
pip install -r requirements.txt
python build_chunks.py     # ingest + clean + chunk  -> chunks.json
python embed_store.py      # embed + store           -> chroma_db/
python test_retrieval.py   # sanity-check retrieval
python app.py              # web UI at http://localhost:7860
```

---

## Domain: NYC Dining — Getting Hard-to-Get Reservations

This system covers the unofficial playbook for landing tables at New York City's
most in-demand restaurants: when reservations "drop" and at exactly what time,
how to use waitlist/Notify features, walk-in and bar-seat timing tricks,
third-party sniping bots, the secondary resale market, and the current legal
landscape around reservation resale.

This knowledge is valuable and hard to find through official channels because the
booking platforms (Resy, Tock, OpenTable, SevenRooms) only tell you whether a slot
is open *right now*. They don't tell you that Tatiana releases tables 28 days out
at noon, that a particular restaurant holds its entire bar for walk-ins, or that
buying a reservation on a resale site is now restricted in New York. That practical
intelligence is scattered across restaurant blogs, food-media guides, help docs,
GitHub repos, community forums, and news reporting — and it goes stale quickly,
which is exactly what a retrieval system over current sources is good for.

---

## Document Sources

Ten sources, chosen to cover complementary subtopics: official platform strategy,
independent food-media guides, community/anecdotal knowledge, technical automation,
and the secondary market plus its legal status. Each was loaded as a `.txt` file in
`documents/`; the filename is carried through the pipeline as the attribution key.

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Resy — "How to Get the Toughest Restaurant Reservations in New York" | Platform blog / strategy guide | https://blog.resy.com/the-one-who-keeps-the-book/toughest-restaurant-reservations-nyc/ — `documents/01_resy_toughest_reservations_nyc.txt` |
| 2 | The Infatuation — "The Toughest Reservations In NYC Right Now" | Editorial guide | https://www.theinfatuation.com/new-york/guides/toughest-restaurant-reservations-nyc — `documents/02_infatuation_toughest_reservations.txt` |
| 3 | Resy Help Desk — "What is Notify and how does it work?" | Product help doc | https://helpdesk.resy.com/what-is-notify-and-how-does-it-work-BJrJzPQLu — `documents/03_resy_notify_how_it_works.txt` |
| 4 | Resy — "What Restaurant Operators Wish You Knew About Reservation Fraud" | Platform blog | https://blog.resy.com/for-restaurants/restaurant-reservation-fraud/ — `documents/04_resy_reservation_fraud.txt` |
| 5 | Alkaar/resy-booking-bot | Code repository (README) | https://github.com/Alkaar/resy-booking-bot — `documents/05_github_resy_booking_bot.txt` |
| 6 | Appointment Trader | Secondary marketplace | https://appointmenttrader.com/ — `documents/06_appointment_trader_marketplace.txt` |
| 7 | NBC News — "Reservations at top NYC restaurants are selling for hundreds" | News article | https://www.nbcnews.com/news/us-news/reservations-top-new-york-city-restaurants-are-selling-hundreds-dollar-rcna151702 — `documents/07_nbc_reservations_sell_for_hundreds.txt` |
| 8 | Columbia News Service — "New York Banned Reservation Resales…" | News article | https://columbianewsservice.com/2025/07/28/new-york-banned-reservation-resales-now-appointment-trader-is-testing-the-law-with-ai/ — `documents/08_ny_reservation_resale_law.txt` |
| 9 | r/FoodNYC | Community forum (tips, summarized) | https://www.reddit.com/r/FoodNYC/ — `documents/09_reddit_foodnyc_tips.txt` |
| 10 | r/finedining | Community forum (tips, summarized) | https://www.reddit.com/r/finedining/ — `documents/10_reddit_finedining_tips.txt` |

Notes on collection: sources 1 and 2 are live web guides whose substantive article
text was extracted and cleaned (nav, ads, image captions, and CTAs removed).
Sources 3 and 5 are JavaScript-rendered / would not return clean article text, so
their content was assembled from the page's stated facts. Sources 9 and 10 are
labeled in-file as *summarized* community tips rather than verbatim posts, because
the subreddits are not cleanly scrapable; they paraphrase recurring, widely-shared
advice rather than quoting individuals.

---

## Chunking Strategy

**Chunk size:** ~225 tokens (≈900 characters)
**Overlap:** ~40 tokens (≈150 characters, ~15%)

**Preprocessing before chunking** (`pipeline/ingest.py`): unescape HTML entities
(`&amp;`, `&nbsp;`, `&#39;`), remove markdown image syntax, convert markdown links
to their visible text, strip any remaining HTML tags, drop boilerplate lines
(navigation, "Book Now"/"Reserve a table" CTAs, "Photo by…/Photo courtesy of…"
credits, and bare-URL lines), and normalize whitespace.

**Method** (`pipeline/chunker.py`): paragraph-aware. The chunker splits on blank
lines and packs whole paragraphs (which, in the two big guides, are one
self-contained block per restaurant) up to the size limit. A single paragraph that
alone exceeds the limit is hard-split with a sliding window. A short overlap tail is
carried into the next chunk so a restaurant name is never orphaned from the timing
detail that follows it. An empty-chunk guard (`len(chunk) > 0`) and a
minimum-length merge prevent fragments; documents shorter than the chunk size are
kept whole.

**Why these choices fit the documents:** the corpus is *entry-structured* — short
per-restaurant or per-tip blocks rather than long flowing prose. ~900-character
chunks keep one complete entry (name + drop time + walk-in advice + pro tip)
together while holding only 2–3 entries per chunk, so a query for one restaurant
isn't diluted by six others. (See the revision note below — this size was chosen
deliberately after testing a larger one.)

**Final chunk count:** **55 chunks** across 10 documents (min 339 / avg 648 /
max 889 characters; 0 empty chunks). Distribution: 18 chunks from the Resy guide,
8 from the Infatuation guide, and 3–4 from each of the remaining eight documents.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dimensional),
stored in ChromaDB with **cosine** distance and embeddings normalized at index and
query time. Retrieval returns **top-k = 5**.

It is the right default here: it runs locally with no API key and no rate limits,
it is fast on CPU, and it is accurate enough on short English passages, which is
exactly what this corpus is.

**Production tradeoff reflection:** if I were deploying this for real users and cost
weren't a constraint, I'd weigh a larger hosted model such as OpenAI
`text-embedding-3-large` or a Voyage retrieval model. The tradeoffs: (1) **accuracy
on domain-specific text** — bigger models embed jargon like "drop," "Notify,"
"walk-in," and specific restaurant names more reliably, which would directly help
the entity-ambiguity failure described below; (2) **context length** — MiniLM
truncates around 256 tokens, so a longer-context model could embed bigger, more
coherent chunks; (3) **multilingual support** — some reviews/posts aren't in
English, and a multilingual model would index rather than drop them; (4) **latency
and cost** — hosted models add per-call cost and network round-trips; and (5)
**privacy / dependency** — local MiniLM keeps everything offline with no API
dependency. For a hobby-scale guide, local MiniLM wins; for a paid product I'd pay
for a larger hosted model and accept the latency/cost in exchange for better recall
on domain terms.

---

## Grounded Generation

**System prompt grounding instruction** (`query.py`): the model is told to answer
using **only** the provided context, with these exact rules — (1) use only facts
found in the context, no outside or prior knowledge and no guessing; (2) if the
context is insufficient, reply with exactly *"I don't have enough information on
that."*; (3) cite the source filename(s) used inline; (4) be concise and quote
concrete details. Generation runs at **temperature 0** to reduce drift away from
the context.

Grounding is enforced **structurally**, not just suggested:

1. **Strict system prompt** (above).
2. **Relevance gate:** before calling the LLM, if the closest retrieved chunk has a
   cosine distance above **0.65**, the system returns the refusal sentence
   *without calling the model at all*. This means an out-of-domain question (e.g.
   "best ski resort in Colorado?") cannot be answered from the model's training
   knowledge — there is no model call to hallucinate from.
3. **Programmatic source attribution:** the source list returned to the UI is built
   in code from the retrieved chunks' metadata, so attribution does not depend on
   the model remembering to cite. If the model itself returns the refusal, the code
   suppresses the source list so we never attribute a non-answer.

**How source attribution is surfaced:** the Gradio interface (`app.py`) shows the
answer and a separate "Retrieved from" panel listing the unique source filenames
the answer was drawn from. The model is also instructed to cite inline, so the user
sees attribution both in the answer text and in the dedicated panel.

This behavior is unit-tested without an API key in `selftest_grounding.py`
(12/12 checks pass): an in-domain query is answered with the chunk text actually
passed to the LLM and sources derived in code; an out-of-domain query refuses with
zero LLM calls; and a model-emitted refusal surfaces no sources.

---

## Evaluation Report

The five questions and expected answers come from `planning.md`. **Retrieval
quality** and the cosine distances below are from an actual `python test_retrieval.py`
run (top-k = 5); all five queries cleared the checkpoint with a top result under 0.5.
**System response** summarizes the grounded answer the retrieved context supports,
and **Response accuracy** is the honest judgment.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | How far in advance do reservations at Tatiana drop, and at what time of day? | 28 days in advance, released at 12 noon. | Tatiana drops reservations 28 days in advance at 12 noon (source: `01_resy_toughest_reservations_nyc.txt`). The fact lives in the retrieved chunk #15. | **Partially relevant** — top result (0.39) was a *different* restaurant (Bistrot Ha); the correct Tatiana chunk was rank 3 (0.46). | **Accurate** — the correct chunk was in-context, so grounded generation answers correctly. |
| 2 | What is Resy Notify and how does it help me get a sold-out table? | A modern waitlist: you set an alert for a sold-out day/time, and when a cancellation opens a slot Resy notifies you so you can claim and book it immediately. | Notify is Resy's waitlist; you set an alert for a sold-out slot and get notified when a cancellation opens one, then claim it fast (source: `03_resy_notify_how_it_works.txt`). | **Relevant** — top 3 results are all the Notify doc (best 0.31). | **Accurate.** |
| 3 | What walk-in strategy is recommended for Ha's Snack Bar? | Get there early before the 5:30 p.m. open; use the bar/walk-in path. | The retrieved Ha's chunk says get there early before the 5:30 p.m. open; surrounding context is generic "bar seats / arrive before doors open" advice (sources: `01_resy_...txt`, `09_reddit_foodnyc_tips.txt`). | **Partially relevant** — top 3 results are *generic* walk-in advice; the Ha's-specific chunk was only rank 4 (0.48, near the 0.5 line). See Failure Case. | **Partially accurate** — correct but thin/generic; the answer leans on general advice rather than Ha's-specific detail. |
| 4 | Is it legal to buy a restaurant reservation on Appointment Trader in New York? | New York passed the Restaurant Reservation Anti-Piracy Act banning unauthorized reservation resale; the platform was shut down under the law and is trying to relaunch with an AI interface. | New York's Restaurant Reservation Anti-Piracy Act bans unauthorized resale; Appointment Trader was forced to shut down its NY activity and is testing a relaunch with an AI interface (sources: `06_appointment_trader_marketplace.txt`, `08_ny_reservation_resale_law.txt`). | **Relevant** — top 2 are the marketplace + the NY-law doc (0.28). | **Accurate.** |
| 5 | If I can't get a table reservation, what's one tactic to still eat at a hard-to-book spot? | Go for bar seating, which is often first-come-first-served and usually serves the full menu — arrive at or just before opening. | Try bar seats / walk-ins (often first-come, first-served, full menu); arrive before doors open, and use Notify for cancellations (sources: `09_reddit_foodnyc_tips.txt`, `02_infatuation_toughest_reservations.txt`). | **Relevant** — top results are community/guide tips on the exact tactic (0.27). | **Accurate.** |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

> Note: the **System response** cells summarize the grounded answer the retrieved
> chunks support at temperature 0; verify exact wording by asking the same five
> questions in `python app.py`. The **Retrieval quality** judgments and distances
> are from the real `test_retrieval.py` run.

**Out-of-domain check (grounding):** asking something the corpus doesn't cover (e.g.
"What's the best ramen in Tokyo?") returns *"I don't have enough information on
that."* with no sources — the relevance gate refuses before the LLM is ever called.

---

## Failure Case Analysis

**Question that failed:** "What walk-in strategy is recommended for **Ha's Snack
Bar**?" (Test question 3) — a *partial* retrieval failure.

**What the system returned:** the top three retrieved chunks were all **generic**
walk-in advice, not Ha's-specific: rank 1 was the Infatuation guide's general-tactics
paragraph (distance 0.41), and ranks 2–3 were the r/FoodNYC "Walk-ins — worth it?"
tips (0.46, 0.48). The chunk that actually contains the **Ha's Snack Bar** entry
("get there early — they open at 5:30 p.m.") was only **rank 4, at distance 0.48** —
right at the edge of the 0.5 quality line. So the system can answer, but it answers
mostly from generic advice with the entity-specific chunk barely making the cut.

**Root cause (tied to a specific pipeline stage):** this is a **chunking +
embedding-granularity failure**, not the entity-name conflation I had originally
predicted. (Worth noting honestly: I expected the look-alike restaurant *Bistrot Ha*
to be retrieved and mislead the answer — but its chunk did **not** appear in the
top 5 for this query, so that specific failure never fired.) What actually happens is
that each ~900-character chunk packs **2–3 restaurants and often begins mid-entry
with an unrelated one**. The chunk holding Ha's Snack Bar (chunk #12) literally opens
with the tail of the **Golden Diner** entry ("…for weekend brunch. Pro Tip: Don't put
your phone down…") before reaching Ha's. Its embedding is therefore a *blend* of
Golden Diner + Ha's, so the restaurant name is a small fraction of the vector's
meaning. Meanwhile the query phrase "walk-in strategy" matches dedicated
walk-in-advice chunks much more strongly than the diluted entity chunk — pushing the
chunk that actually names Ha's Snack Bar down to rank 4. The same effect shows up in
Q1: for "Tatiana," a *different* restaurant (Bistrot Ha) took the top spot and
Tatiana's own chunk landed at rank 3. The common cause is multi-restaurant chunks
diluting each entity's identity.

**What I would change to fix it:** make each chunk carry a single restaurant's
identity so the embedding is dominated by the right entity. Concretely: (a) chunk the
entry-structured guides **one restaurant per chunk** and prefix each chunk with the
restaurant name as a header (e.g. "Ha's Snack Bar — walk-in / reservation notes:"),
so the name is embedded *with* its own details and never blended with a neighbor's;
(b) store the restaurant name in chunk metadata for optional filtering; and/or (c)
use a stronger embedding model (see the Embedding tradeoff section). Option (a) is the
cheapest and most direct, and it flows back through `build_chunks.py` without touching
retrieval or generation. (A related data-quality note this exercise surfaced: the
`planning.md` expected answer for this question originally borrowed Bistrot Ha's
"4:45 p.m." detail and mis-attributed it to Ha's Snack Bar; the corpus doesn't support
that, so the expected answer was corrected.)

---

## Spec Reflection

**One way the spec helped you during implementation:** writing the Chunking Strategy
and Retrieval Approach sections of `planning.md` before any code meant the
implementation had concrete targets to hit rather than guesses. `chunk_text()` was
written directly against the planned size/overlap, and `test_retrieval.py` was
written directly against the five evaluation questions defined in the plan. Because
the eval questions existed up front, retrieval could be tested the moment the vector
store was built, and the questions doubled as the embedding/generation test
harness — there was never a "what do I even test this with?" gap.

**One way your implementation diverged from the spec, and why:** the plan originally
specified ~500-token (~2,000-character) chunks. When I implemented that and inspected
the output (Milestone 3), it produced only 20 chunks and a single chunk merged ~7
different restaurants — the "too large / diluted" failure mode, where a query for one
restaurant matches a chunk crowded with six others. Because this corpus is
entry-structured rather than long-form prose, I reduced the chunk size to ~900
characters (55 chunks), and I updated `planning.md`'s Chunking Strategy section to
record the change and the reasoning. A second, smaller divergence: the plan didn't
specify a structural grounding gate, but during Milestone 5 I added one (refuse
without calling the LLM when the best retrieval distance exceeds 0.65) because a
prompt instruction alone is a weaker guarantee than simply not calling the model when
there is no real context.

---

## AI Usage

**Instance 1 — Ingestion + chunking**

- *What I gave the AI:* the Documents and Chunking Strategy sections of
  `planning.md` plus the pipeline diagram, and asked it to implement document
  loading, cleaning, and a `chunk_text()` matching my specified size and overlap.
- *What it produced:* a working `pipeline/ingest.py` (HTML/entity/boilerplate
  cleaning with source-filename metadata) and a paragraph-aware `pipeline/chunker.py`
  at the originally-specified ~2,000-character size.
- *What I changed or overrode:* after running it and printing sample chunks, I saw
  that 2,000-character chunks merged ~7 restaurants each and produced only 20 chunks.
  I overrode the chunk size down to ~900 characters (and overlap to ~150) to fit the
  entry-structured corpus, which raised the count to 55 and made each chunk
  self-contained. I updated `planning.md` to document the change.

**Instance 2 — Grounded generation**

- *What I gave the AI:* the grounding requirement (answers from retrieved context
  only, with source attribution), the desired output format (answer + source list),
  and the Groq + Gradio skeleton, and asked it to wire generation to retrieval.
- *What it produced:* a first version that enforced grounding through the system
  prompt alone and relied on the model to cite its sources in the answer text.
- *What I changed or overrode:* I directed it to make grounding *structural* rather
  than prompt-only — adding a relevance gate that returns the refusal without
  calling the LLM when retrieval distance is too high, and deriving the source list
  programmatically from chunk metadata instead of trusting the model to cite. I also
  added `selftest_grounding.py` to prove these guarantees hold without an API key.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key (free tier, no card): https://console.groq.com
cp .env.example .env        # then edit .env and set GROQ_API_KEY

# 3. Build the corpus and index
python build_chunks.py      # documents/ -> chunks.json   (ingest + clean + chunk)
python embed_store.py       # chunks.json -> chroma_db/   (embed + store)

# 4. Verify retrieval, then launch the UI
python test_retrieval.py
python app.py               # http://localhost:7860
```
