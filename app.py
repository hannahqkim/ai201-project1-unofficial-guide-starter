"""Milestone 5: Gradio web interface.

Run:  python app.py     then open http://localhost:7860

A viewer should be able to use this without narration: type a question about
getting hard NYC restaurant reservations, click Ask (or press Enter), and read
the grounded answer plus the documents it was retrieved from.
"""

from __future__ import annotations

import gradio as gr

from query import ask

EXAMPLES = [
    "How far in advance do Tatiana reservations drop, and at what time?",
    "What is Resy Notify and how does it help me get a sold-out table?",
    "What walk-in strategy is recommended for Ha's Snack Bar?",
    "Is it legal to buy a restaurant reservation on Appointment Trader in New York?",
    "If I can't get a reservation, how do I still eat at a hard-to-book spot?",
]


def handle_query(question: str):
    result = ask(question)
    if result["sources"]:
        sources = "\n".join(f"• {s}" for s in result["sources"])
    else:
        sources = "(no sources — the guide doesn't cover this)"
    return result["answer"], sources


with gr.Blocks(title="The Unofficial Guide: Hard NYC Reservations") as demo:
    gr.Markdown(
        "# The Unofficial Guide — Hard NYC Restaurant Reservations\n"
        "Ask how to get into NYC's toughest tables. Answers come **only** from the "
        "guide's source documents, with the sources shown. If the guide doesn't "
        "cover something, it will say so."
    )
    with gr.Row():
        inp = gr.Textbox(
            label="Your question",
            placeholder="e.g. How do I get a reservation at Tatiana?",
            lines=2,
            scale=4,
        )
        btn = gr.Button("Ask", variant="primary", scale=1)

    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    gr.Examples(examples=EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
