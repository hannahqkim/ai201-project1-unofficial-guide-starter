"""Ingestion + cleaning.

Loads every document from the documents/ folder and strips it down to the
substantive content we want the RAG system to retrieve over. Cleaning is kept
deliberately conservative: it removes boilerplate (HTML tags/entities, markdown
image and link syntax, photo credits, nav/CTA lines like "Book Now") while
preserving the actual reservation advice.

Supported file types: .txt, .md, and (optionally) .pdf if pdfplumber is installed.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

# Default location of the corpus, relative to the repo root.
DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}

# Lines that are pure boilerplate / call-to-action / navigation and carry no
# domain meaning. Matched case-insensitively against a fully-stripped line.
_BOILERPLATE_LINE_PATTERNS = [
    r"^book now$",
    r"^reserve a table.*$",
    r"^reserve a table with$",
    r"^read more.*$",
    r"^find more.*$",
    r"^read on\.?$",
    r"^share$",
    r"^photo by .*$",
    r"^photo courtesy of .*$",
    r"^exclusive reservations with.*$",
    r"^perfect for:.*$",
    r"^\d+\.\s*$",            # stray list numbers left over from nav
]
_BOILERPLATE_LINE_RE = re.compile("|".join(_BOILERPLATE_LINE_PATTERNS), re.IGNORECASE)

# Inline patterns.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")          # ![alt](url)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")          # [text](url) -> text
_BARE_URL_LINE_RE = re.compile(r"^\s*(https?://|www\.)\S+\s*$", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Return a cleaned version of ``raw`` document text.

    Steps, in order:
      1. Unescape HTML entities (&amp;, &nbsp;, &#39; -> &, space, ').
      2. Remove markdown image syntax entirely.
      3. Convert markdown links to just their visible text.
      4. Strip any remaining HTML tags.
      5. Drop boilerplate / nav / CTA / photo-credit lines and bare-URL lines.
      6. Normalize whitespace (collapse runs of spaces, cap blank lines at one).
    """
    text = html.unescape(raw)

    text = _MD_IMAGE_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub("", text)
    # Non-breaking spaces and zero-width junk that survive unescaping.
    text = text.replace("\xa0", " ").replace("​", "")

    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")  # preserve paragraph breaks
            continue
        if _BOILERPLATE_LINE_RE.match(stripped):
            continue
        if _BARE_URL_LINE_RE.match(stripped):
            continue
        kept_lines.append(stripped)

    text = "\n".join(kept_lines)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF, if pdfplumber is available."""
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            f"{path.name} is a PDF but pdfplumber is not installed. "
            "Add pdfplumber to requirements.txt and `pip install` it, "
            "or convert the file to .txt."
        ) from exc

    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def _read_raw(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8", errors="replace")


def load_documents(documents_dir: Path | str = DOCUMENTS_DIR) -> list[dict]:
    """Load and clean every supported document in ``documents_dir``.

    Returns a list of dicts, one per document, each shaped like::

        {
            "source":   "01_resy_toughest_reservations_nyc.txt",  # filename = attribution
            "path":     "/abs/path/...",
            "raw":      "<original text>",
            "text":     "<cleaned text>",
            "raw_len":  12345,
            "clean_len": 11000,
        }

    The ``source`` field is what gets attached to every chunk as metadata so we
    can always trace a retrieved chunk back to its origin.
    """
    documents_dir = Path(documents_dir)
    if not documents_dir.exists():
        raise FileNotFoundError(f"documents directory not found: {documents_dir}")

    docs: list[dict] = []
    for path in sorted(documents_dir.iterdir()):
        if path.name.startswith("."):  # skip .gitkeep, .DS_Store, etc.
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        raw = _read_raw(path)
        cleaned = clean_text(raw)
        if not cleaned:
            print(f"  ! {path.name}: empty after cleaning — skipping")
            continue

        docs.append(
            {
                "source": path.name,
                "path": str(path),
                "raw": raw,
                "text": cleaned,
                "raw_len": len(raw),
                "clean_len": len(cleaned),
            }
        )

    if not docs:
        raise RuntimeError(
            f"No documents loaded from {documents_dir}. "
            "Add .txt/.md/.pdf files and try again."
        )
    return docs
