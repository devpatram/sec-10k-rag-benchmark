"""Chunking: one PDF page per chunk, each carrying its document name and page number."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DOCUMENTS = ("NVIDIA_FY2026_10-K", "AMD_FY2025_10-K", "Intel_FY2025_10-K")


@dataclass(frozen=True)
class Page:
    chunk_id: int
    document: str
    page: int  # physical page in the PDF, starting at 1
    text: str

    def label(self) -> str:
        return f"{self.document}, page {self.page}"


def load_pages(pdf_dir: str | Path, documents=DOCUMENTS) -> list[Page]:
    """Reads every page of each PDF. A page is the unit: no token windows and no overlap."""
    try:
        import pymupdf
    except ImportError:  # older PyMuPDF
        import fitz as pymupdf
    pages: list[Page] = []
    for name in documents:
        path = Path(pdf_dir) / f"{name}.pdf"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. See data/README.md for how to get the filings.")
        with pymupdf.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                pages.append(Page(len(pages), name, i, page.get_text("text").strip()))
    return pages


def normalize(text: str) -> str:
    """Collapses whitespace so an anchor still matches when the PDF breaks a line inside it."""
    return " ".join(text.split())


def anchor_pattern(anchor: str) -> re.Pattern:
    """An anchor is a regular expression (alternatives with |); a plain space matches any run of whitespace.

    It must not start in the middle of a number, so "12 million shares" does not match "112 million shares".
    """
    return re.compile(r"(?<![\w,.])(?:" + r"\s+".join(anchor.split(" ")) + ")", re.I)


def evidence_pages(pages: list[Page], document: str, anchor: str) -> set[int]:
    """The pages of one document whose text contains the anchor.

    Evidence is located by its text, not by fixed page numbers, so the metrics still hold if the filings are
    printed with different page breaks.
    """
    pattern = anchor_pattern(anchor)
    return {p.page for p in pages if p.document == document and pattern.search(normalize(p.text))}
