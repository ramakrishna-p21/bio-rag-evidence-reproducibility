"""Original-PDF search hints; extracted text never establishes scientific gold."""

from pathlib import Path

from pydantic import Field

from biorag_repro.benchmark.models import Record
from biorag_repro.core.hashing import verify_file_sha256


class PdfHit(Record):
    page: int = Field(ge=1)
    search_term: str
    region: list[float]
    nearby_text: str
    status: str = "location_hint_only"


def locate_pdf(
    path: Path, expected_sha256: str, terms: list[str], *, max_hits: int = 30
) -> list[PdfHit]:
    """Locate terms in verified frozen PDF bytes for direct page inspection."""
    import pymupdf  # type: ignore[import-untyped]

    verify_file_sha256(path, expected_sha256)
    hits: list[PdfHit] = []
    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, 1):
            for term in dict.fromkeys(terms):
                if not term.strip():
                    continue
                for rectangle in page.search_for(term):
                    nearby = page.get_textbox(rectangle + (-50, -30, 50, 30)).strip()
                    hits.append(
                        PdfHit(
                            page=page_number,
                            search_term=term,
                            region=[rectangle.x0, rectangle.y0, rectangle.x1, rectangle.y1],
                            nearby_text=nearby,
                        )
                    )
                    if len(hits) >= max_hits:
                        return hits
    return hits
