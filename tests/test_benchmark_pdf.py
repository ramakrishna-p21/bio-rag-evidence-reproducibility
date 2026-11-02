from pathlib import Path

import pymupdf

from biorag_repro.benchmark.pdf import locate_pdf
from biorag_repro.core.hashing import sha256_file


def test_pdf_locator_returns_page_bounds_not_confirmation(tmp_path: Path) -> None:
    path = tmp_path / "source.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((70, 70), "BRCA1 effect was 1.2 under condition A.")
    document.save(path)
    document.close()
    hits = locate_pdf(path, sha256_file(path), ["1.2", "BRCA1"])
    assert hits and all(hit.page == 1 for hit in hits)
    assert all(hit.region[0] < hit.region[2] for hit in hits)
    assert all(hit.status == "location_hint_only" for hit in hits)
    assert locate_pdf(path, sha256_file(path), ["missing phrase"]) == []
