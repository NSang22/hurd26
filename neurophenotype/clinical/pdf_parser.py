"""
PDF Parser for prior genetic testing reports.

Strategy:
1. pdfplumber for digital PDFs
2. OCR fallback for scanned PDFs

Also provides lightweight post-processing to extract a structured summary
of likely test type, result class, and genes mentioned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedPage:
    page_num: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    ocr_used: bool = False


def parse_pdf(path: str | Path) -> list[ParsedPage]:
    """Parse a PDF into pages, falling back to OCR when digital text is sparse."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = _parse_with_pdfplumber(path)

    avg_len = sum(len(page.text) for page in pages) / max(len(pages), 1)
    if avg_len < 50:
        pages = _parse_with_ocr(path)

    return pages


def _parse_with_pdfplumber(path: Path) -> list[ParsedPage]:
    import pdfplumber

    results: list[ParsedPage] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            clean_tables = [
                [[cell or "" for cell in row] for row in table]
                for table in tables
            ]
            results.append(
                ParsedPage(
                    page_num=i + 1,
                    text=text,
                    tables=clean_tables,
                    ocr_used=False,
                )
            )
    return results


def _parse_with_ocr(path: Path) -> list[ParsedPage]:
    """Convert PDF pages to images then OCR with Tesseract."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ImportError("pytesseract and Pillow are required for OCR fallback") from exc

    try:
        import fitz

        doc = fitz.open(str(path))
        results: list[ParsedPage] = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, config="--psm 6")
            results.append(
                ParsedPage(
                    page_num=i + 1,
                    text=text,
                    tables=[],
                    ocr_used=True,
                )
            )
        return results
    except ImportError:
        from pdf2image import convert_from_path

        images = convert_from_path(str(path), dpi=200)
        results: list[ParsedPage] = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, config="--psm 6")
            results.append(
                ParsedPage(
                    page_num=i + 1,
                    text=text,
                    tables=[],
                    ocr_used=True,
                )
            )
        return results


def stitch_pages(pages: list[ParsedPage]) -> str:
    """Merge all page text into one normalized string."""
    combined = "\n".join(page.text for page in pages)
    return re.sub(r"\n{3,}", "\n\n", combined)


def extract_all_tables(pages: list[ParsedPage]) -> list[list[list[str]]]:
    """Return every table across pages as a flat list."""
    tables: list[list[list[str]]] = []
    for page in pages:
        tables.extend(page.tables)
    return tables


def has_handwriting_hint(pages: list[ParsedPage]) -> bool:
    """Conservative placeholder for future OCR confidence-based checks."""
    if not any(page.ocr_used for page in pages):
        return False
    return False


def summarize_genetic_test_report(pages: list[ParsedPage]) -> dict[str, object]:
    """
    Extract a minimal structured summary from parsed report text.

    This is intentionally rule-based for hackathon reliability. Claude can refine
    this later, but the pipeline needs deterministic fields immediately.
    """
    text = stitch_pages(pages)
    text_lower = text.lower()

    genes = sorted(set(re.findall(r"\b(?:MECP2|SCN1A|UBE3A|CDKL5|FOXG1)\b", text, flags=re.IGNORECASE)))
    genes = [gene.upper() for gene in genes]

    if "whole exome" in text_lower or "wes" in text_lower or "exome sequencing" in text_lower:
        test_type = "exome"
    elif "methylation" in text_lower:
        test_type = "methylation"
    elif "microarray" in text_lower or "array cgh" in text_lower:
        test_type = "array_cgh"
    elif "panel" in text_lower:
        test_type = "targeted_panel"
    else:
        test_type = "unknown"

    if "variant of uncertain significance" in text_lower or re.search(r"\bvus\b", text_lower):
        result_class = "vus"
    elif "negative" in text_lower or "no pathogenic variant" in text_lower:
        result_class = "negative"
    elif "incomplete" in text_lower or "limited genes" in text_lower:
        result_class = "incomplete_panel"
    elif "pathogenic" in text_lower or "likely pathogenic" in text_lower:
        result_class = "positive"
    else:
        result_class = "unknown"

    date_match = re.search(r"\b(20\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01]))\b", text)
    if not date_match:
        date_match = re.search(r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/20\d{2}\b", text)

    summary_lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = " ".join(summary_lines[:3])[:500]

    return {
        "test_type": test_type,
        "result_class": result_class,
        "date": date_match.group(0) if date_match else None,
        "genes_mentioned": genes,
        "ocr_used": any(page.ocr_used for page in pages),
        "summary": summary,
    }
