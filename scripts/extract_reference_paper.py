from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "references" / "ssrn-4007466.pdf"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "exports" / "ssrn-4007466_extracted.txt"


@dataclass(frozen=True)
class ExtractionResult:
    backend: str
    page_count: int
    pages: list[str]


def _non_empty_page_count(pages: list[str]) -> int:
    return sum(bool(page.strip()) for page in pages)


def _extract_with_pypdf(pdf_path: Path) -> ExtractionResult:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:
            raise RuntimeError(f"pypdf failed on page {page_number}: {exc}") from exc

    return ExtractionResult(backend="pypdf", page_count=len(reader.pages), pages=pages)


def _extract_with_pymupdf(pdf_path: Path) -> ExtractionResult:
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    doc = pymupdf.open(str(pdf_path))
    try:
        pages = [page.get_text("text") or "" for page in doc]
        return ExtractionResult(backend="pymupdf", page_count=doc.page_count, pages=pages)
    finally:
        doc.close()


def extract_pdf_text(pdf_path: Path) -> ExtractionResult:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Reference paper not found: {pdf_path}")

    errors: list[str] = []
    for extractor in (_extract_with_pypdf, _extract_with_pymupdf):
        try:
            result = extractor(pdf_path)
            if result.page_count <= 0:
                raise RuntimeError("PDF has no pages")
            if _non_empty_page_count(result.pages) == 0:
                raise RuntimeError("extracted zero non-empty pages")
            return result
        except Exception as exc:
            errors.append(f"{extractor.__name__}: {exc}")

    joined_errors = "\n".join(f"- {error}" for error in errors)
    raise RuntimeError(f"PDF text extraction failed for {pdf_path}:\n{joined_errors}")


def format_extracted_text(result: ExtractionResult, pdf_path: Path) -> str:
    header = (
        "Local text extraction artifact for methodology audit.\n"
        f"Source PDF: {pdf_path.relative_to(PROJECT_ROOT)}\n"
        f"Extraction backend: {result.backend}\n"
        f"Page count: {result.page_count}\n\n"
        "Copyright note: do not commit full extracted copyrighted text if this repository "
        "may become public. This artifact is intended for local audit use only.\n"
    )

    parts = [header]
    for page_number, text in enumerate(result.pages, start=1):
        page_text = text.strip() or "[No extractable text on this page]"
        parts.append(f"\n\n===== PAGE {page_number} =====\n\n{page_text}\n")
    return "".join(parts)


def main() -> None:
    result = extract_pdf_text(PDF_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(format_extracted_text(result, PDF_PATH), encoding="utf-8")

    print(f"Backend: {result.backend}")
    print(f"Page count: {result.page_count}")
    print(f"Non-empty pages: {_non_empty_page_count(result.pages)}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
