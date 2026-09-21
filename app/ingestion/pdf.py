import re
import fitz


def _validate_open_document(doc, max_pages: int | None = None) -> int:
    """Validate an already-open PDF and return its page count."""
    page_count = len(doc)
    if page_count == 0:
        raise ValueError("PDF contains no pages")
    if max_pages is not None and page_count > max_pages:
        raise ValueError(f"PDF contains {page_count} pages; maximum allowed is {max_pages}")
    return page_count


def _open_pdf_bytes(content: bytes):
    if b"%PDF-" not in content[:1024]:
        raise ValueError("Uploaded file is not a valid PDF")
    try:
        return fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("Uploaded file is corrupt or unreadable as a PDF") from exc


def _metadata(doc):
    meta = doc.metadata or {}
    first_text = doc[0].get_text("text")[:16000] if len(doc) else ""
    title = (meta.get("title") or "").strip() or None
    if not title and first_text:
        lines = [re.sub(r"\s+", " ", x).strip() for x in first_text.splitlines() if x.strip()]
        for line in lines[:30]:
            if 15 <= len(line) <= 180 and not re.search(r"abstract|keywords|university|department|proceedings|arxiv", line, re.I):
                title = line
                break
    author_raw = (meta.get("author") or "").strip()
    authors = [a.strip() for a in re.split(r"[,;]|\band\b", author_raw) if a.strip()] or None
    if not authors and first_text:
        m = re.search(r"(?im)^(?:authors?|by)\s*[:\-]?\s*(.+)$", first_text)
        if m:
            authors = [a.strip() for a in re.split(r",|;|\band\b", m.group(1)) if a.strip()]
    year = None
    publication_patterns = [
        r"(?im)\b(?:published online|online publication|date published|publication date|published|publication|online)\b[^\n]{0,80}?\b((?:19|20)\d{2})\b",
    ]
    for pattern in publication_patterns:
        m = re.search(pattern, first_text[:12000])
        if m:
            candidate = int(m.group(1))
            if 1900 <= candidate <= 2100:
                year = candidate
                break
    if year is None:
        arxiv = re.search(r"\barXiv:\d{4}\.(?:\d{4,5})(?:v\d+)?\b", first_text, re.I)
        if arxiv:
            year = 2000 + int(arxiv.group(0)[6:8])
    abstract = None
    m = re.search(r"(?is)\babstract\b\s*[:\n]\s*(.+?)(?:\n\s*(?:keywords?|1\.?\s+introduction|introduction)\b|$)", first_text)
    if m:
        abstract = re.sub(r"\s+", " ", m.group(1)).strip()[:10000]
    doi = None
    m = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", first_text, re.I)
    if m:
        doi = m.group(0).rstrip(".,;)")
    return {"title": title, "authors": authors, "year": year, "abstract": abstract, "doi": doi}


def _ocr_page(page):
    import pytesseract
    from PIL import Image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img).strip()


def extract_pdf(path: str, enable_ocr: bool = False, ocr_min_chars: int = 80, max_pages: int | None = None):
    """Open, validate, and extract a PDF in one document pass."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(1024)
        if b"%PDF-" not in header:
            raise ValueError("Uploaded file is not a valid PDF")
        doc = fitz.open(path)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Uploaded file is corrupt or unreadable as a PDF") from exc
    try:
        _validate_open_document(doc, max_pages)
        pages = []
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            extracted_chars = len(text)
            ocr_used = False
            ocr_error = None
            if enable_ocr and extracted_chars < ocr_min_chars:
                try:
                    ocr_text = _ocr_page(page)
                    if len(ocr_text) > len(text):
                        text = ocr_text
                        ocr_used = True
                except Exception as exc:
                    ocr_error = type(exc).__name__
            pages.append({"page": page_no, "text": text, "extracted_chars": extracted_chars,
                          "ocr_used": ocr_used, "ocr_error": ocr_error, "has_text": bool(text)})
        metadata = _metadata(doc)
    finally:
        doc.close()
    return pages, metadata
