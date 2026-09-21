import re
import uuid
from dataclasses import dataclass


@dataclass
class ParentRecord:
    id: str
    text: str
    page_start: int
    page_end: int
    section: str | None
    metadata: dict


@dataclass
class ChunkRecord:
    text: str
    page: int
    page_start: int
    page_end: int
    section: str | None
    parent_id: str
    metadata: dict


_HEADING_RE = re.compile(r"^(?:(?:\d+(?:\.\d+)*\.?)\s+)?[A-Z][A-Za-z0-9 ,:/&()'\-–—.]{2,100}$")
_ABBREVIATIONS = {"e.g.", "i.e.", "etc.", "et al.", "fig.", "eq.", "dr.", "mr.", "mrs.", "ms.", "prof.", "vs.", "no.", "u.s.", "u.k."}


def is_heading(line: str) -> bool:
    s = re.sub(r"\s+", " ", line.strip())
    if not s or len(s) > 110 or len(s.split()) > 14 or s.endswith((".", ",", ";", ":")):
        return False
    if not _HEADING_RE.match(s):
        return False
    if re.match(r"^\d+(?:\.\d+)*\.?\s+", s):
        return True
    letters = [ch for ch in s if ch.isalpha()]
    if letters and sum(ch.isupper() for ch in letters) / len(letters) >= 0.70:
        return True
    words = [w.strip("()[]{}") for w in s.split() if w.strip("()[]{}")]
    significant = [w for w in words if w.lower() not in {"a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "with"}]
    return bool(significant) and all(w[:1].isupper() for w in significant if w)


def _sentence_spans(text: str):
    spans = []
    start = 0
    for match in re.finditer(r"[.!?]+(?=\s+|$)", text):
        end = match.end()
        prefix = text[start:end].rstrip()
        if not prefix:
            continue
        last_token = prefix.split()[-1].lower()
        if last_token in _ABBREVIATIONS:
            continue
        if match.start() > 0 and match.end() < len(text):
            left, right = text[match.start() - 1], text[match.end()]
            if left.isdigit() and right.isdigit():
                continue
        value = text[start:end].strip()
        if value:
            left_trim = len(text[start:end]) - len(text[start:end].lstrip())
            right_trim = len(text[start:end].rstrip())
            spans.append((value, start + left_trim, start + right_trim))
        start = end
    tail = text[start:].strip()
    if tail:
        left_trim = len(text[start:]) - len(text[start:].lstrip())
        spans.append((tail, start + left_trim, len(text)))
    return spans


def _word_spans(text: str):
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _page_for_offset(page_ranges, offset):
    for page, start, end in page_ranges:
        if start <= offset < end:
            return page
    if page_ranges and offset == page_ranges[-1][2]:
        return page_ranges[-1][0]
    return page_ranges[-1][0] if page_ranges else 1


def _pages_for_span(page_ranges, start, end):
    pages = [page for page, ps, pe in page_ranges if start < pe and end > ps]
    if not pages:
        p = _page_for_offset(page_ranges, start)
        return p, p
    return min(pages), max(pages)


def _relative_page_ranges(page_ranges, start, end, text_length):
    """Return page spans relative to a chunk's own text coordinates."""
    ranges = []
    for page, ps, pe in page_ranges:
        if ps >= end or pe <= start:
            continue
        ranges.append((page, max(0, ps - start), min(text_length, pe - start)))
    return ranges


def _section_records(pages):
    current_section = "Unknown"
    lines = []
    for page in pages:
        for raw in page["text"].splitlines():
            line = raw.strip()
            if not line:
                if lines and lines[-1][0] != "":
                    lines.append(("", page["page"]))
                continue
            if is_heading(line):
                if lines:
                    yield current_section, lines
                    lines = []
                current_section = line
            else:
                lines.append((line, page["page"]))
    if lines:
        yield current_section, lines


def _build_section_text(lines):
    pieces = []
    page_ranges = []
    pos = 0
    first = True
    for line, page in lines:
        if not line:
            if pieces and pieces[-1] != "\n":
                pieces.append("\n")
                pos += 1
            continue
        if not first and pieces and pieces[-1] != "\n":
            pieces.append(" ")
            pos += 1
        start = pos
        pieces.append(line)
        pos += len(line)
        page_ranges.append((page, start, pos))
        first = False
    return "".join(pieces), page_ranges


def _split_by_words(text: str, start: int, end: int, size: int, overlap: int):
    result = []
    words = [(w, a, b) for w, a, b in _word_spans(text) if a >= start and b <= end]
    if not words:
        return result
    i = 0
    while i < len(words):
        chunk_start = words[i][1]
        j = i
        while j + 1 < len(words) and words[j + 1][2] - chunk_start <= size:
            j += 1
        chunk_end = words[j][2]
        result.append((text[chunk_start:chunk_end].strip(), chunk_start, chunk_end))
        if chunk_end >= end:
            break
        target = max(chunk_start + 1, chunk_end - overlap)
        next_i = i + 1
        while next_i < len(words) and words[next_i][1] < target:
            next_i += 1
        i = max(i + 1, next_i)
    return result


def _make_parents(text, page_ranges, section, parent_size):
    spans = _sentence_spans(text) or [(text, 0, len(text))]
    parents = []
    i = 0
    while i < len(spans):
        start = spans[i][1]
        j = i
        while j + 1 < len(spans) and spans[j + 1][2] - start <= parent_size:
            j += 1
        end = spans[j][2]
        pieces = _split_by_words(text, start, end, parent_size, 0) if end - start > parent_size and i == j else [(text[start:end].strip(), start, end)]
        for parent_text, ps, pe in pieces:
            if not parent_text:
                continue
            # Correct for stripping whitespace at the parent boundaries.
            leading = len(text[ps:pe]) - len(text[ps:pe].lstrip())
            actual_start = ps + leading
            actual_end = pe - (len(text[ps:pe]) - len(text[ps:pe].rstrip()))
            parent_text = text[actual_start:actual_end]
            page_start, page_end = _pages_for_span(page_ranges, actual_start, actual_end)
            rel_ranges = _relative_page_ranges(page_ranges, actual_start, actual_end, len(parent_text))
            parents.append((parent_text, page_start, page_end, actual_start, actual_end, rel_ranges))
        i = j + 1
    return parents


def _child_chunks(text, page_ranges, child_size, overlap):
    spans = _sentence_spans(text) or [(text, 0, len(text))]
    chunks = []
    i = 0
    while i < len(spans):
        start = spans[i][1]
        j = i
        while j + 1 < len(spans) and spans[j + 1][2] - start <= child_size:
            j += 1
        end = spans[j][2]
        if end - start > child_size and i == j:
            chunks.extend(_split_by_words(text, start, end, child_size, overlap))
            i = j + 1
            continue
        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        actual_start = start + leading
        actual_end = end - trailing
        chunks.append((text[actual_start:actual_end], actual_start, actual_end))
        if j == len(spans) - 1:
            break
        if overlap <= 0:
            i = j + 1
            continue
        target = max(start + 1, end - overlap)
        # Choose the earliest sentence whose end reaches the target. This
        # preserves a real sentence-level overlap instead of always jumping to j+1.
        next_i = j
        while next_i > i and spans[next_i - 1][2] > target:
            next_i -= 1
        # If the target falls inside the final sentence, retain that sentence;
        # otherwise retain the latest preceding sentence that overlaps target.
        i = max(i + 1, next_i)
    output = []
    for piece, start, end in chunks:
        if not piece:
            continue
        ps, pe = _pages_for_span(page_ranges, start, end)
        output.append((piece, ps, pe, start, end))
    return output


def chunk_pages(pages: list[dict], parent_size: int = 1800, child_size: int = 900, overlap: int = 120):
    if parent_size <= 0 or child_size <= 0 or overlap < 0 or overlap >= child_size:
        raise ValueError("Invalid chunk sizes/overlap")
    if parent_size < child_size:
        raise ValueError("parent_size must be >= child_size")
    parents, children = [], []
    for section, lines in _section_records(pages):
        text, page_ranges = _build_section_text(lines)
        if not text:
            continue
        for parent_text, ps, pe, start, end, parent_page_ranges in _make_parents(text, page_ranges, section, parent_size):
            parent_id = str(uuid.uuid4())
            parent_metadata = {
                "section": section,
                "page_start": ps,
                "page_end": pe,
                # Coordinates are relative to parent_text.
                "parent_page_ranges": [[p, a, b] for p, a, b in parent_page_ranges],
            }
            parents.append(ParentRecord(parent_id, parent_text, ps, pe, section, parent_metadata))
            for piece, cps, cpe, cstart, cend in _child_chunks(parent_text, parent_page_ranges, child_size, overlap):
                child_ranges = _relative_page_ranges(parent_page_ranges, cstart, cend, cend - cstart)
                children.append(ChunkRecord(
                    piece, cps, cps, cpe, section, parent_id,
                    {
                        "page": cps,
                        "page_start": cps,
                        "page_end": cpe,
                        "section": section,
                        "parent_page_start": ps,
                        "parent_page_end": pe,
                        # Coordinates are relative to this child chunk's text.
                        "child_page_ranges": [[p, a, b] for p, a, b in child_ranges],
                    },
                ))
    return parents, children
