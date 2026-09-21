import time
import re
from sqlalchemy import select
from app.db.models import Chunk, ParentChunk
from app.core.config import settings
from app.retrieval.bm25 import BM25Index
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.retrieval.vector import vector_search
from app.services.query import rewrite_queries, hyde_query
from app.services.compression import compress
from app.services.citations import verify_citations
from app.services.llm import embed, generate
from app.core.logging import logger

_bm25 = BM25Index()
_bm25_dirty = True


def invalidate_bm25():
    global _bm25_dirty
    _bm25_dirty = True


def rebuild_bm25(db):
    global _bm25_dirty
    if not _bm25_dirty:
        return len(_bm25.rows)
    rows = db.execute(
        select(
            Chunk.id, Chunk.text, Chunk.document_id, Chunk.parent_id, Chunk.page,
            Chunk.page_start, Chunk.page_end, Chunk.section, Chunk.metadata_json
        ).order_by(Chunk.id)
    ).all()
    _bm25.build(rows)
    _bm25_dirty = False
    return len(rows)


def _parent_expand(db, candidates: list[dict], document_ids=None):
    parent_ids = list(dict.fromkeys(c.get("parent_id") for c in candidates if c.get("parent_id")))
    if not parent_ids:
        return candidates
    stmt = select(ParentChunk).where(ParentChunk.id.in_(parent_ids))
    if document_ids is not None:
        if not document_ids:
            return []
        stmt = stmt.where(ParentChunk.document_id.in_(document_ids))
    parents = {p.id: p for p in db.execute(stmt).scalars().all()}
    out = []
    for c in candidates:
        p = parents.get(c.get("parent_id"))
        if p:
            out.append({
                **c,
                "parent_text": p.text,
                "parent_page_start": p.page_start,
                "parent_page_end": p.page_end,
                "parent_section": p.section,
                "parent_metadata": p.metadata_json or {},
                "child_page_ranges": c.get("metadata", {}).get("child_page_ranges", []),
            })
    return out


def retrieve_candidates(db, question: str, document_ids=None, use_hyde=False, candidate_k=24, use_multi_query=None, use_bm25=True, use_vector=True):
    if use_multi_query is None:
        use_multi_query = settings.enable_multi_query
    stage_status = {"multi_query": "disabled" if not use_multi_query else "ok",
                    "hyde": "disabled" if not use_hyde else "ok"}
    if use_bm25:
        rebuild_bm25(db)
    if use_vector:
        try:
            queries = rewrite_queries(question, use_multi_query)
        except Exception:
            logger.exception("Query rewriting failed; falling back to the original query")
            queries = [question]
            if use_multi_query:
                stage_status["multi_query"] = "fallback"
        if use_hyde:
            try:
                queries.append(hyde_query(question))
            except Exception:
                logger.exception("HyDE generation failed; continuing without HyDE")
                stage_status["hyde"] = "fallback"
    else:
        queries = [question]
        if use_multi_query:
            stage_status["multi_query"] = "skipped_no_vector"
        if use_hyde:
            stage_status["hyde"] = "skipped_no_vector"
    dense_lists = []
    if use_vector:
        vectors = embed(queries)
        for q, qvec in zip(queries, vectors):
            rows = vector_search(db, q, document_ids, settings.top_k_vector, query_vector=qvec)
            dense_lists.append([(r["id"], r["score"], r) for r in rows])
    lexical = _bm25.search(question, settings.top_k_bm25, document_ids) if use_bm25 else []
    lexical_list = [
        (r[0], r[1], {
            "id": r[0], "text": r[1], "document_id": r[2], "parent_id": r[3],
            "page": r[4], "page_start": r[5], "page_end": r[6], "section": r[7], "metadata": r[8]
        })
        for r in lexical
    ]
    fused = reciprocal_rank_fusion(*dense_lists, lexical_list)
    candidates = []
    for rank, (item_id, rrf_score, payload) in enumerate(fused[:candidate_k], start=1):
        candidates.append({**payload, "rrf_rank": rank, "rrf_score": float(rrf_score), "ranking_source": "rrf"})
    return candidates, queries, stage_status


def _matching_spans(text: str, evidence: str):
    """Find every whitespace-tolerant occurrence of evidence in text."""
    words = evidence.split()
    if not words:
        return []
    pattern = r"\s+".join(re.escape(w) for w in words)
    return [m.span() for m in re.finditer(pattern, text, flags=re.IGNORECASE)]


def _span_pages(ranges, start, end):
    pages = [int(r[0]) for r in ranges if start < int(r[2]) and end > int(r[1])]
    return (min(pages), max(pages)) if pages else None


def _annotate_evidence_pages(source: dict):
    evidence = str(source.get("evidence", "")).strip()
    if not evidence:
        return source

    # Prefer an exact occurrence inside the retrieved child. This is the most
    # precise evidence locator because child_page_ranges use child-text coordinates.
    retrieval_text = str(source.get("retrieval_text", ""))
    child_ranges = source.get("child_page_ranges") or source.get("metadata", {}).get("child_page_ranges") or []
    child_matches = _matching_spans(retrieval_text, evidence) if retrieval_text else []
    if child_matches and child_ranges:
        pages = _span_pages(child_ranges, *child_matches[0])
        if pages:
            source["evidence_page_start"], source["evidence_page_end"] = pages
            source["evidence_page_source"] = "retrieved_child"
            return source

    # Otherwise locate all occurrences in the context text. If several exist,
    # prefer the occurrence overlapping the retrieved child's source pages.
    text = str(source.get("text", ""))
    ranges = source.get("page_ranges") or []
    matches = _matching_spans(text, evidence)
    if not matches:
        return source
    candidates = []
    retrieval_pages = set()
    if source.get("retrieval_page_start") is not None:
        retrieval_pages.update(range(int(source["retrieval_page_start"]), int(source.get("retrieval_page_end") or source["retrieval_page_start"]) + 1))
    for span in matches:
        pages = _span_pages(ranges, *span)
        if pages:
            overlap = bool(retrieval_pages.intersection(range(pages[0], pages[1] + 1)))
            candidates.append((0 if overlap else 1, span[0], pages))
    if not candidates:
        return source
    _, _, pages = min(candidates)
    source["evidence_page_start"], source["evidence_page_end"] = pages
    source["evidence_page_source"] = "context_preferred" if len(matches) > 1 else "context"
    return source

def answer_question(db, question: str, document_ids=None, top_k=None, use_hyde=False):
    top_k = top_k or settings.default_top_k
    rerank_pool_multiplier = 3
    retrieval_k = max(settings.retrieval_candidate_k, top_k * 4)
    timings = {}
    pipeline_status = {"multi_query": "enabled" if settings.enable_multi_query else "disabled",
                       "hyde": "enabled" if use_hyde else "disabled",
                       "reranker": "ok",
                       "compression": {"configured": bool(settings.enable_compression), "used": False,
                                       "status": "configured" if settings.enable_compression else "disabled"},
                       "citation_audit": "disabled" if not settings.enable_citation_audit else "ok"}
    start = time.perf_counter()
    candidates, queries, retrieval_status = retrieve_candidates(db, question, document_ids, use_hyde, retrieval_k)
    pipeline_status.update(retrieval_status)
    timings["retrieval_ms"] = round((time.perf_counter() - start) * 1000, 1)

    start = time.perf_counter()
    rerank_k = min(len(candidates), max(top_k * rerank_pool_multiplier, top_k))
    try:
        reranked = rerank(question, candidates, rerank_k)
    except Exception:
        logger.exception("Reranker unavailable; falling back to RRF ordering")
        pipeline_status["reranker"] = "fallback"
        reranked = [{**c, "ranking_source": "rrf"} for c in candidates[:rerank_k]]
    else:
        reranked = [{**c, "ranking_source": "reranker"} for c in reranked]
    expanded = _parent_expand(db, reranked, document_ids)

    # Keep the strongest child hit for each parent, then choose final parents.
    unique = {}
    for item in expanded:
        key = item.get("parent_id") or item["id"]
        score = item.get("rerank_score") if item.get("ranking_source") == "reranker" else item.get("rrf_score", float("-inf"))
        current = unique.get(key)
        current_score = (current.get("rerank_score") if current.get("ranking_source") == "reranker"
                         else current.get("rrf_score", float("-inf"))) if current else float("-inf")
        if current is None or score > current_score:
            unique[key] = item
    if any(x.get("ranking_source") == "reranker" for x in unique.values()):
        reranked = sorted(unique.values(), key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)[:top_k]
    else:
        reranked = sorted(unique.values(), key=lambda x: (-x.get("rrf_score", float("-inf")), x.get("rrf_rank", 10**9)))[:top_k]

    for item in reranked:
        item["retrieval_text"] = item["text"]
        item["retrieval_page_start"] = item.get("page_start")
        item["retrieval_page_end"] = item.get("page_end")
        if item.get("parent_text"):
            item["text"] = item["parent_text"]
            item["context_page_start"] = item["parent_page_start"]
            item["context_page_end"] = item["parent_page_end"]
            item["page_ranges"] = item.get("parent_metadata", {}).get("parent_page_ranges", [])
            item["section"] = item.get("parent_section") or item.get("section")
    timings["rerank_ms"] = round((time.perf_counter() - start) * 1000, 1)

    start = time.perf_counter()
    try:
        compressed = compress(question, reranked) if settings.enable_compression else []
        if settings.enable_compression and compressed:
            pipeline_status["compression"] = {"configured": True, "used": True, "status": "used"}
        elif settings.enable_compression:
            pipeline_status["compression"] = {"configured": True, "used": False, "status": "fallback"}
    except Exception:
        logger.exception("Compression failed; using full retrieved source text")
        compressed = []
        pipeline_status["compression"] = {"configured": True, "used": False, "status": "fallback"}
    sources = compressed if compressed else reranked
    source_representation = "compressed" if compressed else "retrieved"
    # Citation auditing must work whether compression is enabled, disabled,
    # or unavailable.  Reranked sources normally have only retrieval text, so
    # use that exact source text as verified evidence when no compressor
    # produced an evidence span.
    for source in sources:
        source["source_representation"] = source_representation
        if not str(source.get("evidence", "")).strip():
            source["evidence"] = str(source.get("text", ""))
            source["evidence_verified"] = bool(source["evidence"].strip())
        _annotate_evidence_pages(source)
    timings["compression_ms"] = round((time.perf_counter() - start) * 1000, 1)

    context_parts = []
    for i, s in enumerate(sources):
        s["citation_id"] = f"S{i+1}"
        page_start = s.get("evidence_page_start") or s.get("retrieval_page_start") or s.get("context_page_start") or s.get("page_start") or "?"
        page_end = s.get("evidence_page_end") or s.get("retrieval_page_end") or s.get("context_page_end") or s.get("page_end") or page_start
        page_label = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        context_parts.append(
            f"[S{i+1}] Document={s.get('document_id','unknown')} Pages={page_label} "
            f"Section={s.get('section') or s.get('parent_section') or '?'}\n"
            f"{s.get('evidence', s.get('text', ''))}"
        )

    if not context_parts:
        return {
            "answer": "I could not find relevant evidence in the selected documents.",
            "sources": [],
            "citation_verification": {"valid": False, "status": "failed", "reason": "no_retrieved_evidence"},
            "pipeline_status": pipeline_status,
            "queries": queries,
            "timings_ms": {**timings, "generation_ms": 0.0, "citation_verification_ms": 0.0,
                           "total_ms": round(sum(timings.values()), 1)},
        }

    prompt = (
        "Answer the research question using ONLY the supplied evidence. Compare documents when asked. "
        "If evidence is insufficient, explicitly say so. Every factual claim must have one or more citations "
        "in [S1], [S2] format. Never cite a source that does not support the claim.\n\n"
        f"Question: {question}\n\nEvidence:\n\n" + "\n\n".join(context_parts)
    )
    start = time.perf_counter()
    answer = generate(prompt, system="You are a rigorous research assistant. Use only supplied evidence. Cite every factual claim. If evidence is insufficient, say so.")
    timings["generation_ms"] = round((time.perf_counter() - start) * 1000, 1)

    start = time.perf_counter()
    verification = verify_citations(answer, sources) if settings.enable_citation_audit else {"valid": None, "status": "disabled"}
    if settings.enable_citation_audit:
        if verification.get("audit_unavailable"):
            pipeline_status["citation_audit"] = "unavailable"
            verification["status"] = "unavailable"
        elif verification.get("valid"):
            verification["status"] = "passed"
        else:
            verification["status"] = "failed"
    if (settings.enable_citation_audit and not verification.get("valid")
            and not verification.get("audit_unavailable") and sources):
        repair_prompt = (
            "Rewrite the answer so every factual claim is explicitly supported by the supplied evidence. "
            "Use only the evidence. Keep or remove claims as needed; never invent facts. Every factual claim "
            "must have citations like [S1]. Return only the corrected answer.\n\n"
            f"Original answer:\n{answer}\n\nAudit:\n{verification}\n\nEvidence:\n" +
            "\n".join(f"[S{i+1}] {s.get('evidence', s.get('text', ''))}" for i, s in enumerate(sources))
        )
        original_answer = answer
        original_verification = verification
        try:
            repaired = generate(repair_prompt, system="You are a strict research editor. Produce a grounded answer with valid citations only.")
            repaired_verification = verify_citations(repaired, sources)
            # Never replace the original answer with an unverified repair.
            if repaired_verification.get("valid"):
                repaired_verification["status"] = "passed"
                answer, verification = repaired, repaired_verification
            else:
                answer, verification = original_answer, original_verification
        except Exception:
            answer, verification = original_answer, original_verification
    timings["citation_verification_ms"] = round((time.perf_counter() - start) * 1000, 1)
    timings["total_ms"] = round(sum(timings.values()), 1)
    return {"answer": answer, "sources": sources, "citation_verification": verification,
            "pipeline_status": pipeline_status, "queries": queries, "timings_ms": timings}
