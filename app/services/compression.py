

def _normalize(s: str) -> str:
    return " ".join(str(s).split())


def _exact_evidence(source: str, requested: str) -> str | None:
    requested = str(requested).strip()
    if not requested:
        return None
    if requested in source:
        return requested
    # Accept whitespace-only normalization while returning the original source span.
    target = _normalize(requested)
    words = source.split()
    for i in range(len(words)):
        for j in range(i + 1, min(len(words), i + len(requested.split()) + 12) + 1):
            candidate = " ".join(words[i:j])
            if _normalize(candidate) == target:
                return candidate
    return None


def compress(question: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    evidence = "\n\n".join(f"CHUNK_{i}: {c['text']}" for i, c in enumerate(candidates))
    prompt = f"""For each supplied chunk, extract only minimal evidence that directly helps answer the question. The returned text MUST be copied verbatim from its chunk (whitespace normalization is allowed). Return JSON with key 'evidence', an array of objects with 'index' and 'text'. Omit chunks with no useful evidence.\nQuestion: {question}\n\n{evidence}"""
    from app.services.llm import generate_json
    data = generate_json(prompt, system="You are an evidence extractor. Never invent or paraphrase source text. Return JSON only.")
    result = []
    for item in data.get("evidence", []) if isinstance(data, dict) else []:
        try:
            idx = int(item["index"])
            requested = str(item["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= idx < len(candidates)):
            continue
        verified = _exact_evidence(candidates[idx]["text"], requested)
        if verified:
            result.append({**candidates[idx], "evidence": verified, "evidence_verified": True})
    return result
