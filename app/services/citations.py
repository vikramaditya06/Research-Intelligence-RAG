import re


def citation_ids(answer: str) -> set[str]:
    return set(re.findall(r"\[(S\d+)\]", answer or ""))


def verify_evidence_spans(sources: list[dict]):
    failures = []
    for i, source in enumerate(sources, 1):
        evidence = str(source.get("evidence", "")).strip()
        text = str(source.get("text", ""))
        if not evidence:
            failures.append(f"S{i}")
            continue
        if evidence in text or " ".join(evidence.split()) in " ".join(text.split()):
            continue
        failures.append(f"S{i}")
    return failures


def _audit_result_is_valid(data, valid_ids):
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list) or not data["claims"]:
        return False, []
    claims = []
    for claim in data["claims"]:
        if not isinstance(claim, dict) or not str(claim.get("claim", "")).strip():
            return False, []
        ids = claim.get("citations")
        if not isinstance(ids, list):
            return False, []
        normalized = [str(x).strip() for x in ids if str(x).strip()]
        if any(x not in valid_ids for x in normalized):
            return False, []
        if not isinstance(claim.get("supported"), bool):
            return False, []
        claim["citations"] = normalized
        claims.append(claim)
    return True, claims


def verify_citations(answer: str, sources: list[dict]):
    valid_ids = {f"S{i+1}" for i in range(len(sources))}
    cited = citation_ids(answer)
    invalid = sorted(cited - valid_ids)
    span_failures = verify_evidence_spans(sources)
    if invalid or span_failures or not answer.strip() or not sources:
        return {"valid": False, "cited": sorted(cited), "invalid": invalid,
                "evidence_span_failures": span_failures, "claims": [],
                "citation_completeness": 0.0, "support_rate": 0.0}

    source_text = "\n\n".join(f"[S{i+1}] {s.get('evidence', s.get('text', ''))}" for i, s in enumerate(sources))
    prompt = f"""Audit citation support. Break the answer into atomic factual claims. For each claim, list citation IDs that support it and whether the supplied evidence explicitly supports it. Identify factual claims with no citation. Return JSON: {{\"claims\":[{{\"claim\":string,\"citations\":[string],\"supported\":boolean,\"reason\":string}}]}}. Citation IDs must be only from {sorted(valid_ids)}. Do not infer facts beyond the supplied evidence. Do not return an empty claims array for a substantive answer.\n\nANSWER:\n{answer}\n\nEVIDENCE:\n{source_text}"""
    try:
        from app.services.llm import generate_json
        data = generate_json(prompt, system="You are a strict citation auditor. Evidence support must be explicit. Return JSON only.")
        audit_ok, factual = _audit_result_is_valid(data, valid_ids)
        if not audit_ok:
            raise ValueError("Malformed or unsafe citation audit")
        supported = sum(bool(c.get("supported")) for c in factual)
        cited_claims = sum(bool(c.get("citations")) for c in factual)
        valid = all(bool(c.get("supported")) and bool(c.get("citations")) for c in factual)
        return {"valid": valid, "cited": sorted(cited), "invalid": invalid,
                "evidence_span_failures": span_failures, "claims": factual,
                "citation_completeness": cited_claims / len(factual),
                "support_rate": supported / len(factual)}
    except Exception as exc:
        return {"valid": False, "cited": sorted(cited), "invalid": invalid,
                "evidence_span_failures": span_failures, "claims": [],
                "citation_completeness": 0.0, "support_rate": 0.0,
                "audit_unavailable": True,
                "error": f"Citation audit unavailable: {type(exc).__name__}"}
