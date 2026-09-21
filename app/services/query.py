from app.services.llm import generate_json, generate


def rewrite_queries(question: str, multi_query: bool = True):
    """
    Generate alternative retrieval queries.

    Multi-query is an enhancement, not a dependency. If the LLM JSON
    transformation fails, fall back to the original question so retrieval
    can still continue.
    """
    if not multi_query:
        return [question]

    try:
        data = generate_json(
            (
                "Generate exactly 3 alternative search queries for this "
                "research question. Preserve technical terms. "
                "Return JSON with key 'queries' containing an array of strings.\n"
                f"Question: {question}"
            ),
            system="You generate concise retrieval queries. Return JSON only.",
        )

        queries = data.get("queries", [])

        if not isinstance(queries, list):
            return [question]

        clean = [
            str(q).strip()
            for q in queries
            if str(q).strip()
        ]

        return [question] + clean[:3]

    except Exception:
        # Multi-query rewriting must never break the main RAG pipeline.
        return [question]


def hyde_query(question: str):
    """
    HyDE is also optional. The caller already handles failures and can
    continue without the hypothetical query.
    """
    return generate(
        "Write a short hypothetical technical passage that could answer "
        "this research question. It will be embedded for retrieval only, "
        "not used as evidence.\n"
        f"Question: {question}"
    )