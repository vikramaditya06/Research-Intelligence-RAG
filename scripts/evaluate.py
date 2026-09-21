"""Run retrieval ablations on a manually labelled dataset.

Dataset format:
[
  {
    "question": "...",
    "relevant_chunk_ids": ["uuid", "uuid"],
    "relevant_parent_ids": ["uuid"],
    "document_ids": ["uuid"]
  }
]

Examples without the requested labels are skipped. The script never fabricates
benchmark values. Reranker configurations require the sentence-transformers model.
"""
import json
from pathlib import Path
from statistics import mean
from app.db.session import SessionLocal
from app.services.pipeline import retrieve_candidates
from app.retrieval.reranker import rerank
from app.services.evaluation import precision_at_k, recall_at_k, reciprocal_rank, hit_rate_at_k, ndcg_at_k


def _metrics(retrieved, relevant, k):
    return {
        "precision": precision_at_k(retrieved, relevant, k),
        "recall": recall_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant),
        "hit_rate": hit_rate_at_k(retrieved, relevant, k),
        "ndcg": ndcg_at_k(retrieved, relevant, k),
    }


def evaluate(data, db, k=10, use_hyde=False, use_multi_query=False,
             use_bm25=True, use_vector=True, use_reranker=False,
             label_key="relevant_chunk_ids", retrieval_id_key="id"):
    rows = []
    reranker_error = None
    for item in data:
        relevant = set(item.get(label_key, []))
        if not relevant:
            continue
        candidates, _, _ = retrieve_candidates(
            db, item["question"], item.get("document_ids") or None,
            use_hyde=use_hyde, candidate_k=max(k * 4, 24),
            use_multi_query=use_multi_query, use_bm25=use_bm25,
            use_vector=use_vector,
        )
        if use_reranker:
            try:
                candidates = rerank(item["question"], candidates, min(len(candidates), k * 3))
            except Exception as exc:
                # Never report fallback RRF results as reranker benchmark results.
                # A failed optional benchmark component is explicitly unavailable.
                reranker_error = type(exc).__name__
                continue
        retrieved = [c.get(retrieval_id_key) for c in candidates if c.get(retrieval_id_key)]
        rows.append(_metrics(retrieved, relevant, k))
    if use_reranker and reranker_error is not None:
        return {"status": "unavailable", "metrics": None,
                "reason": f"reranker unavailable: {reranker_error}"}
    if not rows:
        return None
    return {"status": "ok",
            "metrics": {key: round(mean(row[key] for row in rows), 4) for key in rows[0]}}


if __name__ == "__main__":
    data = json.loads(Path("evals/sample_dataset.json").read_text(encoding="utf-8"))
    with SessionLocal() as db:
        configs = {
            "dense": {"use_multi_query": False, "use_bm25": False, "use_hyde": False, "use_vector": True, "use_reranker": False},
            "bm25": {"use_multi_query": False, "use_bm25": True, "use_hyde": False, "use_vector": False, "use_reranker": False},
            "hybrid_rrf": {"use_multi_query": False, "use_bm25": True, "use_hyde": False, "use_vector": True, "use_reranker": False},
            "hybrid_multi_query": {"use_multi_query": True, "use_bm25": True, "use_hyde": False, "use_vector": True, "use_reranker": False},
            "hybrid_reranker": {"use_multi_query": False, "use_bm25": True, "use_hyde": False, "use_vector": True, "use_reranker": True},
            "hybrid_multi_query_reranker": {"use_multi_query": True, "use_bm25": True, "use_hyde": False, "use_vector": True, "use_reranker": True},
            "hybrid_multi_query_hyde_reranker": {"use_multi_query": True, "use_bm25": True, "use_hyde": True, "use_vector": True, "use_reranker": True},
        }
        results = {}
        for name, cfg in configs.items():
            result = evaluate(data, db, k=10, **cfg)
            if result is not None:
                results[name] = result
        if not results:
            raise SystemExit("No real labelled examples found. Add actual relevant_chunk_ids before claiming benchmark results.")
        output = {"child_retrieval": results}
        parent_results = {}
        for name, cfg in configs.items():
            result = evaluate(data, db, k=10, label_key="relevant_parent_ids", retrieval_id_key="parent_id", **cfg)
            if result is not None:
                parent_results[name] = result
        if parent_results:
            output["parent_context_retrieval"] = parent_results
        print(json.dumps(output, indent=2))
