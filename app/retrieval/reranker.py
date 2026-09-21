from sentence_transformers import CrossEncoder
from app.core.config import settings

_model = None

def rerank(query: str, candidates: list[dict], top_k: int):
    global _model
    if not candidates:
        return []
    if _model is None:
        _model = CrossEncoder(settings.reranker_model)
    pairs = [(query, c["text"]) for c in candidates]
    scores = _model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
    return [{**c, "rerank_score": float(score)} for c, score in ranked[:top_k]]
