from sqlalchemy import select
from app.db.models import Chunk
from app.services.llm import embed


def vector_search(db, query: str, document_ids: list[str] | None, k: int, query_vector=None):
    if k <= 0:
        return []
    qvec = query_vector if query_vector is not None else embed([query])[0]
    distance = Chunk.embedding.cosine_distance(qvec).label("distance")
    stmt = select(Chunk, distance).where(Chunk.embedding.is_not(None))
    if document_ids is not None:
        if not document_ids:
            return []
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    stmt = stmt.order_by(distance).limit(k)
    rows = db.execute(stmt).all()
    return [{
        "id": r.id, "document_id": r.document_id, "parent_id": r.parent_id,
        "page": r.page, "page_start": r.page_start, "page_end": r.page_end,
        "section": r.section, "text": r.text, "metadata": r.metadata_json,
        "score": 1.0 - float(distance_value),
    } for r, distance_value in rows]
