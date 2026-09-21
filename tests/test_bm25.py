from app.retrieval.bm25 import BM25Index


def test_bm25_filters_documents():
    index = BM25Index()
    index.build([
        ("a", "transformer attention model", "doc-a", "pa", 1, "Methods", {}),
        ("b", "transformer attention model", "doc-b", "pb", 2, "Results", {}),
    ])
    rows = index.search("transformer", k=10, document_ids=["doc-b"])
    assert [r[0] for r in rows] == ["b"]
    assert rows[0][2] == "doc-b"
