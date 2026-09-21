from app.ingestion.chunker import chunk_pages


def test_child_chunks_keep_page_ranges_and_parents():
    pages = [
        {"page": 1, "text": "INTRODUCTION\nThis is the first sentence. This is the second sentence."},
        {"page": 2, "text": "This is the third sentence. This is the fourth sentence."},
    ]
    parents, children = chunk_pages(pages, parent_size=1000, child_size=70, overlap=10)
    assert parents
    assert children
    assert all(c.parent_id in {p.id for p in parents} for c in children)
    assert all(c.page_start <= c.page_end for c in children)
    assert any(c.page_start == 2 or c.page_end == 2 for c in children)


def test_sentence_chunking_has_real_overlap_when_boundary_allows_it():
    pages = [{"page": 1, "text": "METHODS\n" + "First sentence with enough useful words. " + "Second sentence with enough useful words. " + "Third sentence with enough useful words. " + "Fourth sentence with enough useful words."}]
    _, children = chunk_pages(pages, parent_size=1000, child_size=90, overlap=20)
    assert len(children) >= 2
    texts = [c.text for c in children]
    for a, b in zip(texts, texts[1:]):
        shared = set(a.split()) & set(b.split())
        assert len(shared) >= 4


def test_long_sentences_split_on_word_boundaries():
    pages = [{"page": 1, "text": "METHODS\n" + ("word " * 500) + "."}]
    parents, children = chunk_pages(pages, parent_size=300, child_size=150, overlap=20)
    assert parents and children
    assert all(len(c.text) <= 150 or " " not in c.text for c in children)
    assert all(c.text == c.text.strip() for c in children)
