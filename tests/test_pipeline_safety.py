import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_citation_repair_only_accepts_verified_repair():
    source = (PROJECT_ROOT / "app/services/pipeline.py").read_text(encoding="utf-8")
    assert 'if repaired_verification.get("valid"):' in source
    assert 'answer, verification = original_answer, original_verification' in source


def test_pipeline_keeps_retrieval_and_context_page_ranges():
    source = (PROJECT_ROOT / "app/services/pipeline.py").read_text(encoding="utf-8")
    assert 'item["retrieval_page_start"]' in source
    assert 'item["context_page_start"]' in source
    assert 'evidence_page_start' in source


def test_compression_disabled_has_verified_evidence_fallback():
    source = (PROJECT_ROOT / "app/services/pipeline.py").read_text(encoding="utf-8")
    assert 'if not str(source.get("evidence", "")).strip():' in source
    assert 'source["evidence"] = str(source.get("text", ""))' in source
    assert 'source["evidence_verified"] = bool(source["evidence"].strip())' in source


def test_json_generation_only_falls_back_for_response_format_compatibility():
    source = (PROJECT_ROOT / "app/services/llm.py").read_text(encoding="utf-8")
    assert 'except BadRequestError as exc:' in source
    assert 'if not _response_format_unsupported(exc):' in source


def test_embedding_client_is_reused_across_batches():
    source = (PROJECT_ROOT / "app/services/llm.py").read_text(encoding="utf-8")
    assert 'api_client = client()' in source
    assert 'api_client.embeddings.create' in source
    assert 'client().embeddings.create' not in source


def test_embedding_batches_are_streaming():
    source = (PROJECT_ROOT / "app/services/llm.py").read_text(encoding="utf-8")
    assert 'def embed_batches(' in source
    assert 'yield _embed_batch(api_client, batch)' in source


def test_upload_uses_streaming_embedding_batches():
    source = (PROJECT_ROOT / "app/api/routes.py").read_text(encoding="utf-8")
    assert 'from app.services.llm import embed_batches' in source
    assert 'embed_batches((c.text for c in chunks))' in source
    assert 'vectors = embed([c.text for c in chunks])' not in source


def test_upload_handles_concurrent_duplicate_without_deleting_winner():
    source = (PROJECT_ROOT / "app/api/routes.py").read_text(encoding="utf-8")
    assert 'except IntegrityError:' in source
    assert 'existing = db.scalar(select(Document).where(Document.file_hash == file_hash))' in source
    assert 'if path != Path(existing.path):' in source


def test_pdf_validation_uses_canonical_open_document_validation():
    source = (PROJECT_ROOT / "app/ingestion/pdf.py").read_text(encoding="utf-8")
    assert 'def _validate_open_document(' in source
    assert '_validate_open_document(doc, max_pages)' in source
    assert 'validate_pdf_bytes(content)' not in (PROJECT_ROOT / "app/api/routes.py").read_text(encoding="utf-8")


def test_llm_retry_does_not_retry_bad_request():
    source = (PROJECT_ROOT / "app/services/llm.py").read_text(encoding="utf-8")
    retry_source = source[source.index("def _with_retry"):source.index("def embed_batches")]
    assert "except BadRequestError:" in retry_source


def test_citation_audit_failure_does_not_trigger_repair():
    source = (PROJECT_ROOT / "app/services/pipeline.py").read_text(encoding="utf-8")
    assert 'not verification.get("audit_unavailable")' in source


def test_pdf_validation_is_page_limit_aware():
    source = (PROJECT_ROOT / "app/ingestion/pdf.py").read_text(encoding="utf-8")
    assert 'b"%PDF-" not in content[:1024]' in source
    assert 'page_count > max_pages' in source


def test_ready_uses_dynamic_alembic_heads():
    source = (PROJECT_ROOT / "app/api/routes.py").read_text(encoding="utf-8")
    assert 'ScriptDirectory.from_config' in source
    assert '.get_heads()' in source
    assert 'db_versions != heads' in source
    assert 'EXPECTED_SCHEMA_REVISION' not in source


def test_query_has_explicit_no_vector_stage_statuses():
    source = (PROJECT_ROOT / "app/services/pipeline.py").read_text(encoding="utf-8")
    assert 'stage_status["multi_query"] = "skipped_no_vector"' in source
    assert 'stage_status["hyde"] = "skipped_no_vector"' in source


def test_ranking_metadata_preserves_rrf_fallback_semantics():
    source = (PROJECT_ROOT / "app/services/pipeline.py").read_text(encoding="utf-8")
    assert '"rrf_rank": rank' in source
    assert '"rrf_score": float(rrf_score)' in source
    assert '"ranking_source": "rrf"' in source
    assert '"ranking_source": "reranker"' in source


def test_compression_status_distinguishes_configured_used_and_fallback():
    source = (PROJECT_ROOT / "app/services/pipeline.py").read_text(encoding="utf-8")
    assert '"configured": True, "used": True, "status": "used"' in source
    assert '"configured": True, "used": False, "status": "fallback"' in source
    assert 'source["source_representation"] = source_representation' in source


def test_storage_reconciliation_does_not_delete_arbitrary_user_pdfs():
    source = (PROJECT_ROOT / "app/services/storage.py").read_text(encoding="utf-8")
    assert 'len(prefix) != 16' in source
    assert 'any(ch not in "0123456789abcdef"' in source


def test_startup_storage_reconciliation_exists():
    source = (PROJECT_ROOT / "app/services/storage.py").read_text(encoding="utf-8")
    assert 'def reconcile_document_storage' in source
    assert '.uploading' in source
    main = (PROJECT_ROOT / "app/main.py").read_text(encoding="utf-8")
    assert 'reconcile_storage_on_startup' in main


def test_invalid_pdf_errors_are_returned_as_422():
    source = (PROJECT_ROOT / "app/api/routes.py").read_text(encoding="utf-8")
    assert 'except ValueError as exc:' in source
    assert 'raise HTTPException(422, str(exc)) from exc' in source


def test_embedding_dimension_cannot_diverge_from_migration_schema():
    source = (PROJECT_ROOT / "app/core/config.py").read_text(encoding="utf-8")
    assert 'EMBEDDING_DIMENSION must be 1536' in source
    assert 'if value != 1536:' in source


def test_health_and_ready_have_distinct_roles():
    source = (PROJECT_ROOT / "app/api/routes.py").read_text(encoding="utf-8")
    assert 'def health()' in source and 'return {"status": "ok"}' in source
    assert 'def ready(' in source and 'HTTPException(503' in source


def test_vector_search_uses_database_distance_value():
    source = (PROJECT_ROOT / "app/retrieval/vector.py").read_text(encoding="utf-8")
    assert 'distance = Chunk.embedding.cosine_distance(qvec).label("distance")' in source
    assert 'select(Chunk, distance)' in source
    assert '1.0 - float(distance_value)' in source
    assert 'r.embedding.cosine_distance' not in source


def test_ready_checks_critical_schema_and_vector_dimension():
    source = (PROJECT_ROOT / "app/api/routes.py").read_text(encoding="utf-8")
    assert "information_schema.columns" in source
    assert "documents" in source and "parent_chunks" in source and "chunks" in source
    assert "chunks.embedding" in source
    assert "format_type(a.atttypid, a.atttypmod)" in source
    assert "settings.embedding_dimension" in source


def test_pdf_has_single_public_extraction_entrypoint():
    source = (PROJECT_ROOT / "app/ingestion/pdf.py").read_text(encoding="utf-8")
    assert "def validate_pdf_bytes" not in source
    assert "def extract_pdf" in source


def test_chunk_page_ranges_have_explicit_coordinate_systems():
    source = (PROJECT_ROOT / "app/ingestion/chunker.py").read_text(encoding="utf-8")
    assert '"parent_page_ranges"' in source
    assert '"child_page_ranges"' in source
    assert '"page_ranges"' not in source


def test_evidence_locator_checks_all_occurrences_and_prefers_retrieved_child():
    source = (PROJECT_ROOT / "app/services/pipeline.py").read_text(encoding="utf-8")
    assert "def _matching_spans" in source
    assert "for span in matches" in source
    assert '"retrieved_child"' in source
    assert '"context_preferred"' in source


def test_evaluator_marks_reranker_failure_unavailable():
    source = (PROJECT_ROOT / "scripts/evaluate.py").read_text(encoding="utf-8")
    assert '"status": "unavailable"' in source
    assert '"metrics": None' in source
    assert "Never report fallback RRF results as reranker benchmark results" in source


def test_pipeline_uses_clear_default_top_k_and_rerank_pool_multiplier():
    config = (PROJECT_ROOT / "app/core/config.py").read_text(encoding="utf-8")
    pipeline = (PROJECT_ROOT / "app/services/pipeline.py").read_text(encoding="utf-8")
    assert "default_top_k: int = Field(default=8" in config
    assert "settings.default_top_k" in pipeline
    assert "rerank_pool_multiplier = 3" in pipeline
    assert "settings.top_k_rerank" not in pipeline


def test_mocked_end_to_end_pipeline_without_external_services(monkeypatch):
    pytest = __import__("pytest")
    pytest.importorskip("pgvector")
    import app.services.pipeline as pipeline

    class DummyDB:
        pass

    candidates = [{
        "id": "c1", "text": "The model achieves 92% accuracy.",
        "document_id": "d1", "parent_id": "p1", "page": 5,
        "page_start": 5, "page_end": 5, "section": "Results",
        "metadata": {"child_page_ranges": [[5, 0, 33]]},
        "rrf_rank": 1, "rrf_score": 1.0, "ranking_source": "rrf",
    }]

    monkeypatch.setattr(pipeline, "retrieve_candidates", lambda *args, **kwargs: (candidates, ["question"], {"multi_query": "disabled", "hyde": "disabled"}))
    monkeypatch.setattr(pipeline, "rerank", lambda q, cs, k: cs[:k])
    monkeypatch.setattr(pipeline, "_parent_expand", lambda db, cs, document_ids=None: [{
        **cs[0], "parent_text": cs[0]["text"], "parent_page_start": 5, "parent_page_end": 5,
        "parent_section": "Results", "parent_metadata": {"parent_page_ranges": [[5, 0, 33]]},
    }])
    monkeypatch.setattr(pipeline, "generate", lambda *args, **kwargs: "The model achieves 92% accuracy [S1].")

    monkeypatch.setattr(pipeline.settings, "enable_compression", False)
    monkeypatch.setattr(pipeline.settings, "enable_citation_audit", False)

    result = pipeline.answer_question(DummyDB(), "question", top_k=1)
    assert result["answer"].endswith("[S1].")
    assert result["sources"][0]["source_representation"] == "retrieved"
    assert result["sources"][0]["evidence_verified"] is True
    assert result["sources"][0]["evidence_page_start"] == 5
    assert result["pipeline_status"]["compression"]["status"] == "disabled"
