from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from app.services.citations import citation_ids, verify_evidence_spans, _audit_result_is_valid


def test_citation_ids():
    assert citation_ids("Claim [S1]. Another [S2][S1].") == {"S1", "S2"}


def test_evidence_span_verification():
    assert verify_evidence_spans([{"text": "Original evidence here.", "evidence": "evidence here."}]) == []
    assert verify_evidence_spans([{"text": "Original evidence here.", "evidence": "invented evidence."}]) == ["S1"]


def test_auditor_rejects_unknown_citation_ids_and_empty_claims():
    assert _audit_result_is_valid({"claims": []}, {"S1"})[0] is False
    assert _audit_result_is_valid({"claims": [{"claim": "x", "citations": ["S99"], "supported": True}]}, {"S1"})[0] is False


def test_auditor_rejects_non_boolean_supported_field():
    ok, claims = _audit_result_is_valid(
        {"claims": [{"claim": "x", "citations": ["S1"], "supported": "true"}]},
        {"S1"},
    )
    assert ok is False
    assert claims == []


def test_evidence_locator_is_import_independent_in_unit_suite():
    source = (PROJECT_ROOT / "app/services/pipeline.py").read_text(encoding="utf-8")
    assert "def _matching_spans" in source
    assert "child_page_ranges" in source
    assert "evidence_page_source" in source
