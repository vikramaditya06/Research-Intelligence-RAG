from app.api.schemas import QueryRequest
import pytest


def test_query_rejects_too_short_question():
    with pytest.raises(Exception):
        QueryRequest(question="x")


def test_query_accepts_metadata_filters():
    req = QueryRequest(question="What is the contribution?", year_from=2020, year_to=2026)
    assert req.year_from == 2020
    assert req.year_to == 2026


def test_query_rejects_blank_question_at_route_boundary():
    from app.api.routes import query
    from unittest.mock import Mock
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        query(QueryRequest(question="   "), Mock())
    assert exc.value.status_code == 400



def test_query_rejects_inverted_year_range():
    with pytest.raises(Exception, match="year_from must be less than or equal to year_to"):
        QueryRequest(question="What happened?", year_from=2025, year_to=2020)
