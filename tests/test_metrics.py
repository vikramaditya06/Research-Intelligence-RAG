from app.services.evaluation import precision_at_k, recall_at_k, reciprocal_rank, hit_rate_at_k, ndcg_at_k


def test_metrics():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert hit_rate_at_k(retrieved, relevant, 2) == 1.0
    assert 0 < ndcg_at_k(retrieved, relevant, 4) <= 1


def test_precision_at_k_uses_k_as_denominator_when_fewer_results_are_returned():
    retrieved = ["a", "b"]
    relevant = {"a", "b"}
    assert precision_at_k(retrieved, relevant, 10) == 0.2


def test_precision_at_k_empty_retrieval_is_zero():
    assert precision_at_k([], {"a"}, 10) == 0.0


def test_precision_at_k_non_positive_k_is_zero():
    assert precision_at_k(["a"], {"a"}, 0) == 0.0


def test_evaluation_metrics_do_not_fabricate_unlabelled_examples():
    """An empty benchmark is explicit and cannot produce fabricated metrics."""
    from pathlib import Path
    import json
    dataset = json.loads((Path(__file__).resolve().parents[1] / "evals" / "sample_dataset.json").read_text(encoding="utf-8"))
    assert dataset == []
    evaluator = (Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    assert "No real labelled examples found" in evaluator
