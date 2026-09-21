import math


def _unique(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def precision_at_k(retrieved, relevant, k):
    """Standard Precision@K: relevant items in the top K divided by K.

    If there are no retrieved items, or K is non-positive, precision is 0.0.
    Importantly, fewer than K retrieved results do not shrink the denominator.
    """
    if k <= 0:
        return 0.0
    top = _unique(retrieved)[:k]
    if not top:
        return 0.0
    return sum(x in relevant for x in top) / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    top = _unique(retrieved)[:k]
    return sum(x in relevant for x in top) / len(relevant)


def reciprocal_rank(retrieved, relevant):
    for i, x in enumerate(_unique(retrieved), 1):
        if x in relevant:
            return 1 / i
    return 0.0


def hit_rate_at_k(retrieved, relevant, k):
    return float(any(x in relevant for x in _unique(retrieved)[:k]))


def ndcg_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    top = _unique(retrieved)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(top) if x in relevant)
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg else 0.0
