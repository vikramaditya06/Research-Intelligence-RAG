def reciprocal_rank_fusion(*ranked_lists, k=60):
    scores = {}
    payload = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            item_id = item[0]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in payload:
                payload[item_id] = item[2]
    ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
    return [(item_id, scores[item_id], payload[item_id]) for item_id in ordered]
