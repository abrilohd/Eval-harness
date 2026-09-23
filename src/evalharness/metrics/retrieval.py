def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> bool:
    return bool(set(retrieved[:k]) & set(expected))


def reciprocal_rank(retrieved: list[str], expected: list[str], k: int) -> float:
    wanted = set(expected)
    for rank, source_id in enumerate(retrieved[:k], 1):
        if source_id in wanted:
            return 1.0 / rank
    return 0.0
