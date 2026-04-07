from typing import Any, Iterable


def _is_matrix_2d(x: Any) -> bool:
    return (
        isinstance(x, (list, tuple))
        and bool(x)
        and isinstance(x[0], (list, tuple))
    )


def validate_weight_shapes(payload: dict[str, Any], num_features: int, num_rules: int, length_centroids: int) -> None:
    """Optional helper for dict payloads; API uses pydantic WeightPayload instead."""
    for field in ("c", "p", "s"):
        m = payload.get(field)
        if m is None:
            raise ValueError(f"Missing field '{field}' in payload.")
        if len(m) != num_features:
            raise ValueError(f"Field '{field}' must have {num_features} rows.")
        for row in m:
            if len(row) != num_rules:
                raise ValueError(f"Field '{field}' rows must have length {num_rules}.")
    if len(payload.get("q", [])) != num_rules:
        raise ValueError(f"Field 'q' must have length {num_rules}.")
    for field in ("cluster_aggressive", "cluster_normal", "cluster_calm"):
        v = payload.get(field)
        if v is None:
            raise ValueError(f"Missing field '{field}' in payload.")
        if len(v) != length_centroids:
            raise ValueError(f"Field '{field}' must have size {length_centroids}.")


def average_vectors(vectors: Iterable[Any]) -> list[float] | list[list[float]]:
    """Element-wise mean of 1D vectors or 2D matrices (list-of-row-lists)."""
    vectors = list(vectors)
    if not vectors:
        return []

    first = vectors[0]
    if _is_matrix_2d(first):
        rows = len(first)
        cols = len(first[0])
        for matrix in vectors:
            if not _is_matrix_2d(matrix) or len(matrix) != rows:
                raise ValueError("2D weight matrices must share the same shape.")
            for r in matrix:
                if len(r) != cols:
                    raise ValueError("2D weight matrices must share the same shape.")
        sums = [[0.0] * cols for _ in range(rows)]
        for matrix in vectors:
            for i in range(rows):
                for j in range(cols):
                    sums[i][j] += float(matrix[i][j])
        count = float(len(vectors))
        return [[v / count for v in row] for row in sums]

    width = len(first)
    sums = [0.0] * width
    for vector in vectors:
        if _is_matrix_2d(vector):
            raise ValueError("Cannot mix 1D and 2D vectors in the same average.")
        if len(vector) != width:
            raise ValueError("Vectors must have the same length to be averaged.")
        for idx, value in enumerate(vector):
            sums[idx] += float(value)

    count = float(len(vectors))
    return [value / count for value in sums]