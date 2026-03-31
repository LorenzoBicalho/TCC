from typing import Iterable


EXPECTED_VECTOR_SIZES = {
    "c": 10,
    "p": 50,
    "s": 50,
    "q": 50,
    "cluster_aggressive": 15,
    "cluster_normal": 15,
    "cluster_calm": 15,
}


def validate_weight_shapes(payload: dict[str, list[float]]) -> None:
    for field, expected_size in EXPECTED_VECTOR_SIZES.items():
        values = payload.get(field)
        if values is None:
            raise ValueError(f"Missing field '{field}' in payload.")
        if len(values) != expected_size:
            raise ValueError(f"Field '{field}' must have size {expected_size}.")


def average_vectors(vectors: Iterable[list[float]]) -> list[float]:
    vectors = list(vectors)
    if not vectors:
        return []

    width = len(vectors[0])
    sums = [0.0] * width
    for vector in vectors:
        if len(vector) != width:
            raise ValueError("Vectors must have the same length to be averaged.")
        for idx, value in enumerate(vector):
            sums[idx] += float(value)

    count = float(len(vectors))
    return [value / count for value in sums]