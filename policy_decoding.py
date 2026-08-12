"""Canonical decoding of BO vectors into simulator policy parameters."""
from __future__ import annotations

import numpy as np


BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
CATEGORIES = [
    "Hazard",
    "Illegal Tree Damage",
    "Other",
    "Prune",
    "Remove Tree",
    "Root/Sewer/Sidewalk",
]


def normalize_simplex(values, minimum: float, axis: int | None = None) -> np.ndarray:
    """Floor and normalize finite nonnegative weights without rounding drift."""
    array = np.asarray(values, dtype=float).copy()
    if not np.isfinite(array).all():
        raise ValueError("Policy weights must be finite")
    array = np.maximum(array, float(minimum))
    totals = array.sum(axis=axis, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Policy weights must have positive total mass")
    return array / totals


def infer_policy_class(dim: int) -> dict:
    if dim == 35:
        return {"city_budget": False, "byborough_policy": True, "drop_by_age": True}
    if dim == 65:
        return {"city_budget": False, "byborough_policy": True, "drop_by_age": False}
    if dim == 60:
        return {"city_budget": True, "byborough_policy": True, "drop_by_age": False}
    if dim == 12:
        return {"city_budget": True, "byborough_policy": False, "drop_by_age": False}
    raise ValueError(f"Unsupported policy dimension {dim}")


def decode_policy_vector(values) -> dict:
    """Decode every supported BO vector using the same execution semantics."""
    vector = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(vector).all():
        raise ValueError("Policy vector must contain only finite values")
    if np.any(vector < 0) or np.any(vector > 1):
        raise ValueError("Policy vector entries must lie in the BO domain [0, 1]")
    policy_class = infer_policy_class(len(vector))
    city_budget = policy_class["city_budget"]
    byborough_policy = policy_class["byborough_policy"]

    service_fractions = np.full((len(BOROUGHS), len(CATEGORIES)), 0.9, dtype=float)
    borough_budgets = None
    if len(vector) == 35:
        borough_budgets = normalize_simplex(vector[:5], minimum=0.01)
        inspection_policies = normalize_simplex(
            vector[5:35].reshape(len(BOROUGHS), len(CATEGORIES)),
            minimum=0.0005,
            axis=1,
        )
    elif len(vector) == 65:
        borough_budgets = normalize_simplex(vector[:5], minimum=0.01)
        inspection_policies = normalize_simplex(
            vector[5:35].reshape(len(BOROUGHS), len(CATEGORIES)),
            minimum=0.0005,
            axis=1,
        )
        service_fractions = np.maximum(
            vector[35:65].reshape(len(BOROUGHS), len(CATEGORIES)), 0.10
        )
    elif len(vector) == 60:
        inspection_policies = normalize_simplex(
            vector[:30].reshape(len(BOROUGHS), len(CATEGORIES)),
            minimum=0.0001,
        )
        service_fractions = np.maximum(
            vector[30:60].reshape(len(BOROUGHS), len(CATEGORIES)), 0.10
        )
    elif len(vector) == 12:
        inspection_policies = normalize_simplex(vector[:6], minimum=0.01)
        service_fractions = np.maximum(vector[6:12], 0.10)
    else:  # pragma: no cover - infer_policy_class rejects unsupported dimensions.
        raise ValueError(f"Unsupported policy dimension {len(vector)}")

    return {
        **policy_class,
        "borough_budgets": borough_budgets,
        "inspection_policies": inspection_policies,
        "service_fractions": service_fractions,
        "city_budget": city_budget,
        "byborough_policy": byborough_policy,
    }
