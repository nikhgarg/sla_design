"""Efficiency and equity losses used in the paper."""

import numpy as np
import pandas as pd


def _numeric(frame, column):
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{column} must be numeric and finite")
    return values.astype(float)


def score_all_request(cells, delay_column="delay_median", drop_cost=100):
    """Return cell costs, aggregate efficiency loss, and equity loss."""
    required = {
        "BoroughCode", "SRCategory", "SRCount", "RiskRating",
        "FracInspected", delay_column,
    }
    missing = required.difference(cells.columns)
    if missing:
        raise ValueError(f"Missing objective columns: {sorted(missing)}")
    if not np.isfinite(drop_cost) or drop_cost < 0:
        raise ValueError("drop_cost must be finite and nonnegative")
    if cells.empty:
        raise ValueError("Objective table has no cells")
    identifiers = cells[["BoroughCode", "SRCategory"]]
    if identifiers.isna().any().any() or identifiers.astype(str).apply(
        lambda column: column.str.strip().eq("").any()
    ).any():
        raise ValueError("Objective cells require nonblank Borough and category")
    if cells.duplicated(["BoroughCode", "SRCategory"]).any():
        raise ValueError("Objective cells must be unique by Borough and category")

    result = cells.copy()
    for column in ("SRCount", "RiskRating", "FracInspected", delay_column):
        result[column] = _numeric(result, column)
    if (result["SRCount"] < 0).any() or not np.isclose(
        result["SRCount"], np.round(result["SRCount"])
    ).all():
        raise ValueError("SRCount must contain nonnegative integers")
    result = result.loc[result["SRCount"] > 0].copy()
    if result.empty:
        raise ValueError("Objective table has no active cells")
    if (result["RiskRating"] <= 0).any():
        raise ValueError("RiskRating must be positive")
    if not result["FracInspected"].between(0, 1).all():
        raise ValueError("FracInspected must lie in [0, 1]")
    if (result[delay_column] < 0).any():
        raise ValueError(f"{delay_column} must be nonnegative")

    result["all_request_burden"] = result["RiskRating"] * (
        result["FracInspected"] * result[delay_column]
        + (1 - result["FracInspected"]) * drop_cost
    )
    result["delay_cost"] = (
        result["SRCount"] * result["RiskRating"]
        * result["FracInspected"] * result[delay_column]
    )
    result["drop_cost"] = (
        result["SRCount"] * result["RiskRating"]
        * (1 - result["FracInspected"]) * drop_cost
    )
    result["total_cost"] = result["delay_cost"] + result["drop_cost"]

    equity = result.groupby("SRCategory")["all_request_burden"].agg(
        lambda values: values.max() - values.min()
    ).sum()
    return result, float(result["total_cost"].sum()), float(equity)
