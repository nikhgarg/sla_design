#!/usr/bin/env python3
"""Evaluate one selected SLA policy using the compact public release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from objective_scoring import score_all_request
from simulator import simulator


BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
CATEGORIES = [
    "Hazard",
    "Illegal Tree Damage",
    "Other",
    "Prune",
    "Remove Tree",
    "Root/Sewer/Sidewalk",
]
DROP_FREQUENCIES = [28, 80, 26, 27, 36]
INPUT_SHA256 = {
    "paper_data/primary/selected_policy_parameters.csv": "07e6daab5c799383f1b51693b3f83b803ac4a76cc0213217b7396e245aad5e2d",
    "paper_data/primary/selected_policy_scores_by_seed.csv": "1e82ea8ffab6fab0913526a77f289d7a652cfa277d6494ceed76c33dd278c123",
    "data_clean/mean_risk_rating.csv": "90c22c73613ab73983826f0f6a3144863b6e72250836550c861a270be51e96c3",
    "data_clean/merged_data_mergeTrue.csv": "b0bc1fa2f8079de2112d34b81c5d8869fc662f0dae8cb4f8b831745205b55dbb",
    "data_raw/Forestry_Service_Requests_20231201.csv": "4ab140497fee69582d0c7790c53a55d6040072de3a3d580b740a4d818e9f893d",
    "data_raw/Forestry_Inspections_20231201.csv": "14cc86eabe206519208bfdc2d3d61d7ade236717f6ced353119b0c1ff73931de",
}


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.readline().startswith(b"version https://git-lfs.github.com/spec/v1")
    except FileNotFoundError:
        return False


def _require_hash(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    observed = digest.hexdigest()
    if observed != expected:
        raise ValueError(f"Input hash mismatch: {path}")


def preflight_data(repo_root: Path) -> str:
    for relative_path in (
        "paper_data/primary/selected_policy_parameters.csv",
        "paper_data/primary/selected_policy_scores_by_seed.csv",
    ):
        path = repo_root / relative_path
        if not path.is_file() or _is_lfs_pointer(path):
            raise FileNotFoundError(f"Required compact input is missing: {path}.")
        _require_hash(path, INPUT_SHA256[relative_path])

    data_clean = repo_root / "data_clean"
    risk_path = data_clean / "mean_risk_rating.csv"
    if not risk_path.is_file() or _is_lfs_pointer(risk_path):
        raise FileNotFoundError(
            f"Required public input is missing: {risk_path}."
        )
    _require_hash(risk_path, INPUT_SHA256["data_clean/mean_risk_rating.csv"])
    merged = data_clean / "merged_data_mergeTrue.csv"
    if merged.is_file() and not _is_lfs_pointer(merged):
        _require_hash(merged, INPUT_SHA256["data_clean/merged_data_mergeTrue.csv"])
        return "authenticated_merged_cache"
    for name in (
        "Forestry_Service_Requests_20231201.csv",
        "Forestry_Inspections_20231201.csv",
    ):
        path = repo_root / "data_raw" / name
        if not path.is_file() or _is_lfs_pointer(path):
            raise FileNotFoundError(
                f"Required raw snapshot is missing: {path}. "
                "Follow data_raw/README.md and verify data_raw/SHA256SUMS."
            )
        _require_hash(path, INPUT_SHA256[f"data_raw/{name}"])
    return "archived_raw_snapshots"


def load_policy(path: Path, role: str) -> dict:
    # Exact floating-point recovery matters because these values parameterize
    # seeded multinomial draws; sub-ulp parsing changes can alter a trajectory.
    frame = pd.read_csv(path, float_precision="round_trip")
    required = {
        "role",
        "policy_id",
        "vector_hash",
        "budget_group",
        "borough",
        "category",
        "borough_budget_share",
        "inspection_priority_weight",
        "per_review_retention_probability",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Policy table is missing columns: {sorted(missing)}")
    selected = frame.loc[frame["role"].astype(str).eq(role)].copy()
    if selected.empty:
        choices = sorted(frame["role"].astype(str).unique())
        raise ValueError(f"Unknown role {role!r}; choose from {choices}")
    for column in ("policy_id", "vector_hash", "budget_group"):
        if selected[column].nunique(dropna=False) != 1:
            raise ValueError(f"Role {role!r} does not have one {column}")
    if len(selected) != len(BOROUGHS) * len(CATEGORIES):
        raise ValueError(f"Role {role!r} must have exactly 30 policy cells")
    if selected.duplicated(["borough", "category"]).any():
        raise ValueError(f"Role {role!r} has duplicate Borough-category cells")
    support = set(map(tuple, selected[["borough", "category"]].to_numpy()))
    expected = {(borough, category) for borough in BOROUGHS for category in CATEGORIES}
    if support != expected:
        raise ValueError(f"Role {role!r} does not have exact 5 x 6 support")

    selected = selected.set_index(["borough", "category"]).loc[
        [(borough, category) for borough in BOROUGHS for category in CATEGORIES]
    ]
    priorities = pd.to_numeric(
        selected["inspection_priority_weight"], errors="raise"
    ).to_numpy(dtype=float).reshape(len(BOROUGHS), len(CATEGORIES))
    retentions = pd.to_numeric(
        selected["per_review_retention_probability"], errors="raise"
    ).to_numpy(dtype=float).reshape(len(BOROUGHS), len(CATEGORIES))
    if not np.isfinite(priorities).all() or (priorities < 0).any():
        raise ValueError("Inspection priority weights must be finite and nonnegative")
    if not np.isfinite(retentions).all() or not np.logical_and(
        retentions >= 0, retentions <= 1
    ).all():
        raise ValueError("Retention probabilities must be finite and in [0, 1]")

    budget_group = str(selected["budget_group"].iloc[0])
    if budget_group == "borough":
        budget_by_borough = selected.reset_index().groupby("borough", sort=False)[
            "borough_budget_share"
        ]
        if not (budget_by_borough.nunique(dropna=False) == 1).all():
            raise ValueError("Borough budget share must be constant within each Borough")
        budgets = np.array(
            [float(budget_by_borough.get_group(b).iloc[0]) for b in BOROUGHS]
        )
        if not np.isfinite(budgets).all() or (budgets < 0).any():
            raise ValueError("Borough budget shares must be finite and nonnegative")
        if not np.isclose(budgets.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError("Borough budget shares must sum to one")
        if not np.allclose(priorities.sum(axis=1), 1.0, rtol=0, atol=1e-12):
            raise ValueError("Borough priority weights must sum to one within Borough")
        city_budget = False
    elif budget_group == "city":
        if selected["borough_budget_share"].notna().any():
            raise ValueError("City policies must not contain Borough budget shares")
        budgets = None
        if not np.isclose(priorities.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError("City priority weights must sum to one globally")
        city_budget = True
    else:
        raise ValueError(f"Unknown budget_group {budget_group!r}")

    return {
        "role": role,
        "policy_id": str(selected["policy_id"].iloc[0]),
        "vector_hash": str(selected["vector_hash"].iloc[0]),
        "budget_group": budget_group,
        "city_budget": city_budget,
        "borough_budgets": budgets,
        "inspection_policies": priorities,
        "service_fractions": retentions,
    }


def evaluate(repo_root: Path, policy: dict, seed: int, year: int) -> dict:
    previous_directory = Path.cwd()
    try:
        os.chdir(repo_root)
        sim = simulator(
            boroughs=BOROUGHS,
            categories=CATEGORIES,
            borough_budgets=policy["borough_budgets"],
            inspection_policies=policy["inspection_policies"],
            service_fractions=policy["service_fractions"],
            city_budget=policy["city_budget"],
            byborough_policy=True,
            recursive=3,
            date_start=f"{year}-01-01",
            date_end=f"{year}-12-31",
            fcfs_violation=0.15,
            only_consider_inspected=False,
            dropping_frequency=DROP_FREQUENCIES,
            drop_age=100,
            drop_using_age=False,
            save_logs=False,
            sla_objective="delay_median",
            drop_cost=100,
            equity="allrequest",
            random_seed=seed,
        )
        sim.simulate()
    finally:
        os.chdir(previous_directory)
    _, efficiency, equity = score_all_request(
        sim.df_objectives,
        delay_column="delay_median",
        drop_cost=100,
    )
    return {
        "schema": "sla-fixed-policy-evaluation-v1",
        "calendar_year": year,
        "random_seed": seed,
        "role": policy["role"],
        "policy_id": policy["policy_id"],
        "vector_hash": policy["vector_hash"],
        "efficiency_loss": efficiency,
        "equity_loss": equity,
        "settings": {
            "recursive": 3,
            "rho": 0.15,
            "dropping_frequency_days": DROP_FREQUENCIES,
            "drop_cost": 100,
            "objective": "delay_median",
            "equity": "allrequest",
            "capacity_multiplier": 1.0,
        },
    }


def compare(result: dict, scores_path: Path) -> dict:
    scores = pd.read_csv(scores_path, float_precision="round_trip")
    target = scores.loc[
        scores["policy_id"].astype(str).eq(result["policy_id"])
        & pd.to_numeric(scores["random_seed"], errors="coerce").eq(result["random_seed"])
        & pd.to_numeric(scores["calendar_year"], errors="coerce").eq(result["calendar_year"])
    ]
    if target.empty:
        return {
            "comparison_status": "no_checked_seed_score_for_policy_and_year",
        }
    if len(target) != 1:
        raise ValueError("Expected exactly one checked-in score for policy and seed")
    row = target.iloc[0]
    if str(row["vector_hash"]) != str(result["vector_hash"]):
        raise ValueError("Checked-in score has a different policy vector hash")
    expected_efficiency = float(row["efficiency_loss"])
    expected_equity = float(row["equity_loss"])
    return {
        "comparison_status": "compared",
        "expected_efficiency_loss": expected_efficiency,
        "expected_equity_loss": expected_equity,
        "efficiency_difference": result["efficiency_loss"] - expected_efficiency,
        "equity_difference": result["equity_loss"] - expected_equity,
        "exact_within_float_tolerance": bool(
            np.isclose(result["efficiency_loss"], expected_efficiency, rtol=0, atol=1e-8)
            and np.isclose(result["equity_loss"], expected_equity, rtol=0, atol=1e-10)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--year", type=int, choices=(2019, 2021, 2022), default=2019)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("--seed must lie in [0, 2**32)")
    repo_root = args.repo_root.resolve()
    code_root = Path(__file__).resolve().parent
    if repo_root != code_root:
        parser.error("--repo-root must identify the checkout containing this script")
    input_mode = preflight_data(repo_root)
    policy = load_policy(
        repo_root / "paper_data" / "primary" / "selected_policy_parameters.csv",
        args.role,
    )
    result = evaluate(repo_root, policy, args.seed, args.year)
    result["input_mode"] = input_mode
    result.update(
        compare(
            result,
            repo_root / "paper_data" / "primary" / "selected_policy_scores_by_seed.csv",
        )
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
