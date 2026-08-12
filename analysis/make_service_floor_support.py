"""Build the selection-only 20 percent minimum-service support diagnostic.

This module post-processes compact aggregate outputs.  It does not run Bayesian
optimization or simulation, and it does not use inference outcomes to select a
policy.  A policy is eligible when, in every active category--Borough cell, its
mean raw inspection fraction over selection seeds 311--320 is at least 0.20.
Nondominance is then computed separately within the Borough- and City-budget
eligible supports using the same ten-seed selection means.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SERVICE_FLOOR = 0.20
SELECTION_SEEDS = tuple(range(311, 321))
INFERENCE_SEEDS = tuple(range(321, 346))
SETTING_ID = "d100_delay_median_cap1"
BUDGET_ORDER = ("borough", "city")
BUDGET_COLORS = {"borough": "#ffa500", "city": "#0000ff"}
HISTORICAL_COLOR = "#ff0000"
BOROUGHS = ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")
CATEGORIES = (
    "Hazard",
    "Illegal Tree Damage",
    "Other",
    "Prune",
    "Remove Tree",
    "Root/Sewer/Sidewalk",
)
EXPECTED_CELL_KEYS = {
    (borough, category) for borough in BOROUGHS for category in CATEGORIES
}
EXPECTED_POLICY_COUNTS = {"borough": 274, "city": 264}
EXPECTED_ELIGIBLE_COUNTS = {"borough": 9, "city": 16}
EXPECTED_RESTRICTED_POLICY_IDS = {
    "borough": "candidate_d100_delay_median_cap1_borough_d06a017e6305154d",
    "city": "candidate_d100_delay_median_cap1_city_a324bbb3e5e086a4",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")

CELL_COLUMNS = (
    "policy_id",
    "budget_group",
    "BoroughCode",
    "SRCategory",
    "inspection_fraction_mean",
    "inspection_fraction_std",
    "n_selection_seeds",
    "inspection_fraction_sem",
    "selection_seed_start",
    "selection_seed_end",
)
SUMMARY_COLUMNS = (
    "policy_id",
    "budget_group",
    "vector_hash",
    "decoded_policy_hash",
    "minimum_cell_borough",
    "minimum_cell_category",
    "minimum_cell_mean_inspection_fraction",
    "minimum_cell_inspection_fraction_std",
    "n_selection_seeds",
    "minimum_cell_inspection_fraction_sem",
    "selection_seed_start",
    "selection_seed_end",
    "selection_mean_efficiency",
    "selection_mean_equity",
    "selection_seed_count",
    "service_floor",
    "satisfies_20pct_floor",
    "efficiency_ratio_historical",
    "equity_ratio_historical",
    "evidence_layer",
)


def _read(path: Path, required: Iterable[str]) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        keep_default_na=False,
        float_precision="round_trip",
    )
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    return frame


def _numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{label} contains a nonfinite {column}")
        frame[column] = values


def _budget_counts(frame: pd.DataFrame, label: str) -> dict[str, int]:
    counts = (
        frame[["policy_id", "budget_group"]]
        .drop_duplicates()
        .groupby("budget_group")
        .size()
        .astype(int)
        .to_dict()
    )
    if counts != EXPECTED_POLICY_COUNTS:
        raise ValueError(f"{label} policy counts changed: {counts}")
    return counts


def _load_cell_means(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "robustness" / "policy_cell_selection_means.csv"
    frame = _read(path, CELL_COLUMNS)
    if tuple(frame.columns) != CELL_COLUMNS:
        raise ValueError("Policy-cell input has an unexpected ordered schema")
    if len(frame) != 16_140:
        raise ValueError("Policy-cell input must contain exactly 16,140 rows")
    if frame.duplicated(
        ["policy_id", "budget_group", "BoroughCode", "SRCategory"]
    ).any():
        raise ValueError("Policy-cell input contains duplicate policy-cell rows")
    if set(frame["budget_group"].astype(str)) != set(BUDGET_ORDER):
        raise ValueError("Policy-cell input must contain Borough and City policies")
    _budget_counts(frame, "Policy-cell input")

    numeric = (
        "inspection_fraction_mean",
        "inspection_fraction_std",
        "n_selection_seeds",
        "inspection_fraction_sem",
        "selection_seed_start",
        "selection_seed_end",
    )
    _numeric(frame, numeric, "Policy-cell input")
    if not frame["inspection_fraction_mean"].between(0.0, 1.0).all():
        raise ValueError("Policy-cell means must be inspection fractions")
    if (frame[["inspection_fraction_std", "inspection_fraction_sem"]] < 0).any().any():
        raise ValueError("Policy-cell uncertainty values must be nonnegative")
    if set(frame["n_selection_seeds"].astype(int)) != {len(SELECTION_SEEDS)}:
        raise ValueError("Policy-cell means must use ten selection seeds")
    if set(frame["selection_seed_start"].astype(int)) != {min(SELECTION_SEEDS)} or set(
        frame["selection_seed_end"].astype(int)
    ) != {max(SELECTION_SEEDS)}:
        raise ValueError("Policy-cell means must use selection seeds 311--320")
    expected_sem = frame["inspection_fraction_std"] / math.sqrt(len(SELECTION_SEEDS))
    if not np.allclose(
        expected_sem,
        frame["inspection_fraction_sem"],
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError("Policy-cell standard errors do not match ten-seed standard deviations")

    expected_signature = frozenset(EXPECTED_CELL_KEYS)
    signatures = frame.groupby("policy_id", sort=False)[
        ["BoroughCode", "SRCategory"]
    ].apply(
        lambda group: frozenset(
            zip(group["BoroughCode"].astype(str), group["SRCategory"].astype(str))
        ),
    )
    if len(signatures) != sum(EXPECTED_POLICY_COUNTS.values()) or not signatures.map(
        lambda cells: cells == expected_signature
    ).all():
        raise ValueError("Every policy must contain the exact 30 active cells")
    if frame.groupby("policy_id")["budget_group"].nunique().max() != 1:
        raise ValueError("A policy appears under more than one budget class")
    return frame


def _load_historical(data_dir: Path) -> dict[str, float]:
    frame = _read(
        data_dir / "primary" / "historical_reference.csv",
        {
            "efficiency_total_cost_delay_median",
            "allrequest_equity_delay_median",
            "efficiency_pct_historical",
            "equity_pct_historical",
        },
    )
    if len(frame) != 1:
        raise ValueError("Historical reference must contain exactly one row")
    _numeric(
        frame,
        (
            "efficiency_total_cost_delay_median",
            "allrequest_equity_delay_median",
            "efficiency_pct_historical",
            "equity_pct_historical",
        ),
        "Historical reference",
    )
    if not np.allclose(
        frame[["efficiency_pct_historical", "equity_pct_historical"]],
        100.0,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Historical reference must normalize to (1,1)")
    efficiency = float(frame.iloc[0]["efficiency_total_cost_delay_median"])
    equity = float(frame.iloc[0]["allrequest_equity_delay_median"])
    if efficiency <= 0 or equity <= 0:
        raise ValueError("Historical objectives must be positive")
    return {"efficiency": efficiency, "equity": equity}


def _load_selection(data_dir: Path, historical: dict[str, float]) -> pd.DataFrame:
    required = {
        "setting_id",
        "policy_id",
        "budget_group",
        "efficiency_total_cost",
        "allrequest_equity",
        "decoded_policy_hash",
        "policy_vector_hash",
        "replication_count",
        "evidence_layer",
        "efficiency_pct_historical",
        "equity_pct_historical",
    }
    frame = _read(data_dir / "primary" / "selection_means.csv", required)
    if len(frame) != sum(EXPECTED_POLICY_COUNTS.values()):
        raise ValueError("Selection means must contain exactly 538 policies")
    if frame.duplicated(["policy_id", "budget_group"]).any():
        raise ValueError("Selection means contain duplicate policies")
    _budget_counts(frame, "Selection means")
    if set(frame["setting_id"].astype(str)) != {SETTING_ID}:
        raise ValueError("Selection means contain a non-primary setting")
    if set(pd.to_numeric(frame["replication_count"], errors="raise").astype(int)) != {
        len(SELECTION_SEEDS)
    }:
        raise ValueError("Selection means must use ten replications")
    if set(frame["evidence_layer"].astype(str)) != {"selection_seed_mean"}:
        raise ValueError("Selection means have an unexpected evidence layer")
    for column in ("decoded_policy_hash", "policy_vector_hash"):
        if not frame[column].astype(str).map(lambda value: bool(HEX64.fullmatch(value))).all():
            raise ValueError(f"Selection means contain an invalid {column}")
    _numeric(
        frame,
        (
            "efficiency_total_cost",
            "allrequest_equity",
            "efficiency_pct_historical",
            "equity_pct_historical",
        ),
        "Selection means",
    )
    if (frame[["efficiency_total_cost", "allrequest_equity"]] <= 0).any().any():
        raise ValueError("Selection objectives must be positive")
    expected_efficiency = 100.0 * frame["efficiency_total_cost"] / historical["efficiency"]
    expected_equity = 100.0 * frame["allrequest_equity"] / historical["equity"]
    if not np.allclose(
        expected_efficiency,
        frame["efficiency_pct_historical"],
        rtol=1e-12,
        atol=1e-10,
    ) or not np.allclose(
        expected_equity,
        frame["equity_pct_historical"],
        rtol=1e-12,
        atol=1e-10,
    ):
        raise ValueError("Selection normalization does not match the historical reference")
    return frame


def _load_inference(data_dir: Path, historical: dict[str, float]) -> pd.DataFrame:
    required = {
        "policy_id",
        "budget_group",
        "efficiency_total_cost_delay_median",
        "allrequest_equity_delay_median",
        "n_seeds",
        "efficiency_pct_historical",
        "equity_pct_historical",
    }
    frame = _read(data_dir / "primary" / "inference_means.csv", required)
    if len(frame) != 12 or frame.duplicated(["policy_id", "budget_group"]).any():
        raise ValueError("Inference means must contain 12 unique locked policies")
    if frame.groupby("budget_group").size().astype(int).to_dict() != {
        "borough": 6,
        "city": 6,
    }:
        raise ValueError("Inference means must contain six policies per budget class")
    _numeric(
        frame,
        (
            "efficiency_total_cost_delay_median",
            "allrequest_equity_delay_median",
            "n_seeds",
            "efficiency_pct_historical",
            "equity_pct_historical",
        ),
        "Inference means",
    )
    if set(frame["n_seeds"].astype(int)) != {len(INFERENCE_SEEDS)}:
        raise ValueError("Inference means must use 25 locked replications")
    expected_efficiency = (
        100.0 * frame["efficiency_total_cost_delay_median"] / historical["efficiency"]
    )
    expected_equity = (
        100.0 * frame["allrequest_equity_delay_median"] / historical["equity"]
    )
    if not np.allclose(
        expected_efficiency,
        frame["efficiency_pct_historical"],
        rtol=1e-12,
        atol=1e-10,
    ) or not np.allclose(
        expected_equity,
        frame["equity_pct_historical"],
        rtol=1e-12,
        atol=1e-10,
    ):
        raise ValueError("Inference normalization does not match the historical reference")
    frame = frame.copy()
    frame["efficiency_ratio_historical"] = expected_efficiency / 100.0
    frame["equity_ratio_historical"] = expected_equity / 100.0
    return frame


def nondominated_min(frame: pd.DataFrame, *, x: str, y: str) -> pd.DataFrame:
    """Return deterministic nondominated rows for a two-objective minimization."""
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    _numeric(work, (x, y), "Pareto input")
    work = work.sort_values([x, y, "policy_id"]).drop_duplicates([x, y], keep="first")
    values = work[[x, y]].to_numpy(dtype=float)
    keep = np.ones(len(work), dtype=bool)
    for index, value in enumerate(values):
        keep[index] = not bool(
            np.any(np.all(values <= value, axis=1) & np.any(values < value, axis=1))
        )
    return work.loc[keep].sort_values([x, y, "policy_id"]).reset_index(drop=True)


def compute_service_floor_support(
    cell_means: pd.DataFrame,
    selection: pd.DataFrame,
    historical: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return all-policy floors, eligible support, and within-class Pareto rows."""
    cell_policies = cell_means[["policy_id", "budget_group"]].drop_duplicates()
    selection_policies = selection[["policy_id", "budget_group"]].drop_duplicates()
    if set(map(tuple, cell_policies.itertuples(index=False, name=None))) != set(
        map(tuple, selection_policies.itertuples(index=False, name=None))
    ):
        raise ValueError("Policy-cell input does not match the primary selection pool")

    floors = (
        cell_means.sort_values(
            ["policy_id", "inspection_fraction_mean", "BoroughCode", "SRCategory"]
        )
        .groupby("policy_id", sort=True)
        .first()
        .reset_index()
        .rename(
            columns={
                "BoroughCode": "minimum_cell_borough",
                "SRCategory": "minimum_cell_category",
                "inspection_fraction_mean": "minimum_cell_mean_inspection_fraction",
                "inspection_fraction_std": "minimum_cell_inspection_fraction_std",
                "inspection_fraction_sem": "minimum_cell_inspection_fraction_sem",
            }
        )
    )
    metadata = selection[
        [
            "policy_id",
            "budget_group",
            "policy_vector_hash",
            "decoded_policy_hash",
            "efficiency_total_cost",
            "allrequest_equity",
        ]
    ].rename(
        columns={
            "policy_vector_hash": "vector_hash",
            "efficiency_total_cost": "selection_mean_efficiency",
            "allrequest_equity": "selection_mean_equity",
        }
    )
    summary = metadata.merge(
        floors.drop(columns="budget_group"),
        on="policy_id",
        validate="one_to_one",
    )
    summary["selection_seed_count"] = len(SELECTION_SEEDS)
    summary["service_floor"] = SERVICE_FLOOR
    summary["satisfies_20pct_floor"] = (
        summary["minimum_cell_mean_inspection_fraction"] >= SERVICE_FLOOR
    )
    summary["efficiency_ratio_historical"] = (
        summary["selection_mean_efficiency"] / historical["efficiency"]
    )
    summary["equity_ratio_historical"] = (
        summary["selection_mean_equity"] / historical["equity"]
    )
    summary["evidence_layer"] = "selection_seed_mean_311_320"
    summary = summary.loc[:, SUMMARY_COLUMNS]
    summary = summary.sort_values(["budget_group", "policy_id"]).reset_index(drop=True)

    eligible = summary.loc[summary["satisfies_20pct_floor"]].copy()
    counts = eligible.groupby("budget_group").size().astype(int).to_dict()
    if counts != EXPECTED_ELIGIBLE_COUNTS:
        raise ValueError(f"Eligible finite-support counts changed: {counts}")

    restricted = []
    for budget in BUDGET_ORDER:
        frontier = nondominated_min(
            eligible.loc[eligible["budget_group"] == budget],
            x="selection_mean_efficiency",
            y="selection_mean_equity",
        )
        frontier["frontier_rank"] = np.arange(1, len(frontier) + 1)
        restricted.append(frontier)
    pareto = pd.concat(restricted, ignore_index=True)
    observed_ids = {
        budget: pareto.loc[pareto["budget_group"] == budget, "policy_id"].tolist()
        for budget in BUDGET_ORDER
    }
    expected_ids = {
        budget: [policy_id] for budget, policy_id in EXPECTED_RESTRICTED_POLICY_IDS.items()
    }
    if observed_ids != expected_ids:
        raise ValueError(f"Restricted Pareto support changed: {observed_ids}")
    return summary, eligible, pareto


def _main_frontier(inference: pd.DataFrame) -> pd.DataFrame:
    frontiers = []
    for budget in BUDGET_ORDER:
        frontier = nondominated_min(
            inference.loc[inference["budget_group"] == budget],
            x="efficiency_total_cost_delay_median",
            y="allrequest_equity_delay_median",
        )
        frontier["frontier_rank"] = np.arange(1, len(frontier) + 1)
        frontiers.append(frontier)
    result = pd.concat(frontiers, ignore_index=True)
    if result.groupby("budget_group").size().astype(int).to_dict() != {
        "borough": 6,
        "city": 6,
    }:
        raise ValueError("Locked main-text frontier support changed")
    return result


def _overlay_coordinates(
    inference: pd.DataFrame,
    main_frontier: pd.DataFrame,
    eligible: pd.DataFrame,
    restricted: pd.DataFrame,
) -> pd.DataFrame:
    main_ids = set(main_frontier["policy_id"].astype(str))
    restricted_ids = set(restricted["policy_id"].astype(str))
    rows = []
    for row in inference.to_dict(orient="records"):
        rows.append(
            {
                "evidence_layer": "locked_inference_mean_321_345",
                "budget_group": str(row["budget_group"]),
                "policy_id": str(row["policy_id"]),
                "replication_count": len(INFERENCE_SEEDS),
                "satisfies_20pct_floor_on_selection_means": "not_evaluated_for_this_layer",
                "within_layer_nondominated": str(row["policy_id"]) in main_ids,
                "efficiency_ratio_historical": float(row["efficiency_ratio_historical"]),
                "equity_ratio_historical": float(row["equity_ratio_historical"]),
            }
        )
    for row in eligible.to_dict(orient="records"):
        rows.append(
            {
                "evidence_layer": "min20_selection_eligible_mean_311_320",
                "budget_group": str(row["budget_group"]),
                "policy_id": str(row["policy_id"]),
                "replication_count": len(SELECTION_SEEDS),
                "satisfies_20pct_floor_on_selection_means": True,
                "within_layer_nondominated": str(row["policy_id"]) in restricted_ids,
                "efficiency_ratio_historical": float(row["efficiency_ratio_historical"]),
                "equity_ratio_historical": float(row["equity_ratio_historical"]),
            }
        )
    rows.append(
        {
            "evidence_layer": "historical_2019",
            "budget_group": "historical",
            "policy_id": "historical_2019",
            "replication_count": 1,
            "satisfies_20pct_floor_on_selection_means": "not_applicable",
            "within_layer_nondominated": True,
            "efficiency_ratio_historical": 1.0,
            "equity_ratio_historical": 1.0,
        }
    )
    return pd.DataFrame(rows)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.use14corefonts": False,
            "text.usetex": False,
            "mathtext.fontset": "dejavusans",
        }
    )


def _style_axis(ax) -> None:
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="out", width=0.8, length=3.0)
    ax.set_xlabel("Efficiency Loss (relative to historical)")
    ax.set_ylabel("All-request Equity Loss (relative to historical)")


def _draw_panel(
    ax,
    *,
    main_frontier: pd.DataFrame,
    eligible: pd.DataFrame,
    restricted: pd.DataFrame,
    show_eligible: bool,
) -> None:
    for budget in BUDGET_ORDER:
        label = budget.title()
        color = BUDGET_COLORS[budget]
        main = main_frontier.loc[main_frontier["budget_group"] == budget]
        ax.plot(
            main["efficiency_ratio_historical"],
            main["equity_ratio_historical"],
            color=color,
            marker="o",
            markersize=3.4,
            linewidth=2.0,
            label=f"{label} main frontier (25 inference seeds)",
            zorder=3,
        )
        if show_eligible:
            cloud = eligible.loc[eligible["budget_group"] == budget]
            ax.scatter(
                cloud["efficiency_ratio_historical"],
                cloud["equity_ratio_historical"],
                s=24,
                facecolors="none",
                edgecolors=color,
                linewidths=0.8,
                alpha=0.42,
                zorder=2,
            )
        point = restricted.loc[restricted["budget_group"] == budget].iloc[0]
        ax.scatter(
            [point["efficiency_ratio_historical"]],
            [point["equity_ratio_historical"]],
            color=color,
            marker="D",
            edgecolors="black",
            linewidths=0.6,
            s=42,
            label=f"{label} 20% finite-support singleton (10 selection seeds)",
            zorder=5,
        )
        ax.annotate(
            f"{label} singleton",
            (point["efficiency_ratio_historical"], point["equity_ratio_historical"]),
            xytext=(5, -10 if budget == "borough" else 7),
            textcoords="offset points",
            fontsize=7.2,
            color=color,
            va="top" if budget == "borough" else "bottom",
        )
    ax.scatter(
        [1.0],
        [1.0],
        s=52,
        color=HISTORICAL_COLOR,
        edgecolors="none",
        label="Historical inspections",
        zorder=6,
    )
    _style_axis(ax)


def _padded_limits(values: Iterable[float]) -> tuple[float, float]:
    numeric = np.asarray(list(values), dtype=float)
    lower, upper = float(numeric.min()), float(numeric.max())
    padding = max(0.04, 0.06 * max(upper - lower, abs(upper), 1.0))
    return lower - padding, upper + padding


def _assert_pdf_fonts(path: Path) -> None:
    executable = shutil.which("pdffonts")
    if executable is None:
        raise RuntimeError("pdffonts is required for the publication font gate")
    result = subprocess.run(
        [executable, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdffonts failed for {path}: {result.stderr.strip()}")
    invalid = []
    for row in [line.strip() for line in result.stdout.splitlines()[2:] if line.strip()]:
        fields = row.split()
        if len(fields) < 6 or fields[-5].lower() != "yes" or "type 3" in row.lower():
            invalid.append(row)
    if invalid:
        raise RuntimeError(f"PDF contains Type 3 or unembedded fonts: {' | '.join(invalid)}")


def _render_overlay(
    main_frontier: pd.DataFrame,
    eligible: pd.DataFrame,
    restricted: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    _draw_panel(
        axes[0],
        main_frontier=main_frontier,
        eligible=eligible,
        restricted=restricted,
        show_eligible=True,
    )
    axes[0].set_title("Eligible finite support (selection only)")
    axes[0].text(
        0.02,
        0.98,
        "Hollow points: policies meeting 20% in all 30 cells",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
    )
    full_x = pd.concat(
        [
            main_frontier["efficiency_ratio_historical"],
            eligible["efficiency_ratio_historical"],
            pd.Series([1.0]),
        ],
        ignore_index=True,
    )
    full_y = pd.concat(
        [
            main_frontier["equity_ratio_historical"],
            eligible["equity_ratio_historical"],
            pd.Series([1.0]),
        ],
        ignore_index=True,
    )
    axes[0].set_xlim(*_padded_limits(full_x))
    axes[0].set_ylim(*_padded_limits(full_y))

    _draw_panel(
        axes[1],
        main_frontier=main_frontier,
        eligible=eligible,
        restricted=restricted,
        show_eligible=False,
    )
    axes[1].set_title("Nondominated finite support (one point per class)")
    focus_x = pd.concat(
        [
            main_frontier["efficiency_ratio_historical"],
            restricted["efficiency_ratio_historical"],
            pd.Series([1.0]),
        ],
        ignore_index=True,
    )
    focus_y = pd.concat(
        [
            main_frontier["equity_ratio_historical"],
            restricted["equity_ratio_historical"],
            pd.Series([1.0]),
        ],
        ignore_index=True,
    )
    axes[1].set_xlim(*_padded_limits(focus_x))
    axes[1].set_ylim(*_padded_limits(focus_y))

    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [
        "Historical inspections",
        "Borough main frontier (25 inference seeds)",
        "City main frontier (25 inference seeds)",
        "Borough 20% finite-support singleton (10 selection seeds)",
        "City 20% finite-support singleton (10 selection seeds)",
    ]
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.suptitle(
        "20% minimum-service finite-support diagnostic (no constrained reoptimization)",
        fontsize=10.5,
    )
    fig.legend(
        [by_label[label] for label in order],
        order,
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=7.4,
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 0.95))
    pdf = output_dir / "service_floor_support_overlay.pdf"
    png = output_dir / "service_floor_support_overlay.png"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"Creator": "SLA public replication", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        png,
        bbox_inches="tight",
        pad_inches=0.03,
        dpi=300,
        metadata={"Software": "SLA public replication"},
    )
    plt.close(fig)
    _assert_pdf_fonts(pdf)
    return pdf, png


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")
    return path


def _caption(eligible: pd.DataFrame, restricted: pd.DataFrame) -> str:
    eligible_counts = eligible.groupby("budget_group").size().astype(int).to_dict()
    pareto_counts = restricted.groupby("budget_group").size().astype(int).to_dict()
    return (
        "Twenty-percent minimum-service finite-support diagnostic for the primary "
        "2019 D=100, median-delay, rho=0.15 specification. Solid lines show the "
        "existing main-text Borough- and City-budget frontiers based on locked "
        "25-seed inference means (seeds 321--345). Hollow points show policies in "
        "the 538-policy selection pool whose mean raw inspection fraction over "
        "selection seeds 311--320 is at least 20% in every one of the 30 active "
        "category--Borough cells. Diamonds show the within-class nondominated "
        "subset of those ten-seed selection means. The eligible finite support "
        f"contains {eligible_counts.get('borough', 0)} Borough and "
        f"{eligible_counts.get('city', 0)} City policies; its nondominated subsets "
        f"contain {pareto_counts.get('borough', 0)} and "
        f"{pareto_counts.get('city', 0)} points, respectively. Both subsets are "
        "singletons and both points are worse than historical inspections at "
        "(1,1). This selection-only post-processing diagnostic is not a newly "
        "optimized constrained frontier and does not use inference outcomes for "
        "policy selection.\n"
    )


def build_service_floor_support(data_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    """Build all machine-readable and rendered service-floor diagnostics."""
    if data_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output directories must differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    historical = _load_historical(data_dir)
    cells = _load_cell_means(data_dir)
    selection = _load_selection(data_dir, historical)
    inference = _load_inference(data_dir, historical)
    summary, eligible, restricted = compute_service_floor_support(
        cells,
        selection,
        historical,
    )
    main_frontier = _main_frontier(inference)
    coordinates = _overlay_coordinates(inference, main_frontier, eligible, restricted)

    outputs = [
        _write_csv(summary, output_dir / "service_floor_policy_summary.csv"),
        _write_csv(eligible, output_dir / "service_floor_eligible_support.csv"),
        _write_csv(
            restricted,
            output_dir / "service_floor_restricted_pareto_support.csv",
        ),
        _write_csv(
            coordinates,
            output_dir / "service_floor_overlay_coordinates.csv",
        ),
    ]
    caption = output_dir / "service_floor_support_caption.txt"
    caption.write_text(_caption(eligible, restricted), encoding="utf-8")
    outputs.append(caption)
    outputs.extend(_render_overlay(main_frontier, eligible, restricted, output_dir))
    for path in outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected nonempty output: {path}")
    return tuple(outputs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the selection-only 20 percent service-floor diagnostic."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("paper_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/robustness"))
    arguments = parser.parse_args()
    for result in build_service_floor_support(arguments.data_dir, arguments.output_dir):
        print(result)
