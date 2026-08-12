"""Build the final appendix robustness figure and service comparison table."""

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .make_paper_figures import _load_capacity


SETTING_ORDER = (
    "d100_delay_median_cap1",
    "d100_delay_75_cap1",
    "d200_delay_median_cap1",
    "d200_delay_75_cap1",
)
DISPLAY_SETTING_ORDER = SETTING_ORDER[1:]
SETTING_SPECS = {
    "d100_delay_median_cap1": (100, "delay_median", "Median delay"),
    "d100_delay_75_cap1": (100, "delay_75", "75th-percentile delay"),
    "d200_delay_median_cap1": (200, "delay_median", "Median delay"),
    "d200_delay_75_cap1": (200, "delay_75", "75th-percentile delay"),
}
SEARCH_COUNTS = {
    ("d100_delay_median_cap1", "borough"): 2304,
    ("d100_delay_median_cap1", "city"): 2304,
    ("d100_delay_75_cap1", "borough"): 3136,
    ("d100_delay_75_cap1", "city"): 3200,
    ("d200_delay_median_cap1", "borough"): 2624,
    ("d200_delay_median_cap1", "city"): 2624,
    ("d200_delay_75_cap1", "borough"): 2624,
    ("d200_delay_75_cap1", "city"): 2624,
}
INFERENCE_COUNTS = {
    ("d100_delay_median_cap1", "borough"): 6,
    ("d100_delay_median_cap1", "city"): 6,
    ("d100_delay_75_cap1", "borough"): 3,
    ("d100_delay_75_cap1", "city"): 11,
    ("d200_delay_median_cap1", "borough"): 8,
    ("d200_delay_median_cap1", "city"): 12,
    ("d200_delay_75_cap1", "borough"): 8,
    ("d200_delay_75_cap1", "city"): 7,
}
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
ROLE_SPECS = {
    "borough_most_efficient": ("borough", "most_efficient", "Most efficient"),
    "borough_balanced": (
        "borough",
        "balanced",
        "Balanced",
    ),
    "borough_most_equitable": ("borough", "most_equitable", "Most equitable"),
    "city_most_efficient": ("city", "most_efficient", "Most efficient"),
    "city_balanced": (
        "city",
        "balanced",
        "City Balanced",
    ),
    "city_most_equitable": ("city", "most_equitable", "Most equitable"),
}


def _read_exact(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if list(frame.columns) != columns:
        raise ValueError(f"{path.name} has unexpected columns: {list(frame.columns)}")
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    return frame


def _finite_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values in {columns}")


def _load_frontiers(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    robustness = data_dir / "robustness"
    search = _read_exact(
        robustness / "appendix_reoptimized_frontier_search.csv",
        [
            "setting_id",
            "budget_group",
            "policy_id",
            "efficiency_ratio_historical",
            "equity_ratio_historical",
        ],
    )
    inference = _read_exact(
        robustness / "appendix_reoptimized_frontier_inference_means.csv",
        [
            "setting_id",
            "policy_id",
            "budget_group",
            "n_seeds",
            "objective",
            "efficiency_ratio_historical",
            "equity_ratio_historical",
        ],
    )
    historical = _read_exact(
        robustness / "appendix_reoptimized_frontier_historical_reference.csv",
        [
            "setting_id",
            "drop_cost",
            "objective",
            "efficiency_historical",
            "allrequest_equity_historical",
            "historical_x",
            "historical_y",
        ],
    )

    for name, frame in (("search", search), ("inference", inference)):
        if frame[["setting_id", "budget_group", "policy_id"]].eq("").any().any():
            raise ValueError(f"{name} support contains an empty identifier")
        if set(frame["setting_id"]) != set(SETTING_ORDER):
            raise ValueError(f"{name} support has the wrong settings")
        if set(frame["budget_group"]) != set(BUDGET_COLORS):
            raise ValueError(f"{name} support has the wrong policy classes")
        if frame.duplicated(["setting_id", "budget_group", "policy_id"]).any():
            raise ValueError(f"{name} support contains duplicate policies")
        _finite_numeric(
            frame,
            ["efficiency_ratio_historical", "equity_ratio_historical"],
        )
        if (
            frame[["efficiency_ratio_historical", "equity_ratio_historical"]]
            .le(0.0)
            .any()
            .any()
        ):
            raise ValueError(f"{name} support contains a nonpositive loss ratio")

    search_counts = search.groupby(["setting_id", "budget_group"]).size().to_dict()
    if search_counts != SEARCH_COUNTS:
        raise ValueError(f"Search support changed: {search_counts}")
    inference_counts = (
        inference.groupby(["setting_id", "budget_group"]).size().to_dict()
    )
    if inference_counts != INFERENCE_COUNTS:
        raise ValueError(f"Locked inference support changed: {inference_counts}")
    _finite_numeric(inference, ["n_seeds"])
    if set(inference["n_seeds"].astype(int)) != {25}:
        raise ValueError("Locked-policy means must use 25 inference simulations")
    for setting_id, (_, objective, _) in SETTING_SPECS.items():
        if set(inference.loc[inference["setting_id"].eq(setting_id), "objective"]) != {
            objective
        }:
            raise ValueError(f"Inference objective does not match {setting_id}")

    if historical["setting_id"].tolist() != list(SETTING_ORDER):
        raise ValueError("Historical references have the wrong setting order")
    _finite_numeric(
        historical,
        [
            "drop_cost",
            "efficiency_historical",
            "allrequest_equity_historical",
            "historical_x",
            "historical_y",
        ],
    )
    if (
        historical[["efficiency_historical", "allrequest_equity_historical"]]
        .le(0.0)
        .any()
        .any()
        or set(historical["historical_x"]) != {1.0}
        or set(historical["historical_y"]) != {1.0}
    ):
        raise ValueError("Historical references are not positive denominators at (1, 1)")
    for row in historical.itertuples(index=False):
        drop_cost, objective, _ = SETTING_SPECS[row.setting_id]
        if row.drop_cost != drop_cost or row.objective != objective:
            raise ValueError(f"Historical reference does not match {row.setting_id}")
    return search, inference, historical


def _load_capacity_panel(
    data_dir: Path, inference: pd.DataFrame, historical: pd.DataFrame
) -> pd.DataFrame:
    capacity = _load_capacity(data_dir)
    reference = historical.loc[
        historical["setting_id"].eq("d100_delay_median_cap1")
    ]
    if len(reference) != 1:
        raise ValueError("Capacity panel lacks the primary historical reference")
    efficiency_historical = float(reference.iloc[0]["efficiency_historical"])
    equity_historical = float(reference.iloc[0]["allrequest_equity_historical"])
    for scenario in ("primary", "capacity125"):
        capacity[f"efficiency_ratio_historical_{scenario}"] = (
            capacity[f"efficiency_mean_{scenario}"] / efficiency_historical
        )
        capacity[f"equity_ratio_historical_{scenario}"] = (
            capacity[f"allrequest_equity_mean_{scenario}"] / equity_historical
        )
    ratio_columns = [
        f"{metric}_ratio_historical_{scenario}"
        for scenario in ("primary", "capacity125")
        for metric in ("efficiency", "equity")
    ]
    if (
        not np.isfinite(capacity[ratio_columns].to_numpy(dtype=float)).all()
        or capacity[ratio_columns].le(0.0).any().any()
    ):
        raise ValueError("Capacity panel contains invalid historical ratios")

    primary = capacity[
        [
            "policy_id",
            "budget_group",
            "efficiency_ratio_historical_primary",
            "equity_ratio_historical_primary",
        ]
    ]
    locked = inference.loc[
        inference["setting_id"].eq("d100_delay_median_cap1"),
        [
            "policy_id",
            "budget_group",
            "efficiency_ratio_historical",
            "equity_ratio_historical",
        ],
    ]
    joined = primary.merge(
        locked,
        on=["policy_id", "budget_group"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != 12 or not np.allclose(
        joined[
            [
                "efficiency_ratio_historical_primary",
                "equity_ratio_historical_primary",
            ]
        ].to_numpy(dtype=float),
        joined[
            ["efficiency_ratio_historical", "equity_ratio_historical"]
        ].to_numpy(dtype=float),
        rtol=1e-11,
        atol=1e-12,
    ):
        raise ValueError("Capacity panel primary support differs from the main frontier")
    return capacity


def _pareto(frame: pd.DataFrame) -> pd.DataFrame:
    x, y = "efficiency_ratio_historical", "equity_ratio_historical"
    ordered = frame.sort_values([x, y, "policy_id"]).drop_duplicates([x, y])
    retained: list[int] = []
    best_y = math.inf
    for index, row in ordered.iterrows():
        if float(row[y]) < best_y:
            retained.append(index)
            best_y = float(row[y])
    return ordered.loc[retained]


def _render_frontiers_in_clean_context(
    search: pd.DataFrame,
    inference: pd.DataFrame,
    capacity: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
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
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    x, y = "efficiency_ratio_historical", "equity_ratio_historical"
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.35))

    capacity_axis = axes.ravel()[0]
    for budget in ("borough", "city"):
        group = capacity.loc[capacity["budget_group"].eq(budget)].copy()
        for scenario, marker in (("primary", "o"), ("capacity125", "D")):
            plotted = group.assign(
                efficiency_ratio_historical=group[
                    f"efficiency_ratio_historical_{scenario}"
                ],
                equity_ratio_historical=group[
                    f"equity_ratio_historical_{scenario}"
                ],
            )
            if scenario == "primary":
                capacity_axis.scatter(
                    plotted[x],
                    plotted[y],
                    s=10,
                    color=BUDGET_COLORS[budget],
                    alpha=0.12,
                    edgecolors="none",
                    zorder=1,
                )
            else:
                capacity_axis.scatter(
                    plotted[x],
                    plotted[y],
                    s=14,
                    facecolors="none",
                    edgecolors=BUDGET_COLORS[budget],
                    linewidths=0.8,
                    alpha=0.22,
                    marker=marker,
                    zorder=1,
                )
            frontier = _pareto(plotted)
            capacity_axis.plot(
                frontier[x],
                frontier[y],
                color=BUDGET_COLORS[budget],
                linestyle="-",
                linewidth=2.0,
                marker=marker,
                markerfacecolor=(
                    BUDGET_COLORS[budget] if scenario == "primary" else "white"
                ),
                markeredgecolor=BUDGET_COLORS[budget],
                markeredgewidth=1.0,
                markersize=3.2,
                zorder=3,
            )
    capacity_axis.scatter(
        [1.0],
        [1.0],
        s=58,
        color=HISTORICAL_COLOR,
        edgecolors="none",
        zorder=4,
    )
    capacity_axis.set_title("D = 100; Median delay (capacity comparison)")

    for axis, setting_id in zip(axes.ravel()[1:], DISPLAY_SETTING_ORDER):
        panel_search = search.loc[search["setting_id"].eq(setting_id)]
        panel_inference = inference.loc[inference["setting_id"].eq(setting_id)]
        for budget in ("borough", "city"):
            cloud = panel_search.loc[panel_search["budget_group"].eq(budget)]
            locked = panel_inference.loc[panel_inference["budget_group"].eq(budget)]
            axis.scatter(
                cloud[x],
                cloud[y],
                s=5,
                color=BUDGET_COLORS[budget],
                alpha=0.03,
                edgecolors="none",
                rasterized=True,
                zorder=1,
            )
            frontier = _pareto(locked)
            axis.plot(
                frontier[x],
                frontier[y],
                color=BUDGET_COLORS[budget],
                linestyle="-",
                linewidth=2.0,
                marker="o",
                markersize=3.2,
                zorder=3,
            )
        axis.scatter(
            [1.0],
            [1.0],
            s=58,
            color=HISTORICAL_COLOR,
            edgecolors="none",
            zorder=4,
        )
        drop_cost, _, objective_label = SETTING_SPECS[setting_id]
        axis.set_title(f"D = {drop_cost}; {objective_label}")

    for axis in axes.ravel():
        axis.set_xlabel("Efficiency Loss (relative to historical)")
        axis.set_ylabel("All-request Equity Loss (relative to historical)")
        axis.set_xlim(0.4, 1.2)
        axis.set_ylim(0.0, 1.5)
        axis.set_facecolor("white")
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(
            axis="both", which="both", direction="out", width=0.8, length=3.0
        )

    legend_handles = [
        Line2D(
            [], [], linestyle="none", marker="o", markersize=7.5,
            markerfacecolor=HISTORICAL_COLOR, markeredgecolor="none",
            label="Historical inspections",
        ),
        Line2D(
            [], [], color=BUDGET_COLORS["borough"], linewidth=2.0,
            marker="o", markersize=3.2,
            label="Pareto frontier of Borough-budget policy",
        ),
        Line2D(
            [], [], color=BUDGET_COLORS["city"], linewidth=2.0,
            marker="o", markersize=3.2,
            label="Pareto frontier of City-budget policy",
        ),
        Line2D(
            [], [], color="0.25", linewidth=2.0, marker="D", markersize=3.2,
            markerfacecolor="white", markeredgecolor="0.25",
            label="Same policies at 1.25× capacity",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        framealpha=0.92,
        fontsize=9.0,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "appendix_robustness_reoptimized_frontiers_2026-08-10"
    pdf, png = base.with_suffix(".pdf"), base.with_suffix(".png")
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        png,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"Software": "authenticated manuscript frontier renderer"},
    )
    plt.close(fig)
    return pdf, png


def _render_frontiers(
    search: pd.DataFrame,
    inference: pd.DataFrame,
    capacity: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    # Earlier builders use seaborn and intentionally change process-global
    # rcParams. Render from Matplotlib defaults in a local context so running
    # the complete reproduction pipeline is identical to rendering this figure
    # by itself, without changing the style of any later builder.
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        return _render_frontiers_in_clean_context(
            search, inference, capacity, output_dir
        )


def _load_cell_comparison(data_dir: Path) -> pd.DataFrame:
    columns = [
        "d100_setting_id",
        "d200_setting_id",
        "policy_role",
        "budget_group",
        "display_role",
        "Borough",
        "Category",
        "d100_source_policy_id",
        "d100_vector_hash",
        "d100_n_inference_seeds",
        "d100_realized_inspection_fraction",
        "d200_source_policy_id",
        "d200_vector_hash",
        "d200_n_inference_seeds",
        "d200_realized_inspection_fraction",
        "inspection_fraction_change_d200_minus_d100",
        "historical_inspection_fraction",
    ]
    comparison = _read_exact(
        data_dir / "robustness" / "d100_d200_median_cell_service_comparison.csv",
        columns,
    )
    strings = [
        column
        for column in columns
        if column
        not in {
            "d100_n_inference_seeds",
            "d100_realized_inspection_fraction",
            "d200_n_inference_seeds",
            "d200_realized_inspection_fraction",
            "inspection_fraction_change_d200_minus_d100",
            "historical_inspection_fraction",
        }
    ]
    if comparison[strings].eq("").any().any():
        raise ValueError("Cell comparison contains an empty identifier")
    if set(comparison["d100_setting_id"]) != {"d100_delay_median_cap1"} or set(
        comparison["d200_setting_id"]
    ) != {"d200_delay_median_cap1"}:
        raise ValueError("Cell comparison does not compare the two median-delay settings")
    if set(comparison["policy_role"]) != set(ROLE_SPECS):
        raise ValueError("Cell comparison lacks the six locked display roles")
    if set(comparison["Borough"]) != set(BOROUGHS) or set(
        comparison["Category"]
    ) != set(CATEGORIES):
        raise ValueError("Cell comparison lacks the 30 active cells")
    if len(comparison) != 180 or comparison.duplicated(
        ["policy_role", "Borough", "Category"]
    ).any():
        raise ValueError("Cell comparison must contain 180 unique role-cell rows")
    numeric = [
        "d100_n_inference_seeds",
        "d100_realized_inspection_fraction",
        "d200_n_inference_seeds",
        "d200_realized_inspection_fraction",
        "inspection_fraction_change_d200_minus_d100",
        "historical_inspection_fraction",
    ]
    _finite_numeric(comparison, numeric)
    for prefix in ("d100", "d200"):
        if set(comparison[f"{prefix}_n_inference_seeds"].astype(int)) != {25}:
            raise ValueError(f"{prefix} cell means must use 25 inference simulations")
        if not comparison[f"{prefix}_realized_inspection_fraction"].between(
            0.0, 1.0
        ).all():
            raise ValueError(f"{prefix} inspection fractions must lie in [0, 1]")
    if not comparison["historical_inspection_fraction"].between(0.0, 1.0).all():
        raise ValueError("Historical inspection fractions must lie in [0, 1]")
    expected_change = (
        comparison["d200_realized_inspection_fraction"]
        - comparison["d100_realized_inspection_fraction"]
    )
    if not np.allclose(
        comparison["inspection_fraction_change_d200_minus_d100"],
        expected_change,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Cell comparison contains an inconsistent D=200 minus D=100 change")
    if (
        comparison.groupby(["Borough", "Category"])[
            "historical_inspection_fraction"
        ].nunique()
        != 1
    ).any():
        raise ValueError("Historical inspection fraction changes across policy roles")
    for policy_role, (budget, display_role, _) in ROLE_SPECS.items():
        rows = comparison.loc[comparison["policy_role"].eq(policy_role)]
        if set(rows["budget_group"]) != {budget} or set(rows["display_role"]) != {
            display_role
        }:
            raise ValueError(f"Cell comparison role identity changed: {policy_role}")
        for prefix in ("d100", "d200"):
            for identity in ("source_policy_id", "vector_hash"):
                if rows[f"{prefix}_{identity}"].nunique() != 1:
                    raise ValueError(
                        f"{policy_role} has multiple {prefix} {identity} values"
                    )
    return comparison


def _write_brooklyn_prune_table(
    comparison: pd.DataFrame, output_dir: Path
) -> tuple[Path, Path]:
    rows = comparison.loc[
        comparison["budget_group"].eq("borough")
        & comparison["Borough"].eq("Brooklyn")
        & comparison["Category"].eq("Prune")
    ].copy()
    role_order = {
        "borough_most_efficient": (0, "Most efficient"),
        "borough_balanced": (1, "Balanced"),
        "borough_most_equitable": (2, "Most equitable"),
    }
    if len(rows) != 3 or set(rows["policy_role"]) != set(role_order):
        raise ValueError("Cell comparison lacks the three Brooklyn Prune Borough roles")
    if rows["historical_inspection_fraction"].nunique() != 1:
        raise ValueError("Brooklyn Prune has conflicting historical fractions")
    rows["_order"] = rows["policy_role"].map(
        lambda value: role_order[str(value)][0]
    )
    rows = rows.sort_values("_order")
    table = pd.DataFrame(
        {
            "Policy role": rows["policy_role"].map(
                lambda value: role_order[str(value)][1]
            ),
            "Historical inspection fraction": rows[
                "historical_inspection_fraction"
            ].to_numpy(),
            "D=100 inspection fraction": rows[
                "d100_realized_inspection_fraction"
            ].to_numpy(),
            "D=200 inspection fraction": rows[
                "d200_realized_inspection_fraction"
            ].to_numpy(),
        }
    ).round(3)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "appendix_brooklyn_prune_d100_d200.csv"
    tex_path = output_dir / "appendix_brooklyn_prune_d100_d200.tex"
    table.to_csv(csv_path, index=False, float_format="%.3f")
    tex = [
        "% Generated from historical and role-matched simulation outputs.",
        r"\begin{table}[htb]",
        r"\small",
        r"\centering",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Policy role & Historical & $D=100$ & $D=200$ \\",
        r"\midrule",
    ]
    for row in table.to_dict(orient="records"):
        tex.append(
            f"{row['Policy role']} & "
            f"{float(row['Historical inspection fraction']):.3f} & "
            f"{float(row['D=100 inspection fraction']):.3f} & "
            f"{float(row['D=200 inspection fraction']):.3f} " + r"\\"
        )
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Historical and role-matched inspection fractions for Brooklyn Prune requests under the median-delay Borough-budget policies. Proposed-policy values are means across 25 simulations; Historical is the observed eventual inspection fraction.}",
            r"\label{tab:d200-low-service}",
            r"\end{table}",
            "",
        ]
    )
    tex_path.write_text("\n".join(tex), encoding="utf-8")
    return csv_path, tex_path


def _write_cell_change_summary(comparison: pd.DataFrame, output_dir: Path) -> Path:
    rows = comparison.copy()
    change = rows["inspection_fraction_change_d200_minus_d100"]
    rows["increased"] = change.gt(1e-12)
    rows["decreased"] = change.lt(-1e-12)
    rows["tied"] = change.abs().le(1e-12)
    for prefix in ("d100", "d200"):
        fraction = rows[f"{prefix}_realized_inspection_fraction"]
        rows[f"{prefix}_below_10pct"] = fraction.lt(0.10)
        rows[f"{prefix}_below_15pct"] = fraction.lt(0.15)
    summary = (
        rows.groupby(["policy_role", "budget_group", "display_role"], sort=False)
        .agg(
            active_cells=("Borough", "size"),
            increased_cells=("increased", "sum"),
            decreased_cells=("decreased", "sum"),
            tied_cells=("tied", "sum"),
            d100_minimum_inspection_fraction=(
                "d100_realized_inspection_fraction",
                "min",
            ),
            d200_minimum_inspection_fraction=(
                "d200_realized_inspection_fraction",
                "min",
            ),
            d100_cells_below_10pct=("d100_below_10pct", "sum"),
            d200_cells_below_10pct=("d200_below_10pct", "sum"),
            d100_cells_below_15pct=("d100_below_15pct", "sum"),
            d200_cells_below_15pct=("d200_below_15pct", "sum"),
            minimum_change_d200_minus_d100=(
                "inspection_fraction_change_d200_minus_d100",
                "min",
            ),
            mean_change_d200_minus_d100=(
                "inspection_fraction_change_d200_minus_d100",
                "mean",
            ),
        )
        .reset_index()
    )
    summary.insert(
        3,
        "paper_role",
        summary["policy_role"].map(
            {policy_role: spec[2] for policy_role, spec in ROLE_SPECS.items()}
        ),
    )
    integer_columns = [
        "active_cells",
        "increased_cells",
        "decreased_cells",
        "tied_cells",
        "d100_cells_below_10pct",
        "d200_cells_below_10pct",
        "d100_cells_below_15pct",
        "d200_cells_below_15pct",
    ]
    summary[integer_columns] = summary[integer_columns].astype(int)
    if set(summary["active_cells"]) != {30} or not (
        summary["increased_cells"]
        + summary["decreased_cells"]
        + summary["tied_cells"]
    ).eq(30).all():
        raise ValueError("D=100/D=200 direction counts do not reconcile to 30 cells")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "appendix_d100_d200_cell_change_summary.csv"
    summary.to_csv(path, index=False, float_format="%.10g")
    return path


def build_appendix_robustness(data_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    """Build App. Fig. 10, the supporting D comparison, and summary counts."""
    search, inference, historical = _load_frontiers(data_dir)
    capacity = _load_capacity_panel(data_dir, inference, historical)
    comparison = _load_cell_comparison(data_dir)
    return (
        *_render_frontiers(search, inference, capacity, output_dir / "figures"),
        *_write_brooklyn_prune_table(comparison, output_dir / "tables"),
        _write_cell_change_summary(comparison, output_dir / "tables"),
    )
