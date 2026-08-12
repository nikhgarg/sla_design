"""Build the active Pareto--Hazard manuscript figure from compact public data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"borough": "#ffa500", "city": "#0000ff", "historical": "#ff0000"}
HIST_COLORS = {"borough": "#ff7f0e", "city": "#1f77b4"}
BOROUGHS = ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")
HAZARD_ROLES = {
    "borough_most_efficient": "borough",
    "city_most_efficient": "city",
    "historical_2019": "historical",
}


def _read(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    return frame


def _ratios(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    for metric in ("efficiency", "equity"):
        ratio = f"{metric}_ratio_historical"
        percent = f"{metric}_pct_historical"
        if ratio in result:
            result[ratio] = pd.to_numeric(result[ratio], errors="raise")
        elif percent in result:
            result[ratio] = pd.to_numeric(result[percent], errors="raise") / 100.0
        else:
            raise ValueError(f"{name} lacks {ratio} or {percent}")
        values = result[ratio].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} contains invalid {metric} ratios")
    return result


def _pareto(
    frame: pd.DataFrame,
    x: str = "efficiency_ratio_historical",
    y: str = "equity_ratio_historical",
) -> pd.DataFrame:
    ordered = frame.sort_values([x, y]).drop_duplicates([x, y])
    keep, best = [], np.inf
    for index, row in ordered.iterrows():
        if float(row[y]) < best:
            keep.append(index)
            best = float(row[y])
    return ordered.loc[keep]


def _load_hazard(path: Path) -> pd.DataFrame:
    required = {
        "borough",
        "role",
        "bin_index",
        "bin_left_days",
        "bin_right_days",
        "bin_label",
        "inspected_delay_count",
        "terminal_uninspected_count",
        "display_count",
        "all_request_count",
        "display_share_percent",
    }
    frame = _read(path, required)
    numeric = required - {"borough", "role", "bin_label", "bin_right_days"}
    frame[list(numeric)] = frame[list(numeric)].apply(pd.to_numeric, errors="raise")
    frame["bin_right_days"] = pd.to_numeric(frame["bin_right_days"], errors="coerce")
    missing_right = frame["bin_right_days"].isna()
    if not (frame.loc[missing_right, "bin_label"] == ">=20 days or not inspected").all():
        raise ValueError("Only the terminal Hazard bin may omit its right edge")
    frame.loc[missing_right, "bin_right_days"] = frame.loc[missing_right, "bin_left_days"] + 1
    if set(frame["role"].astype(str)) != set(HAZARD_ROLES):
        raise ValueError("Hazard bins do not contain the three active figure roles")
    frame["role_kind"] = frame["role"].astype(str).map(HAZARD_ROLES)
    if set(frame["borough"].astype(str)) != set(BOROUGHS):
        raise ValueError("Hazard bins need the five manuscript Boroughs")
    if set(frame["role_kind"]) != set(COLORS):
        raise ValueError("Hazard bins need Borough, City, and historical roles")
    if frame.duplicated(["borough", "role_kind", "bin_index"]).any():
        raise ValueError("Hazard bins contain duplicate Borough/role/bin rows")
    expected_grid = {
        (borough, role) for borough in BOROUGHS for role in COLORS
    }
    if set(frame.groupby(["borough", "role_kind"]).size().index) != expected_grid:
        raise ValueError("Hazard bins do not contain the complete five-by-three grid")
    counts = frame[["inspected_delay_count", "terminal_uninspected_count", "display_count"]]
    if (counts < 0).any().any() or not (
        frame["display_count"]
        == frame["inspected_delay_count"] + frame["terminal_uninspected_count"]
    ).all():
        raise ValueError("Hazard display counts do not reconcile")
    if (frame.loc[frame["bin_index"] < 20, "terminal_uninspected_count"] != 0).any():
        raise ValueError("Uninspected Hazard requests must appear only in the terminal bin")
    expected = 100.0 * frame["display_count"] / frame["all_request_count"]
    if (frame["all_request_count"] <= 0).any() or not np.allclose(
        expected, frame["display_share_percent"], rtol=1e-8, atol=1e-10
    ):
        raise ValueError("Hazard display shares do not reconcile")
    for key, group in frame.groupby(["borough", "role_kind"]):
        ordered = group.sort_values("bin_index")
        if ordered["all_request_count"].nunique() != 1:
            raise ValueError(f"Hazard denominator changes within {key}")
        if ordered["bin_index"].astype(int).tolist() != list(range(21)):
            raise ValueError(f"Hazard bins are not the expected 0--20 grid for {key}")
        if not np.allclose(ordered["bin_left_days"], range(21)) or not np.allclose(
            ordered["bin_right_days"], range(1, 22)
        ):
            raise ValueError(f"Hazard bin edges are invalid for {key}")
    return frame


def _set_appendix_style() -> None:
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
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
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


def _save_pair(fig, base: Path) -> tuple[Path, Path]:
    pdf, png = base.with_suffix(".pdf"), base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return pdf, png


def _load_capacity(data_dir: Path) -> pd.DataFrame:
    required = {
        "policy_id", "budget_group", "period_primary", "period_capacity125",
        "efficiency_mean_primary", "allrequest_equity_mean_primary",
        "efficiency_mean_capacity125", "allrequest_equity_mean_capacity125",
        "n_seeds_primary", "n_seeds_capacity125", "objective_primary",
        "objective_capacity125", "efficiency_pct_primary", "equity_pct_primary",
        "efficiency_pct_capacity125", "equity_pct_capacity125", "coverage_interpretation",
    }
    frame = _read(data_dir / "robustness" / "capacity125_policy_means.csv", required)
    if frame.groupby("budget_group").size().to_dict() != {"borough": 6, "city": 6}:
        raise ValueError("Capacity robustness needs six locked policies per budget group")
    if frame.duplicated(["budget_group", "policy_id"]).any():
        raise ValueError("Capacity robustness contains duplicate locked policies")
    if set(frame["period_primary"].astype(int)) != {2019} or set(
        frame["period_capacity125"].astype(int)
    ) != {2019}:
        raise ValueError("Capacity robustness must compare 2019 with 2019")
    if set(frame["n_seeds_primary"].astype(int)) != {25} or set(
        frame["n_seeds_capacity125"].astype(int)
    ) != {25}:
        raise ValueError("Capacity robustness must use 25 seeds in both scenarios")
    if set(frame["objective_primary"].astype(str)) != {"delay_median"} or set(
        frame["objective_capacity125"].astype(str)
    ) != {"delay_median"}:
        raise ValueError("Capacity robustness must use the primary median-delay objective")
    expected_interpretation = (
        "fixed_support_no_global_coverage_certificate_no_reoptimization"
    )
    if set(frame["coverage_interpretation"].astype(str)) != {expected_interpretation}:
        raise ValueError("Capacity robustness is not labeled as a fixed-support diagnostic")
    values = [
        f"{metric}_{scenario}"
        for scenario in ("primary", "capacity125")
        for metric in ("efficiency_mean", "allrequest_equity_mean", "efficiency_pct", "equity_pct")
    ]
    frame[values] = frame[values].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[values].to_numpy()).all() or (frame[values] <= 0).any().any():
        raise ValueError("Capacity robustness contains invalid objective values")

    locked = _read(
        data_dir / "primary" / "inference_means.csv",
        {
            "policy_id", "budget_group", "efficiency_total_cost_delay_median",
            "allrequest_equity_delay_median", "n_seeds",
        },
    )
    keys = ["budget_group", "policy_id"]
    if set(map(tuple, frame[keys].itertuples(index=False, name=None))) != set(
        map(tuple, locked[keys].itertuples(index=False, name=None))
    ):
        raise ValueError("Capacity robustness support differs from locked inference support")
    joined = frame.merge(locked, on=keys, how="inner", validate="one_to_one")
    if not np.allclose(
        joined["efficiency_mean_primary"],
        pd.to_numeric(joined["efficiency_total_cost_delay_median"], errors="raise"),
        rtol=1e-12,
        atol=1e-8,
    ) or not np.allclose(
        joined["allrequest_equity_mean_primary"],
        pd.to_numeric(joined["allrequest_equity_delay_median"], errors="raise"),
        rtol=1e-12,
        atol=1e-10,
    ):
        raise ValueError("Capacity robustness primary values differ from locked inference means")
    if not np.array_equal(joined["n_seeds_primary"].astype(int), joined["n_seeds"].astype(int)):
        raise ValueError("Capacity robustness seed counts differ from locked inference support")
    reference = frame.loc[
        (frame["budget_group"] == "borough")
        & np.isclose(frame["efficiency_pct_primary"], 100.0)
        & np.isclose(frame["equity_pct_primary"], 100.0)
    ]
    if len(reference) != 1:
        raise ValueError("Capacity robustness lacks one primary Borough-efficient reference")
    efficiency_reference = float(reference.iloc[0]["efficiency_mean_primary"])
    equity_reference = float(reference.iloc[0]["allrequest_equity_mean_primary"])
    for scenario in ("primary", "capacity125"):
        for metric, reference_value in (
            ("efficiency", efficiency_reference), ("equity", equity_reference)
        ):
            raw = "efficiency_mean" if metric == "efficiency" else "allrequest_equity_mean"
            expected = 100.0 * frame[f"{raw}_{scenario}"] / reference_value
            if not np.allclose(expected, frame[f"{metric}_pct_{scenario}"], rtol=1e-9, atol=1e-10):
                raise ValueError("Capacity robustness percentages do not match the declared reference")

    diagnostic = _read(
        data_dir / "robustness" / "capacity125_diagnostic.csv",
        {"budget_group", "locked_support_policy_count", "interpretation", "full_capacity125_BO_authorized"},
    )
    if set(diagnostic["budget_group"]) != {"borough", "city"} or set(
        diagnostic["locked_support_policy_count"].astype(int)
    ) != {6}:
        raise ValueError("Capacity diagnostic does not describe the same locked support")
    authorized = diagnostic["full_capacity125_BO_authorized"].astype(str).str.lower()
    if set(authorized) != {"false"} or not diagnostic["interpretation"].astype(str).str.contains(
        "not_global_coverage", regex=False
    ).all():
        raise ValueError("Capacity diagnostic incorrectly implies a new BO coverage claim")
    return frame


def _capacity_figure_in_clean_context(
    data_dir: Path, output_dir: Path
) -> tuple[Path, Path]:
    frame = _load_capacity(data_dir)
    historical = _read(
        data_dir / "primary" / "historical_reference.csv",
        {
            "efficiency_total_cost_delay_median",
            "allrequest_equity_delay_median",
            "efficiency_pct_historical",
            "equity_pct_historical",
        },
    )
    if len(historical) != 1:
        raise ValueError("Historical reference must contain exactly one row")
    efficiency_historical = float(historical.iloc[0]["efficiency_total_cost_delay_median"])
    equity_historical = float(historical.iloc[0]["allrequest_equity_delay_median"])
    if (
        not np.isfinite([efficiency_historical, equity_historical]).all()
        or efficiency_historical <= 0
        or equity_historical <= 0
        or not np.allclose(
            historical.iloc[0][["efficiency_pct_historical", "equity_pct_historical"]]
            .astype(float)
            .to_numpy(),
            [100.0, 100.0],
        )
    ):
        raise ValueError("Historical reference is invalid")
    _set_appendix_style()
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 13,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(6.35, 5.25))
    for budget in ("borough", "city"):
        color = COLORS[budget]
        for scenario, linestyle, marker, label in (
            (
                "primary",
                "-",
                "o",
                f"{budget.title()} budget, historical capacity",
            ),
            (
                "capacity125",
                "-",
                "D",
                f"{budget.title()} budget, 1.25× capacity",
            ),
        ):
            group = frame.loc[frame["budget_group"] == budget].copy()
            x, y = "_efficiency_ratio", "_equity_ratio"
            group[x] = group[f"efficiency_mean_{scenario}"] / efficiency_historical
            group[y] = group[f"allrequest_equity_mean_{scenario}"] / equity_historical
            if scenario == "primary":
                ax.scatter(
                    group[x], group[y], s=12, color=color, alpha=0.16,
                    edgecolors="none", zorder=1,
                )
            else:
                ax.scatter(
                    group[x], group[y], s=17, facecolors="none", edgecolors=color,
                    linewidths=0.8, alpha=0.32, marker=marker, zorder=1,
                )
            frontier = _pareto(group, x, y)
            ax.plot(
                frontier[x], frontier[y], color=color, linestyle=linestyle, marker=marker,
                markerfacecolor=color if scenario == "primary" else "white",
                markeredgecolor=color, markeredgewidth=1.0,
                markersize=3.6, linewidth=2.0, label=label, zorder=3,
            )
    ax.scatter(
        [1.0], [1.0], s=58, color=COLORS["historical"], edgecolors="none",
        label="Historical inspections", zorder=5,
    )
    ax.set(
        xlabel="Efficiency Loss (relative to historical)",
        ylabel="All-request Equity Loss (relative to historical)",
        xlim=(0.40, 1.20),
        ylim=(0.0, 1.25),
    )
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [handles[-1], *handles[:-1]], [labels[-1], *labels[:-1]],
        frameon=True, framealpha=0.92, fontsize=8.4, ncol=2,
        loc="lower center", bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    return _save_pair(
        fig, output_dir / "fixed_support_capacity125_tradeoff_comparison_2026-08-04"
    )


def _capacity_figure(data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        return _capacity_figure_in_clean_context(data_dir, output_dir)


def _load_progress(data_dir: Path) -> pd.DataFrame:
    required = {
        "arm_name", "budget_group", "search_mode", "global_batch", "observations",
        "target_observations", "endpoint_best_loss", "hypervolume_pct_final", "elapsed_hours",
    }
    frame = _read(data_dir / "primary" / "bo_search_progress.csv", required)
    expected = {
        (mode, budget)
        for mode in ("efficiency", "equity", "qnehvi")
        for budget in ("borough", "city")
    }
    if set(zip(frame["search_mode"], frame["budget_group"])) != expected:
        raise ValueError("BO progress lacks the six primary search arms")
    numeric = [
        "global_batch", "observations", "target_observations",
        "hypervolume_pct_final", "elapsed_hours",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    frame["endpoint_best_loss"] = pd.to_numeric(
        frame["endpoint_best_loss"].replace("", np.nan), errors="raise"
    )
    if frame[["global_batch", "observations", "target_observations", "hypervolume_pct_final", "elapsed_hours"]].isna().any().any():
        raise ValueError("BO progress contains missing required values")
    for (mode, budget), group in frame.groupby(["search_mode", "budget_group"]):
        group = group.sort_values("observations")
        final = 2304 if mode == "qnehvi" else 3200
        expected_observations = list(range(64, final + 1, 64))
        if group["observations"].astype(int).tolist() != expected_observations or group[
            "global_batch"
        ].astype(int).tolist() != list(range(len(group))):
            raise ValueError(f"BO progress has an incomplete batch grid for {budget}/{mode}")
        if set(group["target_observations"].astype(int)) != {3200}:
            raise ValueError("BO progress target changed")
        if (group["elapsed_hours"] < 0).any() or not group["elapsed_hours"].is_monotonic_increasing:
            raise ValueError("BO elapsed time must be nonnegative and cumulative")
        if not group["hypervolume_pct_final"].is_monotonic_increasing or not np.isclose(
            group["hypervolume_pct_final"].iloc[-1], 100.0
        ):
            raise ValueError("BO hypervolume must be monotone and end at 100 percent")
        endpoint = group["endpoint_best_loss"]
        if mode == "qnehvi":
            if endpoint.notna().any():
                raise ValueError("qNEHVI progress must not claim a scalar endpoint loss")
        elif endpoint.isna().any() or (endpoint <= 0).any() or not endpoint.is_monotonic_decreasing:
            raise ValueError("Endpoint-search best loss must be positive and monotone")
    return frame


def _progress_figure(data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    frame = _load_progress(data_dir)
    _set_appendix_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.45))
    panels = (
        ("efficiency", "Efficiency-targeted search", "Efficiency loss (millions)"),
        ("equity", "Equity-targeted search", "Equity loss"),
    )
    for ax, (mode, title, ylabel) in zip(axes[:2], panels):
        for budget in ("borough", "city"):
            group = frame.loc[
                (frame["search_mode"] == mode) & (frame["budget_group"] == budget)
            ].sort_values("observations")
            ax.plot(
                group["observations"], group["endpoint_best_loss"], color=COLORS[budget],
                linewidth=2.0, label=f"{budget.title()} Budget",
            )
        ax.set_title(title)
        ax.set_xlabel("Policy evaluations")
        ax.set_ylabel(ylabel)
        _style_axis(ax)
    for budget in ("borough", "city"):
        group = frame.loc[
            (frame["search_mode"] == "qnehvi") & (frame["budget_group"] == budget)
        ].sort_values("observations")
        axes[2].plot(
            group["observations"], group["hypervolume_pct_final"], color=COLORS[budget],
            linewidth=2.0, label=f"{budget.title()} Budget",
        )
    axes[2].set_title("Frontier search (qNEHVI)")
    axes[2].set_xlabel("Policy evaluations")
    axes[2].set_ylabel("Hypervolume (% of final value)")
    _style_axis(axes[2])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=True, fontsize=8.0, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.26, top=0.88, wspace=0.38)
    output_dir.mkdir(parents=True, exist_ok=True)
    return _save_pair(fig, output_dir / "primary_bo_search_progress_2026-08-06")


def build_figure(data_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    primary = data_dir / "primary"
    search = _read(
        primary / "search_cloud.csv",
        {
            "setting_id",
            "budget_group",
            "arm_name",
            "source_index",
            "efficiency_ratio_historical",
            "equity_ratio_historical",
            "plot_search_nondominated",
        },
    )
    selection = _read(
        primary / "selection_means.csv",
        {
            "setting_id",
            "policy_id",
            "budget_group",
            "efficiency_pct_historical",
            "equity_pct_historical",
        },
    )
    inference = _read(
        primary / "inference_means.csv",
        {
            "budget_group",
            "display_roles",
            "n_seeds",
            "policy_id",
            "selection_efficiency_endpoint",
            "efficiency_pct_historical",
            "equity_pct_historical",
        },
    )
    settings = set(search["setting_id"].astype(str))
    if settings != {"d100_delay_median_cap1"} or set(
        selection["setting_id"].astype(str)
    ) != settings:
        raise ValueError("Search and selection files must use the primary D=100 setting")
    setting = next(iter(settings))
    if not inference["policy_id"].astype(str).str.contains(setting, regex=False).all():
        raise ValueError("Inference policy IDs do not match the primary setting")
    search, selection, inference = (
        _ratios(search, "search_cloud.csv"),
        _ratios(selection, "selection_means.csv"),
        _ratios(inference, "inference_means.csv"),
    )
    if search.groupby("budget_group").size().to_dict() != {
        "borough": 2304,
        "city": 2304,
    } or search.duplicated(["budget_group", "source_index"]).any():
        raise ValueError("Search cloud must contain 2,304 unique rows per budget group")
    if selection.groupby("budget_group").size().to_dict() != {
        "borough": 274,
        "city": 264,
    } or selection.duplicated(["budget_group", "policy_id"]).any():
        raise ValueError("Selection means do not have the locked policy support")
    if inference.groupby("budget_group").size().to_dict() != {
        "borough": 6,
        "city": 6,
    } or inference.duplicated(["budget_group", "policy_id"]).any():
        raise ValueError("Inference means must contain six unique policies per budget group")
    if set(inference["n_seeds"].astype(int)) != {25}:
        raise ValueError("Inference means must use the declared 25 seeds")
    for name, frame in (("search", search), ("selection", selection), ("inference", inference)):
        if set(frame["budget_group"].astype(str)) != {"borough", "city"}:
            raise ValueError(f"{name} needs Borough and City evidence")
    hazard = _load_hazard(primary / "hazard_histogram_bins.csv")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 13,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": False,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.use14corefonts": False,
            "text.usetex": False,
            "mathtext.fontset": "dejavusans",
        }
    )
    fig = plt.figure(figsize=(10.14, 6.3))
    outer = fig.add_gridspec(
        1, 2, width_ratios=[1.90, 1.0], wspace=0.14,
        left=0.07, right=0.985, bottom=0.11, top=0.91,
    )
    ax = fig.add_subplot(outer[0, 0])
    x, y = "efficiency_ratio_historical", "equity_ratio_historical"
    endpoints = []
    for budget in ("borough", "city"):
        color = COLORS[budget]
        cloud = search.loc[search["budget_group"].astype(str) == budget]
        selected = selection.loc[selection["budget_group"].astype(str) == budget]
        locked = inference.loc[inference["budget_group"].astype(str) == budget]
        ax.scatter(cloud[x], cloud[y], s=5, color=color, alpha=0.055,
                   edgecolors="none", zorder=1)
        ax.scatter(selected[x], selected[y], s=10, color=color, alpha=0.16,
                   edgecolors="none", zorder=2)
        ax.scatter(locked[x], locked[y], s=14, color=color, alpha=0.34,
                   edgecolors="none", zorder=3)
        frontier = _pareto(locked)
        ax.plot(
            frontier[x], frontier[y], color=color, marker="o", markersize=3.2,
            linewidth=2.0, label=f"Pareto frontier of {budget.title()}-budget policy",
            zorder=4,
        )
        endpoint_flag = locked["selection_efficiency_endpoint"].astype(str).str.lower()
        if not endpoint_flag.isin({"true", "false"}).all():
            raise ValueError("Inference endpoint flags are not Boolean")
        endpoint = locked.loc[endpoint_flag == "true"]
        role_endpoint = locked["display_roles"].astype(str) == "most_efficient"
        if not endpoint.index.equals(locked.index[role_endpoint]):
            raise ValueError("Most-efficient role and endpoint flag disagree")
        if len(endpoint) != 1:
            raise ValueError(f"Expected one most-efficient {budget} inference row")
        ax.scatter(endpoint[x], endpoint[y], s=72, color=color, edgecolors="black",
                   linewidths=0.9, zorder=7)
        endpoints.append((budget, float(endpoint.iloc[0][x]), float(endpoint.iloc[0][y])))
    ax.scatter([1.0], [1.0], s=58, color=COLORS["historical"], edgecolors="none",
               label="Historical inspections", zorder=6)
    ax.set(
        xlabel="Efficiency Loss (relative to historical)",
        ylabel="All-request Equity Loss (relative to historical)",
        xlim=(0.70, 1.20), ylim=(0.0, 1.25),
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1], *handles[:-1]], [labels[-1], *labels[:-1]],
              loc="upper left", fontsize=10.5, framealpha=0.92)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="out", width=0.8, length=3.0)
    for _, endpoint_x, endpoint_y in endpoints:
        ax.plot([endpoint_x, 1.195, 1.195], [endpoint_y, endpoint_y, 0.70],
                color="black", linewidth=0.9, linestyle="--", zorder=6)
    ax.annotate("", xy=(1.225, 0.70), xytext=(1.195, 0.70),
                arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.9,
                            "linestyle": "--", "mutation_scale": 12},
                annotation_clip=False)

    subgrid = outer[0, 1].subgridspec(len(BOROUGHS), 1, hspace=0.32)
    axes = []
    for index, borough in enumerate(BOROUGHS):
        panel = fig.add_subplot(subgrid[index, 0], sharex=axes[0] if axes else None)
        axes.append(panel)
        for kind in ("city", "borough", "historical"):
            group = hazard.loc[
                (hazard["borough"].astype(str) == borough) & (hazard["role_kind"] == kind)
            ].sort_values("bin_index")
            left = group["bin_left_days"].to_numpy(float)
            right = group["bin_right_days"].to_numpy(float)
            values = group["display_share_percent"].to_numpy(float)
            edges = np.r_[left, right[-1]]
            label = ("Historical" if kind == "historical" else f"{kind.title()} Budget") if index == 4 else None
            if kind == "historical":
                panel.stairs(values, edges, color=COLORS[kind], linewidth=1.6, label=label, zorder=4)
            else:
                panel.bar(left, values, width=right - left, align="edge", color=HIST_COLORS[kind], alpha=0.50, label=label)
        panel.text(0.23, 0.55, "Staten\nIsland" if borough == "Staten Island" else borough,
                   transform=panel.transAxes, ha="center", va="center", fontsize=12)
        panel.set(xlim=(0, 21), ylim=(0, 100), yticks=[0, 100], yticklabels=["0%", "100%"])
        panel.spines[["top", "right"]].set_visible(False)
        panel.tick_params(axis="both", which="both", direction="out", width=0.8, length=3.0)
        if index == 0:
            panel.set_title("Hazard Inspection\nOutcomes", pad=7)
        if index < 4:
            panel.tick_params(labelbottom=False)
        else:
            panel.set_xticks([0, 5, 10, 20], ["0", "5", "10", "$\\geq 20$ or\nnot inspected"])
            panel.get_xticklabels()[-1].set_ha("right")
            panel.set_xlabel("Inspection delay (days)")
            handles, labels = panel.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            order = ("City Budget", "Borough Budget", "Historical")
            panel.legend([by_label[label] for label in order], order,
                         loc="upper right", fontsize=8.8, framealpha=0.92)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "pareto_frontier_with_hazard_delay_allrequest"
    pdf_path, png_path = base.with_suffix(".pdf"), base.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return (
        pdf_path,
        png_path,
        *_capacity_figure(data_dir, output_dir),
        *_progress_figure(data_dir, output_dir),
    )
