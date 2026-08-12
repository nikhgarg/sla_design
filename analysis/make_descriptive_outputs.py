"""Build active descriptive manuscript outputs from compact public aggregates."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BOROUGHS = ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")
INTRO_BOROUGHS = ("Bronx", "Manhattan")
CATEGORIES = (
    "Hazard",
    "Illegal Tree Damage",
    "Other",
    "Prune",
    "Remove Tree",
    "Root/Sewer/Sidewalk",
)
YEARS = (2019, 2020, 2021, 2022)
ANNUAL_REQUEST_TOTALS = (75_076, 139_084, 82_381, 76_800)
ANNUAL_INSPECTION_TOTALS = (48_340, 102_217, 56_221, 51_976)
YEAR_COLORS = ("#440154", "#31688e", "#35b779", "#fde725")
INTRO_COLORS = {"Bronx": "#1f77b4", "Manhattan": "#ff7f0e"}
DELAY_SPECS = {
    "Hazard": (14, "hazard_delay_hist_allrequest_2026-08-04"),
    "Illegal Tree Damage": (8, "itd_delay_hist_allrequest_2026-08-04"),
}


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if list(frame.columns) != columns:
        raise ValueError(f"{path.name} has unexpected columns: {list(frame.columns)}")
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    return frame


def _setup_style(font_size: float) -> None:
    try:
        import seaborn as sns

        sns.set_theme(
            style="white",
            context="paper",
            rc={
                "axes.grid": False,
                "axes.spines.top": False,
                "axes.spines.right": False,
            },
        )
    except ImportError:
        pass
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.use14corefonts": False,
            "text.usetex": False,
            "mathtext.fontset": "dejavusans",
        }
    )


def _style_axis(axis) -> None:
    axis.set_facecolor("white")
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_color("0.2")
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(
        axis="both", which="both", direction="out", color="black",
        width=0.8, length=3.0,
    )


def _save(fig, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(
        pdf,
        metadata={
            "Creator": "analysis/make_descriptive_outputs.py",
            "Title": stem,
            "Subject": "SLA descriptive manuscript output from compact public data",
        },
    )
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def _load_delays(bin_path: Path, summary_path: Path) -> pd.DataFrame:
    columns = [
        "category", "borough", "bin_left_days", "bin_right_days",
        "right_edge_inclusive", "row_count", "density_per_day",
    ]
    bins = _read(bin_path, columns)
    numeric = ["bin_left_days", "bin_right_days", "row_count", "density_per_day"]
    bins[numeric] = bins[numeric].apply(pd.to_numeric, errors="raise")
    if bins.duplicated(["category", "borough", "bin_left_days"]).any():
        raise ValueError("Delay histogram contains duplicate cells")
    if set(bins["category"].astype(str)) != set(CATEGORIES):
        raise ValueError("Delay histogram does not contain the six modeled categories")
    if set(bins["borough"].astype(str)) != set(INTRO_BOROUGHS):
        raise ValueError("Delay histogram does not contain Bronx and Manhattan")
    for key, group in bins.groupby(["category", "borough"], sort=False):
        ordered = group.sort_values("bin_left_days")
        if not np.array_equal(ordered["bin_left_days"].to_numpy(float), np.arange(50)):
            raise ValueError(f"Delay histogram has invalid left edges for {key}")
        if not np.array_equal(ordered["bin_right_days"].to_numpy(float), np.arange(1, 51)):
            raise ValueError(f"Delay histogram has invalid right edges for {key}")
        counts = ordered["row_count"].to_numpy(float)
        density = ordered["density_per_day"].to_numpy(float)
        if (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
            raise ValueError(f"Delay histogram has invalid counts for {key}")
        if counts.sum() <= 0 or not np.allclose(density, counts / counts.sum()):
            raise ValueError(f"Delay histogram densities do not reconcile for {key}")

    summary_columns = [
        "category", "borough", "matching_rows", "unique_service_requests",
        "repeated_rows", "valid_delay_rows", "negative_delay_rows",
        "displayed_delay_rows", "above_display_range_rows",
        "display_upper_edge_days", "first_day_density",
        "median_displayed_delay_days", "mean_displayed_delay_days",
        "documented_sla_days",
    ]
    summary = _read(summary_path, summary_columns)
    expected = [(category, borough) for category in CATEGORIES for borough in INTRO_BOROUGHS]
    if list(zip(summary["category"].astype(str), summary["borough"].astype(str))) != expected:
        raise ValueError("Delay summary has the wrong category/Borough grid")
    numeric = summary_columns[2:-1]
    summary[numeric] = summary[numeric].apply(pd.to_numeric, errors="raise")
    summary["documented_sla_days"] = pd.to_numeric(
        summary["documented_sla_days"], errors="coerce"
    )
    by_key = summary.set_index(["category", "borough"])
    for key, group in bins.groupby(["category", "borough"], sort=False):
        row = by_key.loc[key]
        if (
            row["matching_rows"] != row["unique_service_requests"] + row["repeated_rows"]
            or row["valid_delay_rows"]
            != row["negative_delay_rows"] + row["displayed_delay_rows"] + row["above_display_range_rows"]
            or row["displayed_delay_rows"] != group["row_count"].sum()
            or row["display_upper_edge_days"] != 50
            or not np.isclose(row["first_day_density"], group.loc[group["bin_left_days"].eq(0), "density_per_day"].iloc[0])
        ):
            raise ValueError(f"Delay summary does not reconcile for {key}")
        expected_sla = DELAY_SPECS.get(str(key[0]), (np.nan,))[0]
        if not (np.isnan(expected_sla) and np.isnan(row["documented_sla_days"])) and row["documented_sla_days"] != expected_sla:
            raise ValueError(f"Delay SLA metadata does not reconcile for {key}")
    return bins


def _plot_delays(bins: pd.DataFrame, output_dir: Path) -> list[Path]:
    _setup_style(18.0)
    outputs = []
    for category, (sla, stem) in DELAY_SPECS.items():
        fig, axis = plt.subplots(figsize=(6.4, 4.8))
        category_rows = bins.loc[bins["category"].astype(str).eq(category)]
        for borough in INTRO_BOROUGHS:
            rows = category_rows.loc[
                category_rows["borough"].astype(str).eq(borough)
            ].sort_values("bin_left_days")
            axis.bar(
                rows["bin_left_days"], rows["density_per_day"], width=1.0,
                align="edge", alpha=0.5, color=INTRO_COLORS[borough],
                edgecolor="white", linewidth=0.45, label=borough,
                zorder=2 if borough == "Bronx" else 1,
            )
        maximum = float(category_rows["density_per_day"].max())
        axis.axvline(sla, color="red", linestyle="--", linewidth=1.2)
        axis.text(sla + 1.0, maximum * 0.90, "SLA", color="red", ha="left", va="center")
        axis.set(xlim=(0, 50), ylim=(0, maximum * 1.16))
        axis.set_xticks(np.arange(0, 51, 10))
        axis.set_xlabel(f"Inspection delay, {category}\nincidents (days)")
        axis.set_ylabel("Density")
        axis.legend(title="Borough", title_fontsize=18.0, loc="upper right", frameon=False)
        _style_axis(axis)
        fig.subplots_adjust(left=0.20, right=0.95, bottom=0.23, top=0.96)
        outputs.extend(_save(fig, output_dir, stem))
    return outputs


def _load_weekly(path: Path) -> pd.DataFrame:
    frame = _read(path, ["year", "week", "merged_row_count"])
    frame[["year", "week", "merged_row_count"]] = frame[
        ["year", "week", "merged_row_count"]
    ].apply(pd.to_numeric, errors="raise")
    if frame.duplicated(["year", "week"]).any() or set(frame["year"].astype(int)) != set(YEARS):
        raise ValueError(f"{path.name} must have one row per observed ISO week in 2019--2022")
    expected_weeks = {2019: 52, 2020: 53, 2021: 52, 2022: 52}
    for year, count in expected_weeks.items():
        weeks = frame.loc[frame["year"].eq(year), "week"].astype(int).tolist()
        if weeks != list(range(1, count + 1)):
            raise ValueError(f"{path.name} has invalid ISO weeks for {year}")
    counts = frame["merged_row_count"].to_numpy(float)
    if (counts <= 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError(f"{path.name} contains invalid row counts")
    return frame


def _plot_weekly(
    weekly: pd.DataFrame,
    *,
    ylabel: str,
    ylim: tuple[float, float],
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(4.667, 3.468))
    for year, color in zip(YEARS, YEAR_COLORS):
        rows = weekly.loc[weekly["year"].eq(year)].sort_values("week")
        axis.plot(rows["week"], rows["merged_row_count"], color=color, linewidth=1.5, label=str(year))
    axis.set_yscale("log")
    axis.set(xlim=(0, 55), ylim=ylim)
    axis.set_xticks([0, 10, 20, 30, 40, 50])
    axis.set_xlabel("Week of year")
    axis.set_ylabel(ylabel)
    axis.legend(title="Year", loc="upper right", frameon=True)
    _style_axis(axis)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.20, top=0.97)
    return _save(fig, output_dir, stem)


def _load_hazard_risk(count_path: Path, summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = _read(count_path, ["BoroughCode", "risk_rating", "row_count"])
    summary_columns = [
        "BoroughCode", "row_count", "unique_service_requests", "mean_risk_rating",
        "median_risk_rating", "min_risk_rating", "max_risk_rating", "repeated_rows",
    ]
    summary = _read(summary_path, summary_columns)
    counts[["risk_rating", "row_count"]] = counts[["risk_rating", "row_count"]].apply(
        pd.to_numeric, errors="raise"
    )
    summary[summary_columns[1:]] = summary[summary_columns[1:]].apply(pd.to_numeric, errors="raise")
    if counts.duplicated(["BoroughCode", "risk_rating"]).any():
        raise ValueError("Hazard risk counts contain duplicate cells")
    if counts["BoroughCode"].drop_duplicates().astype(str).tolist() != list(BOROUGHS):
        raise ValueError("Hazard risk counts have the wrong Borough order")
    if summary["BoroughCode"].astype(str).tolist() != list(BOROUGHS):
        raise ValueError("Hazard risk summary has the wrong Borough order")
    indexed = summary.set_index("BoroughCode")
    for borough in BOROUGHS:
        rows = counts.loc[counts["BoroughCode"].astype(str).eq(borough)].sort_values("risk_rating")
        if not np.array_equal(rows["risk_rating"].to_numpy(int), np.arange(13)):
            raise ValueError(f"Hazard risk counts have invalid rating support for {borough}")
        values = rows["row_count"].to_numpy(float)
        if (values < 0).any() or values.sum() != indexed.loc[borough, "row_count"]:
            raise ValueError(f"Hazard risk counts do not reconcile for {borough}")
        mean = np.average(rows["risk_rating"], weights=values)
        if not np.isclose(mean, indexed.loc[borough, "mean_risk_rating"]):
            raise ValueError(f"Hazard risk mean does not reconcile for {borough}")
        if indexed.loc[borough, "row_count"] - indexed.loc[borough, "unique_service_requests"] != indexed.loc[borough, "repeated_rows"]:
            raise ValueError(f"Hazard risk repeated rows do not reconcile for {borough}")
    return counts, summary


def _plot_hazard_risk(counts: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(
        1, len(BOROUGHS), figsize=(8.434, 4.843), sharey=True,
        gridspec_kw={"wspace": 0.045},
    )
    means = summary.set_index("BoroughCode")["mean_risk_rating"]
    for index, (axis, borough) in enumerate(zip(axes, BOROUGHS)):
        rows = counts.loc[counts["BoroughCode"].astype(str).eq(borough)].sort_values("risk_rating")
        axis.barh(rows["risk_rating"], rows["row_count"], height=0.08, color="#0000ff", alpha=0.70, linewidth=0)
        axis.axhline(float(means.loc[borough]), color="#d62728", alpha=0.80, linestyle="--", linewidth=1.2)
        axis.set_title(borough)
        axis.set_xlabel("Frequency")
        axis.set_ylim(-0.5, 12.5)
        axis.set_yticks(range(0, 13, 2))
        axis.margins(x=0)
        _style_axis(axis)
        if index:
            axis.tick_params(axis="y", labelleft=False, length=0)
    axes[0].set_ylabel("Risk Rating")
    axes[-1].legend(
        handles=[Line2D([0], [0], color="#d62728", alpha=0.80, linestyle="--", label="Mean")],
        loc="lower left", frameon=True,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.18, top=0.90, wspace=0.045)
    return _save(fig, output_dir, "hazard_risk_rating_distplot_allrequest_2026-08-04")


def _annual_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    lines = [
        "% AUTO-GENERATED FILE. DO NOT EDIT BY HAND.",
        "% Generated from compact public descriptive data.",
        "\\begin{table}[tb]", "\\small", "\\centering",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        " & " + " & ".join(BOROUGHS) + r" \\", "\\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(f"{int(row[0])} & " + " & ".join(str(int(value)) for value in row[1:]) + r" \\")
    lines.extend([
        "\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}",
        f"\\label{{{label}}}", "\\end{table}", "",
    ])
    return "\n".join(lines)


def _load_annual(path: Path, expected_totals: tuple[int, ...]) -> pd.DataFrame:
    frame = _read(path, ["year", *BOROUGHS])
    frame[["year", *BOROUGHS]] = frame[["year", *BOROUGHS]].apply(pd.to_numeric, errors="raise")
    if frame["year"].astype(int).tolist() != list(YEARS):
        raise ValueError(f"{path.name} has the wrong year order")
    values = frame[list(BOROUGHS)].to_numpy(float)
    if (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{path.name} contains invalid counts")
    totals = frame[list(BOROUGHS)].sum(axis=1).astype(int).tolist()
    if totals != list(expected_totals):
        raise ValueError(
            f"{path.name} has annual totals {totals}, expected {list(expected_totals)}"
        )
    return frame


def _risk_priority_table(frame: pd.DataFrame) -> str:
    lines = [
        "% AUTO-GENERATED FILE. DO NOT EDIT BY HAND.",
        "% Generated from compact public descriptive data.",
        "\\begin{table}[tb]", "\\small", "\\centering",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        " & " + " & ".join(BOROUGHS) + r" & Assigned category priority weights\\",
        "\\midrule",
    ]
    for row in frame.to_dict("records"):
        values = " & ".join(f"{float(row[borough]):.2f}" for borough in BOROUGHS)
        weight = float(row["Assigned category priority weight"])
        lines.append(f"{row['category']} & {values} & {weight:g}" + r" \\")
    lines.extend([
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Mean observed risk rating among linked inspected-and-rated rows by incident category and Borough, 2019--2022, alongside the assigned category priority weights used in the loss functions. Recorded risk ratings and assigned policy weights are distinct objects; the assigned weights are model inputs, not request-level ex ante measurements.}",
        "\\label{tab:avgrisk}", "\\end{table}", "",
    ])
    return "\n".join(lines)


def _load_risk_priority(path: Path) -> pd.DataFrame:
    columns = ["category", *BOROUGHS, "Assigned category priority weight"]
    frame = _read(path, columns)
    if frame["category"].astype(str).tolist() != list(CATEGORIES):
        raise ValueError("Risk-priority table has the wrong category order")
    numeric = [*BOROUGHS, "Assigned category priority weight"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    values = frame[numeric].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any() or (frame[list(BOROUGHS)] > 12).any().any():
        raise ValueError("Risk-priority table contains invalid values")
    return frame


def build_descriptive_outputs(data_dir: Path, output_dir: Path) -> list[Path]:
    """Build five figures and three LaTeX tables under ``output_dir``."""
    source = data_dir / "descriptive"
    figure_dir, table_dir = output_dir / "figures", output_dir / "tables"
    outputs = _plot_delays(
        _load_delays(
            source / "observed_delay_histogram_bins.csv",
            source / "observed_delay_histogram_summary.csv",
        ),
        figure_dir,
    )

    _setup_style(10.5)
    outputs.extend(
        _plot_weekly(
            _load_weekly(source / "weekly_service_requests.csv"),
            ylabel="Number of service requests", ylim=(100, 100_000),
            output_dir=figure_dir, stem="srs_per_week_allrequest_2026-08-04",
        )
    )
    outputs.extend(
        _plot_weekly(
            _load_weekly(source / "weekly_inspections.csv"),
            ylabel="Number of inspections", ylim=(100, 30_000),
            output_dir=figure_dir, stem="inspections_per_week_allrequest_2026-08-04",
        )
    )
    outputs.extend(
        _plot_hazard_risk(
            *_load_hazard_risk(
                source / "hazard_risk_counts.csv", source / "hazard_risk_summary.csv"
            ),
            figure_dir,
        )
    )

    table_dir.mkdir(parents=True, exist_ok=True)
    request = table_dir / "appendix_annual_request_rows.tex"
    request.write_text(
        _annual_table(
            _load_annual(
                source / "annual_simulation_request_rows_by_borough.csv",
                ANNUAL_REQUEST_TOTALS,
            ),
            "Number of requests in each Borough in each year (for the categories in our simulation). Requests linked to multiple inspections are duplicated.",
            "tab:number_srs_by_borough_year",
        ),
        encoding="utf-8",
    )
    inspection = table_dir / "appendix_annual_inspection_rows.tex"
    inspection.write_text(
        _annual_table(
            _load_annual(
                source / "annual_simulation_inspection_rows_by_borough.csv",
                ANNUAL_INSPECTION_TOTALS,
            ),
            "Number of completed inspections by Borough in each year (for the categories in our simulation).",
            "tab:number_inspections_by_borough_year",
        ),
        encoding="utf-8",
    )
    risk = table_dir / "appendix_risk_priority.tex"
    risk.write_text(
        _risk_priority_table(
            _load_risk_priority(source / "observed_risk_and_priority_by_category.csv")
        ),
        encoding="utf-8",
    )
    outputs.extend([request, inspection, risk])
    return outputs
