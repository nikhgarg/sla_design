"""Build the manuscript policy-performance table from compact public scores."""

from pathlib import Path

import numpy as np
import pandas as pd


ROLES = ("most_efficient", "most_equitable", "balanced")
BOROUGHS = ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")
CATEGORIES = (
    "Hazard",
    "Illegal Tree Damage",
    "Other",
    "Prune",
    "Remove Tree",
    "Root/Sewer/Sidewalk",
)
ROLE_LABELS = {
    "most_efficient": "Most efficient",
    "most_equitable": "Most equitable",
    "balanced": "Balanced",
}
CELL_ROLES = (
    "historical_2019",
    "borough_most_efficient",
    "borough_most_equitable",
    "borough_balanced",
)
CELL_PREFIXES = dict(zip(CELL_ROLES, ("historical", "most_efficient", "most_equitable", "balanced")))
COMBINED_CELL_COLUMNS = (
    "borough",
    "category",
    *(
        column
        for prefix in CELL_PREFIXES.values()
        for column in (
            f"{prefix}_realized_median_inspection_delay_days",
            f"{prefix}_inspection_fraction",
        )
    ),
)
BOROUGH_TABLE_METADATA = {
    "Bronx": (
        "bronx",
        "SLAs and inspection fractions in the Bronx, under proposed Borough-budget "
        "policies, compared with outcomes of historical inspections. Values are means "
        "across 25 simulations.",
        "tab:bronxsla",
    ),
    "Brooklyn": (
        "brooklyn",
        "SLAs and inspection fractions in Brooklyn, under the optimal Borough budget "
        "policies, compared with outcomes of historical inspections. Values are means "
        "across 25 simulations.",
        "tab:brooklyn",
    ),
    "Manhattan": (
        "manhattan",
        "SLAs and inspection fractions in Manhattan, under proposed Borough-budget "
        "policies, compared with outcomes of historical inspections. Values are means "
        "across 25 simulations.",
        "tab:manhattan",
    ),
    "Queens": (
        "queens",
        "SLAs and inspection fractions in Queens, under proposed Borough-budget "
        "policies, compared with outcomes of historical inspections. Values are means "
        "across 25 simulations.",
        "tab:queens",
    ),
    "Staten Island": (
        "staten_island",
        "SLAs and inspection fractions in Staten Island, under proposed Borough-budget "
        "policies, compared with outcomes of historical inspections. Values are means "
        "across 25 simulations.",
        "tab:staten",
    ),
}
YEARS = (2019, 2021, 2022)
REQUIRED = {
    "calendar_year",
    "random_seed",
    "display_role",
    "policy_id",
    "vector_hash",
    "efficiency_loss",
    "equity_loss",
    "historical_efficiency_loss",
    "historical_equity_loss",
    "efficiency_ratio_historical",
    "equity_ratio_historical",
}


def _load_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED - set(scores.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
    if scores.empty:
        raise ValueError(f"{path.name} is empty")

    numeric = [
        "calendar_year",
        "random_seed",
        "efficiency_loss",
        "equity_loss",
        "historical_efficiency_loss",
        "historical_equity_loss",
        "efficiency_ratio_historical",
        "equity_ratio_historical",
    ]
    scores[numeric] = scores[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(scores[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Policy-score numeric fields must be finite")
    if set(scores["calendar_year"].astype(int)) != set(YEARS):
        raise ValueError(f"Expected calendar years {YEARS}")
    if set(scores["display_role"].astype(str)) != set(ROLES):
        raise ValueError(f"Expected display roles {ROLES}")
    if scores.duplicated(["calendar_year", "random_seed", "display_role"]).any():
        raise ValueError("Policy scores contain duplicate year/seed/role rows")

    for role, group in scores.groupby("display_role"):
        if group["policy_id"].nunique() != 1 or group["vector_hash"].nunique() != 1:
            raise ValueError(f"{role} does not identify one fixed policy vector")
    for (year, seed), group in scores.groupby(["calendar_year", "random_seed"]):
        if set(group["display_role"].astype(str)) != set(ROLES):
            raise ValueError(f"Roles do not share seeds in {year}/{seed}")
        for column in ("historical_efficiency_loss", "historical_equity_loss"):
            if group[column].nunique() != 1:
                raise ValueError(f"Historical reference differs by role in {year}/{seed}")

    for metric in ("efficiency", "equity"):
        reference = scores[f"historical_{metric}_loss"]
        if (reference <= 0).any() or (scores[f"{metric}_loss"] < 0).any():
            raise ValueError(f"{metric.title()} losses need positive references")
        recomputed = scores[f"{metric}_loss"] / reference
        reported = scores[f"{metric}_ratio_historical"]
        if not np.allclose(recomputed, reported, rtol=1e-9, atol=1e-12):
            raise ValueError(f"Reported {metric} ratios do not match score/reference")
    return scores


def _budget_table(data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    columns = (
        "Borough",
        "fraction_of_2019_requests",
        "historical_eventual_inspection_share",
        "borough_most_efficient_budget_share",
        "borough_most_equitable_budget_share",
        "balanced_budget_share",
    )
    frame = pd.read_csv(
        data_dir / "primary" / "budget_allocation.csv", float_precision="round_trip"
    )
    if tuple(frame.columns) != columns or frame["Borough"].tolist() != list(BOROUGHS):
        raise ValueError("Budget allocation does not have the manuscript schema and order")
    values = frame[list(columns[1:])].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all() or not values.apply(
        lambda column: column.between(0, 1).all()
    ).all():
        raise ValueError("Budget shares must be finite and between zero and one")
    if not np.allclose(values.sum(axis=0), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Every budget-allocation column must sum to one")

    rows = []
    for _, row in frame.iterrows():
        borough = "The Bronx" if row["Borough"] == "Bronx" else row["Borough"]
        rows.append(
            f"{borough} & "
            + " & ".join(f"{float(row[column]):.3f}" for column in columns[1:])
            + r" \\"
        )
    tex = "\n".join(
        [
            "% AUTO-GENERATED FILE. DO NOT EDIT BY HAND.",
            "% Generated from paper_data/primary/budget_allocation.csv.",
            r"\begin{table}[tb]",
            r"\centering",
            r"\small",
            r"\resizebox{.8\textwidth}{!}{%",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"\multicolumn{1}{c}{\multirow{2}{*}{Borough}} &",
            r"  \multirow{2}{*}{\begin{tabular}[c]{@{}c@{}}Fraction of\\ inspection requests\end{tabular}} &",
            r"  \multicolumn{4}{c}{Budget allocation} \\ \cmidrule{3-6}",
            r"\multicolumn{1}{c}{} &",
            r"  & Historical & Most Efficient & Most Equitable & Balanced\\ \midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\caption{Comparison of the historical allocation of inspections, and the budget allocations of the most efficient, equitable, and Balanced Borough budget policies with the fraction of inspection requests received in each Borough in 2019.}",
            r"\label{tab:budgetalloc}",
            r"\end{table}",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "main_budget_allocation.csv"
    tex_path = output_dir / "main_budget_allocation.tex"
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(tex)
    return csv_path, tex_path


def _format_historical_sla(value: float) -> str:
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}"


def _borough_tables(data_dir: Path, output_dir: Path) -> list[Path]:
    required = {
        "scope", "role", "policy_id", "vector_hash", "budget_group", "display_role",
        "borough", "category", "sla_days_mean", "inspection_fraction_mean",
        "sla_days_std", "inspection_fraction_std", "n_evaluations", "sla_days_sem",
        "inspection_fraction_sem",
    }
    cells = pd.read_csv(
        data_dir / "primary" / "selected_policy_cells.csv",
        keep_default_na=False,
        float_precision="round_trip",
    )
    if required - set(cells.columns):
        raise ValueError("Selected-policy cells lack the public manuscript schema")
    cells = cells.loc[cells["role"].isin(CELL_ROLES)].copy()
    if len(cells) != len(CELL_ROLES) * len(BOROUGHS) * len(CATEGORIES):
        raise ValueError("Selected-policy cells lack the four-by-five-by-six paper grid")
    if cells.duplicated(["role", "borough", "category"]).any():
        raise ValueError("Selected-policy cells contain duplicate paper-grid rows")
    if set(cells["borough"]) != set(BOROUGHS) or set(cells["category"]) != set(CATEGORIES):
        raise ValueError("Selected-policy cells contain another Borough/category support")
    means = ["sla_days_mean", "inspection_fraction_mean", "n_evaluations"]
    uncertainty = [
        "sla_days_std", "inspection_fraction_std", "sla_days_sem",
        "inspection_fraction_sem",
    ]
    numeric = means + uncertainty
    cells[numeric] = cells[numeric].apply(pd.to_numeric, errors="raise")
    history = cells["role"] == "historical_2019"
    if not np.isfinite(cells[means].to_numpy()).all() or not np.isfinite(
        cells.loc[~history, uncertainty].to_numpy()
    ).all():
        raise ValueError("Selected-policy cell statistics must be finite where defined")
    if (cells["sla_days_mean"] < 0).any() or not cells["inspection_fraction_mean"].between(0, 1).all():
        raise ValueError("Selected-policy cell means are outside their valid ranges")
    if set(cells.loc[history, "n_evaluations"].astype(int)) != {1} or set(
        cells.loc[~history, "n_evaluations"].astype(int)
    ) != {25}:
        raise ValueError("Historical and simulated cells have unexpected evaluation counts")
    for role, group in cells.loc[~history].groupby("role"):
        if group["policy_id"].nunique() != 1 or group["vector_hash"].nunique() != 1:
            raise ValueError(f"{role} does not identify one fixed policy")

    outputs = []
    combined_records = []
    combined_rows = []
    for borough in BOROUGHS:
        subset = cells.loc[cells["borough"] == borough]
        indexed = subset.set_index(["category", "role"])
        rows = []
        records = []
        for category in CATEGORIES:
            record = {"category": category}
            combined_record = {"borough": borough, "category": category}
            rendered = []
            for role in CELL_ROLES:
                cell = indexed.loc[(category, role)]
                prefix = CELL_PREFIXES[role]
                sla = float(cell["sla_days_mean"])
                fraction = float(cell["inspection_fraction_mean"])
                record[f"{prefix}_sla_days"] = sla
                record[f"{prefix}_inspection_fraction"] = fraction
                combined_record[
                    f"{prefix}_realized_median_inspection_delay_days"
                ] = sla
                combined_record[f"{prefix}_inspection_fraction"] = fraction
                rendered.extend(
                    [
                        _format_historical_sla(sla) if role == "historical_2019" else f"{sla:.2f}",
                        f"{fraction:.2f}",
                    ]
                )
            records.append(record)
            combined_records.append(combined_record)
            rows.append(f"{category} & " + " & ".join(rendered) + r" \\")
        combined_rows.append(
            rf"\multicolumn{{9}}{{@{{}}l}}{{\textbf{{{borough}}}}} \\"
        )
        combined_rows.extend(rows)
        if borough != BOROUGHS[-1]:
            combined_rows.append(r"\addlinespace[1.5pt]")
        stem, caption, label = BOROUGH_TABLE_METADATA[borough]
        header = [
            r"\begin{table}[htb]", r"\small", r"\centering",
            r"\resizebox{.8\columnwidth}{!}{%", r"\begin{tabular}{@{}lrrrrrrrr@{}}",
            r"\toprule",
            r"\multicolumn{1}{c}{\multirow{2}{*}{Category}} & \multicolumn{2}{c}{Historical} & \multicolumn{2}{c}{Most Efficient} & \multicolumn{2}{c}{Most Equitable} & \multicolumn{2}{c}{Balanced}\\ \cmidrule(l){2-9}",
            r"\multicolumn{1}{c}{} &",
        ]
        subheads = []
        for index in range(8):
            label_text = r"SLAs\\ (days)" if index % 2 == 0 else r"Inspection\\ Fraction"
            suffix = r" &" if index < 7 else r"\\ \midrule"
            subheads.append(
                r"  \multicolumn{1}{c}{\begin{tabular}[c]{@{}c@{}}"
                + label_text + r"\end{tabular}}" + suffix
            )
        tex = "\n".join(
            [
                "% AUTO-GENERATED FILE. DO NOT EDIT BY HAND.",
                "% Generated from paper_data/primary/selected_policy_cells.csv.",
                *header, *subheads, *rows, r"\bottomrule", r"\end{tabular}%", r"}",
                f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}", "",
            ]
        )
        csv_path = output_dir / f"appendix_borough_sla_{stem}.csv"
        tex_path = output_dir / f"appendix_borough_sla_{stem}.tex"
        pd.DataFrame(records).to_csv(csv_path, index=False)
        tex_path.write_text(tex)
        outputs.extend([csv_path, tex_path])

    combined = pd.DataFrame(combined_records, columns=COMBINED_CELL_COLUMNS)
    expected_keys = [
        (borough, category) for borough in BOROUGHS for category in CATEGORIES
    ]
    if list(combined[["borough", "category"]].itertuples(index=False, name=None)) != expected_keys:
        raise ValueError("Combined Borough table does not have the canonical row order")

    combined_csv = output_dir / "appendix_borough_sla_outcomes.csv"
    combined_tex = output_dir / "appendix_borough_sla_outcomes.tex"
    combined.to_csv(combined_csv, index=False)

    subheads = []
    for index in range(8):
        label_text = (
            r"Delay\\ (days)"
            if index % 2 == 0
            else r"Inspection\\ fraction"
        )
        suffix = r" &" if index < 7 else r"\\ \midrule"
        subheads.append(
            r"  \multicolumn{1}{c}{\begin{tabular}[c]{@{}c@{}}"
            + label_text
            + r"\end{tabular}}"
            + suffix
        )
    tex = "\n".join(
        [
            "% AUTO-GENERATED FILE. DO NOT EDIT BY HAND.",
            "% Generated from paper_data/primary/selected_policy_cells.csv.",
            r"\begin{table}[p]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{0.93}",
            r"\begin{tabular}{@{}lrrrrrrrr@{}}",
            r"\toprule",
            r"\multicolumn{1}{c}{\multirow{2}{*}{Category}} & \multicolumn{2}{c}{Historical} & \multicolumn{2}{c}{Most Efficient} & \multicolumn{2}{c}{Most Equitable} & \multicolumn{2}{c}{Balanced}\\ \cmidrule(l){2-9}",
            r"\multicolumn{1}{c}{} &",
            *subheads,
            *combined_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Realized median inspection delays and inspection fractions by Borough and category for historical inspections and three selected Borough-budget policies. Simulated values are means across 25 simulations.}",
            r"\label{tab:borough-sla-outcomes}",
            r"\end{table}",
            "",
        ]
    )
    combined_tex.write_text(tex)
    outputs.extend([combined_csv, combined_tex])
    return outputs


def build_table(data_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    scores = _load_scores(data_dir / "primary" / "selected_policy_scores_by_seed.csv")
    counts = scores.groupby(["calendar_year", "display_role"])["random_seed"].nunique()
    if counts.groupby(level=0).nunique().ne(1).any():
        raise ValueError("Each role must have the same seeds within a year")

    means = (
        scores.groupby(["calendar_year", "display_role"], sort=False)[
            ["efficiency_ratio_historical", "equity_ratio_historical"]
        ]
        .mean()
        .rename(columns=lambda value: value.removesuffix("_ratio_historical"))
    )
    rows = []
    for year in YEARS:
        for metric in ("efficiency", "equity"):
            row = {"calendar_year": year, "objective": f"{metric.title()} loss"}
            row.update({role: means.loc[(year, role), metric] for role in ROLES})
            rows.append(row)
    table = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "main_policy_performance.csv"
    tex_path = output_dir / "main_policy_performance.tex"
    table.to_csv(csv_path, index=False, float_format="%.6f")

    body = []
    for index, row in table.iterrows():
        values = np.array([row[role] for role in ROLES], dtype=float)
        cells = [
            (f"\\textbf{{{value:.3f}}}" if np.isclose(value, values.min()) else f"{value:.3f}")
            for value in values
        ]
        year = f"\\multirow{{2}}{{*}}{{{int(row['calendar_year'])}}}" if index % 2 == 0 else ""
        rule = " \\\\ \\midrule" if index % 2 else "\\\\"
        body.append(f"{year} & {row['objective']} & " + " & ".join(cells) + rule)
    body[-1] = body[-1].replace(" \\\\ \\midrule", " \\\\ \\bottomrule")

    year_counts = counts.groupby(level=0).first().astype(int).to_dict()
    if year_counts[2021] != year_counts[2022]:
        raise ValueError("The two out-of-sample years must use the same seed count")
    out_of_sample_count = (
        "five" if year_counts[2021] == 5 else str(year_counts[2021])
    )
    caption = (
        "Performance of the most efficient, most equitable, and Balanced Borough "
        "budget GPS policies, when evaluated on data from 2019 (``training set''), "
        "2021, and 2022 (``test sets''), compared with efficiency and equity loss "
        "functions evaluated from historical inspection data. A lower value indicates "
        "better performance, and the best performer of each row is indicated in bold. "
        "We omit 2020 due to a large storm dominating the number of incidents; during "
        "such emergency periods, the city activates cross-agency resources and does not "
        "follow its regular operations. Values are means across "
        f"{year_counts[2019]} simulations for 2019 and {out_of_sample_count} simulations "
        "for each out-of-sample year."
    )
    columns = " &\n  ".join(
        [
            "\\begin{tabular}[c]{@{}c@{}}Data source \\\\ (Calendar year)\\end{tabular}",
            "\\begin{tabular}[c]{@{}c@{}}Objective\\\\ function\\end{tabular}",
            *[
                f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{ROLE_LABELS[role]} policy\\\\ (Relative to historical)\\end{{tabular}}"
                for role in ROLES
            ],
        ]
    )
    tex = (
        "% Generated from paper_data/primary/selected_policy_scores_by_seed.csv.\n"
        "\\begin{table}[tb]\n\\centering\n\\small\n"
        "\\resizebox{.9\\textwidth}{!}{%\n\\begin{tabular}{clrrr}\n\\toprule\n"
        f"{columns}\\\\ \\midrule\n" + "\n".join(body) + "\n"
        "\\end{tabular}%\n}\n"
        f"\\caption{{{caption}}}\n\\label{{tab:robustnesstab}}\n\\end{{table}}\n"
    )
    tex_path.write_text(tex)
    return csv_path, tex_path, *_budget_table(data_dir, output_dir), *_borough_tables(data_dir, output_dir)
