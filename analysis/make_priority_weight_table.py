"""Build the appendix priority-weight sensitivity table."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROLES = ("most_efficient", "most_equitable", "balanced")
METRICS = ("efficiency", "equity")
WEIGHTS = (
    ("assigned", "Assigned"),
    ("equal", "Equal"),
    ("observed_mean_recorded_risk_2019_2022", "Average recorded risk"),
    ("power_0p25", r"Power $\eta=0.25$"),
    ("power_0p50", r"Power $\eta=0.5$"),
    ("power_0p75", r"Power $\eta=0.75$"),
    ("power_1p25", r"Power $\eta=1.25$"),
    ("power_1p50", r"Power $\eta=1.5$"),
    ("power_2p00", r"Power $\eta=2$"),
)


def _read_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    robustness = data_dir / "robustness"
    named = pd.read_csv(robustness / "priority_weight_named_scores.csv")
    random = pd.read_csv(robustness / "priority_weight_random_summary.csv")
    named_columns = {
        "weight_id",
        "display_role",
        "policy_id",
        "efficiency_relative_to_historical",
        "equity_relative_to_historical",
    }
    random_columns = {"display_role", "metric", "draw_count", "maximum", "p01", "p99"}
    for frame, required, label in (
        (named, named_columns, "named scores"),
        (random, random_columns, "random summary"),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Priority-weight {label} lacks columns: {sorted(missing)}")

    expected_named = set(product((weight_id for weight_id, _ in WEIGHTS), ROLES))
    named_keys = list(zip(named["weight_id"], named["display_role"]))
    if len(named_keys) != len(set(named_keys)) or set(named_keys) != expected_named:
        raise ValueError("Named scores must contain one row per weight and policy role")
    if named.groupby("display_role")["policy_id"].nunique().ne(1).any():
        raise ValueError("Each policy role must identify one fixed policy")

    expected_random = set(product(ROLES, METRICS))
    random_keys = list(zip(random["display_role"], random["metric"]))
    if len(random_keys) != len(set(random_keys)) or set(random_keys) != expected_random:
        raise ValueError("Random summary must contain one row per policy role and metric")
    if random["draw_count"].nunique() != 1 or random["draw_count"].iloc[0] <= 0:
        raise ValueError("Random-summary rows must use one positive draw count")

    numeric = [f"{metric}_relative_to_historical" for metric in METRICS]
    named[numeric] = named[numeric].apply(pd.to_numeric, errors="raise")
    random[["maximum", "p01", "p99"]] = random[["maximum", "p01", "p99"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(named[numeric].to_numpy()).all():
        raise ValueError("Named priority-weight scores must be finite")
    if not np.isfinite(random[["maximum", "p01", "p99"]].to_numpy()).all():
        raise ValueError("Random priority-weight summaries must be finite")
    if (random["p01"] > random["p99"]).any() or (random["p99"] > random["maximum"]).any():
        raise ValueError("Random-summary quantiles and maxima are inconsistent")
    return (
        named.set_index(["weight_id", "display_role"]),
        random.set_index(["display_role", "metric"]),
    )


def build_priority_weight_table(data_dir: Path, output_dir: Path) -> Path:
    named, random = _read_inputs(data_dir)
    rows = []
    for weight_id, label in WEIGHTS:
        values = [
            named.loc[(weight_id, role), f"{metric}_relative_to_historical"]
            for role in ROLES
            for metric in METRICS
        ]
        price = values[2] / values[0] - 1
        rows.append(
            f"{label} & "
            + " & ".join(f"{value:.3f}" for value in values)
            + f" & {100 * price:.1f}\\% \\\\"
        )

    interval = [
        f"[{random.loc[(role, metric), 'p01']:.3f}, {random.loc[(role, metric), 'p99']:.3f}]"
        for role in ROLES
        for metric in METRICS
    ]
    maximum = [
        f"{random.loc[(role, metric), 'maximum']:.3f}"
        for role in ROLES
        for metric in METRICS
    ]
    rows.extend(
        [
            "\\midrule",
            "Sampled 1st--99th percentile & " + " & ".join(interval) + " & --- \\\\",
            "Sampled componentwise maximum & " + " & ".join(maximum) + " & --- \\\\",
        ]
    )
    tex = (
        "% AUTO-GENERATED FILE. DO NOT EDIT BY HAND.\n"
        "% Generated from compact public summaries.\n"
        "\\begin{table}[tb]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
        "& \\multicolumn{2}{c}{Most Efficient} & \\multicolumn{2}{c}{Most Equitable} & \\multicolumn{2}{c}{Balanced} & \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Priority-weight specification & Efficiency & Equity & Efficiency & Equity & Efficiency & Equity & Price of equity \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}}\n"
        "\\caption{Fixed-policy priority-weight sensitivity. Efficiency and equity entries are policy loss divided by historical loss under the same weights. The price of equity is the proportional increase in efficiency loss from the most efficient to the most equitable fixed policy. In short, we find that equity gains from our optimized policies are more robust across weights, while efficiency gains are small when the category weights are more equal across categories.}\n"
        "\\label{tab:priority-weight-sensitivity}\n\\end{table}\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "priority_weight_sensitivity.tex"
    output.write_text(tex)
    return output
