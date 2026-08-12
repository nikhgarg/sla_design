"""Build the active three-panel tract-cost map from compact public data."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd


ROLES = (
    "historical_2019",
    "borough_most_efficient",
    "borough_balanced",
    "borough_most_equitable",
    "city_most_efficient",
    "city_balanced",
    "city_most_equitable",
)
MAIN_ROLES = (
    "historical_2019",
    "borough_most_efficient",
    "borough_most_equitable",
)
TITLES = {
    "historical_2019": "Historical inspections",
    "borough_most_efficient": "Borough: most efficient",
    "borough_most_equitable": "Borough: most equitable",
}
HASH = re.compile(r"[0-9a-f]{64}")
COST_COLUMNS = ("cost_delay", "cost_drop", "total_cost")
SHAPEFILE_HASHES = {
    ".shp": "dc301d74e4935ef8b441ea89b428c860ae733b7a9aa4e5fb029adde08680ae85",
    ".shx": "c6a73302660307a584ea46124791755d3b89098c9968321f8f0c9abb2799ef1e",
    ".dbf": "42f5b05ed41c4043c1f408b3677ee7c8826065033aebee36b0780f20c2fd3d09",
    ".prj": "496a9166511a534dc8eff570dac27cdc91c817fe605e463b7ca70b25e94fbfd9",
}


def _read_costs(path: Path) -> pd.DataFrame:
    required = {"role", "policy_id", "vector_hash", "GEOID", *COST_COLUMNS}
    frame = pd.read_csv(path, dtype={"GEOID": "string"}, keep_default_na=False)
    if set(frame.columns) != required or frame.empty:
        raise ValueError(f"{path.name} has an invalid schema or is empty")
    if set(frame["role"].astype(str)) != set(ROLES):
        raise ValueError("Tract costs do not contain exactly the seven locked roles")
    frame["GEOID"] = frame["GEOID"].astype(str).str.strip()
    if not frame["GEOID"].str.fullmatch(r"\d{11}").all():
        raise ValueError("Tract costs contain an invalid 11-digit GEOID")
    if frame.duplicated(["role", "GEOID"]).any():
        raise ValueError("Tract costs contain duplicate role/GEOID rows")
    for column in COST_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        values = frame[column].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Tract costs contain invalid {column} values")
    if not np.allclose(
        frame["total_cost"],
        frame["cost_delay"] + frame["cost_drop"],
        rtol=1e-9,
        atol=1e-8,
    ):
        raise ValueError("Tract costs violate total_cost = cost_delay + cost_drop")

    support = None
    for role in ROLES:
        rows = frame.loc[frame["role"] == role]
        if rows["policy_id"].nunique() != 1 or not str(rows["policy_id"].iloc[0]).strip():
            raise ValueError(f"Map role {role} does not identify exactly one policy")
        hashes = rows["vector_hash"].astype(str).unique().tolist()
        if len(hashes) != 1:
            raise ValueError(f"Map role {role} does not identify exactly one vector")
        vector_hash = hashes[0]
        if role == "historical_2019":
            if vector_hash:
                raise ValueError("Historical map rows must have a blank vector hash")
        elif HASH.fullmatch(vector_hash) is None:
            raise ValueError(f"Map role {role} has an invalid vector hash")
        role_support = set(rows["GEOID"])
        if support is None:
            support = role_support
        elif role_support != support:
            raise ValueError(f"GEOID support for {role} differs from the other map roles")
    return frame


def _canonical_hash(role: str, vector_hash: str) -> str:
    return vector_hash or f"role:{role}"


def _read_scale(path: Path, costs: pd.DataFrame) -> float:
    required = {
        "role",
        "vector_hash",
        "n_tracts_with_cost",
        "raw_pooled_quantile",
        "pooled_quantile_level",
        "shared_map_cap",
        "display_cap_rounding",
        "linear_colorbar_tick_count",
    }
    summary = pd.read_csv(path, keep_default_na=False)
    if set(summary.columns) != required or len(summary) != len(ROLES):
        raise ValueError(f"{path.name} has an invalid schema or row count")
    if set(summary["role"].astype(str)) != set(ROLES) or summary["role"].duplicated().any():
        raise ValueError("Map scale summary does not contain exactly one row per role")

    hashes = {
        role: _canonical_hash(
            role,
            str(costs.loc[costs["role"] == role, "vector_hash"].iloc[0]),
        )
        for role in ROLES
    }
    for role, row in summary.set_index("role").iterrows():
        if str(row["vector_hash"]) != hashes[role]:
            raise ValueError(f"Map scale vector hash disagrees for {role}")
        expected_rows = int((costs["role"] == role).sum())
        if int(row["n_tracts_with_cost"]) != expected_rows:
            raise ValueError(f"Map scale tract count disagrees for {role}")

    constant_columns = (
        "raw_pooled_quantile",
        "pooled_quantile_level",
        "shared_map_cap",
        "display_cap_rounding",
        "linear_colorbar_tick_count",
    )
    if any(summary[column].nunique() != 1 for column in constant_columns):
        raise ValueError("Map scale settings differ across roles")
    quantile = float(summary["pooled_quantile_level"].iloc[0])
    raw_quantile = float(summary["raw_pooled_quantile"].iloc[0])
    cap = float(summary["shared_map_cap"].iloc[0])
    if (
        not np.isfinite([quantile, raw_quantile, cap]).all()
        or not 0 < quantile <= 1
        or raw_quantile <= 0
        or cap <= 0
        or summary["display_cap_rounding"].iloc[0] != "ceil_to_nearest_1000"
    ):
        raise ValueError("Map scale settings are invalid")

    representatives: dict[str, str] = {}
    for role in ROLES:
        vector_hash = hashes[role]
        if vector_hash in representatives:
            first = costs.loc[costs["role"] == representatives[vector_hash]].sort_values("GEOID")
            current = costs.loc[costs["role"] == role].sort_values("GEOID")
            if not np.allclose(first["total_cost"], current["total_cost"], rtol=1e-9, atol=1e-8):
                raise ValueError("Roles sharing a vector hash have different tract costs")
        else:
            representatives[vector_hash] = role
    pooled = np.concatenate(
        [costs.loc[costs["role"] == role, "total_cost"].to_numpy(float)
         for role in representatives.values()]
    )
    observed_quantile = float(np.quantile(pooled, quantile))
    expected_cap = float(math.ceil(observed_quantile / 1000.0) * 1000.0)
    if not np.isclose(raw_quantile, observed_quantile, rtol=1e-12, atol=1e-8):
        raise ValueError("Map scale pooled quantile does not match tract costs")
    if not np.isclose(cap, expected_cap, rtol=0, atol=1e-10):
        raise ValueError("Map scale cap is not the rounded pooled quantile")
    if int(summary["linear_colorbar_tick_count"].iloc[0]) != len(_map_ticks(cap)):
        raise ValueError("Map scale tick count does not match the shared cap")
    return cap


def _load_shapes(shapefile: Path, costs: pd.DataFrame) -> dict[str, object]:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("Cost-map reproduction requires geopandas") from exc
    required_files = [shapefile.with_suffix(suffix) for suffix in SHAPEFILE_HASHES]
    missing = [path.name for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Census-tract shapefile bundle lacks: {missing}")
    for path in required_files:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != SHAPEFILE_HASHES[path.suffix]:
            raise ValueError(f"Census-tract shapefile component changed: {path.name}")
    base = gpd.read_file(shapefile)
    if base.empty or base.crs is None or not {"GEOID", "geometry"}.issubset(base.columns):
        raise ValueError("Census-tract shapefile is empty or lacks CRS/GEOID/geometry")
    base = base.to_crs(epsg=4326)[["GEOID", "geometry"]].copy()
    base["GEOID"] = base["GEOID"].astype(str).str.strip()
    if (
        not base["GEOID"].str.fullmatch(r"\d{11}").all()
        or base["GEOID"].duplicated().any()
        or base.geometry.isna().any()
        or base.geometry.is_empty.any()
        or not base.geometry.is_valid.all()
    ):
        raise ValueError("Census-tract shapefile has invalid GEOIDs or geometries")
    support = set(costs.loc[costs["role"] == ROLES[0], "GEOID"])
    missing_geoids = sorted(support - set(base["GEOID"]))
    if missing_geoids:
        raise ValueError(f"Map costs contain GEOIDs absent from the shapefile: {missing_geoids[:5]}")

    shapes = {}
    for role in MAIN_ROLES:
        role_costs = costs.loc[costs["role"] == role, ["GEOID", "total_cost"]]
        merged = base.merge(role_costs, on="GEOID", how="left", validate="one_to_one")
        if len(merged) != len(base) or int(merged["total_cost"].notna().sum()) != len(support):
            raise ValueError(f"Map merge changed the tract support for {role}")
        shapes[role] = merged
    return shapes


def _map_ticks(cap: float) -> list[float]:
    target_step = cap / 4.0
    magnitude = 10.0 ** math.floor(math.log10(target_step))
    candidates = np.asarray([1.0, 2.0, 5.0, 10.0]) * magnitude
    step = float(candidates[np.argmin(np.abs(candidates - target_step))])
    ticks = [float(value) for value in np.arange(0.0, cap, step) if value < cap]
    if not ticks or ticks[0] != 0.0:
        ticks.insert(0, 0.0)
    if len(ticks) > 1 and cap - ticks[-1] < 0.55 * step:
        ticks.pop()
    ticks.append(float(cap))
    if len(ticks) > 5:
        indices = np.linspace(0, len(ticks) - 1, 5).round().astype(int)
        ticks = [ticks[index] for index in sorted(set(indices.tolist()))]
        ticks[-1] = float(cap)
    return ticks


def _cost_tick(value: float, _position: float) -> str:
    if np.isclose(value, 0.0, rtol=0, atol=1e-12):
        return "0"
    scaled = float(value) / 1000.0
    digits = 0 if np.isclose(scaled, round(scaled), rtol=0, atol=1e-9) else 1
    return f"{scaled:.{digits}f}k"


def _assert_fonts(path: Path) -> None:
    executable = shutil.which("pdffonts")
    if executable is None:
        raise RuntimeError("pdffonts is required to verify the cost-map PDF")
    result = subprocess.run([executable, str(path)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"pdffonts could not inspect {path}: {result.stderr.strip()}")
    rows = [row for row in result.stdout.splitlines()[2:] if row.strip()]
    if not rows or any(
        len(row.split()) < 6
        or "type 3" in row.lower()
        or row.split()[-5].lower() != "yes"
        for row in rows
    ):
        raise RuntimeError(f"Cost-map PDF has Type 3 or unembedded fonts: {path}")


def build_cost_map(
    data_dir: Path,
    output_dir: Path,
    shapefile: Path | None = None,
) -> tuple[Path, Path]:
    """Render the active historical/efficient/equitable Borough map trio."""
    primary = data_dir / "primary"
    costs = _read_costs(primary / "tract_policy_costs.csv")
    cap = _read_scale(primary / "map_scale_summary.csv", costs)
    if shapefile is None:
        shapefile = Path(__file__).resolve().parents[1] / "shapefile" / "nyct2020.shp"
    shapes = _load_shapes(shapefile, costs)

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
    cmap = LinearSegmentedColormap.from_list(
        "sla_white_red", ["#ffffff", "#fee5d9", "#fb6a4a", "#a50f15"]
    )
    bounds = shapes[MAIN_ROLES[0]].total_bounds
    if not np.isfinite(bounds).all() or not (bounds[0] < bounds[2] and bounds[1] < bounds[3]):
        raise ValueError("Census-tract map bounds are invalid")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.6), squeeze=False)
    used = []
    for ax, role in zip(axes.ravel(), MAIN_ROLES):
        shapes[role].plot(
            column="total_cost",
            cmap=cmap,
            vmin=0.0,
            vmax=cap,
            linewidth=0.1,
            edgecolor="0.95",
            missing_kwds={"color": "0.75"},
            ax=ax,
        )
        ax.set_title(TITLES[role], pad=4)
        ax.set_xlim(float(bounds[0]), float(bounds[2]))
        ax.set_ylim(float(bounds[1]), float(bounds[3]))
        ax.set_axis_off()
        ax.grid(False)
        used.append(ax)
    scalar = ScalarMappable(cmap=cmap, norm=Normalize(0.0, cap, clip=True))
    scalar.set_array([])
    bar = fig.colorbar(
        scalar,
        ax=used,
        orientation="horizontal",
        shrink=0.8,
        fraction=0.046,
        pad=0.04,
        extend="max",
    )
    bar.set_ticks(_map_ticks(cap))
    bar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(_cost_tick))
    bar.set_label("Tract all-request cost")
    bar.ax.tick_params(labelsize=8.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.15, wspace=0.01, hspace=0.06)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "allrequest_cost_maps_main"
    pdf, png = base.with_suffix(".pdf"), base.with_suffix(".png")
    for path in (pdf, png):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    _assert_fonts(pdf)
    return pdf, png
