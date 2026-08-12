"""Reproduce the supported public manuscript outputs from paper_data only."""

import csv
import hashlib
import json
from pathlib import Path

from .make_appendix_robustness import build_appendix_robustness
from .make_cost_map import build_cost_map
from .make_descriptive_outputs import build_descriptive_outputs
from .make_paper_figures import build_figure
from .make_paper_tables import build_table
from .make_priority_weight_table import build_priority_weight_table
from .make_service_floor_support import build_service_floor_support

INPUTS = (
    "descriptive/annual_simulation_inspection_rows_by_borough.csv",
    "descriptive/annual_simulation_request_rows_by_borough.csv",
    "descriptive/hazard_risk_counts.csv",
    "descriptive/hazard_risk_summary.csv",
    "descriptive/observed_delay_histogram_bins.csv",
    "descriptive/observed_delay_histogram_summary.csv",
    "descriptive/observed_risk_and_priority_by_category.csv",
    "descriptive/weekly_inspections.csv",
    "descriptive/weekly_service_requests.csv",
    "primary/search_cloud.csv",
    "primary/historical_reference.csv",
    "primary/selection_means.csv",
    "primary/inference_means.csv",
    "primary/hazard_histogram_bins.csv",
    "primary/map_scale_summary.csv",
    "primary/selected_policy_scores_by_seed.csv",
    "primary/budget_allocation.csv",
    "primary/selected_policy_cells.csv",
    "primary/bo_search_progress.csv",
    "primary/tract_policy_costs.csv",
    "robustness/capacity125_policy_means.csv",
    "robustness/capacity125_diagnostic.csv",
    "robustness/appendix_reoptimized_frontier_search.csv",
    "robustness/appendix_reoptimized_frontier_inference_means.csv",
    "robustness/appendix_reoptimized_frontier_historical_reference.csv",
    "robustness/d100_d200_median_cell_service_comparison.csv",
    "robustness/priority_weight_named_scores.csv",
    "robustness/priority_weight_random_summary.csv",
    "robustness/policy_cell_selection_means.csv",
)


def _validate_manifest(data_dir: Path) -> None:
    manifest = json.loads((data_dir / "MANIFEST.json").read_text())
    entries = {entry["relative_path"]: entry for entry in manifest["files"]}
    if len(entries) != len(manifest["files"]):
        raise ValueError("Manifest contains duplicate file paths")
    for relative_path in INPUTS:
        if relative_path not in entries:
            raise ValueError(f"Manifest lacks {relative_path}")

    source_ledger = json.loads((data_dir / "SOURCE_ARTIFACTS.json").read_text())
    known_source_ids = {entry["source_id"] for entry in source_ledger["source_files"]}
    for entry in manifest["files"]:
        relative_path = entry["relative_path"]
        path = data_dir / relative_path
        if not path.is_file():
            raise ValueError(f"Manifest file is missing: {relative_path}")
        content = path.read_bytes()
        if len(content) != entry["size_bytes"]:
            raise ValueError(f"Size mismatch for {relative_path}")
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Hash mismatch for {relative_path}")
        unknown_sources = set(entry["source_ids"]) - known_source_ids
        if unknown_sources:
            raise ValueError(f"Unknown source IDs for {relative_path}: {unknown_sources}")
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                rows = csv.reader(handle)
                columns = next(rows)
                row_count = sum(1 for _ in rows)
            if columns != entry["ordered_columns"]:
                raise ValueError(f"Column mismatch for {relative_path}")
            if row_count != entry["row_count"]:
                raise ValueError(f"Row-count mismatch for {relative_path}")


def reproduce(data_dir: Path, output_dir: Path) -> list[Path]:
    if data_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output directories must differ")
    _validate_manifest(data_dir)
    outputs = [
        *build_table(data_dir, output_dir / "tables"),
        build_priority_weight_table(data_dir, output_dir / "tables"),
        *build_figure(data_dir, output_dir / "figures"),
        *build_cost_map(data_dir, output_dir / "figures"),
        *build_descriptive_outputs(data_dir, output_dir),
        *build_appendix_robustness(data_dir, output_dir),
        *build_service_floor_support(data_dir, output_dir / "robustness"),
    ]
    for path in outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected nonempty output: {path}")
    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("paper_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    args = parser.parse_args()
    for result in reproduce(args.data_dir, args.output_dir):
        print(result)
