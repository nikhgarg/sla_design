# Compact paper data

This directory contains aggregate and seed-level outputs for fast paper reproduction.

- `descriptive/` contains the broader seven-category weekly descriptive counts,
  the five-Borough and six-category annual counts used for the simulation-input
  tables, delay-histogram bins, and aggregate recorded-risk summaries.
- `primary/` separates one-seed search points, selection-seed means, locked 25-seed inference means, selected-policy cell summaries, representative-run map and Hazard-histogram aggregates, and policy scores by seed.
- `robustness/` contains the fixed-support capacity and category-priority-weight
  sensitivity summaries, the compact 21,440-row four-setting search support,
  61 locked 25-simulation inference means, four setting-specific historical
  denominators, the role-matched $D=100$/$D=200$ median-delay cell panel, and
  the 16,140 policy--cell selection means used for the 20% minimum-service
  finite-support diagnostic.

The annual request and inspection inputs use the same five Boroughs and six
categories as the simulation. The weekly series used for the descriptive
appendix figure intentionally retain the broader seven-category universe.


`MANIFEST.json` records every public file's SHA-256 hash, row count, ordered schema, source artifact IDs, and transformation. `SOURCE_ARTIFACTS.json` records only sanitized source bundle identifiers and hashes. The tract-cost and simulated Hazard-histogram aggregates use representative simulation seed 321; they are not 25-seed inference means.

The three Borough policies used in the paper are fixed by policy ID. The most
efficient (`candidate_d100_delay_median_cap1_borough_c7df3d1fd0b07964`) and
most equitable
(`candidate_d100_delay_median_cap1_borough_9da82092c7ffaf17`) policies are the
deterministic endpoints of the final selection means. The Balanced policy
(`candidate_d100_delay_median_cap1_borough_500e7870e139a8dc`) is the
selected interior paper policy and is exposed as `borough_balanced`.
`robustness/selected_borough_roles.csv` records these distinct selection
statuses. The corresponding City paper alias is `city_balanced`.

The compact cell panel repeats each cell's historical inspection fraction so
that the supporting $D=100$/$D=200$ comparison table is reproducible without
request-level records. It is a post-processing comparison of separately
reoptimized and locked policies, not a fixed-policy change in $D$.

Run the supported public build from the repository root with:

```bash
python -m analysis.reproduce --data-dir paper_data --output-dir build
```

The final robustness additions appear at:

- `build/figures/appendix_robustness_reoptimized_frontiers_2026-08-10.pdf`
  and `.png` (App. Fig. 10), combining the same-policy capacity comparison
  with the three reoptimized delay-statistic and noninspection-penalty panels;
- `build/tables/appendix_brooklyn_prune_d100_d200.tex` and `.csv`
  (a supporting $D=100$/$D=200$ comparison, not an active appendix table);
  and
- `build/tables/appendix_d100_d200_cell_change_summary.csv`, which records the
  increase, decrease, tie, minimum-service, and threshold counts behind the
  surrounding discussion.

The paths below map public build products to the active manuscript or supporting
analysis artifacts. Paths in the right column are relative to the separate
manuscript source tree.

| Public build output | Current manuscript/supporting artifact |
|---|---|
| `build/tables/main_policy_performance.tex` | `tables/main_policy_performance.tex` |
| `build/tables/main_budget_allocation.tex` | `tables/main_budget_allocation.tex` |
| `build/tables/appendix_borough_sla_outcomes.tex` | `tables/appendix_borough_sla_outcomes.tex` |
| `build/tables/appendix_annual_request_rows.tex` | `tables/appendix_annual_request_rows.tex` |
| `build/tables/appendix_annual_inspection_rows.tex` | `tables/appendix_annual_inspection_rows.tex` |
| `build/tables/appendix_risk_priority.tex` | `tables/appendix_risk_priority.tex` |
| `build/tables/priority_weight_sensitivity.tex` | `gpt_documents/docs/generated/priority_weight_robustness_863147_combined.tex` |
| `build/figures/allrequest_cost_maps_main.pdf` | `figures/allrequest_cost_maps_main_2026-08-04.pdf` |
| `build/figures/pareto_frontier_with_hazard_delay_allrequest.pdf` | `figures/pareto_frontier_with_hazard_delay_allrequest_2026-08-04.pdf` |
| `build/figures/primary_bo_search_progress_2026-08-06.pdf` | `figures/primary_bo_search_progress_2026-08-06.pdf` |
| `build/figures/appendix_robustness_reoptimized_frontiers_2026-08-10.pdf` | `figures/appendix_robustness_reoptimized_frontiers_2026-08-10.pdf` |
| `build/figures/srs_per_week_allrequest_2026-08-04.pdf` | `figures/srs_per_week_allrequest_2026-08-04.pdf` |
| `build/figures/inspections_per_week_allrequest_2026-08-04.pdf` | `figures/inspections_per_week_allrequest_2026-08-04.pdf` |
| `build/figures/hazard_delay_hist_allrequest_2026-08-04.pdf` | `figures/hazard_delay_hist_allrequest_2026-08-04.pdf` |
| `build/figures/itd_delay_hist_allrequest_2026-08-04.pdf` | `figures/itd_delay_hist_allrequest_2026-08-04.pdf` |
| `build/figures/hazard_risk_rating_distplot_allrequest_2026-08-04.pdf` | `figures/hazard_risk_rating_distplot_allrequest_2026-08-04.pdf` |
| `build/tables/appendix_brooklyn_prune_d100_d200.tex` | `tables/appendix_brooklyn_prune_d100_d200.tex`, a supporting comparison not used in the active appendix |

CSV and PNG sidecars, the five per-Borough component tables, the D-change
summary, the standalone fixed-support capacity figure, and the service-floor
outputs are audit/support products. The active manuscript uses the combined
Borough table and combined App. Fig. 10.

The App. Fig. 10 renderer resets Matplotlib defaults inside a local style
context, so its output does not depend on which other paper builders ran first.
The supported environment pins Matplotlib 3.8.0 for functional/content
reproduction. The manuscript PDF was rendered with Matplotlib 3.10.9 and has
SHA-256
`292b9c07f9a5fe88a3e24bad4f983663c123b844974083de9b5a2ced591676f3`;
exact PDF-byte reproduction therefore requires 3.10.9. Under 3.8.0, PDF
producer or creation metadata can differ even when normalized drawing content,
embedded images, and the PNG are identical. The manuscript PNG has SHA-256
`d0a8fc3dc22a391bbe5793ff62bc478111b4bfe71dd77bd0bcc6c315d3aaf198`.

All other generated outputs retain their established names under
`build/figures/` and `build/tables/`; the reproduction command prints the
complete output inventory after validation.

The service-floor builder joins `robustness/policy_cell_selection_means.csv` to
the existing primary selection means, locked inference means, and historical
reference. Eligibility is based only on ten-seed selection means; the inference
means are used only to draw the already locked main-text frontiers. The two
policy classes each yield one nondominated eligible point, so the output is
explicitly labeled as a finite-support diagnostic rather than a constrained
Pareto curve.
