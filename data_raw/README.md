# NYC Forestry input snapshots

The simulator expects these two files at their historical paths:

- `Forestry_Service_Requests_20231201.csv`
- `Forestry_Inspections_20231201.csv`

NYC Open Data maintains the source datasets at [Forestry Service Requests](https://data.cityofnewyork.us/Environment/Forestry-Service-Requests/mu46-p9is/about_data) and [Forestry Inspections](https://data.cityofnewyork.us/Environment/Forestry-Inspections/4pt5-3vv4/about_data). The live exports are updated over time and therefore need not match the December 1, 2023 snapshots used for the paper.

The exact archived snapshots are tracked at the paths above through Git LFS. A
normal clone with Git LFS installed downloads them automatically. If the clone
was created with LFS downloads disabled, restore and verify them from the
repository root with:

```bash
git lfs pull --include="data_raw/Forestry_Service_Requests_20231201.csv,data_raw/Forestry_Inspections_20231201.csv"
sha256sum -c data_raw/SHA256SUMS
```

The Inspections and Service Requests snapshots contain about 1.1 GB in total. The fast
compact-output reproduction does not require either snapshot.
