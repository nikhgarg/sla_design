### Redesigning Service Level Agreements
 
This repository contains the simulator, compact outputs, and analysis code used for the public replication of *Redesigning Service Level Agreements: Equity and Efficiency in City Government Operations*.

The latest full text version of the paper can be found at: https://arxiv.org/abs/2410.14825

@inproceedings{10.1145/3670865.3673624,
author = {Liu, Zhi and Garg, Nikhil},
title = {Redesigning Service Level Agreements: Equity and Efficiency in City Government Operations},
year = {2024},
isbn = {9798400707049},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3670865.3673624},
doi = {10.1145/3670865.3673624},
abstract = {In this work, we consider government service allocation - how the government allocates resources (e.g., maintenance of public infrastructure) over time. It is important to make these decisions efficiently and equitably - though these desiderata may conflict. In particular, we consider the design of Service Level Agreements (SLA) in city government operations: promises that incidents such as potholes and fallen trees will be responded to within a certain time.We model the problem of designing a set of SLAs as an optimization problem with different equity and efficiency objectives under a queuing network framework; the city has two decision levers: how to allocate response budgets to different neighborhoods, and how to schedule responses to individual incidents. We:(1) Theoretically analyze a stylized model and find that the "price of equity" is small in realistic settings;(2) Develop a simulation-optimization framework to optimize policies in practice; (3) Apply our framework empirically using data from NYC, finding that: (a) status quo inspections are highly inefficient and inequitable compared to optimal ones, and (b) in practice, the equity-efficiency tradeoff is not substantial: generally, inefficient policies are inequitable, and vice versa.},
booktitle = {Proceedings of the 25th ACM Conference on Economics and Computation},
pages = {309},
numpages = {1},
location = {New Haven, CT, USA},
series = {EC '24}
}


### Fast reproduction from compact outputs

The default reproduction path uses the aggregate files in `paper_data/`; it does not rerun Bayesian optimization or require request-level records. After creating the `sla` Conda environment, run

```bash
conda env create -f environment.yml
conda activate sla
python -m analysis.reproduce --data-dir paper_data --output-dir build
```

This currently rebuilds the main policy-performance and budget tables, the
combined all-Borough SLA table and its five per-Borough component tables, the
appendix priority-weight table, the primary
Pareto--Hazard figure, the tract-cost map, the fixed-support capacity and
BO-progress figures, the combined capacity and reoptimized appendix frontier,
the $D=100$ versus $D=200$ service comparison, the selection-only 20%
minimum-service finite-support diagnostic, and the descriptive figures and
tables used in the introduction and appendix. The input manifest is checked
before any output is created. See `paper_data/README.md` for the evidence layers
and generated-output inventory. For a notebook walkthrough of the compact
reproduction and optional simulation paths, see [`demo.ipynb`](demo.ipynb).

### Fixed-policy simulation

After restoring the archived inputs below, one selected policy and seed can be rerun without Bayesian optimization:

```bash
python evaluate_selected_policy.py --repo-root . \
  --role borough_most_efficient --seed 321 --year 2019 \
  --output build/fixed_policy_2019_seed321.json
```

The runner uses the effective simulator parameters in
`paper_data/primary/selected_policy_parameters.csv`, the paper settings
($D=100$, $\rho=0.15$, median delay, and three input cycles), and in-memory
Borough-level outcomes. For the three
displayed Borough policies, the runner also compares a matching 2019, 2021,
or 2022 seed with the checked-in score.

### Simulation inputs

The simulator and optional full optimization use the two December 1, 2023 NYC
Open Data snapshots tracked under `data_raw/` through Git LFS. A normal clone
with Git LFS installed retrieves them automatically; `data_raw/README.md` gives
source links, exact hashes, and instructions for a clone created with LFS
downloads disabled. The compact-output reproduction above does not read or
download these raw snapshots.

### Full optimization (optional and expensive)

`run_bo.py` is the entry point for new Bayesian-optimization searches. Every
supported 35-, 65-, 60-, or 12-coordinate BO vector is converted to simulator
parameters by the shared canonical decoder in `policy_decoding.py`. The search
requires the simulation inputs described above and writes local checkpoints to
`botorch_log/`. This directory is not part of the compact paper build, and its
generated contents are ignored by Git. Run `python run_bo.py --help` for the
available settings.
