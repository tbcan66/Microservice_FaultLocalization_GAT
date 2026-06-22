# Replication Package — Graph-Based Fault Localization in Microservices using Static Code Metrics

This repository is the replication package for the paper **"Graph-Based Fault Localization in Microservices using Static Code Metrics"** by **Tuğba Can** and **Feza Buzluca** (Istanbul Technical University), accepted at **ECSA 2026**.
 
It contains the data, code, and a self-contained Docker environment needed to
reproduce the GNN and GAT fault-localization results reported in the paper. The study
performs **method-level fault localization** on the
[Train-Ticket](https://github.com/FudanSELab/train-ticket) microservice benchmark
using a parameter-free message-passing baseline (GNN) and a graph attention network
(GAT).

## 1. Relation to the Paper

This package reproduces the GNN and GAT rows of the paper's **Table 2** using
**each method's original published configuration**.

| Paper result      | Script / source                          | Configuration                          | Expected (single seed)              |
|-------------------|------------------------------------------|----------------------------------------|-------------------------------------|
| Table 2 — GNN L1  | `scripts/run_gnn.py`                     | Parameter-free msg-passing, 5 iters    | 11.5        |
| Table 2 — GNN L2  | `scripts/run_gnn.py`                     | Parameter-free msg-passing, 5 iters    | 34.6          |
| Table 2 — GNN L3  | `scripts/run_gnn.py`                     | Parameter-free msg-passing, 5 iters    | 42.3 / 76.9 / 88.5        |
| Table 2 — GAT L1  | `scripts/run_gat.py`                     | 3-layer GAT, 400 epochs, noise excl.   | ~26.9 (seed-dependent)              |
| Table 2 — GAT L2  | `scripts/run_gat.py`                     | 3-layer GAT, 400 epochs, noise excl.   | ~23.1 (seed-dependent)              |
| Table 2 — GAT L3  | `scripts/run_gat.py`                     | 3-layer GAT, 400 epochs, noise excl.   | ~84.6 / 96.2 / 96.2 (seed-dependent)    |
| Tables 4–5        | `data/evaluation-results/*.json`         | LOFO / ablation (pre-computed)         | see paper                           |

Values reported by the Docker run are written to `reference/expected_output.json`,
which is the authoritative reference for this package.

---

## 2. Per-Method Protocol Differences

The two methods in Table 2 were developed and evaluated independently. This
replication package preserves those differences:

| Aspect                | GNN (message-passing)            | GAT (3-layer)                    |
|-----------------------|----------------------------------|----------------------------------|
| Source script         | `run_gnn.py`.                    | `run_gat.py`.                    |
| Input data            | JSON graph + test paths          | JSON graph + test paths          |
| Ranking               | **Per-test** (coverage ranking)  | **Per-test** (coverage ranking)  |
| Ground truth          | `faulty_method` (26 tests)       | `faulty_method` (26 tests)       |
| Deterministic         | Yes                              | No (seed-dependent)              |

---

## 3. Artifact Structure

```
.
├── README.md                     # This file — single entry point
├── LICENSE                       # Code license (see §9)
├── DATA-LICENSE                  # Data license / attribution (see §9–10)
├── Dockerfile                    # CPU-only reproduction environment
├── requirements.txt              # Python dependencies (numpy, scipy)
├── reproduce.sh                  # Entry point: smoke test → GNN → GAT
├── scripts/
│   ├── run_gnn.py                # GNN — parameter-free message passing (all levels)
│   ├── run_gat.py                # GAT — 3-layer Graph Attention Network (all levels)
│   ├── combine_results.py        # Merge outputs + print Table 2 summary
│   └── verify_setup.py           # Smoke test (data + imports)
├── reference/
│   └── expected_output.json      # Reference output from the seed=42 run
└── data/
    ├── graphs/                   # Multi-level microservice call graphs (L1/L2/L3)
    ├── test-paths/               # Integration-test execution paths (210 tests)
    ├── code-metrics/             # 15 static code metrics per method node
    ├── evaluation-results/       # GNN/GAT outputs, ablation study, LOFO cross-validation
    └── README.md                 # Non-executable artifacts/data description
```
---

## 4. Requirements
 
- **Hardware:** CPU only. No GPU required. Approximately 8 GB RAM. The full run
  completes in under ~20 minutes on a standard laptop.
- **Software:** [Docker](https://docs.docker.com/get-docker/) only. No other local installation is needed; all dependencies are pinned inside the image (Python 3.11, PyTorch 2.2.2, PyTorch Geometric 2.5.3).
- **Network:** Not required at run time. The image builds from pinned packages and
  the analysis runs entirely on the committed `data/` artifacts.

---

## 5. Setup
 
Build the Docker image from the repository root:
 
```bash
docker build -t ecsa2026-repro .
```
 
This step installs all pinned dependencies. Expected build time: ~5-10 minutes.

---
 
## 6. Smoke Test (confirm successful installation)
 
Before the full run, verify the environment and data mount:
 
```bash
docker run --rm ecsa2026-repro python scripts/verify_setup.py
```
 
Expected output confirms the L3 graph loads correctly and all packages import:
 
```
[OK] L3 method graph: 1990 nodes, 1060 edges
[OK] PyTorch / PyTorch Geometric imports successful
Smoke test passed.
```

---

## 7. Reproducing the Results

Run the full pipeline (GNN, GAT) each in their published configuration:
 
```bash
docker run --rm ecsa2026-repro ./reproduce.sh
```
 
This runs the smoke test, then both methods independently, and prints a
Table-2-style summary with per-row source script and configuration info.
Results are also written to `outputs/results.json` inside the container.
 
**Reproduced results (seed = 42):**

> The GAT is a learned model, so single-seed values may vary by a few points from
> the paper's seed-averaged figures. The committed `reference/expected_output.json`
> holds the exact values produced by the seed=42 run.

To change the seed:
```bash
docker run --rm ecsa2026-repro python scripts/run_gat.py --seed 7
```

---

## 8. Reduced Configuration

This package is a **scaled-down reproduction** intended to run quickly on a reviewer's
machine. It uses a **single seed** rather than the multi-seed averaging and full
leave-one-fault-out (LOFO) / ablation sweeps reported in the paper. The full
cross-validation results (Tables 4–5) are provided as pre-computed outputs under
`data/evaluation-results/` and can be re-derived from them without retraining.

---
 
## 9. License
 
- **Code** (scripts, Dockerfile): [MIT / Apache-2.0] — see `LICENSE`.
- **Data** (`data/`): [CC BY 4.0] — see `DATA-LICENSE`.

---

## 10. Data Provenance & Attribution
 
The subject system is **Train-Ticket**, an open-source microservice benchmark
(Apache-2.0): https://github.com/FudanSELab/train-ticket
 
The integration-test suite and fault set are derived from:
 
> Gregor, L., Skalski, M., & Pretschner, A. (2025). *Benchmarking component and
> integration testing in microservices: Test suites and fault analysis on TrainTicket.*
> 2025 IEEE International Conference on Service-Oriented System Engineering (SOSE), 39–50.
> https://doi.org/10.1109/SOSE67019.2025.00009
 
Graphs and code metrics in this package were extracted from the Train-Ticket source via
static analysis as described in the paper.
 
---

## 11. Archival Location
 
This package is archived with a persistent identifier:
 
- **DOI:** [10.5281/zenodo.20794220]
- **Repository:** [https://github.com/tbcan66/Microservice_FaultLocalization_GAT]
 
---
 
## 12. How to Cite
 
```bibtex
@misc{can2026artifact,
  author       = {Tuğba Can and Feza Buzluca},
  title        = {Replication Package: Graph-Based Fault Localization in
                  Microservices using Static Code Metrics},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20794220},
  note         = {Artifact for ECSA 2026}
}
```
 
---

## 13. Contact
 
Tuğba Can - cant15@itu.edu.tr — Istanbul Technical University