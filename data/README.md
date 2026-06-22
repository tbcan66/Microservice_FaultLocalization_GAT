# Non-Executable Artifacts — Data Description

This directory contains all non-executable data artifacts produced and consumed by the GNN-based fault localization pipeline. Each subfolder is self-contained and documented below.

**Subject system:** [Train-Ticket](https://github.com/FudanSELab/train-ticket) — a microservice benchmark with 41 services (Java Spring Boot, Node.js, Python Django, Go).

**Dataset Paper:** Gregor, L., Skalski, M., & Pretschner, A. (2025). Benchmarking component and integration testing in microservices: Test suites and fault analysis on TrainTicket. 2025 IEEE International Conference on Service-Oriented System Engineering (SOSE), 39–50. https://doi.org/10.1109/SOSE67019.2025.00009

---

## Directory Structure

```
data/
├── graphs/                  # Multi-level microservice call graphs
├── test-paths/              # Integration test execution paths (3 granularity levels)
├── code-metrics/            # Static code quality metrics for method nodes
├── evaluation-results/      # GNN model outputs, ablation study, LOFO cross-validation
```

---

## 1. `graphs/` — Multi-Level Microservice Call Graphs

These JSON files include the directed call graph of the Train-Ticket system at three levels, extracted via static analysis of the Java source code.

| File | Nodes | Edges | Node Types | Description |
|------|------:|------:|------------|-------------|
| `level1_service_graph.json` | 41 | 135 | SERVICE | Service-to-service dependency graph |
| `level2_api_graph.json` | 283 | 111 | API | REST API endpoint interaction graph |
| `level3_method_graph.json` | 1,990 | 1,060 | METHOD | Java method-level call graph |

**Schema** (all levels):
```json
{
  "nodes": [{"id": "ts-travel-service:TravelServiceImpl.query", "type": "METHOD", "service": "ts-travel-service", "label": "TravelServiceImpl.query", "info": "Service"}],
  "links": [{"source": "...", "target": "...", "relation": "CALLS"}]
}
```
---

## 2. `test-paths/` — Integration Test Execution Paths

Execution paths traced for each of the 210 integration tests (26 failing, 184 passing)
at three granularity levels. Each entry maps a test to the infrastructure nodes it covers
and identifies the ground-truth faulty node.

| File | Level | Records | Description |
|------|-------|--------:|-------------|
| `test_service_paths.json` | L1 Service | 210 | Services traversed by each test |
| `test_api_paths.json` | L2 API | 210 | API endpoints hit by each test |
| `test_method_paths.json` | L3 Method | 210 | Java methods reachable from test entry point |
| `all_failing_tests_with_ground_truth.csv` | — | 41 | Ground-truth mapping for all 26 failing tests |
| `integration_tests_list.json` | 210 integration tests with metadata (test class, method, service, status) |

**Schema (L3 — `test_method_paths.json`):**
```json
{
  "test_id": "ts-travel-service:TripsLeftTest.FAILING_testQueryFullyValidRequestBodyOrderFound",
  "status": "FAILING",
  "source_service": "ts-travel-service",
  "entry_controller_method": "ts-travel-service:TravelController.queryInfo",
  "method_path": ["ts-travel-service:TravelController.queryInfo", "..."],
  "reachable_methods": [{"method_id": "...", "label": "...", "service": "..."}],
  "faulty_method": "ts-travel-service:TravelServiceImpl.query"
}
```
---

## 3. `code-metrics/` — Static Code Quality Metrics

Metrics extracted via static analysis of Java source code for every method node in the L3 graph.

| File | Records | Description |
|------|--------:|-------------|
| `code_metrics_basic.json` | 1,990 | 4 base metrics (cyclomatic, loc, fan_out, param_count), min-max normalised |
| `code_metrics_full.json` | 1,990 | 15 metrics across 5 categories, raw values |
| `metric_importance_stats.json` | 15 | Statistical test results (Mann-Whitney U, Cliff's delta) per metric |

**15 Metrics in 5 categories (in `code_metrics_full.json`):**

| Category | Metrics |
|-----------|---------|
| **Complexity** | `cyclomatic_complexity`, `cognitive_complexity`, `max_nesting_depth`, `return_point_count` |
| **Size** | `loc`, `statement_count`, `param_count` |
| **Coupling** | `fan_out`, `fan_in`, `exception_count`, `external_service_calls` |
| **Cohesion** | `variable_spread`, `comment_density` |
| **Graph Structure** | `pagerank`, `betweenness_centrality` |

**Example Schema (`code_metrics_full.json`):**
```json
{
  "ts-travel-service:TravelServiceImpl.query": {
    "cyclomatic_complexity": 5.0,
    "cognitive_complexity": 8.0,
    "max_nesting_depth": 2.0,
    "return_point_count": 3.0,
    "loc": 25.0,
    "statement_count": 18.0,
    "param_count": 1.0,
    "fan_out": 4.0,
    "fan_in": 3,
    "exception_count": 1.0,
    "external_service_calls": 2.0,
    "variable_spread": 0.45,
    "comment_density": 0.08,
    "pagerank": 0.000523,
    "betweenness_centrality": 0.00312,
    "source_found": true,
    "body_extracted": true
  }
}
```

---

## 4. `evaluation-results/` — GNN Model Outputs

Pre-computed results from the GNN and GAT fault localization pipeline. Each file maps to a specific paper table.

| File | Paper Table | Description |
|------|------------|-------------|
| `gnn_results.json` | Table 2 (GNN row) | Baseline GNN results across all 3 graph levels |
| `gnn_results_optionB_fixed.json` | Table 2 (GAT row) | GAT with edge-typed attention + noise method exclusion |
| `ablation_results.json` | Table 4 | 19 metric configurations × 10 seeds ablation study |
| `lofo_results.json` | Tables 4 & 5 | Leave-One-Fault-Out cross-validation (14 folds × 5 seeds) |

**Paper Table 2 — Baseline Fault Localization Results:**

| Model | Source File | L1 Top-1 | L2 Top-1 | L3 Top-1/Top-3/Top-5 |
|-------|------------|------:|------:|------:|
| GNN | `gnn_results.json` | 11.5% | 34.6% | 42.3% / 76.9% / 88.5% |
| GAT | `gnn_results_optionB_fixed.json` | 26.9% | 23.1% | 84.6% / 96.2% / 96.2% |

**Paper Table 4 — Cross-Validation for Metric-Augmented GAT:**

The `all_5` configuration (size + coupling + betweenness) achieves Top-1 = 93.8%, Top-3 = 100%, ±9.9% std via LOFO cross-validation.

**Paper Table 5 — LOFO Cross-Validation Results (14 folds × 5 seeds = 70 runs):**

| Configuration | Top-1 | Top-3 | Top-5 | ±Std | In-sample Top-1 | Gap |
|--------------|------:|------:|------:|-----:|---------:|----:|
| No metrics (baseline) | 86.7% | 100% | 100% | ±24.8% | 88.5% | −1.8pp |
| all_5 (final) | 93.8% | 100% | 100% | ±9.9% | 96.1% | −2.3pp |

---