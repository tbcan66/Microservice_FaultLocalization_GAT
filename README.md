# Replication Package: Graph-Based Fault Localization in Microservices

This dataset accompanies the thesis:

> **Graph-Based Fault Localization in Microservice Systems Using Graph Attention Networks**  

The dataset supports reproduction of all experiments: the multi-level call graph construction, metric selection, GAT fault localization, ablation study, and LOFO cross-validation, applied to the [Train-Ticket](https://github.com/FudanSELab/train-ticket) microservice benchmark.

---

## Repository Structure

```
Zenodo Upload/
├── Documentation/
│   ├── README.md                        ← this file
│   └── Integration_Tests.xlsx           ← integration test and ground truth inventory spreadsheet
│
├── Graphs/
│   ├── Main Graphs/
│   │   ├── level1_service_graph.json    ← service-level call graph (41 services)
│   │   ├── level2_api_graph.json        ← API endpoint-level call graph
│   │   ├── level3_method_graph.json     ← method-level call graph (1,990 methods + 19 repos)
│   │   └── test_method_paths_fixed.json ← method-level test coverage paths (fixed)
│   │
│   ├── Integration Tests/
│   │   ├── integration_tests_list.json  ← all 210 integration tests with status/metadata
│   │   ├── Service Level Graph with Tests/
│   │   │   ├── test_service_paths.json  ← per-test service-level coverage paths
│   │   │   └── neo4j_import.cypher     ← Cypher script to import L1 graph into Neo4j
│   │   ├── API Level Graph with Tests/
│   │   │   ├── test_api_paths.json      ← per-test API-level coverage paths
│   │   │   └── neo4j_import_level2.cypher
│   │   └── Method Level Graph with Tests/
│   │       ├── test_method_paths.json   ← per-test method-level coverage paths
│   │       └── neo4j_import_level3.cypher
│   │
│   ├── Pruned Method Graph/
│   │   ├── level3_method_graph_pruned.json  ← method graph after helper-node collapse
│   │   └── prune_method_graph.py            ← pruning script
│   │
│   └── Code Metrics/
│       ├── code_metrics.json            ← raw extracted metric values (all methods)
│       └── metric_importance_stats.json ← Mann-Whitney U, Cliff's delta, Spearman rho
│
├── Metrics/
│   ├── code_metrics.json                ← same as above (top-level copy)
│   └── metric_importance_stats.json
│
└── Model Codes/
    ├── run_gnn_all_levels.py            ← GNN baseline (message-passing, all 3 levels)
    ├── run_gnn_gat_optionB.py           ← GAT model (FaultLocGAT, fixed 12-feature)
    ├── ablation_study.py                ← ablation: 19 metric configs × 10 seeds
    ├── analyze_metric_importance.py     ← Mann-Whitney + Cliff's delta + Spearman
    └── lofo_evaluation.py               ← LOFO cross-validation (14 folds)
```

---

## Data Description

### Graph Files (JSON)

All graphs use the same format:

```json
{
  "nodes": [{ "id": "service:ClassName.method", "type": "METHOD", "info": "...", ... }],
  "links": [{ "source": "...", "target": "...", "relation": "CALLS_LOCAL" }]
}
```

| File | Nodes | Links | Node types |
|---|---|---|---|
| `level1_service_graph.json` | 41 services | service→service CALLS | SERVICE |
| `level2_api_graph.json` | API endpoints | HTTP_CALL | API |
| `level3_method_graph.json` | 2,009 (1,990 METHOD + 19 REPOSITORY) | 1,220 (1,060 METHOD_TO_METHOD + 160 METHOD_TO_REPO) | METHOD, REPOSITORY |

### Test Coverage Files (JSON)

Each entry represents one integration test:

```json
{
  "test_id":              "ts-order-service.FAILING_getSoldTickets_1",
  "status":               "FAILING",
  "faulty_method":        "ts-order-service:OrderServiceImpl.getSoldTickets",
  "method_path":          ["ts-order-service:OrderController.getSoldTickets", ...],
  "entry_controller_method": "ts-order-service:OrderController.getSoldTickets",
  "reachable_methods":    [{"method_id": "...", ...}]
}
```

- 210 integration tests total: **26 failing** (covering **14 distinct faulty methods**), 184 passing.
- 1 failing test excluded from evaluation (no method-level coverage path).

### Code Metrics (JSON)

`code_metrics.json` contains per-method metric values for all 1,990 method nodes. Keys used:

| Metric key | Description |
|---|---|
| `loc` | Lines of code (excl. blank/comments) |
| `fan_in` | Number of callers (in-degree in call graph) |
| `fan_out` | Number of callees (out-degree in call graph) |
| `param_count` | Number of method parameters |
| `betweenness_centrality` | NetworkX betweenness centrality (normalised) |

`metric_importance_stats.json` contains the statistical selection results: Mann-Whitney U test p-values, Cliff's delta effect sizes, and Spearman rank correlation with fault labels, for all 15 candidate metrics.

---

## Reproducing the Experiments

### Requirements

```
python >= 3.10
torch >= 2.0
torch-geometric >= 2.3
scipy
numpy
matplotlib
seaborn
networkx
```

Install:
```bash
pip install torch torch-geometric scipy numpy matplotlib seaborn networkx
```

### Step 1 — GNN Baseline (all 3 graph levels)

```bash
python "Model Codes/run_gnn_all_levels.py"
```

Outputs Top-1/3/5 accuracy for service, API, and method levels using message-passing GNN (5 iterations, no PyTorch required).

### Step 2 — GAT Model (method level)

```bash
python "Model Codes/run_gnn_gat_optionB.py"
```

Trains a 2-layer Graph Attention Network (`FaultLocGAT`: GATConv hidden=32, heads=4, epochs=300, lr=0.005) on the method-level graph with 12-dim node features including Ochiai score and 4 code metrics.

### Step 3 — Metric Importance Analysis

```bash
python "Model Codes/analyze_metric_importance.py"
```

Computes Mann-Whitney U + Cliff's delta for all 15 metrics (faulty vs. non-faulty method distributions). Writes results to `Metrics/metric_importance_stats.json`.

### Step 4 — Ablation Study

```bash
python "Model Codes/ablation_study.py"
```

Runs 19 metric configurations × 10 random seeds × 300 epochs using `FaultLocGAT_flexible`. Reports Top-1/3/5 accuracy per configuration, leave-one-out metric impact, and Spearman correlation between statistical importance (Cliff's delta) and GNN contribution.

### Step 5 — LOFO Cross-Validation

```bash
python "Model Codes/lofo_evaluation.py"
```

Leave-One-Fault-Out: trains on 13 of 14 faulty nodes, evaluates on the held-out fault. 14 folds × 5 seeds. Tests GNN generalization across distinct fault types.

---

## Ground Truth Faults

The 14 faulty methods correspond to known issues in Train-Ticket, identified from the test suite (see `integration_tests_list.json` for test-to-fault mappings). Each faulty method is referenced in the `"faulty_method"` field of the test coverage files.

---

## License

The Train-Ticket benchmark system is originally published under the [Apache 2.0 License](https://github.com/FudanSELab/train-ticket/blob/master/LICENSE).  
The additional dataset, graph files, metrics, and model code in this package are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
