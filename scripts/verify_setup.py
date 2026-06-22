#!/usr/bin/env python3
"""
verify_setup.py — Smoke test for the replication environment.

Checks:
  1. Python packages (torch, torch_geometric) are importable
  2. L3 graph has exactly 1,990 METHOD nodes and 1,060 edges
  3. code_metrics_full.json has 1,990 entries
  4. test_method_paths.json has 210 tests (26 FAILING, 184 PASSING)
  5. L2 API graph is loadable

Exit 0 on success, exit 1 on any failure.
"""

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

errors = []

# ── 1. Package imports ────────────────────────────────────────────────────────
print("Checking Python packages...")
try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
except ImportError as e:
    errors.append(f"torch import failed: {e}")

try:
    import torch_geometric
    print(f"  ✓ torch_geometric {torch_geometric.__version__}")
except ImportError as e:
    errors.append(f"torch_geometric import failed: {e}")

try:
    from torch_geometric.nn import GATConv
    print("  ✓ GATConv available")
except ImportError as e:
    errors.append(f"GATConv import failed: {e}")

# ── 2. L3 graph integrity ────────────────────────────────────────────────────
print("\nChecking L3 method graph...")
graph_path = DATA_DIR / "graphs" / "level3_method_graph.json"
try:
    with open(graph_path) as f:
        g3 = json.load(f)
    n_nodes = len(g3["nodes"])
    n_edges = len(g3["links"])
    method_nodes = [n for n in g3["nodes"] if n.get("type") == "METHOD"]
    n_methods = len(method_nodes)

    if n_methods != 1990:
        errors.append(f"Expected 1,990 METHOD nodes, got {n_methods}")
    else:
        print(f"  ✓ {n_methods} METHOD nodes")

    if n_edges != 1060:
        errors.append(f"Expected 1,060 edges, got {n_edges}")
    else:
        print(f"  ✓ {n_edges} edges")

except FileNotFoundError:
    errors.append(f"Graph file not found: {graph_path}")
except Exception as e:
    errors.append(f"Graph load error: {e}")

# ── 3. Code metrics ──────────────────────────────────────────────────────────
print("\nChecking code metrics...")
metrics_path = DATA_DIR / "code-metrics" / "code_metrics_full.json"
try:
    with open(metrics_path) as f:
        metrics = json.load(f)
    n_entries = len(metrics)
    if n_entries != 1990:
        errors.append(f"Expected 1,990 metric entries, got {n_entries}")
    else:
        print(f"  ✓ {n_entries} method metric entries")
except FileNotFoundError:
    errors.append(f"Metrics file not found: {metrics_path}")
except Exception as e:
    errors.append(f"Metrics load error: {e}")

# ── 4. Test paths ────────────────────────────────────────────────────────────
print("\nChecking test paths...")
tests_path = DATA_DIR / "test-paths" / "test_method_paths.json"
try:
    with open(tests_path) as f:
        tests = json.load(f)
    n_tests = len(tests)
    n_fail = sum(1 for t in tests if t["status"] == "FAILING")
    n_pass = sum(1 for t in tests if t["status"] == "PASSING")

    if n_tests != 210:
        errors.append(f"Expected 210 tests, got {n_tests}")
    else:
        print(f"  ✓ {n_tests} integration tests")

    if n_fail != 26:
        errors.append(f"Expected 26 FAILING tests, got {n_fail}")
    else:
        print(f"  ✓ {n_fail} FAILING tests")

    if n_pass != 184:
        errors.append(f"Expected 184 PASSING tests, got {n_pass}")
    else:
        print(f"  ✓ {n_pass} PASSING tests")

except FileNotFoundError:
    errors.append(f"Tests file not found: {tests_path}")
except Exception as e:
    errors.append(f"Tests load error: {e}")

# ── 5. L2 API graph ─────────────────────────────────────────────────────────
print("\nChecking L2 API graph...")
l2_path = DATA_DIR / "graphs" / "level2_api_graph.json"
try:
    with open(l2_path) as f:
        g2 = json.load(f)
    n_api = sum(1 for n in g2["nodes"] if n.get("type") == "API")
    print(f"  ✓ L2 API graph: {n_api} API nodes")
except FileNotFoundError:
    errors.append(f"L2 API graph not found: {l2_path}")
except Exception as e:
    errors.append(f"L2 API graph error: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print("✗ FAILED — the following checks did not pass:")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("[OK] L3 method graph: 1990 nodes, 1060 edges")
    print("[OK] PyTorch / PyTorch Geometric imports successful")
    print("Smoke test passed.")
    sys.exit(0)
