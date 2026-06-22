#!/usr/bin/env python3
"""
run_gnn.py — Reproduce Table 2 GNN row at all three graph levels (L1/L2/L3).

Original script: run_gnn_all_levels.py
Data:            level1_service_graph.json, level2_api_graph.json, level3_method_graph.json
                 test_service_paths.json, test_api_paths.json, test_method_paths.json
Algorithm:       Parameter-free message-passing propagation (5 iterations)
                 Test nodes anchored: FAILING=[0,1], PASSING=[1,0]
                 Infrastructure nodes: mean-aggregation over incoming neighbours

"""

import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data"

ITERATIONS = 5   # message-passing iterations (matches existing code)

# ── Helpers ────────────────────────────────────────────────────────────────────

def graph_propagation(node_ids, is_test, is_failing, edges_src_tgt, iterations=ITERATIONS):
    """
    Args:
        node_ids: list of all node IDs
        is_test: set of test node IDs
        is_failing: set of failing test node IDs
        edges_src_tgt: list of (src, tgt) directed edges
        iterations: number of propagation iterations

    Returns:
        dict { node_id: fail_score }
    """
    # Initialize states
    states = {}
    for nid in node_ids:
        if nid in is_test:
            states[nid] = [0.0, 1.0] if nid in is_failing else [1.0, 0.0]
        else:
            states[nid] = [0.5, 0.5]

    # Build incoming edges index
    incoming = defaultdict(list)
    for src, tgt in edges_src_tgt:
        if src in states and tgt in states:
            incoming[tgt].append(src)

    # Propagate
    for _ in range(iterations):
        new_states = {}
        for nid in node_ids:
            if nid in is_test:
                new_states[nid] = states[nid]   # anchor test nodes
            else:
                nbrs = incoming.get(nid, [])
                if nbrs:
                    ps = sum(states[n][0] for n in nbrs)
                    fs = sum(states[n][1] for n in nbrs)
                    tot = ps + fs
                    if tot > 0:
                        new_states[nid] = [ps / tot, fs / tot]
                    else:
                        new_states[nid] = [0.5, 0.5]
                else:
                    new_states[nid] = states[nid]
        states = new_states

    return {nid: states[nid][1] for nid in node_ids}


def top_k_accuracy(tests, cover_field, faulty_field, gnn_scores, k_values=(1, 3, 5)):
    """Evaluate Top-K using GNN fail scores."""
    failing_with_gt = [t for t in tests if t["status"] == "FAILING" and t.get(faulty_field)]
    hits = defaultdict(int)

    per_test_ranks = []
    for t in failing_with_gt:
        faulty = t[faulty_field]
        covered = t.get(cover_field, [])
        if not covered:
            per_test_ranks.append(None)
            continue
        ranked = sorted(covered, key=lambda n: -gnn_scores.get(n, 0.0))
        rank = ranked.index(faulty) + 1 if faulty in ranked else None
        per_test_ranks.append((t["test_id"], faulty, rank))
        for k in k_values:
            if faulty in ranked[:k]:
                hits[k] += 1

    total = len(failing_with_gt)
    return {k: (hits[k] / total if total else 0.0) for k in k_values}, total, per_test_ranks


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — Services
# ══════════════════════════════════════════════════════════════════════════════

def run_level1():
    with open(DATA_DIR / "test-paths" / "test_service_paths.json") as f:
        tests = json.load(f)
    with open(DATA_DIR / "graphs" / "level1_service_graph.json") as f:
        g1 = json.load(f)

    # Node universe: all services + all test IDs
    service_ids = {n["id"] for n in g1["nodes"]}
    test_ids    = {t["test_id"] for t in tests}
    all_nodes   = service_ids | test_ids

    # Failing tests
    failing_ids = {t["test_id"] for t in tests if t["status"] == "FAILING"}

    # Edges:
    # 1. Test → source_service (RUNS_ON)
    # 2. source_service → each service in calls[] (COVERS)
    # 3. service → service (CALLS from level1 graph)
    edges = []
    for t in tests:
        tid = t["test_id"]
        src = t["source_service"]
        if src in service_ids:
            edges.append((tid, src))
        for svc in t.get("calls", []):
            if svc in service_ids:
                edges.append((tid, svc))
                edges.append((src, svc))
    for lnk in g1["links"]:
        edges.append((lnk["source"], lnk["target"]))

    scores = graph_propagation(list(all_nodes), test_ids, failing_ids, edges)

    # For evaluation, coverage field = "path" (services in path)
    acc, n, ranks = top_k_accuracy(tests, "path", "faulty_service", scores)
    return acc, n, scores, ranks, "L1 Service", service_ids


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 2 — API Endpoints
# ══════════════════════════════════════════════════════════════════════════════

def run_level2():
    with open(DATA_DIR / "test-paths" / "test_api_paths.json") as f:
        tests = json.load(f)
    with open(DATA_DIR / "graphs" / "level2_api_graph.json") as f:
        g2 = json.load(f)

    api_ids   = {n["id"] for n in g2["nodes"] if n.get("type") == "API"}
    test_ids  = {t["test_id"] for t in tests}
    all_nodes = api_ids | test_ids
    failing_ids = {t["test_id"] for t in tests if t["status"] == "FAILING"}

    edges = []
    for t in tests:
        tid = t["test_id"]
        entry = t.get("entry_point_api")
        if entry and entry in api_ids:
            edges.append((tid, entry))
        for d in t.get("downstream_apis", []):
            aid = d["api_id"]
            if aid in api_ids:
                edges.append((tid, aid))
                if entry and entry in api_ids:
                    edges.append((entry, aid))
    # HTTP_CALL edges from graph
    for lnk in g2["links"]:
        if lnk.get("relation") == "HTTP_CALL":
            edges.append((lnk["source"], lnk["target"]))

    scores = graph_propagation(list(all_nodes), test_ids, failing_ids, edges)
    acc, n, ranks = top_k_accuracy(tests, "api_path", "faulty_api", scores)
    return acc, n, scores, ranks, "L2 API Endpoint", api_ids


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 3 — Java Methods
# ══════════════════════════════════════════════════════════════════════════════

def run_level3():
    with open(DATA_DIR / "test-paths" / "test_method_paths.json") as f:
        tests = json.load(f)
    with open(DATA_DIR / "graphs" / "level3_method_graph.json") as f:
        g3 = json.load(f)

    method_ids = {n["id"] for n in g3["nodes"] if n.get("type") == "METHOD"}
    test_ids   = {t["test_id"] for t in tests}
    all_nodes  = method_ids | test_ids
    failing_ids = {t["test_id"] for t in tests if t["status"] == "FAILING"}

    edges = []
    for t in tests:
        tid = t["test_id"]
        entry = t.get("entry_controller_method")
        if entry and entry in method_ids:
            edges.append((tid, entry))
        for rm in t.get("reachable_methods", []):
            mid = rm["method_id"]
            if mid in method_ids:
                edges.append((tid, mid))
    # CALLS/CALLS_LOCAL/CALLS_REPO from graph
    for lnk in g3["links"]:
        edges.append((lnk["source"], lnk["target"]))

    scores = graph_propagation(list(all_nodes), test_ids, failing_ids, edges)
    acc, n, ranks = top_k_accuracy(tests, "method_path", "faulty_method", scores)
    return acc, n, scores, ranks, "L3 Java Method", method_ids


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  GNN (Parameter-Free Message Passing) — All 3 Graph Levels")
    print("  Source: run_gnn_all_levels.py")
    print(f"  Algorithm: Mean-aggregation propagation ({ITERATIONS} iterations)")
    print("  Config: Test anchoring, per-test coverage ranking, no training")
    print("=" * 70)

    results = []
    for run_fn in [run_level1, run_level2, run_level3]:
        acc, n, scores, ranks, label, infra_ids = run_fn()
        results.append((label, acc, n, scores, ranks, infra_ids))

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'Level':<22} {'Tests':>6} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8}")
    print("-" * 56)
    for label, acc, n, _, _, _ in results:
        print(f"{label:<22} {n:>6} {acc[1]:>7.1%} {acc[3]:>8.1%} {acc[5]:>8.1%}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_data = {
        "method": "GNN (parameter-free message passing)",
        "source_script": "run_gnn_all_levels.py",
        "config": {
            "algorithm": f"Mean-aggregation propagation ({ITERATIONS} iterations)",
            "training": "none (parameter-free)",
            "deterministic": True,
        },
    }
    for label, acc, n, scores, ranks, infra_ids in results:
        top_nodes = sorted(
            [(nid, sc) for nid, sc in scores.items() if nid in infra_ids],
            key=lambda x: -x[1]
        )[:30]
        out_data[label] = {
            "top_1": round(acc[1], 4),
            "top_3": round(acc[3], 4),
            "top_5": round(acc[5], 4),
            "n_failing_tested": n,
            "top_30_nodes": [{"node": nid, "score": round(sc, 6)} for nid, sc in top_nodes]
        }

    out_path = out_dir / "gnn_results.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\n  ✅  GNN results saved → {out_path}")
    return out_data


if __name__ == "__main__":
    main()
