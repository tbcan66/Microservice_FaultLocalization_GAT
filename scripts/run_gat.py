#!/usr/bin/env python3
"""
run_gat.py — Reproduce Table 2 GAT row at all three graph levels (L1/L2/L3).

Data:            level1_service_graph.json, level2_api_graph.json, level3_method_graph.json
                 test_service_paths.json, test_api_paths.json, test_method_paths.json
Architecture:    3-layer GAT (GATConv), heads 4→2→1, hidden=32, dropout=0.2
                 BCEWithLogitsLoss with pos_weight, Adam lr=0.005, StepLR(100, 0.5)


Single seed, GAT is a learned model so the results may vary slightly from paper's 
seed-averaged figures.
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data"

EPOCHS   = 400
LR       = 0.005
HIDDEN   = 32
HEADS    = 4
DROPOUT  = 0.2

EDGE_TYPE_MAP = {
    "TEST_RUNS_ON": 1.0,
    "TEST_COVERS":  0.8,
    "CALLS":        0.7,
    "CALLS_REPO":   0.6,
    "HTTP_CALL":    0.5,
    "CALLS_LOCAL":  0.35,
    "DEFAULT":      0.5,
}

# Patterns identifying noise nodes to exclude from L3
NOISE_PATTERNS = [".fallback", ".ok", ".home", ".welcome", ".BCryptPasswordEncoder",
                  ".WebMvcConfigurerAdapter", ".JWTFilter", ".restTemplate"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_noise_method(mid):
    return any(mid.endswith(p) for p in NOISE_PATTERNS)


def compute_ochiai(tests, cover_field):
    nf = sum(1 for t in tests if t["status"] == "FAILING")
    ef, ep = defaultdict(int), defaultdict(int)
    for t in tests:
        for nid in t.get(cover_field, []):
            if t["status"] == "FAILING": ef[nid] += 1
            else:                        ep[nid] += 1
    scores = {}
    for nid in set(ef) | set(ep):
        d = math.sqrt(nf * (ef[nid] + ep[nid]))
        scores[nid] = ef[nid] / d if d else 0.0
    return scores, ef, ep


def top_k_accuracy(tests, cover_field, faulty_field, logit_scores, k_values=(1, 3, 5)):
    failing = [t for t in tests if t["status"] == "FAILING" and t.get(faulty_field)]
    hits = defaultdict(int)
    ranks_out = []
    for t in failing:
        faulty  = t[faulty_field]
        covered = t.get(cover_field, [])
        if not covered:
            ranks_out.append((t["test_id"], faulty, None)); continue
        ranked = sorted(covered, key=lambda n: -logit_scores.get(n, -999.0))
        rank   = ranked.index(faulty) + 1 if faulty in ranked else None
        ranks_out.append((t["test_id"], faulty, rank))
        for k in k_values:
            if faulty in ranked[:k]:
                hits[k] += 1
    total = len(failing)
    return {k: (hits[k] / total if total else 0.0) for k in k_values}, total, ranks_out


# ── Build PyG Data ────────────────────────────────────────────────────────────
def build_pyg_data(tests, cover_field, faulty_field,
                   infra_ids, infra_type_label, edges_with_types,
                   bidirectional=True):
    ochiai, ef, ep = compute_ochiai(tests, cover_field)
    nf_total = max(sum(1 for t in tests if t["status"] == "FAILING"), 1)
    np_total = max(sum(1 for t in tests if t["status"] == "PASSING"), 1)

    faulty_nodes = {t[faulty_field] for t in tests if t.get(faulty_field)}
    test_ids     = {t["test_id"] for t in tests}
    failing_ids  = {t["test_id"] for t in tests if t["status"] == "FAILING"}

    all_ids = list(test_ids | infra_ids)
    idx_map = {nid: i for i, nid in enumerate(all_ids)}

    in_deg  = defaultdict(int)
    out_deg = defaultdict(int)
    for src, tgt, _ in edges_with_types:
        in_deg[tgt]  += 1
        out_deg[src] += 1
    max_in  = max(in_deg.values(),  default=1)
    max_out = max(out_deg.values(), default=1)

    type_enc = {"TEST": 0.0, "SERVICE": 0.25, "API": 0.5, "METHOD": 0.75, "CONTROLLER": 1.0}

    feats, labels = [], []
    for nid in all_ids:
        is_fail  = 1.0 if nid in failing_ids else 0.0
        is_pass  = 1.0 if (nid in test_ids and nid not in failing_ids) else 0.0
        is_inf   = 1.0 if nid in infra_ids else 0.0
        oc       = ochiai.get(nid, 0.0)
        n_ef     = ef.get(nid, 0) / nf_total
        n_ep     = ep.get(nid, 0) / np_total
        indeg    = in_deg.get(nid,  0) / max_in
        outdeg   = out_deg.get(nid, 0) / max_out
        ntype    = type_enc.get(infra_type_label if nid in infra_ids else "TEST", 0.0)
        feats.append([is_fail, is_pass, is_inf, oc, n_ef, n_ep, indeg, outdeg, ntype])
        labels.append(1.0 if (nid in faulty_nodes and nid in infra_ids) else 0.0)

    x = torch.tensor(feats,  dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)

    edge_list, edge_attrs = [], []
    for src, tgt, etype in edges_with_types:
        if src in idx_map and tgt in idx_map:
            w = EDGE_TYPE_MAP.get(etype, EDGE_TYPE_MAP["DEFAULT"])
            edge_list.append([idx_map[src], idx_map[tgt]])
            edge_attrs.append([w])
            if bidirectional:
                edge_list.append([idx_map[tgt], idx_map[src]])
                edge_attrs.append([w * 0.5])

    edge_index = (torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                  if edge_list else torch.zeros((2, 0), dtype=torch.long))
    edge_attr  = (torch.tensor(edge_attrs, dtype=torch.float)
                  if edge_attrs else torch.zeros((0, 1), dtype=torch.float))

    # Training mask: infra nodes + all faulty nodes (always include them)
    covered_set = {nid for t in tests for nid in t.get(cover_field, [])}
    train_set   = (covered_set | faulty_nodes) & infra_ids
    train_mask  = torch.tensor([nid in train_set for nid in all_ids])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=y, train_mask=train_mask), idx_map, all_ids


class FaultLocGAT(torch.nn.Module):
    def __init__(self, in_features=9, hidden=HIDDEN, heads=HEADS, dropout=DROPOUT):
        super().__init__()
        self.conv1   = GATConv(in_features, hidden, heads=heads, edge_dim=1, dropout=dropout)
        self.conv2   = GATConv(hidden * heads, hidden, heads=2,  edge_dim=1, dropout=dropout)
        self.conv3   = GATConv(hidden * 2, hidden,    heads=1,   edge_dim=1, dropout=dropout)
        self.out     = torch.nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, ei, ea))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, ei, ea))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv3(x, ei, ea))
        return self.out(x).squeeze(-1)  # raw logits — no sigmoid


def train_and_predict(data, all_ids, infra_ids, epochs=EPOCHS, lr=LR):
    model     = FaultLocGAT(in_features=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    n_pos = max(data.y[data.train_mask].sum().item(), 1)
    n_neg = data.train_mask.sum().item() - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]) 

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out  = model(data)
        pred = out[data.train_mask]
        true = data.y[data.train_mask]
        loss = criterion(pred, true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(data).numpy()

    return {nid: float(logits[i]) for i, nid in enumerate(all_ids) if nid in infra_ids}


# ── Edge builders ─────────────────────────────────────────────────────────────
def build_edges_l1(tests, service_ids, g1):
    edges = []
    for t in tests:
        tid, src = t["test_id"], t["source_service"]
        if src in service_ids:
            edges.append((tid, src, "TEST_RUNS_ON"))
        for svc in t.get("path", []):
            if svc in service_ids and svc != src:
                edges.append((tid, svc, "TEST_COVERS"))
    for lnk in g1["links"]:
        edges.append((lnk["source"], lnk["target"], "CALLS"))
    return edges


def build_edges_l2(tests, api_ids, g2):
    edges = []
    for t in tests:
        tid   = t["test_id"]
        entry = t.get("entry_point_api")
        if entry and entry in api_ids:
            edges.append((tid, entry, "TEST_RUNS_ON"))
        for d in t.get("downstream_apis", []):
            aid = d["api_id"]
            if aid in api_ids:
                edges.append((tid, aid, "TEST_COVERS"))
    for lnk in g2["links"]:
        rel = lnk.get("relation", "HTTP_CALL")
        edges.append((lnk["source"], lnk["target"], rel if rel in EDGE_TYPE_MAP else "HTTP_CALL"))
    return edges


def build_edges_l3(tests, method_ids, g3):
    edges = []
    for t in tests:
        tid   = t["test_id"]
        entry = t.get("entry_controller_method")
        if entry and entry in method_ids:
            edges.append((tid, entry, "TEST_RUNS_ON"))
        for rm in t.get("reachable_methods", []):
            mid = rm["method_id"]
            if mid in method_ids:
                edges.append((tid, mid, "TEST_COVERS"))
    for lnk in g3["links"]:
        rel = lnk.get("relation", "CALLS")
        src, tgt = lnk["source"], lnk["target"]
        if src in method_ids and tgt in method_ids:
            edges.append((src, tgt, rel if rel in EDGE_TYPE_MAP else "CALLS"))
    return edges


# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Table 2 — GAT fault localization (all levels)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"GAT training epochs (default: {EPOCHS})")
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("\n" + "=" * 70)
    print("  GAT (Graph Attention Network) — All 3 Graph Levels")
    print("  Source: run_gat.py")
    print(f"  Config: 3-layer GAT, heads 4→2→1, {args.epochs} epochs, seed={seed}")
    print(f"  PyTorch {torch.__version__}")
    print("=" * 70)

    with open(DATA_DIR / "graphs" / "level1_service_graph.json") as f: g1 = json.load(f)
    with open(DATA_DIR / "graphs" / "level2_api_graph.json")     as f: g2 = json.load(f)
    with open(DATA_DIR / "graphs" / "level3_method_graph.json")  as f: g3 = json.load(f)
    with open(DATA_DIR / "test-paths" / "test_service_paths.json")  as f: t1 = json.load(f)
    with open(DATA_DIR / "test-paths" / "test_api_paths.json")      as f: t2 = json.load(f)
    with open(DATA_DIR / "test-paths" / "test_method_paths.json")   as f: t3 = json.load(f)

    svc_ids = {n["id"] for n in g1["nodes"]}
    api_ids = {n["id"] for n in g2["nodes"] if n.get("type") == "API"}

    method_ids_raw = {n["id"] for n in g3["nodes"] if n.get("type") == "METHOD"}
    method_ids     = {m for m in method_ids_raw if not is_noise_method(m)}
    excluded = len(method_ids_raw) - len(method_ids)
    print(f"\n  L3: Excluded {excluded} noise/fallback method nodes ({len(method_ids)} remain)")

    # Coverage diagnostics
    for label, tests, cf, ff, iids in [
        ("L1", t1, "path",        "faulty_service", svc_ids),
        ("L2", t2, "api_path",    "faulty_api",     api_ids),
        ("L3", t3, "method_path", "faulty_method",  method_ids),
    ]:
        fn = {t[ff] for t in tests if t.get(ff)}
        cn = {nid for t in tests for nid in t.get(cf, [])}
        vis = fn & cn & iids
        print(f"  {label}: faulty nodes={len(fn)}, visible to trainer={len(vis)}/{len(fn)}")

    levels = [
        # (name, tests, cover, faulty, infra_ids, type, edges, bidirectional)
        ("L1 Service",      t1, "path",        "faulty_service", svc_ids,    "SERVICE", build_edges_l1(t1, svc_ids, g1), True),
        ("L2 API Endpoint", t2, "api_path",    "faulty_api",     api_ids,    "API",     build_edges_l2(t2, api_ids, g2), True),
        ("L3 Java Method",  t3, "method_path", "faulty_method",  method_ids, "METHOD",  build_edges_l3(t3, method_ids, g3), False),
    ]

    results = []
    for name, tests, cfield, ffield, infra, itype, edges, bidir in levels:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        data, idx_map, all_ids = build_pyg_data(
            tests, cfield, ffield, infra, itype, edges, bidirectional=bidir
        )
        n_pos = int(data.y.sum().item())
        print(f"  Nodes={len(all_ids)} | Edges={data.edge_index.shape[1]} | "
              f"Train={data.train_mask.sum().item()} | Faulty labels={n_pos}")
        scores = train_and_predict(data, all_ids, infra, epochs=args.epochs)
        acc, n, ranks = top_k_accuracy(tests, cfield, ffield, scores)
        results.append((name, acc, n, scores, ranks, infra))

    print("\n" + "=" * 70)
    print("RESULTS — GAT (Option B Fixed)")
    print("=" * 70)
    print(f"\n{'Level':<22} {'Tests':>6} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8}")
    print("-" * 56)
    for name, acc, n, _, _, _ in results:
        print(f"{name:<22} {n:>6} {acc[1]:>7.1%} {acc[3]:>8.1%} {acc[5]:>8.1%}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_data = {
        "method": "GAT (Graph Attention Network)",
        "source_script": "run_gat.py",
        "config": {
            "architecture": "3-layer GAT, heads 4→2→1",
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "epochs": args.epochs,
            "lr": LR,
            "loss": "BCEWithLogitsLoss (pos_weight)",
            "seed": seed,
            "noise_exclusion": True,
            "l3_bidirectional": False,
        },
    }
    for name, acc, n, scores, _, infra in results:
        top30 = sorted([(k, v) for k, v in scores.items() if k in infra],
                       key=lambda x: -x[1])[:30]
        out_data[name] = {
            "algorithm": "GAT_Fixed",
            "top_1":  round(acc[1], 4),
            "top_3":  round(acc[3], 4),
            "top_5":  round(acc[5], 4),
            "n_failing_tested": n,
            "top_30_nodes": [{"node": k, "score": round(v, 6)} for k, v in top30]
        }

    out_path = out_dir / "gat_results.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\n  ✅  GAT results saved → {out_path}")
    return out_data


if __name__ == "__main__":
    main()
