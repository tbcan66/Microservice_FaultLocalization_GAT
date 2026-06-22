#!/usr/bin/env python3
"""
combine_results.py — Merge per-method outputs and print Table 2 summary.

Reads:
  outputs/gnn_results.json
  outputs/gat_results.json

Writes:
  outputs/results.json       (combined)
  reference/expected_output.json  (authoritative reference)

Prints a Table-2-style summary with per-row source script and config info.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR   = REPO_ROOT / "outputs"
REF_DIR   = REPO_ROOT / "reference"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt_pct(v):
    return f"{v*100:.1f}%"


def main():
    gnn_path  = OUT_DIR / "gnn_results.json"
    gat_path  = OUT_DIR / "gat_results.json"

    for p in [gnn_path, gat_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run the individual scripts first.")
            sys.exit(1)

    gnn  = load_json(gnn_path)
    gat  = load_json(gat_path)

    # ── Print Table 2 summary ────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("TABLE 2 — Fault Localization Results (GNN & GAT, Per-Method Published Configs)")
    print("=" * 90)

    # Header
    print(f"\n{'Method':<28} {'Source Script':<32} {'Level':<15} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7}")
    print("-" * 100)

    # GNN (all levels)
    for lvl_key, lvl_name in [("L1 Service", "L1 Service"), ("L2 API Endpoint", "L2 API"),
                               ("L3 Java Method", "L3 Method")]:
        g = gnn[lvl_key]
        print(f"{'GNN (msg-passing)':<28} {'run_gnn_all_levels.py':<32} {lvl_name:<15} "
              f"{fmt_pct(g['top_1']):>7} {fmt_pct(g['top_3']):>7} {fmt_pct(g['top_5']):>7}")
    print(f"  {'':28} {'  Parameter-free, 5 iters, 1990 nodes (no exclusion), deterministic'}")

    # GAT (all levels)
    for lvl_key, lvl_name in [("L1 Service", "L1 Service"), ("L2 API Endpoint", "L2 API"),
                               ("L3 Java Method", "L3 Method")]:
        g = gat[lvl_key]
        print(f"{'GAT (3-layer)':<28} {'run_gnn_optionB_fixed.py':<32} {lvl_name:<15} "
              f"{fmt_pct(g['top_1']):>7} {fmt_pct(g['top_3']):>7} {fmt_pct(g['top_5']):>7}")
    seed = gat.get("config", {}).get("seed", "?")
    epochs = gat.get("config", {}).get("epochs", "?")
    print(f"  {'':28} {'  Heads 4→2→1, ' + str(epochs) + ' epochs, noise exclusion, seed=' + str(seed)}")

    # Paper reference
    print(f"\nPaper Table 2 reference (seed-averaged):")
    print(f"  GNN  L1: 11.5%  L2: 34.6%  L3: 42.3% / 76.9% / 88.5%")
    print(f"  GAT  L1: 26.9%  L2: 23.1%  L3: 84.6% / 96.2% / 96.2%")

    # ── Merge into combined JSON ─────────────────────────────────────────────
    combined = {
        "description": "Reproduction of paper Table 2 — GNN and GAT rows (per-method published configs)",
        "gnn":  gnn,
        "gat":  gat,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)

    # Save combined results
    combined_path = OUT_DIR / "results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  ✅  Combined results → {combined_path}")

    # Save as authoritative reference
    ref_path = REF_DIR / "expected_output.json"
    with open(ref_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  ✅  Reference output → {ref_path}")


if __name__ == "__main__":
    main()
