#!/bin/bash
# reproduce.sh — Single entry point for the ECSA 2026 replication kit.
#
# Runs:
#   1. Smoke test (verify environment + data integrity)
#   2. GNN (parameter-free message passing) — all 3 graph levels
#   3. GAT (3-layer Graph Attention Network) — all 3 graph levels
#   4. Combine results and print Table 2 summary
#
# Usage:
#   bash reproduce.sh              # default: seed=42, 400 epochs
#   bash reproduce.sh --seed 123   # custom seed (passed to GAT only)
#
# Expected runtime: ~5-10 minutes on CPU.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ECSA 2026 Replication Kit — Fault Localization in          ║"
echo "║  Microservices using Graph Neural Networks                  ║"
echo "║  Per-method published configurations                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Smoke test ───────────────────────────────────────────────────────
echo "=== Step 1: Environment & Data Verification ==="
python3 scripts/verify_setup.py
echo ""

# ── Step 2: GNN (parameter-free message passing) ────────────────────────────
echo "=== Step 2: GNN (Message Passing) — All 3 Graph Levels ==="
python3 scripts/run_gnn.py
echo ""

# ── Step 3: GAT (Graph Attention Network) ────────────────────────────────────
echo "=== Step 3: GAT (3-Layer) — All 3 Graph Levels ==="
python3 scripts/run_gat.py "$@"
echo ""

# ── Step 4: Combine and summarise ────────────────────────────────────────────
echo "=== Step 4: Combined Table 2 Summary ==="
python3 scripts/combine_results.py
echo ""

echo "✅  Reproduction complete. Results in outputs/results.json"
echo "    Reference output updated in reference/expected_output.json"
