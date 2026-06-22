# ─────────────────────────────────────────────────────────────────────────────
# ECSA 2026 Replication Kit — CPU-only Docker image
# ─────────────────────────────────────────────────────────────────────────────
# Reproduces Table 2: GNN and GAT fault localization
# on the Train-Ticket microservice benchmark.
#
# Build:  docker build -t ecsa2026-repro .
# Run:    docker run --rm ecsa2026-repro
# Smoke:  docker run --rm ecsa2026-repro python3 scripts/verify_setup.py
#
# Expected build time:  ~5-8 min (mostly pip install torch)
# Expected run time:    ~5-10 min (mostly GAT training on CPU)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /replication

# System dependencies (minimal — no C compilation needed for CPU wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# ── Install PyTorch CPU-only (pinned version) ────────────────────────────────
# torch + pyg + scatter/sparse must all match.
# Using torch 2.2.2 CPU + pyg 2.5.3 (well-tested combination).
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# ── Install PyG and extensions ───────────────────────────────────────────────
# torch-scatter and torch-sparse are optional but GATConv works without them
# in recent PyG versions. We install pyg with the CPU find-links index.
RUN pip install --no-cache-dir \
    torch_geometric==2.5.3

# ── Install remaining dependencies ───────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy replication kit ─────────────────────────────────────────────────────
COPY . .

# ── Default command: run full reproduction ───────────────────────────────────
CMD ["bash", "reproduce.sh"]
