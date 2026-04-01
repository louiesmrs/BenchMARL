# Flatland Model Implementation Differences (Base vs Tree)

This document summarizes **how each model family is implemented** in BenchMARL Flatland and reports observed performance for representative runs.

## 1) Implementation differences

## Base models (`mlp`, `gnn`, `lstm`, `gru`)

### `mlp`
- **Files**: `benchmarl/models/mlp.py`, wiring in `fine_tuned/flatland/benchmark.py`.
- **Input path**:
  - `FlatlandMlpBenchmarkTransform` flattens tree observation fields into `("agents","observation","flat_observation")`.
  - Tree-structured keys are excluded from final observation input for base runs.
- **Core architecture**:
  - Concatenate input features and pass through `MultiAgentMLP` (or per-agent `MLP` if not using shared multi-agent input path).
- **No recurrent state**.

### `gnn`
- **Files**: `benchmarl/models/gnn.py`, wiring in `fine_tuned/flatland/benchmark.py`.
- **Input path**:
  - Uses per-agent features with agent dimension present.
- **Core architecture**:
  - In benchmark wiring, `gnn` is a `SequenceModel`: **GNN layer -> MLP layer**.
  - GNN uses PyG message passing (configured to `GATv2Conv` in benchmark script for flatland benchmarks).
  - Supports centralized pooling via mean across agent dimension when configured centralised.
- **No recurrent state**.

### `lstm`
- **Files**: `benchmarl/models/lstm.py`.
- **Input path**:
  - Concatenates per-agent feature tensors.
- **Core architecture**:
  - Multi-agent LSTM core (`MultiAgentLSTM`) followed by MLP projection head.
- **Stateful**:
  - Reads/writes `is_init`, hidden/cell states (`_hidden_lstm_h_*`, `_hidden_lstm_c_*`) via tensordict.

### `gru`
- **Files**: `benchmarl/models/gru.py`.
- **Input path**:
  - Concatenates per-agent feature tensors.
- **Core architecture**:
  - Multi-agent GRU core (`MultiAgentGRU`) followed by MLP projection head.
- **Stateful**:
  - Reads/writes `is_init` and hidden state (`_hidden_gru_*`) via tensordict.

---

## Tree models (`treetransformer`, `treegnn`, `treelstm`)

- **Files**: `benchmarl/models/flatland_treelstm.py`, `benchmarl/models/flatland_tree_modules.py`.
- **Shared tree observation contract** (required keys):
  - `agents_attr`, `node_attr`, `adjacency`, `node_order`, `edge_order`, `valid_actions`.
- **Shared high-level pipeline**:
  1. Encode agent attributes with an MLP (`attr_embedding`).
  2. Encode tree structure with one of:
     - `TreeLSTM` (`treelstm`),
     - `TreeTransformer` (`treetransformer`),
     - Tree-node GATv2 stack (`treegnn`).
  3. Concatenate agent + tree embeddings.
  4. Optional cross-agent attention blocks (`AgentAttentionBlock`, often disabled via `transformer_layers: 0` for stability in these runs).
  5. Build head input with optional **flat-observation gated fusion** (`flat_projection` + sigmoid gate), then `LayerNorm`.
  6. Policy/value heads are shallow MLPs with **residual skip projections** (`head + skip`) for stability.

### `treelstm`
- Tree encoder is child-sum style fixed-arity TreeLSTM over `node_order` / `edge_order` iterations.
- Root-node representation (`node 0`) is used as tree summary per agent.

### `treetransformer`
- Tree encoder is transformer over per-node embeddings plus structural positional encoding derived from adjacency slot/depth information.
- Produces per-agent embedding from flattened node outputs.

### `treegnn`
- Tree encoder is a GATv2 stack over tree nodes.
- Builds graph edges from tree adjacency; optional edge attributes are one-hot child-slot encodings.
- Supports reverse edges and self-loops via config.

### Operational differences vs base models
- Tree runs in benchmark flow use `FlatlandTreeActionMaskTransform`; masking is applied by distribution layer (not inside tree policy forward).
- Tree runs set `disable_value_estimator_vmap=True` in benchmark builder (compat/stability with tree control flow).

---

## 2) Performance results (representative runs)

Metric shown: `eval_info_arrival_ratio_mean` (higher is better).

| Model | Run | Setup | Max eval arrival | Last eval arrival | Last eval deadlock | Final frames |
|---|---|---|---:|---:|---:|---:|
| `mlp` | `2026-03-31_08-06-33__curriculum` phase 02 | curriculum, delay=0/departure=1 | 0.7171 | 0.6679 | 0.0657 | 7,680,000 |
| `gru` | `2026-03-28_22-48-51__curriculum` phase 02 | curriculum, delay=0/departure=1 | 0.7107 | 0.6464 | 0.0743 | 7,680,000 |
| `treelstm` | `2026-03-26_22-05-55__curriculum` phase 01 | curriculum, delay=1/departure=0 | 0.5386 | 0.5271 | 0.3357 | 3,840,000 |
| `treegnn` | `2026-03-27_23-56-25__curriculum` phase 01 | curriculum, delay=1/departure=0 | 0.5193 | 0.4729 | 0.4021 | 1,884,000 |
| `treetransformer` | `2026-03-29_22-48-28__curriculum` phase 02 | curriculum resume, delay=0/departure=1 | 0.7400 | 0.7400 | 0.1229 | 7,680,000 |

### Important comparability caveat
- These are all **7-agent curriculum runs**.
- They still are not fully apples-to-apples because phases differ (`delay=1/departure=0` vs `delay=0/departure=1`) and frame budgets/checkpoint points differ.

Use this table as **run summary**, not as a strict architecture ranking.
