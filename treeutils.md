# Flatland vs `flatland_cutils` backend: run-based comparison

## Scope
This compares two **IPPO + MLP curriculum** runs with identical curriculum/task shape (7 agents, 2 phases), differing in tree observation backend:

- **Flatland backend (python TreeObs)**
  - `fine_tuned/flatland/benchmark_runs/2026-04-08_08-04-48__curriculum`
  - `task_config.tree_observation_backend: flatland`
- **cutils backend**
  - `fine_tuned/flatland/benchmark_runs/2026-04-07_18-52-19__curriculum`
  - `task_config.tree_observation_backend: flatland_cutils`

---

## Implementation difference (what is actually different)

### Backend switch (config + env construction)
`benchmarl/environments/flatland/flatland.py`

```python
observation_backend = str(self.config.get("tree_observation_backend", "flatland_cutils"))
...
if observation_backend == "flatland_cutils":
    from flatland_cutils import TreeObsForRailEnv as TreeCutils
    obs_builder = TreeCutils(tree_num_nodes, tree_predictor_depth)
elif observation_backend == "flatland":
    from flatland.envs.observations import TreeObsForRailEnv as TreePython
    obs_builder = TreePython(max_depth=tree_max_depth)
```

### cutils path
`benchmarl/environments/flatland/torchrl_flatland_env.py`

```python
# direct cutils arrays
"agents_attr": observations[0]
"node_attr": observations[1][0]
"adjacency": observations[1][1]
"node_order": observations[1][2]
"edge_order": observations[1][3]
_, _, valid_actions = self.obs_builder.get_properties()
```

### python-Flatland path
`benchmarl/environments/flatland/python_tree_adapter.py`

```python
# converts TreeObs Node trees -> fixed tensors
agents_attr, node_attr, adjacency, node_order, edge_order, valid_actions = self.tree_adapter.encode(...)
```

Notable: python adapter synthesizes `agents_attr` and computes `valid_actions` from rail transitions. That is semantically close, but not byte-identical to cutils features.

---

## Results summary

## Phase 1 (`delay=1, departure=0`)

| Backend | Eval last arrival | Eval max arrival | Eval last deadlock | Collection mean arrival | Collection mean deadlock |
|---|---:|---:|---:|---:|---:|
| `flatland` | 0.5714 | 0.5714 | **0.0264** | **0.5737** | **0.0462** |
| `flatland_cutils` | **0.7429** | **0.8000** | 0.1571 | 0.5170 | 0.3572 |

Interpretation: cutils spikes higher arrival early, but with much higher deadlock footprint.

## Phase 2 (`delay=0, departure=1`)

| Backend | Eval last arrival | Eval max arrival | Eval last deadlock | Eval last reward | Collection mean arrival | Collection mean deadlock |
|---|---:|---:|---:|---:|---:|---:|
| `flatland` | **0.7393** | 0.7464 | **0.0000** | **4.509** | **0.7253** | **0.0026** |
| `flatland_cutils` | 0.6000 | **0.8714** | 0.0857 | 3.857 | 0.6639 | 0.1101 |

Interpretation: cutils shows higher *peak* but unstable trajectory; flatland backend finishes significantly stronger and cleaner (near-zero deadlocks).

---

## Throughput / cost

Using `(collection_time + training_time) / frames` (less sensitive to eval settings than wall-clock total):

| Backend | Phase 1 sec / 1M frames | Phase 2 sec / 1M frames |
|---|---:|---:|
| `flatland` | **4436.3** | **1948.3** |
| `flatland_cutils` | 4494.1 | 1959.0 |

Difference is small; no meaningful speed advantage for cutils in these runs.

---

## Caveat (important)
These two runs are not perfectly identical in eval sampling:
- `flatland` run used `evaluation_episodes=200`
- `flatland_cutils` run used `evaluation_episodes=10`

So cutils eval curves are noisier and peaks are easier to overestimate.

That said, **collection-side metrics** (which do not depend on eval episode count) also favor `flatland` in both phases, especially deadlock ratio.

---

## Conclusion
For this MLP curriculum setting, **`flatland` backend is better overall**:

1. **Better final policy quality in phase 2** (higher final arrival, lower deadlock, higher reward).
2. **Much better deadlock behavior** in both eval and collection metrics.
3. **No external binary dependency** (`flatland_cutils` optional), which improves reproducibility/portability.
4. **No throughput penalty** observed in this comparison.

`flatland_cutils` still has value if you optimize for short-term peak arrival and want exact cutils feature semantics, but based on these end-to-end curriculum outcomes, it is the weaker default.
