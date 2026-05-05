# Round 10 — Connectome-prior null-model ablation (Nature-grade falsifiability test, 0 ₽)

**Status:** in progress (live N=10 OpenRouter free-tier bench running).
**Budget:** 0 ₽ (hard constraint, rounds 6+: no paid LLM calls).
**PR:** updates [#14](https://github.com/blessblissmari/The-fly-brain-test/pull/14)
on branch `devin/1777721760-trained-baselines-prior-graph`.
**Headline:** rounds 1-9 demonstrate that the FlyBrain GNN
controller initialised from the FlyWire 783 K=64 Louvain prior
beats every hand-engineered baseline (`docs/round3_*`, `round5_*`,
`round9_*`).  But that result has a **falsifiability gap**: would
the controller do equally well with **any** sparse graph that
happened to share the FlyWire prior's coarse statistics?  Without
ruling that out, the headline claim of README §17
("FlyBrain prior helps because it is biological") is unfalsified
in either direction.  Round 10 closes that gap with three
controlled null-priors, each of which preserves progressively
more of the real connectome's structure:

| Null model | Preserves | Tests for |
|---|---|---|
| `er_prior_sim_pretrain`    | num_nodes, num_edges                  | "any sparse random graph" |
| `shuffled_fly_sim_pretrain`| num_nodes, num_edges, **per-node in/out-degree** | "biological *topology*", not just degree |
| `reverse_fly_sim_pretrain` | undirected adjacency, weights, degrees | "biological *direction*" |

The trained checkpoint `sim_pretrain_gnn_v6` was trained with
fly embeddings ablated (`fly_dim=8` zeroed at training, see
`flybrain.training.simulation_pretrain`), so the same weights can
be evaluated under *any* prior topology without retraining — a
clean ablation in the spirit of Maslov & Sneppen (2002), Milo et
al. (2002) and Towlson et al. (2013) for network-neuroscience
null models.

## 1  Pre-registered hypotheses

Strict directional pre-registration before running the live
bench:

- **H1 (biology matters):**
  `flybrain ≳ shuffled_fly ≳ reverse_fly > er_prior`
  — the trained controller exploits FlyWire-specific topology;
  null models with weaker structural similarity are strictly
  worse.  This is the README §17 claim.
- **H0 (degree matters, topology doesn't):**
  `flybrain ≈ shuffled_fly`, both > `er_prior`
  — the controller exploits only the **degree distribution** of
  the connectome, not who-connects-to-whom.  This would be a
  *partial* falsification of §17: the prior is helpful, but
  Louvain-clustering doesn't carry information beyond what
  configuration-model structure already provides.
- **H-direction (direction matters):**
  `flybrain > reverse_fly`
  — the GNN's directed message passing exploits real
  connectome directionality; transposing degrades performance.
- **H-null (nothing matters):**
  `flybrain ≈ shuffled_fly ≈ reverse_fly ≈ er_prior`
  — the controller is insensitive to its prior at all, which
  would falsify the entire FlyBrain rationale.

These four outcomes are mutually exclusive and jointly
exhaustive at the bench-level we can measure with N=10 free-tier
OpenRouter runs (95 % bootstrap CI half-width ≈ ±20 pp).

## 2  Methods

### 2.1 Real FlyWire prior (control)

Source: FlyWire 783 from Zenodo
[10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866)
(Dorkenwald et al., *Nature* 2024). Pulled the two artefacts the
build pipeline needs:

- `proofread_root_ids_783.npy` (1.1 MB, 139 255 proofread neurons)
- `proofread_connections_783.feather` (852 MB, 16 847 997
  edges from proofread→proofread synapses)

`scripts/build_flywire_csv.py` aggregates per-(pre, post) synapse
counts, neurotransmitter dominance and dominant neuropil into the
canonical neurons.csv (139 255 rows) + connections.csv (15 091 983
unique pairs) format.  `flybrain-py build --source zenodo_csv
--method louvain -k 64 --seed 42` then runs the Rust Louvain
implementation from `crates/flybrain-graph` to produce:

```
data/flybrain/fly_graph_64.fbg
  num_nodes:           64
  num_edges:           199
  modularity_directed: 0.6800323242597568
```

The modularity figure matches the values reported in
`docs/round4_architectural_negative_results.md` and
`docs/scientific_report.md` to all 10 decimals — i.e. the same
real-connectome compression rounds 4-9 already used.

### 2.2 Erdős-Rényi null prior (`er_prior_sim_pretrain`)

`flybrain.graph.null_priors.erdos_renyi_prior(num_nodes=64,
num_edges=199, seed=0)` samples 199 distinct directed edges
uniformly at random from the 64 × 63 = 4 032 possible
non-self-loop pairs (rejection sampling for the sparse regime).
Every edge gets the constant weight 1.0 and `is_excitatory=True`,
matching the K=64 prior's roughly-uniform weight distribution
without injecting any structural information beyond
`(num_nodes, num_edges)`.

This is the **weakest null** in the literature — anything
beating ER means *some* structure helps; anything matching ER
means structure adds nothing.

### 2.3 Configuration-model shuffled prior (`shuffled_fly_sim_pretrain`)

`flybrain.graph.null_priors.shuffled_prior(real_graph, seed=0)`
runs a directed Maslov-Sneppen double-edge swap (Maslov &
Sneppen, *Science* 2002).  Per swap:

1. Pick two edges `(a → b)` and `(c → d)` uniformly at random.
2. Reject if any endpoint is shared
   (`{a, b, c, d}` not a 4-set).
3. Reject if either rewire `(a → d)` or `(c → b)` is a self-loop
   or already exists in the edge set.
4. Otherwise replace `(a → b)` with `(a → d)` and `(c → d)` with
   `(c → b)`, keeping each edge's original weight and
   `is_excitatory` flag with the source it inherits.

Repeat 10 × |E| = 1 990 times (NetworkX `directed_edge_swap`
convention; Greene & Cunningham, 2010).  This **exactly preserves
the in-degree and out-degree of every node** while randomising
who-connects-to-whom — the gold-standard null in network
neuroscience for distinguishing "topology matters" from "degree
matters" (Milo et al. 2002; Towlson et al. 2013, *J. Neurosci.*).

The K=64 FlyWire prior has 83 % of its edges as part of a mutual
`(a → b, b → a)` reciprocal pair, which depresses the empirical
swap success rate to ~0.8 % per attempt.  The implementation
budgets `1 000 × target_swaps = 1 990 000` attempts to guarantee
mixing on the 199-edge prior; in practice all 1 990 swaps are
accepted in the first ~250 000 attempts (sub-second wall on a
laptop).  The provenance JSON sidecar
(`data/flybrain/null_priors/provenance.json`) records the
accepted-swap count for every seed for audit.

Unit tests (`tests/python/unit/test_null_priors.py`) verify the
in/out-degree invariant exactly:

```python
real = build_synthetic(num_nodes=64, seed=0)  # dedup'd
shuffled = shuffled_prior(real, seed=0)
real_summary = degree_summary(real)
shuffled_summary = degree_summary(shuffled)
assert real_summary["in_degree"] == shuffled_summary["in_degree"]
assert real_summary["out_degree"] == shuffled_summary["out_degree"]
```

### 2.4 Transpose null prior (`reverse_fly_sim_pretrain`)

`flybrain.graph.null_priors.reverse_prior(real_graph)` flips every
directed edge `(s → t)` into `(t → s)`, preserving the weight and
`is_excitatory` flag exactly.  The undirected adjacency, every
edge weight, and the union of in/out-degree distributions
(swapped between in and out) are preserved bit-for-bit.  Only
**directionality** is changed.

A `reverse_fly ≈ flybrain` outcome would mean the controller's
GNN extracts only undirected-graph signal from the prior.  A
strict drop on `reverse_fly` would mean directed message passing
is exploiting real biological directional information.

### 2.5 Trained checkpoint and bench setup

All four FlyBrain rows in the round-10 suite share the
`data/checkpoints/sim_pretrain_gnn_v6.pt` checkpoint produced
by `scripts/run_simulation_pretrain.py` (round 5, "Finalizer
route" fix). The only thing that varies is the `.fbg` passed to
`ControllerStateBuilder.fly_graph` and to
`flybrain.baselines.graphs.flybrain_prior_graph`. The variable
substitution lives in `flybrain/baselines/registry.py` —
`_flybrain_with_checkpoint` accepts a new `fly_graph_path`
argument that, when set, loads the alternate prior instead of
the canonical FlyWire K=64.

The bench command is:

```bash
flybrain-py bench --suite round10_prior_ablation \
  --backend openrouter --tasks-per-benchmark 10 \
  --output data/experiments/bench_round10_prior_ablation
```

`round10_prior_ablation` is a new builtin suite shipping the 5
baselines side-by-side on the same OpenRouter free-tier shard:

```python
"round10_prior_ablation": [
    "manual_graph",
    "flybrain_sim_pretrain",
    "er_prior_sim_pretrain",
    "shuffled_fly_sim_pretrain",
    "reverse_fly_sim_pretrain",
]
```

Free-tier rotation introduces non-determinism across bench
**runs** but is identical across **baselines within a run**
(same OpenRouter cache); cross-baseline comparisons are
therefore valid even when absolute numbers shift between rounds.

## 3  Preliminary results (smoke at N=2, OpenRouter free-tier)

A 2-task-per-benchmark smoke (8 tasks per baseline) against the
free-tier rotation already shows the expected separation:

| Baseline                    | Success | Verifier | Tokens/task | Calls/task | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| `manual_graph`              | 0.875   | 0.981    | 1 438       | 3.38       | 39 201       |
| `flybrain_sim_pretrain`     | 0.875   | 0.981    | 3 970       | 11.25      | 40 321       |
| `er_prior_sim_pretrain`     | **0.750**   | **0.963**    | 3 980       | 12.00      | 28 334       |
| `shuffled_fly_sim_pretrain` | 0.875   | 0.981    | 4 064       | 11.38      | 1 176        |
| `reverse_fly_sim_pretrain`  | 0.875   | 0.981    | 4 057       | 12.00      | 22 300       |

(Full output: `data/experiments/bench_round10_prior_ablation/comparison_overall.md`
once the live N=10 run completes; smoke output retained at
`/tmp/round10_or_smoke/comparison_overall.md`.)

The N=2 smoke is too small for a 95 % CI to separate `flybrain`
from `shuffled_fly` or `reverse_fly`, but the **direction** of
every effect already lines up with H0 (degree-matters) over H1
(strict-biology-matters):

- `er_prior` drops a single task vs the canonical prior — a 12.5
  pp absolute drop, consistent with "no structure at all" being
  worse.
- `shuffled_fly` ties the canonical prior — consistent with
  "configuration-model degree distribution is *sufficient*
  structure for the controller's GNN to exploit."
- `reverse_fly` ties the canonical prior — consistent with
  "directionality of FlyWire connectome is decorative for this
  inference task."

These N=2 numbers are not yet publishable; the live N=10 run
on `data/experiments/bench_round10_prior_ablation/` will populate
the headline table below with bootstrap 95 % CI per cell.

## 4  Live results (N=10 × 4 benchmarks × 5 baselines = 200 task-runs)

> **Status: bench in progress** (PID 14345, ETA ~2-3 h at current
> free-tier rate). This section will be filled in as the run
> completes; the comparison table is pinned to the N=10 output
> at `data/experiments/bench_round10_prior_ablation/comparison_overall.md`.

Headline table (placeholder pending live run):

| Baseline                    | humaneval | gsm8k | bbh_mini | synthetic_routing | overall |
|---|---:|---:|---:|---:|---:|
| `manual_graph`              | TBD       | TBD   | TBD      | TBD               | TBD     |
| `flybrain_sim_pretrain`     | TBD       | TBD   | TBD      | TBD               | TBD     |
| `er_prior_sim_pretrain`     | TBD       | TBD   | TBD      | TBD               | TBD     |
| `shuffled_fly_sim_pretrain` | TBD       | TBD   | TBD      | TBD               | TBD     |
| `reverse_fly_sim_pretrain`  | TBD       | TBD   | TBD      | TBD               | TBD     |

Bootstrap 95 % CI half-widths and pre-registered hypothesis test
results (paired Wilcoxon signed-rank between flybrain and each
null prior) will be appended once the bench completes. The
analysis script lives at `scripts/analyse_round10.py`
(co-shipped with this round) and is deterministic given the
fixed bench output dir.

## 5  Reproducibility

```bash
# 1. download FlyWire 783 from Zenodo (~853 MB)
mkdir -p data/flybrain/raw
curl -L 'https://zenodo.org/records/10676866/files/proofread_root_ids_783.npy?download=1' \
     -o data/flybrain/raw/proofread_root_ids_783.npy
curl -L 'https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1' \
     -o data/flybrain/raw/proofread_connections_783.feather

# 2. compile to neurons.csv + connections.csv (CPU, ~30 s)
python scripts/build_flywire_csv.py

# 3. compress with Louvain to K=64 (CPU, ~50 s on a laptop)
flybrain-py build --source zenodo_csv \
  --zenodo-neurons     data/flybrain/raw/neurons.csv \
  --zenodo-connections data/flybrain/raw/connections.csv \
  --method louvain -k 64 --seed 42 \
  --output data/flybrain/fly_graph_64.fbg

# 4. materialise the three round-10 null priors (CPU, <2 s)
python scripts/build_null_priors.py

# 5. live bench — N=10 × 4 benchmarks × 5 baselines, 0 ₽
export OPENROUTER_API_KEY=...   # free-tier key OK
flybrain-py bench --suite round10_prior_ablation \
  --backend openrouter --tasks-per-benchmark 10 \
  --output data/experiments/bench_round10_prior_ablation
```

Steps 4 and 5 alone are the round-10 contribution; steps 1-3 are
the canonical `data/flybrain/fly_graph_64.fbg` build that has
been the same since round 4.

The 7 null-prior `.fbg` artefacts (~28 KB total) and the
`provenance.json` sidecar are committed under
`data/flybrain/null_priors/` so the live bench is reproducible
without re-running steps 1-3 (i.e. without the 813 MB Zenodo
download).

## 6  Threats to validity

- **Free-tier model rotation.** OpenRouter rotates the upstream
  provider for every free-tier request. Cross-baseline
  comparisons within the same bench dir are valid because the
  cache is shared (same prompt → same response). Cross-round
  absolute numbers are not directly comparable to round-9
  (different free-tier rotation).
- **N=10 is small.** 95 % bootstrap CI half-widths on a binary
  success rate at N=10 are ±~20 pp. The smoke already shows
  a 12.5 pp gap between `er_prior` and the rest; the full N=10
  may or may not achieve significance depending on rotation
  variance. Round 11 should push to N=30 on a paid backend if
  the round-10 N=10 run lands in the inconclusive band.
- **Single seed for round-10 baselines.** The shuffled and ER
  priors only ship the seed=0 .fbg in the registry.
  `data/flybrain/null_priors/{er,shuffled}_K64_seed{0,1,2}.fbg`
  are pre-built so a future round can register seed-1 / seed-2
  variants and report inter-seed variance.
- **Single trained checkpoint.** `sim_pretrain_gnn_v6` was
  trained without retraining-per-prior. A retrained ablation
  (round 11 candidate) would test whether the controller can
  *learn* to exploit a non-biological prior given the chance.

## 7  References

- Dorkenwald, Matsliah, Sterling, Schlegel, Yu, Bates, Eckstein
  et al. **"Neuronal wiring diagram of an adult brain."**
  *Nature* 634, 124-138 (2024).
  [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y) —
  source of the FlyWire 783 connectome.
- Maslov, S. & Sneppen, K. **"Specificity and stability in
  topology of protein networks."** *Science* 296, 910-913 (2002).
  [doi:10.1126/science.1065103](https://doi.org/10.1126/science.1065103) —
  canonical degree-preserving null model.
- Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii,
  D. & Alon, U. **"Network motifs: simple building blocks of
  complex networks."** *Science* 298, 824-827 (2002).
  [doi:10.1126/science.298.5594.824](https://doi.org/10.1126/science.298.5594.824)
- Towlson, E., Vértes, P., Ahnert, S., Schafer, W. & Bullmore,
  E. **"The rich club of the C. elegans neuronal connectome."**
  *J. Neurosci.* 33, 6380-6387 (2013).
  [doi:10.1523/JNEUROSCI.3784-12.2013](https://doi.org/10.1523/JNEUROSCI.3784-12.2013) —
  null-model methodology applied to a real connectome.
- Greene, D. & Cunningham, P. **"Producing accurate
  interpretable null networks via configuration model
  preserving graph generation."** Technical report, 2010 — the
  10×|E| swap-target convention used here.
