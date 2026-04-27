# LeWM Engine Contracts

Bridge document for the prefork refactor (M-series). Describes both the legacy
`info_dict` shape that is still threaded through `jepa.py`,
`harness/pipeline.py`, `harness/compiled_inference.py`, and
`harness/value_cost.py`, and the new typed contracts in
`harness/contracts.py` that are replacing it.

The legacy shape is **in flux** — it disappears once M3 (Consolidate to one
canonical rollout) lands. New code should import from `harness.contracts`.

---

## 1. Shape conventions

These letters appear in every signature below. They are reproduced from
`harness/contracts.py`.

| Symbol | Meaning |
|--------|---------|
| `B`    | batch — independent planning instances |
| `S`    | samples per CEM batch (action candidates) |
| `T`    | time / sequence length |
| `D`    | embedding dim (currently 192 — see §5) |
| `A`    | action dim |
| `H`    | history length (sliding window the predictor attends to) |

Wherever a function says it returns `(B, S, T, D)`, it means **exactly that** —
no implicit broadcasting, no `None` placeholders. Callers reshape explicitly.

---

## 2. Legacy `info_dict` reference

The dict is mutated in place as it flows through `encode → rollout → criterion`.
Keys appear/disappear at well-defined points; this table documents who creates
each key and who reads it.

| Key             | dtype  | Shape                | Written by                             | Read by                                                      |
|-----------------|--------|----------------------|----------------------------------------|--------------------------------------------------------------|
| `pixels`        | float  | `(B, T, C, H, W)`    | caller (pipeline preprocessing)        | `JEPA.encode`, `JEPA.rollout`, `compiled_inference.optimized_rollout` |
| `emb`           | float  | `(B, T, D)`          | `JEPA.encode` (writes `(b t) d → b t d`) | `JEPA.predict`, callers extracting state embeddings        |
| `action`        | float  | `(B, T_hist, A)` or `(B, S, T_hist, A)` after split | `JEPA.rollout` (`act_0` slice), caller for training | `JEPA.encode` (drives `act_emb`)            |
| `act_emb`       | float  | `(B, T, A_emb)`      | `JEPA.encode` (when `action` present)  | `JEPA.predict`                                              |
| `goal`          | float  | `(B, T, C, H, W)`    | caller                                 | `JEPA.get_cost` (renamed to `pixels` for goal sub-encode), `value_cost.ValueCostModel.get_cost` |
| `goal_*`        | varies | `(B, T, ...)`        | caller (any extra goal-side keys)      | `JEPA.get_cost` strips `goal_` prefix and re-attaches to the goal sub-dict |
| `goal_emb`      | float  | `(B, T_goal, D)`     | `JEPA.get_cost`, `ValueCostModel.get_cost` | `JEPA.criterion`                                          |
| `predicted_emb` | float  | `(B, S, T_full, D)`  | `JEPA.rollout`, `optimized_rollout`    | `JEPA.criterion`, `ValueCostModel.get_cost`                 |

`T_full = T_hist + n_future_steps + 1` (initial states + per-step predictions
\+ a terminal predict). `T_hist` is the rollout-time history (often 1 from a
single observation, or 3 when seeded from a window).

### Lifecycle (legacy path)

```
caller         { pixels, action?, goal, goal_* }
  │
  ├─ JEPA.encode      adds: emb, act_emb
  ├─ JEPA.get_cost    adds: goal_emb (re-encodes goal as pixels)
  ├─ JEPA.rollout     adds: predicted_emb   (consumes emb, action)
  └─ JEPA.criterion   reads:  predicted_emb, goal_emb   →   cost (B, S)
```

Notable quirks the refactor will eliminate:
- `JEPA.get_cost` builds a goal sub-dict by slicing `[:, 0]` off every tensor,
  renames `goal` → `pixels`, strips `goal_` prefixes, drops `action`, then
  re-encodes. This is fragile; it relies on `pixels` being the canonical key.
- `info["action"]` is rebound mid-rollout to the historical-action slice
  (`act_0`), then `act_future` is fed step-by-step. Consumers of the dict
  after rollout see only the historical slice.
- `info["emb"]` is overwritten by `info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)`
  inside `rollout`, so its shape changes from `(B, T, D)` to `(B, S, T, D)` mid-call.

---

## 3. New typed contracts (`harness/contracts.py`)

Five dataclasses + one Protocol replace the dict.

### `RolloutRequest`
Inputs to a model rollout. Validated in `__post_init__`.

```python
RolloutRequest(
    state=obs_emb,                   # (B, T_hist, D)
    actions=candidates,              # (B, S, T_horizon, A)
    history_size=3,
)
```

### `RolloutResult`
Output of a model rollout.

```python
result = RolloutResult.from_trajectory(traj)   # traj: (B, S, T, D)
result.terminal                                # (B, S, D)
```

### `Cost` (Protocol)
Pluggable scoring over a rolled-out trajectory. **Lower is better.**

```python
class Cost(Protocol):
    def __call__(self, trajectory: Tensor, goal: Tensor) -> Tensor: ...
    # trajectory: (B, S, T, D), goal: (B, T_goal, D) → (B, S)
```

Known implementers: `TerminalMSECost`, `ValueCost` (`harness/value_cost.py`),
`DreamScorerCost` (`harness/dream_scorer.py`).

### `TerminalMSECost`
Default cost — MSE between final predicted state and last goal step. Mirrors
the historical `pipeline._evaluate_candidates` default.

```python
cost = TerminalMSECost()(trajectory, goal)   # (B, S)
```

### `PlanRequest`
Solver input for `CEMSolver` and Dream Tree.

```python
PlanRequest(state=obs_emb, goal=goal_emb, horizon=5)
```

### `PlanOutcome`
Solver output at the engine layer (one level below `PlanResult`, which is the
**planner's** output in `harness/plan_result.py` and adds confidence /
timing / replan signals on top).

```python
PlanOutcome(
    action=best_action,    # (B, A)
    terminal=term_emb,     # (B, D)
    cost=best_cost,        # (B,)
)
```

---

## 4. Migration table

For each legacy `info_dict` use, what the new contract replaces it with.

| Legacy use                              | New contract                                 |
|-----------------------------------------|----------------------------------------------|
| `info["pixels"]`                        | Caller-side; encoder takes a raw tensor, no dict. |
| `info["emb"]` (post-encode)             | `RolloutRequest.state`                       |
| `info["action"]` (history slice)        | First `T_hist` slice of `RolloutRequest.actions` |
| `info["act_emb"]`                       | Computed inside the predictor; not exposed.  |
| `info["goal"]`, `info["goal_*"]`        | Caller pre-encodes; passes embedding directly. |
| `info["goal_emb"]`                      | `PlanRequest.goal` and `Cost.__call__(goal=…)` |
| `info["predicted_emb"]`                 | `RolloutResult.trajectory`                   |
| Terminal slice `predicted_emb[..., -1, :]` | `RolloutResult.terminal`                  |
| `JEPA.criterion` (default MSE)          | `TerminalMSECost`                            |
| `JEPA.get_cost(...)` return value       | `Cost(trajectory, goal) → (B, S)`            |
| `pipeline._cem_plan` action + cost      | `PlanOutcome`                                |
| `pipeline._evaluate_candidates`         | `Solver` over `RolloutRequest` + `Cost`      |

The big simplification: **encode happens at the caller, not inside the cost
path.** That removes the goal/pixels rename gymnastics in `JEPA.get_cost`
and `ValueCostModel.get_cost`.

---

## 5. Dimension constants

Currently hardcoded throughout the codebase:

| Constant   | Value | Where it lives                                                                 |
|------------|-------|--------------------------------------------------------------------------------|
| `embed_dim` | 192  | `harness/goal_adapter.py` (`(B, 192)` projection target), `harness/language_encoder.py` (`(1, 1, 192)` output), `harness/pipeline.py` docstrings (`(1, 1, 192)`), `harness/dream_scorer.py` (`ckpt.get("embed_dim", 192)` default) |
| `action_dim` | varies | `pipeline._action_dim` infers from `model.action_encoder.patch_embed.in_channels` at runtime |

`embed_dim=192` is a property of the trained encoder (LeJEPA), not a free
choice — but it appears as a literal in projection layers, language encoders,
and goal adapters. **Flagged for M4+S5**: promote `embed_dim` and `action_dim`
to spec fields on `LeWMSpec` so swapping checkpoints with different widths
does not require source edits. Until then, treat 192 as a soft constant: any
new code that needs it should read `model.predictor.embed_dim` (or equivalent)
rather than hardcoding the literal.

---

## See also

- `harness/contracts.py` — source of truth for §1 and §3.
- `harness/plan_result.py` — `PlanResult`, the planner-level output (one layer
  above `PlanOutcome`).
- `docs/HARNESS.md` — overall harness architecture.
- `docs/STATE.md` — current refactor status across M1-M5.
