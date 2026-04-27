# Imagination Engine — Build Plan

A model-agnostic, paradigm-agnostic inference runtime for learned world models. Given a starting state and a batch of candidate action sequences, it produces predicted future trajectories — and optionally pixels, costs, values, and confidence signals — fast enough to live inside an MPC / CEM / MCTS inner loop. Generalizes the role LeHarness plays for LeWM into a substrate that also runs diffusion video models (Waypoint-class), discrete-token AR models (Genie-class), and compact latent dynamics (Dreamer/TD-MPC2-class) through a single API.

**One-liner:** vLLM, but the unit of work is a *rollout*, not a token.

---

## 0. Strategic thesis

There are two valid theses; the build plan diverges based on which one is committed to.

**Thesis A — imagination engine for LeWM-class planners.**
Batched deterministic rollout with confidence/cost outputs. Different KV semantics (rollouts diverge from a shared prefix), different solver (one-step prediction, not diffusion ODE), batched B≫1 mandatory.

**Thesis B — "vLLM for interactive video WMs."**
Multi-tenant server, continuous batching across user sessions, model-agnostic registry, streaming output. Closest open-ecosystem moat — no one has shipped this.

This document covers both. Sections marked **(A)** or **(B)** are thesis-specific; everything else is shared.

---

## 1. Current state — what exists right now

The pre-fork refactor is committed at `cdae34a` on `main` (and `prefork-refactor`). Everything described below is in the tree today.

### Engine surface (will move to fork verbatim)

| File | LOC | What it provides |
|---|---|---|
| `harness/contracts.py` | 201 | `RolloutRequest`, `RolloutResult`, `Cost` Protocol, `TerminalMSECost`, `PlanRequest`, `PlanOutcome` |
| `harness/cem.py` | 333 | `CEMSolver` — pipeline-free CEM with `plan` / `plan_batched` / `score_state` / `evaluate_candidates` |
| `harness/rollout.py` | 147 | `rollout_buffered` + `ModelAdapter` Protocol + `JepaAdapter`. Single canonical rollout |
| `harness/dims.py` | 17 | `LEWM_EMBED_DIM` single source of truth |
| `harness/compiled_inference.py` | 68 | `optimize_model(adapter_cls=...)` — torch.compile + buffer rollout patch |
| `harness/specs/base.py` | 93 | `ModelSpec`, `RunnerKind`, `OutputKind`, `EncoderSpec`, `CacheSpec`, `SchedulerSpec` |
| `harness/specs/lewm_spec.py` | 82 | `LEWM_SPEC` — first declarative spec |
| `docs/CONTRACTS.md` | 192 | Bridge document for typed contracts and legacy info_dict |

These modules are import-clean: `python -c "import harness.{contracts,cem,rollout,dims,specs,compiled_inference}"` succeeds without `stable_pretraining` or `stable_worldmodel`. Verified.

### Planner surface (stays as LeHarness package)

`harness/pipeline.py` (now a thin facade), `dream_*.py`, `drift_detector.py`, `goal_adapter.py`, `language_encoder.py`, `projections.py`, `protocols.py`, `plan_result.py`, `s15_loop.py`, `sim_components.py`, `subgoal_sequencer.py`, `value_function.py`, `value_cost.py`, `adaptive_solver.py`. All consume the engine via the public API (`pipeline.solver`, `pipeline.action_dim`, `pipeline.obs_emb`, `Cost` protocol). No private reach-throughs.

### Test coverage

198 passing tests on CPU. 69 are new engine-side tests covering:
- `tests/test_contracts.py` — typed contract shape validation (14)
- `tests/test_cem.py` — `CEMSolver` plan/plan_batched/score_state/evaluate_candidates (16)
- `tests/test_rollout.py` — `JEPA.rollout` shapes, determinism, parameter handling (16)
- `tests/test_compiled_inference.py` — `optimize_model`, parity vs legacy, `ModelAdapter` (16)
- `tests/test_specs.py` — `LEWM_SPEC` field validation (7)

Plus the original 129 planner tests (drift, goal_adapter, plan_result, s15_loop, sim_components).

### What's still on-pod-only

`harness/pipeline.py` itself (and anything that imports it) loads `stable_worldmodel` + `stable_pretraining` which require GPU torchvision. Tests of `PlanningPipeline` end-to-end run on the RTX 4090, not on CPU dev machines. Not an issue for the fork — the engine modules are already clean.

---

## 2. Architecture

Three layers. Bottom is forked from le-wm. Middle is new. Top is per-model.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer C — Model Specs   (per-architecture, declarative)             │
│  WaypointSpec │ LeWMSpec │ GenieSpec │ DreamerSpec │ TDMPC2Spec │ … │
└─────────────────────────┬────────────────────────────────────────────┘
                          │ registers
┌─────────────────────────▼────────────────────────────────────────────┐
│  Layer B — Runners       (per execution paradigm)                    │
│  DiffusionRunner │ DeterministicRunner │ ARTokenRunner               │
│  SearchRunner   (wraps any of the above for batched search)          │
└─────────────────────────┬────────────────────────────────────────────┘
                          │ uses
┌─────────────────────────▼────────────────────────────────────────────┐
│  Layer A — Primitives    (model-agnostic infrastructure)             │
│                                                                      │
│  • Compiled step harness  (fullgraph + cudagraphs)                   │
│  • Cache subsystem        (ring / ring+tail / paged / contiguous)    │
│  • Quantization registry  (int8 / fp8 / nvfp4)                       │
│  • Conditioning encoders  (text / image / vlm-emb / action)          │
│  • Scheduler interface    (rectified-flow / EDM / DDPM)              │
│  • Snapshot / Restore     (state handles, copy-on-fork)              │
│  • Request scheduler      (continuous batching, server mode)  (B)    │
└──────────────────────────────────────────────────────────────────────┘
```

### Layer A — Primitives

What we already have (lift from le-wm):
- Compiled step harness pattern (`compiled_inference.py`).
- Cost protocol + TerminalMSECost.
- ModelAdapter abstraction (proven by custom-adapter test).
- `rollout_buffered` canonical rollout.
- `LEWM_EMBED_DIM` single source of truth.

What we need to add:
- **Generalized cache subsystem.** LeWM has no KV cache. The fork must support `ring+tail` (Waypoint, world_engine.git), `paged` (continuous batching), `none` (LeWM, JEPA family).
- **Snapshot / restore.** First-class `fork(state) → handle / restore(handle)`. Used for branching search and (B) for per-stream isolation.
- **Quantization registry.** Lift world_engine.git's int8 (gemlite), FP8 (`torch._scaled_mm`), NVFP4 (FlashInfer) stack. Adapt the post-load patch pattern so quantization is opt-in per spec.
- **Scheduler interface.** Uniform `Scheduler.steps()` + `Scheduler.advance(x, v, σ_t, σ_{t+1})` so rectified-flow / EDM / DDPM / consistency all conform. Bake the world_engine.git bf16-LUT trick automatically when sigmas are fixed.
- **Conditioning encoders.** Typed registry: text (UMT5/T5/CLIP), image (SigLIP/DINO/CLIP), VLM-embedding (with projection MLPs — already in le-wm), action (continuous / discrete / controller).
- **Request scheduler (B).** Continuous-batching server — packs concurrent rollout requests into B>1 batches.

### Layer B — Runners

Each runner shapes the inner loop. The runner exposes `step(state, conditioning, cache) → output, new_state` and `rollout(state, action_seq) → trajectory` (cudagraph-captured composition of N steps).

- **`DeterministicRunner`** — single forward, no inner loop. JEPA, V-JEPA, DINO-WM, Dreamer-decode. Cheapest. Default for control. **What we have today is exactly this shape — `JEPA.rollout` + `predict` is one Deterministic step.**
- **`DiffusionRunner`** — runs an ODE/SDE solver via the scheduler interface. Waypoint-1.5, Cosmos-Predict. Includes a distillation hook (1-step inference if checkpoint supports it).
- **`ARTokenRunner`** — autoregressive over discrete latents. Genie-style, GameNGen. KV-cache-shaped, sampler-driven.
- **`SearchRunner`** — wraps any of the above for **batched parallel rollouts from a shared prefix**, with a pluggable cost. **`CEMSolver` today is roughly a SearchRunner specialized to a Deterministic inner runner.**

### Layer C — Model Specs

Each model is one declarative spec. We already have `LEWM_SPEC`. Adding a model is a one-file change. Sketch:

```python
@register_model("waypoint-1.5")
class WaypointSpec(ModelSpec):
    runner = DiffusionRunner
    cache  = CacheSpec(layout="ring+tail",
                       layer_pattern="local|global@period",
                       local_window=W_l, global_window=W_g,
                       global_dilation=...)
    encoders = {"prompt": UMT5("google/umt5-xl"),
                "ctrl":   ControllerEmbed(64, 2, 1)}
    scheduler = RectifiedFlow(sigmas_from_config=True)
    output = OutputKind.IMAGE
    quant_blacklist = ["ctrl_emb","out_norm","unpatchify"]
    inference_patches = ["cached_cond","merged_qkv","split_mlpfusion"]
    builder = WorldDiT
```

---

## 3. The fork

### Files that move (verbatim, no edits)

```
harness/contracts.py            → world_engine_le/contracts.py
harness/cem.py                  → world_engine_le/cem.py            [becomes SearchRunner]
harness/rollout.py              → world_engine_le/rollout.py        [becomes DeterministicRunner]
harness/dims.py                 → world_engine_le/dims.py
harness/compiled_inference.py   → world_engine_le/compile.py
harness/specs/                  → world_engine_le/specs/
docs/CONTRACTS.md               → world_engine_le/docs/CONTRACTS.md

tests/test_contracts.py         → world_engine_le/tests/
tests/test_cem.py               → world_engine_le/tests/
tests/test_rollout.py           → world_engine_le/tests/
tests/test_compiled_inference.py → world_engine_le/tests/
tests/test_specs.py             → world_engine_le/tests/
tests/_engine_fixtures.py       → world_engine_le/tests/
```

### What stays in le-wm (becomes the planner package consuming the fork)

`pipeline.py`, `dream_*.py`, `drift_detector.py`, `goal_adapter.py`, `language_encoder.py`, `projections.py`, `protocols.py`, `plan_result.py`, `s15_loop.py`, `sim_components.py`, `subgoal_sequencer.py`, `value_function.py`, `value_cost.py`, `adaptive_solver.py`. Imports change from `from harness.cem import CEMSolver` to `from world_engine_le.cem import CEMSolver`.

### Repo skeleton for the fork

```
world_engine_le/
  __init__.py              # exports: Engine, Spec registry, runners
  contracts.py             # lifted
  cem.py                   # lifted; later promoted into runners/search.py
  rollout.py               # lifted; later promoted into runners/deterministic.py
  dims.py                  # lifted
  compile.py               # lifted
  
  runners/
    __init__.py
    base.py                # Runner Protocol + step/rollout signatures
    deterministic.py       # consumes rollout.py
    diffusion.py           # NEW: scheduler-driven ODE solver
    ar_token.py            # NEW: discrete-token AR
    search.py              # consumes cem.py (later: + iCEM, MPPI, MCTS)

  cache/
    __init__.py
    base.py                # CacheSpec → cache instance dispatch
    none.py                # for JEPA-style models
    ring.py                # ring buffer (lifted from world_engine.git)
    paged.py               # paged for continuous batching (B)
    snapshot.py            # fork/restore primitive

  schedulers/
    __init__.py
    rectified_flow.py
    edm.py
    ddpm.py
    bf16_lut.py            # cached conditioning trick from world_engine.git

  encoders/
    __init__.py
    text.py                # UMT5 / T5 / CLIP
    image.py               # SigLIP / DINO / CLIP
    vlm.py                 # projection MLPs (lift from le-wm projections.py)
    action.py              # continuous / discrete / controller

  quant/                   # lifted/adapted from world_engine.git
    __init__.py
    int8.py
    fp8.py
    nvfp4.py
    registry.py

  specs/
    base.py                # lifted
    lewm.py                # lifted
    waypoint.py            # NEW
    tdmpc2.py              # NEW

  server/                  # (B) only
    __init__.py
    scheduler.py           # request scheduling + continuous batching
    grpc.py                # transport

  docs/
    CONTRACTS.md           # lifted
    ARCHITECTURE.md        # this doc, restructured for the fork repo
```

---

## 4. Build steps in priority order

No timelines. Each step has an **acceptance gate** (concrete signal it's done).

### 4.1 Pre-fork (remaining in le-wm)

**Step P1 — On-pod parity benchmark.**
Run `scripts/final_benchmark.py` and `eval.py --config-name=pusht policy=pusht/lejepa solver=cem solver.num_samples=128 solver.n_steps=15 solver.topk=25 eval.num_eval=50` on the RTX 4090. Compare against the pre-refactor numbers in `docs/PROJECT.md` and `docs/S1_5.md`.
*Acceptance:* planning latency within 5%, success rate within 2 percentage points.

**Step P2 — Migrate `value_cost.py` and `dream_scorer.py` to `Cost` protocol.**
Both currently implement their own `get_cost(info_dict, ...)` interface. Wrapping them as `Cost` makes them pluggable into `CEMSolver(cost=...)` directly.
*Acceptance:* `tests/test_contracts.py::TestCost` extended to cover both, all 198 tests still green, scripts unchanged.

**Step P3 — Promote `pipeline.n_steps`/`num_samples`/`horizon`/`topk`/`history_size` to solver-backed properties.**
Eliminates the "sync hparams every shim call" idiom.
*Acceptance:* shim methods become one-line delegates; tests still green.

### 4.2 Fork bootstrap

**Step F1 — Initialize fork repo skeleton.**
Create `world_engine_le/` with the skeleton above. `pyproject.toml` declares deps: `torch`, `einops`, plus optional `[diffusion]`, `[quant]`, `[server]` extras.
*Acceptance:* `pip install -e .` works; `python -c "import world_engine_le"` succeeds.

**Step F2 — Lift engine modules verbatim.**
Move the 9 files listed in §3 + 6 test files. No edits beyond import-path rewrites. CI runs the 69 engine tests.
*Acceptance:* engine tests pass on CPU in CI; `LEWM_SPEC` and `CEMSolver` work end-to-end (see the demo in §6.1).

**Step F3 — Le-wm consumes the fork.**
Update `harness/pipeline.py` (and other callers) to import from `world_engine_le` instead of `harness.cem` etc. Delete the duplicated files in le-wm.
*Acceptance:* le-wm 198 tests still pass; `eval.py` still works on-pod.

### 4.3 Runners and the second spec

**Step F4 — Promote `Runner` taxonomy.**
Define `runners/base.py::Runner` Protocol. `rollout.py` becomes `runners/deterministic.py`. `cem.py` becomes `runners/search.py`. Existing tests adjust import paths.
*Acceptance:* test suite green; existing API surface unchanged from outside.

**Step F5 — Add second spec.** Pick one:
- **F5a (Thesis A path):** `tdmpc2.py` — different-shaped deterministic runner with continuous control. Forces the conditioning encoder abstraction (continuous action vectors instead of buttons/mouse).
- **F5b (Thesis B path):** `waypoint.py` — DiffusionRunner + ring+tail KV cache + scheduler. Forces both the cache subsystem and the scheduler interface.

The **second spec is the abstraction-pressure exercise**. Don't extract a registry until it's working.
*Acceptance:* spec loads, runs end-to-end on a stub model, registers in `world_engine_le.specs`. Spec-specific eval suite (§5.3) passes.

**Step F6 — Extract registry from the two specs.**
Now there are two implementations side by side; the shared structure is obvious. Promote it.
*Acceptance:* both specs are loaded via `Engine(spec="lewm")` and `Engine(spec="waypoint-1.5")` from a unified entry point.

### 4.4 Performance and the third paradigm

**Step F7 — Distillation pipeline (Thesis B mainly).**
Bake a 1-step distillation hook into `DiffusionRunner` for Waypoint-class models. Without this, diffusion-WM is unusable inside a planner.
*Acceptance:* a distilled Waypoint checkpoint runs end-to-end at 1 step with measured quality drop < 5% on internal quality benchmark.

**Step F8 — Cross-frame activation cache (TeaCache-style) (B).**
Block-residual reuse for video DiT. Detect via L1 delta on per-block output.
*Acceptance:* idle-control latency drops by ≥1.5x with no measurable quality regression.

**Step F9 — Third spec — discrete AR (Genie-class).**
Pressure-tests the runner abstraction with a third paradigm. Forces the AR runner + paged cache + sampler.
*Acceptance:* third spec ships, registry holds three working models.

### 4.5 Continuous batching and server (Thesis B only)

**Step F10 — Un-bake `B==1` everywhere.**
Touches model forward, KV cache, prep_inputs, pos_ids. Every existing CPU test must run with B>1 too.
*Acceptance:* batched B=4 path produces same numerical output as B=1 looped (parity test added).

**Step F11 — Paged cache + request scheduler.**
PagedKV-style snapshot/restore. Scheduler aligns frame deadlines across streams and packs into B>1 batches.
*Acceptance:* serve 4 concurrent rollout streams from one model with combined throughput ≥ 1.5x single-stream.

**Step F12 — Server transport (gRPC or HTTP+SSE).**
External entry point.
*Acceptance:* round-trip latency under target threshold (model-dependent), graceful shutdown, request quotas enforced.

### 4.6 Operationalization

**Step F13 — KV cache quantization.** int8 KV → 2x cache memory, longer context.
**Step F14 — NVFP4 broader coverage.** MoE `F.grouped_mm` weights covered.
**Step F15 — Async VAE decode** (Thesis B). CUDA streams + double buffer.
**Step F16 — Multi-GPU pipeline parallelism** (Thesis B, optional).

---

## 5. Evaluation strategy

Every step above has an acceptance gate. Beyond per-step gates, four standing eval suites maintain quality across the build.

### 5.1 Engine evals (run on every PR to the fork)

Located in `world_engine_le/tests/`. CPU-runnable. ~69 tests today; this is the regression net.

| Eval | Asserts | Today |
|---|---|---|
| `test_contracts.py` | shape contracts on RolloutRequest/Result, Cost protocol compliance | 14 ✓ |
| `test_rollout.py` | JEPA.rollout shapes, determinism, history_size acceptance | 16 ✓ |
| `test_cem.py` | CEMSolver plan/plan_batched/score_state shapes, determinism, pluggable Cost | 16 ✓ |
| `test_compiled_inference.py` | optimize_model wraps predictor/encoder, ModelAdapter custom-adapter rollout | 16 ✓ |
| `test_specs.py` | LEWM_SPEC field validation, build_lewm_spec overrides | 7 ✓ |

What to add as the engine grows:
- `test_diffusion_runner.py` — scheduler-driven ODE step parity vs reference solver.
- `test_ar_token_runner.py` — sampler determinism under fixed seed; KV cache invariants.
- `test_cache_ring.py` — ring buffer correctness (block-mask alignment, written invariants).
- `test_cache_paged.py` — paged allocator: alloc/free/snapshot/restore round-trip.
- `test_quant.py` — int8/fp8/nvfp4 wrapper round-trip (quantize → dequantize → max-abs-error bound).

### 5.2 Planner evals (run on le-wm CI; the planner consuming the engine)

Existing 129 tests in `tests/`. These guard the planner contract:
- `test_drift_detector.py` (DriftDetector signal correctness)
- `test_goal_adapter.py` (image / text / VLM ingestion)
- `test_plan_result.py` (numpy array protocol, needs_replan threshold)
- `test_s15_loop.py` (S2↔S1 control loop integration)
- `test_sim_components.py` (SimVLM, SimMotorPolicy)

Acceptance after fork: all 129 pass with the engine consumed via `world_engine_le` imports.

### 5.3 Per-spec evals (run when a new spec is added)

Each spec ships with an integration eval that proves it actually runs through the engine surface. Template:

```python
def test_spec_smoke_<name>():
    spec = build_<name>_spec()
    engine = Engine(spec=spec)
    obs = make_synthetic_obs(spec)
    goal = make_synthetic_goal(spec)
    out = engine.step(obs, conditioning={"goal": goal})
    assert out.shape_matches(spec.expected_output_shape)
```

Plus spec-specific shape and behavior tests:
- **LeWM:** existing 16 CEM tests + 16 rollout tests cover this.
- **Waypoint-1.5:** ring+tail cache invariants, scheduler step parity, KV-cache write-on-final-step (if folded), quant round-trips.
- **TD-MPC2:** continuous action vector handling, terminal value head signature.
- **Genie-class:** sampler determinism, paged cache snapshot/restore, KV growth bound.

### 5.4 Integration evals (run on-pod, gates fork merges that touch LeWM behavior)

Real LeWM checkpoints, real environments, real numbers.

| Eval | Command | Pass criterion |
|---|---|---|
| PushT success rate | `eval.py --config-name=pusht policy=pusht/lejepa eval.num_eval=50` | ≥ 94% (current baseline) |
| PushT planning latency | `scripts/final_benchmark.py --policy pusht/lejepa` | ≤ 95 ms p50 (within 5% of 89 ms baseline) |
| TwoRoom S1.5 success | `scripts/eval_s15_integration.py` | ≥ 64% (current baseline) |
| Dream Tree latency | `scripts/eval_dream_tree.py` | within 5% of pre-refactor wall-clock |

Run after F3 (le-wm consumes fork), again after F4 (Runner taxonomy), again after F10 (un-baked B==1), and on every release.

### 5.5 Performance evals (continuous, gate F7-F16)

| Eval | Pass criterion | Baseline |
|---|---|---|
| Single-stream PushT latency | ≤ 95 ms p50 | 89 ms |
| 4-stream batched throughput | ≥ 1.5× single-stream | n/a (new) |
| Distilled diffusion 1-step quality | ≤ 5% degradation vs 4-step | n/a (new) |
| KV cache memory @ 64-frame context | ≤ 50% of pre-quant | n/a (new) |
| TeaCache-style idle-frame speedup | ≥ 1.5× | n/a (new) |

---

## 6. Concrete demos for each milestone

### 6.1 Engine standalone (after F2)

```python
import torch
from world_engine_le.cem import CEMSolver
from world_engine_le.contracts import TerminalMSECost
from world_engine_le.specs import LEWM_SPEC

# Verified on cdae34a — works against a TinyJEPA today.
model = make_lewm_compatible_model()
solver = CEMSolver(model, action_dim=LEWM_SPEC.action_dim,
                   horizon=5, num_samples=128, n_steps=15, topk=25,
                   cost=TerminalMSECost())

action, terminal, cost = solver.plan(obs_emb, goal_emb,
                                     return_terminal_emb=True, return_cost=True)
```

### 6.2 Two specs side by side (after F5+F6)

```python
from world_engine_le import Engine

lewm  = Engine(spec="lewm")
waypoint = Engine(spec="waypoint-1.5", quant="nvfp4")

action = lewm.plan(obs_emb, goal_emb)            # JEPA + CEM
frame  = waypoint.gen_frame(prompt="...", ctrl=ctrl)  # DiT + ring KV
```

### 6.3 SearchRunner over any inner runner (after F4)

```python
from world_engine_le.runners import SearchRunner, DeterministicRunner

inner = DeterministicRunner(model=lewm_model, history_size=3)
planner = SearchRunner(inner, num_samples=128, n_steps=15, topk=25,
                       cost=TerminalMSECost())
```

### 6.4 Continuous batching (after F11, Thesis B)

```python
from world_engine_le.server import RolloutServer

server = RolloutServer(spec="waypoint-1.5", max_concurrent=8, batch_window_ms=10)
async for frame in server.stream(session_id, ctrl_inputs):
    yield frame
```

---

## 7. Risks and open questions

### 7.1 Risks

- **Abstraction trap.** Designing `Runner` / spec registry before the second spec exists produces something that fits LeWM and one imaginary model. Mitigation: F4 happens *after* F5b's spec is wired in; the registry is a refactor, not a design.
- **Continuous batching is its own project.** Touches every layer. Honest scope: significant rewrite of cache + scheduling. Only worth it for Thesis B.
- **Distillation is upstream.** A 1-step Waypoint model is a *training* deliverable. Either Overworld ships one, or distillation runs in-house.
- **flex_attention is torch-version-coupled and CUDA-only.** Multi-runtime support requires SDPA fallback with measurable performance loss.
- **State-dict munging across model versions** (the Waypoint-1→1.5 migration tax in world_engine.git's `load_state_dict`) is a tax we should *not* inherit; offline conversion scripts only.
- **Test coverage on cache and quantization.** The fork inherits world_engine.git's KV cache + quant ideas, which were untested in the source. The fork must add tests before relying on them.

### 7.2 Open questions

- Naming: "Imagination Engine" is descriptive but generic. Project needs a real name before public release.
- License of forked components from world_engine.git (Apache-2.0; check attribution requirements).
- Whether to start with Thesis A (LeWM-aligned) or Thesis B (server) — affects sequencing of F10–F12.
- Whether to vendor world_engine.git's `gemlite`/`flashinfer`/`taehv` deps or re-implement.
- Whether to support eager non-compiled fallback for rapid iteration, or always require warmup. (Recommendation: eager fallback as default, compiled as opt-in.)

---

## 8. Success criteria for v1

The fork is "v1.0" when all of the following hold simultaneously:

1. Engine modules import cleanly with only torch + einops as required deps.
2. LeWM and one other spec (TD-MPC2 *or* Waypoint-1.5) both run end-to-end through `Engine(spec=...)`.
3. All engine-side tests pass on CPU CI; all planner tests pass on GPU CI.
4. PushT success rate (50 ep) within 2 pp of pre-refactor 94%.
5. PushT planning latency p50 within 5% of pre-refactor 89 ms (single-stream, RTX 4090).
6. TwoRoom S1.5 integration eval within 2 pp of pre-refactor 64%.
7. Public API surface stable: `Engine`, `Runner`, `Cost`, `ModelSpec`, `RolloutRequest`/`Result` are all the entry points downstream code touches.
8. README + architecture doc + per-spec docs published.

Thesis B targets are additive on top of v1.0:
9. (B) Continuous batching: 4-stream throughput ≥ 1.5× single-stream.
10. (B) Server transport runs reliably under quota and graceful shutdown.

---

## 9. Index of related docs

- `docs/CONTRACTS.md` — typed contracts reference (legacy info_dict + new `RolloutRequest`/`Result`/`Cost`).
- `docs/PROJECT.md` — LeHarness project overview, phase history.
- `docs/STATE.md` — how the world model works end-to-end.
- `docs/S1_5.md` — S1.5 spec, on-pod validation results, N-series next steps.
- `docs/DREAM_ENGINE.md` — Dream Tree, language conditioning, integration results.
- `docs/VOLUME.md` — on-pod network volume layout.
