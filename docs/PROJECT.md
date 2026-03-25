# LeHarness: Project Overview

## What Is This

**LeHarness** is a planning harness around [LeWM](https://github.com/lucas-maes/le-wm), a 15M-parameter JEPA world model that predicts the future in latent space. LeWM is small, fast to train, and its representations encode real physical structure. But out of the box, it plans slowly (1-2 seconds per decision) and scores trajectories with raw embedding distance — a blunt instrument.

LeHarness wraps LeWM in an optimized planning infrastructure that makes it plan better and plan faster, turning it from a research artifact into something that can run onboard a robot and make decisions in real time.

The end state: a system where LeWM imagines candidate futures, a learned value function picks the best one, and a low-level policy executes it — running at 10-50 Hz on an RTX 4090, with a clear path to onboard Jetson deployment.

## Why This Matters

The robotics field is converging on two approaches to robot intelligence:

1. **VLM-backbone VLAs** (RT-2, OpenVLA, Pi0) — Large vision-language models that map observations directly to actions. They understand semantics but don't model physics. They're slow (5-10 Hz on cloud GPUs) and expensive to deploy.

2. **World-model-based planning** (Dreamer, TD-MPC, LeWM) — Small dynamics models that imagine futures and search for good actions. They understand physics but are limited by their planning loop.

This project demonstrates the latency and efficiency advantages of small world models for real-time robotic planning. A 15M-param model that plans at 10-50 Hz on a Jetson is a fundamentally different deployment profile than a 7-55B VLM requiring cloud GPUs. This is a stepping stone toward using world models as the planning backbone in a dual-system VLA architecture — the System 1 (fast reactive control) and System 2 (deliberate planning) that systems like GR00T N1 and Fast-in-Slow VLA are converging on.

**What LeHarness is:** An efficient, real-time, image-goal-conditioned planner with a learned value function and dual-mode execution.

**What LeHarness is not (yet):** A full VLA backbone. That would additionally require language conditioning, multi-task generalization, and cross-embodiment support — all valid future directions that build on the infrastructure created here.

## What We're Building

Four layers, each building on the last:

### Layer 1 — Faster Planning
Take the existing CEM planner from 300 samples x 30 iterations (45,000 predictor forward passes, ~1-2s) down to 64 samples x 5 iterations (~1,600 forward passes, ~50-100ms) without losing task success. This is done through:
- iCEM solver (colored noise, elite retention from prior timestep)
- Adaptive early stopping (exit when cost converges)
- `torch.compile(backend="tensorrt")` for the encoder and predictor
- CUDA Graphs on the single-step predictor call (not the outer CEM loop)
- Pre-allocated rollout buffers (eliminate per-step `torch.cat` allocations)

### Layer 2 — Smarter Trajectory Scoring
Replace the current cost function (last-step MSE between predicted embedding and goal embedding) with a learned per-step value function V(z_t, z_goal). A ~200-500K parameter MLP that takes a single 192-dim state embedding and the goal embedding, and outputs a progress score. Trajectory cost = sum of V(z_t, z_goal) over rollout steps. This approach:
- Is what TD-MPC2, Dreamer, and Value Summation (Shao 2022) use — proven to outperform trajectory-level scoring
- Captures intermediate progress, not just terminal distance
- Extends effective planning horizon beyond the rollout (terminal value)
- Avoids causal confusion that trajectory-level models suffer from
- Requires training on both expert data AND synthetic failures (not expert-only)

### Layer 3 — Dual-Mode Execution
A small amortized policy (~2-5M params) trained via DAgger (not pure behavioral cloning — BC fails on multimodal planner outputs). Handles routine states at 50+ Hz. The full world model + value function planning loop activates for novel or high-stakes states at 10-20 Hz. Mode switching uses state-novelty detection + value function ensemble disagreement (not policy ensemble disagreement, which conflates multimodality with uncertainty). Switching hysteresis prevents oscillation. Safety constraints (action bounds, fallback stop) on all outputs.

### Layer 4 — Offline Improvement Pipeline
A server-side pipeline that continuously improves all three components:
- Runs large-budget CEM to generate expert (state, action) pairs for DAgger-style policy training
- Collects value function training data: expert trajectories + synthetic failures + predictor-generated rollouts
- Optionally uses a VLM (Qwen-VL-4B) to label subtle quality dimensions not captured by environment state
- Fine-tunes the world model predictor on task-specific data if needed

## How We Evaluate Success

### Primary Metrics

**Task success rate** — Does the robot complete the task? Measured on PushT (2D manipulation), TwoRoom (navigation), Cube (3D manipulation), and Reacher (continuous control). The baseline is LeWM's current eval: 50 episodes per task with CEM planning.

**Planning latency (ms/decision)** — How long does a single planning step take? Current baseline: ~1000-2500ms on desktop GPU. Target: <100ms.

**Control frequency (Hz)** — How many decisions per second? Current: <1 Hz. Target: 10-20 Hz in deliberate mode, 50+ Hz in fast mode.

### Secondary Metrics

**Planning efficiency curve** — Success rate as a function of CEM sample budget. Shows whether the value function lets us achieve the same success with fewer samples.

**Anytime performance** — Success rate if we stop planning after k CEM iterations. Validates adaptive early stopping.

**Value-task correlation** — Does the learned value function's score actually predict task success? Measured as Pearson r between trajectory value and binary outcome.

**Representation utility** — Linear probing accuracy for physical quantities (position, angle, velocity) from LeWM's latent space. Inherited from the original paper; we track it to ensure harness modifications don't degrade the representation.

**Prediction fidelity** — MSE between predicted and actual latent embeddings at rollout depths 1, 3, 5, 10. Monitors whether the world model's predictions are reliable enough for the planning budget we're using.

**Amortization gap** — Success rate of the distilled fast policy vs. the full planner. Measures how much we lose by skipping planning.

### What Success Looks Like

| Milestone | Success Criterion |
|-----------|-------------------|
| M0: Validated model ✅ | 98% success, depth-5 ratio 0.139 (3.6x below threshold) |
| M1: Faster planning | Same success rate at 5x fewer forward passes (iCEM at reduced budget) |
| M2: Value function | Higher success rate than MSE cost at equal budget, OR equal success at half budget |
| M3: Real-time | <100ms per planning step on desktop GPU |
| M4: Fast policy | >80% of planner success rate at 20+ Hz effective frequency |
| M5: Integrated system | Dual-mode harness on RTX 4090 at 20+ Hz with full pipeline end-to-end |

### What Failure Looks Like

- **Prediction error explodes at horizon 3-5** — the world model can't support multi-step planning. Fix the model (fine-tune predictor, increase embed_dim) before optimizing the planner.
- **Reducing CEM budget tanks success rate even with iCEM** — predictions are too noisy for efficient planning. Same fix: improve the model.
- **Value function doesn't outperform MSE cost** — the latent space is already well-structured for distance-based planning. Skip Layer 2 and proceed to speed optimization; the MSE cost is sufficient.
- **Value function is exploited by the planner (reward hacking)** — the ensemble disagrees wildly on CEM's top-ranked trajectories. Increase ensemble size, add training data from the exploit region, or add a regularization term penalizing trajectories far from the training distribution.
- **Amortized policy can't capture the planner's behavior** — the action distribution is multimodal. Switch from MSE loss to a mixture density network or diffusion policy head. If still failing, use DAgger more aggressively.
- **TensorRT/torch.compile breaks the autoregressive rollout** — fall back to `torch.jit.script` or keep PyTorch eager mode and rely on INT8 quantization for speed.

## Phases

Each phase has a hard gate. Do not proceed to the next phase until the gate is passed. If a gate fails, the failure mode tells you what to fix before retrying.

---

### Phase 0: Baseline ✅ COMPLETE

Establish ground truth numbers. Everything that follows is measured against these.

**Results (RTX 4090, PushT):**

| Metric | Value |
|--------|-------|
| LeWM success rate | **98.0%** (49/50 episodes) |
| Random success rate | **8.0%** (4/50 episodes) |
| Planning latency | **1,310 ms/step** (0.76 Hz) |
| Single encode | **4.5 ms** |
| Forward passes/step | 45,000 (300 x 30 x 5) |

**Gate: PASS.** 98% vs 8% — LeWM massively outperforms random.

**Artifacts:** `/workspace/data/results/phase0_*.log`, 50 rollout videos on volume.

---

### Phase 1: Model Fidelity Audit ✅ COMPLETE

Verify that the world model's predictions are reliable enough to support reduced planning budgets.

**Results:**

| Depth | Prediction MSE | Ratio (vs inter-state dist) | Status |
|-------|---------------|----------------------------|--------|
| 1 | 0.010 | 0.006 | PASS |
| 2 | 0.028 | 0.017 | PASS |
| 3 | 0.071 | 0.043 | PASS |
| 5 | 0.229 | **0.139** | PASS |

Linear probing: positions R²=0.93-0.98, block angle R²=0.79, velocities ~0.

**Gate: PASS.** Ratio 0.139 is 3.6x below the 0.50 threshold. The model's predictions are highly reliable at the planning horizon. This means aggressive budget reduction is viable — the predictor has substantial margin.

**Key implication for Phase 2:** The comfortable 0.139 ratio suggests we can test budgets as low as 16 samples x 3 iterations without the model being the bottleneck.

**Artifacts:** `/workspace/data/results/phase1_fidelity.json`, `/workspace/data/results/phase1_fidelity.log`.

---

### Phase 2: Planning Budget Reduction

Determine how much planning compute can be cut without losing performance. This is pure configuration sweeps — no new code beyond a sweep script and logging.

**Steps:**
1. Sweep sample count: run eval at `solver.num_samples=` {300, 128, 64, 32, 16} with 30 iterations each. Record success rate and planning time for each.
2. Sweep iteration count: run eval at `solver.n_steps=` {30, 15, 10, 5, 3} with 300 samples each. Record success rate and planning time for each.
3. Sweep both jointly at the promising intersection points from steps 1-2.
4. Swap CEM for iCEM: run eval with `ICEMSolver` at the same sample/iteration grid. Compare against CEM at equal budgets.
5. Log CEM cost at each iteration to identify convergence behavior — this informs whether adaptive early stopping is viable.

**Artifacts:**
- Planning efficiency table: success rate vs. (samples, iterations, solver)
- Cost convergence curves per task
- Identified minimum viable planning budget (samples x iterations) for iCEM

**Gate:** There exists a configuration where success rate is within 5% of baseline AND total predictor forward passes are reduced by at least 5x. The baseline is 45,000 forward passes (300 samples x 30 iterations x 5 predictor steps per rollout). The target is ~9,000 or fewer (e.g., 64 samples x 5 iterations x 5 steps = 1,600). If no such configuration exists, the world model's predictions are too noisy for efficient planning — return to Phase 1 findings and consider fine-tuning the predictor before proceeding.

---

### Phase 3: Adaptive Early Stopping

Build the first piece of harness code. The planner should stop spending compute when it has already converged.

**Steps:**
1. Using the cost convergence data from Phase 2, define a stopping criterion: exit CEM when the relative cost improvement between iterations drops below a threshold (e.g., `|cost[i] - cost[i-1]| / cost[i-1] < epsilon`)
2. Implement adaptive stopping as a wrapper around the solver (does not modify stable-worldmodel internals)
3. Run full eval with adaptive stopping enabled at the best fixed budget from Phase 2
4. Measure: actual iterations used per step (mean, p50, p95), success rate, planning time

**Artifacts:**
- `harness/adaptive_solver.py` — Solver wrapper with early stopping
- Distribution of actual iterations used (most steps should exit early)
- Updated planning latency numbers

**Gate:** Adaptive stopping reduces mean iterations by at least 30% compared to fixed budget while maintaining success rate within 2% of Phase 2's best fixed configuration. If most steps use the full budget (the cost never converges early), the CEM landscape is too flat — this means the cost function lacks gradient signal, which motivates Phase 4.

---

### Phase 4: Learned Value Function

Replace MSE embedding distance with a learned per-step value function that captures task-relevant structure.

**Steps:**
1. **Data collection — expert trajectories:** Run LeWM's encoder over the expert dataset to extract latent trajectories. For each trajectory, compute ground-truth progress from environment state (e.g., PushT's `eval_state` distance-to-goal normalized to [0, 1]). Store as (z_t, z_goal, progress) triples.
2. **Data collection — synthetic failures:** Run CEM at low budgets (16 samples, 3 iterations) and with random actions to generate suboptimal trajectories. Encode them and label with ground-truth progress. Also collect predictor-generated rollouts (from Phase 1 data) labeled against actual outcomes — this prevents distribution shift between training data and planning-time inputs.
3. **Architecture:** Build a per-step value function V(z_t, z_goal): 2-layer MLP, input dim 384 (concatenation of z_t and z_goal), hidden dim 256, LayerNorm + Mish activations, scalar output. ~200-500K params. Train an ensemble of 3-5 copies with different seeds (~1-2M total).
4. **Training:** MSE loss between predicted progress and ground-truth progress. Split 80/20 train/val. Validate per-task, not averaged — the value function may help more on hard tasks than easy ones.
5. **Integration:** Create a new cost model class that wraps the value function ensemble and conforms to the `get_cost()` interface. Cost = negative mean ensemble prediction, summed over rollout steps with 2x weight on the terminal step.
6. **Eval:** Run the full planning eval with the value function as cost, using the best planning budget from Phase 2-3. Compare success rate against MSE baseline at equal sample budgets. Also compare at reduced budgets — does the value function let you plan effectively with even fewer samples?

**Artifacts:**
- `harness/value_function.py` — Value function architecture, ensemble wrapper, and training script
- `harness/value_cost.py` — Cost model wrapper for solver integration
- Trained value function ensemble checkpoint
- Comparison table: success rate at {32, 64, 128} samples with MSE cost vs. learned value function

**Gate:** Learned value function achieves higher success rate than MSE cost at the same planning budget on at least one task, OR achieves equal success rate at half the planning budget. Evaluate per-task — passing on PushT and Cube but tying on Reacher still counts. If neither holds on any task, the latent space is already well-structured for MSE-based planning — skip this layer and proceed directly to speed optimization.

---

### Phase 5: Compilation and Speed Optimization

Make the existing pipeline fast enough for real-time by compiling the neural network components.

**Steps:**
1. Apply `torch.compile(backend="tensorrt")` to the ViT-tiny encoder with fixed input shape (1, 3, 224, 224). Verify output matches eager PyTorch within tolerance. If torch.compile fails, fall back to explicit ONNX → TensorRT FP16 export.
2. Apply `torch.compile` to the single-step predictor forward pass with fixed batch size (64) and sequence length (3). The autoregressive rollout loop stays in Python. If torch.compile fails for the AdaLN-conditioned blocks, fall back to `torch.jit.script` or ONNX export.
3. Compile the value function ensemble (if Phase 4 passed).
4. Replace `torch.cat` in the rollout loop (`jepa.py:94,97`) with writes into pre-allocated buffers.
5. Collect ~500 representative (emb, act_emb) pairs from actual planning runs. Run INT8 post-training quantization calibration. Validate that INT8 cost divergence from FP16 is <1%.
6. Benchmark on target hardware:
   - Desktop GPU: measure ms/decision breakdown by component (use `torch.cuda.Event` timing)
   - Profile with `nsys` (Nsight Systems) to identify kernel launch gaps, memory transfer stalls, Python overhead
   - If kernel launch overhead dominates the predictor calls, capture a CUDA Graph for the single-step predictor forward pass
   - Jetson AGX Orin (if available): build engines on-device (they are hardware-specific), benchmark in MAXN power mode (60W)

**Artifacts:**
- Compiled encoder, predictor, and value function modules
- INT8 calibration dataset and quantized engines (if applicable)
- `harness/compiled_inference.py` — Compiled inference wrapper
- Benchmark table: ms/decision breakdown by component on each hardware target
- `nsys` timeline profiles identifying remaining bottlenecks

**Gate:** Full deliberate planning step (encode + rollout + score + select) completes in <100ms on desktop GPU. On AGX Orin (if tested), <200ms at FP16 or <150ms at INT8. The gate is about latency, not the specific compilation method — torch.compile, TensorRT, torch.jit.script, or even eager PyTorch with INT8 are all acceptable if they hit the target. If no method hits the target, reduce the planning budget further (Phase 2 data shows the tradeoff) or accept a lower control frequency for deliberate mode.

---

### Phase 6: Policy Distillation and Dual-Mode Execution

Train a fast reactive policy and build the mode-switching harness.

**Steps:**
1. **Data generation:** Run the optimized planner (iCEM + value function + compiled) on a large and diverse set of (start_state, goal_state) pairs. Sample from the dataset AND from synthetic perturbations (shifted positions, rotated goals). For each, record (encoded_observation, encoded_goal, planned_action, planner_success). Generate at least 50K pairs. Include a data scaling experiment (10K, 25K, 50K, 100K) to find the actual data efficiency knee.
2. **Policy architecture:** A small network (~2-5M params) that maps (z_obs, z_goal) → action. Input is LeWM's 192-dim embeddings (encoder runs once, shared with planner). Consider a Gaussian mixture output (2-3 components) instead of single Gaussian if CEM's elite distributions show multimodality.
3. **Training via DAgger:** Initial round: behavioral cloning on planner outputs. Then: deploy the policy, collect states where it fails or where the value function predicts low progress, re-plan from those states with the full planner, add corrected (state, action) pairs to the training set. Repeat for 3-5 rounds. Filter out planner trajectories that actually failed (the planner is not an oracle).
4. **Mode switching:** Implement state-novelty detection (Mahalanobis distance on the 192-dim embedding relative to the training distribution) and value function ensemble disagreement. Switch to deliberate planning when either signal exceeds a calibrated threshold. Add minimum dwell time (e.g., 5 steps) in each mode to prevent oscillation. Warm-start the planner with the fast policy's recent actions when switching to deliberate mode.
5. **Safety constraints:** Enforce action bounds on all outputs. Add a fallback stop action when both policy novelty and value uncertainty are high.
6. **Eval:** Measure success rate and effective Hz for: (a) planner only, (b) fast policy only, (c) dual-mode harness. Report per-task.

**Artifacts:**
- `harness/fast_policy.py` — Policy network with mixture output and DAgger training loop
- `harness/dual_mode.py` — Mode-switching controller with novelty detection, hysteresis, and safety
- Trained fast policy checkpoint (after DAgger rounds)
- Comparison table: success rate and effective Hz for planner-only vs. policy-only vs. dual-mode, per task

**Gate:** The dual-mode harness achieves >80% of planner-only success rate while operating at an effective frequency of 20+ Hz (meaning most steps use the fast policy). If the fast policy's success rate is too low (<60% of planner) after DAgger, the action distribution is too complex — try a diffusion policy head, increase model capacity, or increase DAgger rounds. If dual-mode switching causes oscillation or worse-than-either-mode performance, tune hysteresis and thresholds before concluding failure.

---

### Phase 7: End-to-End Integration on RTX 4090

Bring everything together into a single pipeline on the RunPod 4090 and produce the final numbers that demonstrate the system works.

**Steps:**
1. Assemble the full pipeline: observation input → image preprocessing (GPU) → LeWM encoder (compiled) → dual-mode harness (fast policy or planner with value function) → action output. All in a single `harness/pipeline.py` with clean API.
2. End-to-end latency profiling with `nsys`: measure total time from observation to action. Break down by component (encoder, rollout, value scoring, policy forward pass, mode switching overhead).
3. Run the standard eval suite (PushT, and optionally TwoRoom/Cube/Reacher) through the integrated pipeline. Compare success rates to Phase 0 baseline.
4. Stress tests: introduce perturbations not seen during training — shifted start positions, rotated goals, novel obstacle configurations. Measure degradation.
5. Produce the final efficiency comparison: params, FLOPS/decision, memory, latency, and success rate. This positions the system relative to larger approaches (VLM-backbone VLAs) on the metrics where small world models have a structural advantage.
6. Package results: write up findings, publish comparison tables, record demo videos of the planner in action on PushT.

**Artifacts:**
- `harness/pipeline.py` — Full end-to-end inference pipeline
- Final performance table: success rate, Hz, ms/decision, GPU memory usage
- Efficiency comparison against published VLA numbers (params, FLOPS, latency)
- Demo videos showing the world model planning and executing trajectories
- `nsys` profiles and benchmark scripts for reproducibility

**Gate:** The integrated system completes PushT episodes end-to-end on RTX 4090 at 20+ Hz effective frequency (dual-mode) with success rate within 15% of Phase 0 baseline. The full results package (numbers + demos + comparison) is ready to share publicly.

## Definition of Done: What Success Looks Like After All Phases Pass

When every gate from Phase 0 through Phase 7 is green, you have a system that:

### The Numbers (RTX 4090)

| Metric | Baseline (Phase 0) | Target (Phase 7) |
|--------|--------------------|--------------------|
| Planning latency | **1,310 ms/decision** | <50ms deliberate, <5ms fast |
| Control frequency | **0.76 Hz** | 20-50 Hz deliberate, 100+ Hz fast |
| Success rate | **98.0%** (PushT) | Within 15% (>83%) |
| Total model size | 15M (world model only) | ~20-25M (WM + value fn + policy) |
| GPU memory | Unoptimized | <400MB FP16 |
| Forward passes/decision | **45,000** (300 x 30 x 5) | ~1,600 (64 x 5 x 5) — **28x reduction** |

### What You Can Show The World

1. **A 15M-param world model planning at 20-50 Hz** — demonstrate that a tiny model with smart infrastructure competes on speed with systems 100-1000x larger. The demo videos show the model imagining futures in latent space, the value function scoring trajectories, and the dual-mode system switching between fast reactive control and deliberate planning.

2. **Concrete efficiency comparison** — a table showing LeWM harness vs. published VLA numbers (OpenVLA, Pi0, GR-2) on params, FLOPS/decision, latency, and memory. Not a task-success comparison (unfair in both directions), but an efficiency profile that makes the case for small world models.

3. **Reproducible benchmark package** — scripts, configs, checkpoints, and `nsys` profiles that anyone can run on a RunPod 4090 to verify the numbers. Open source the harness code.

4. **Give it a goal image** — "make the scene look like this" — and it plans a sequence of actions to get there. The value function tells it whether it's making progress. The dual-mode system handles routine states reactively and thinks harder on novel situations.

5. **Swap environments** — the harness code (`adaptive_solver`, `value_cost`, `dual_mode`, `pipeline`) is environment-agnostic. Retrain the value function and fast policy on a new task's data; the infrastructure stays the same.

### What You Cannot Do With It (Yet)

- Deploy on a physical robot (needs Jetson integration — see Future Directions)
- Accept language instructions (no text encoder)
- Generalize across tasks without retraining the value function and policy
- Transfer to a real robot without sim-to-real work
- Handle open-vocabulary goals
- Work across different robot morphologies

These are the Future Directions below — each one builds on the harness infrastructure created here.

### Deliverables

```
harness/
  adaptive_solver.py     # Phase 3: Solver wrapper with early stopping
  value_function.py      # Phase 4: Per-step V(z_t, z_goal) ensemble
  value_cost.py          # Phase 4: Cost model wrapper for solver integration
  compiled_inference.py  # Phase 5: torch.compile / TRT inference wrapper
  fast_policy.py         # Phase 6: DAgger-trained policy with mixture output
  dual_mode.py           # Phase 6: Mode switching with novelty, hysteresis, safety
  pipeline.py            # Phase 7: Full end-to-end inference pipeline
  benchmark.py           # Phase 7: Reproducible benchmark script

checkpoints/
  value_ensemble.pt      # Trained value function ensemble
  fast_policy.pt         # DAgger-trained fast policy

results/
  phase0_baseline.txt    # Baseline numbers
  phase1_fidelity.txt    # Prediction error vs depth
  phase2_sweeps.txt      # Planning efficiency table
  phase5_benchmarks.txt  # Latency breakdown by component
  phase7_final.txt       # Final integrated performance + efficiency comparison
  demo_videos/           # PushT planning demos
```

## Infrastructure Guide

### Cost Optimization Strategy

The #1 rule: **never leave a GPU pod idle.** Write code locally, batch GPU work into sessions, script everything to run unattended, and stop the pod the moment a session's work is done. The network volume persists data between sessions so you don't re-download or re-setup.

### Pod Sessions (Batched for Minimal GPU Waste)

Phases are grouped into pod sessions. Each session is one pod spin-up → do all the work → stop pod. This avoids paying for idle time between phases.

| Session | Phases | What Happens | Pod Hours | Cost @ $0.40/hr |
|---------|--------|-------------|-----------|-----------------|
| **Session 1** | 0 + 1 | Download data (once, saved to volume). Run baseline eval + random baseline. Run fidelity audit (encoder + predictor over dataset). All scripted, runs back-to-back. | 2-3 hrs | ~$1 |
| **Session 2** | 2 | Budget sweeps. Write a sweep script locally beforehand that loops over all (samples, iterations, solver) configs and logs results to a file. Launch it and let it run. This is the longest session because of the combinatorial grid (~25 configs x 50 episodes each). | 8-15 hrs | ~$3-6 |
| **Session 3** | 3 + 4 | Adaptive stopping (code written locally, just eval on pod). Value function data collection + training + eval. These are both light — batch into one session. | 3-5 hrs | ~$1-2 |
| **Session 4** | 5 | torch.compile + benchmarking. Compilation can be finicky — budget a session for iteration. INT8 calibration. nsys profiling. | 3-6 hrs | ~$1-2 |
| **Session 5** | 6 | DAgger loop: generate 50K+ planner rollouts, train policy, deploy, collect failures, re-plan, retrain (3-5 rounds). This is compute-heavy but can be scripted to run unattended. | 6-12 hrs | ~$2-5 |
| **Session 6** | 7 | Final integration, full eval, profiling, demo video recording. | 2-4 hrs | ~$1-2 |
| **Total** | | | **~24-45 hrs** | **~$10-18** |

### What To Do Locally (Free) Between Sessions

| Between Sessions | Local Work |
|-----------------|------------|
| Before Session 1 | Nothing — just spin up and go |
| Session 1 → 2 | Review fidelity results. Write the sweep script (`scripts/sweep_budget.py`) that automates all Phase 2 configs. Decide sweep grid based on Phase 1 findings. |
| Session 2 → 3 | Analyze sweep results. Write `harness/adaptive_solver.py`. Write value function architecture (`harness/value_function.py`, `harness/value_cost.py`). All pure Python, no GPU needed. |
| Session 3 → 4 | Review value function results. Write `harness/compiled_inference.py` scaffold. Prepare INT8 calibration script. |
| Session 4 → 5 | Review benchmarks. Write `harness/fast_policy.py` and `harness/dual_mode.py`. Write DAgger loop script. |
| Session 5 → 6 | Review DAgger results. Write `harness/pipeline.py` and `harness/benchmark.py`. Prepare demo recording script. |

### RunPod Setup: Step-by-Step

#### Step 1: Create a Network Volume

This is done ONCE and persists across all pod sessions. It holds the 43GB dataset so you never re-download it.

1. Go to [runpod.io](https://runpod.io) → **Storage** → **Network Volumes** → **+ New Volume**
2. Settings:
   - **Name:** `lewm-data`
   - **Region:** Pick the region with cheapest 4090 availability (check GPU Cloud page). **The volume and pod must be in the same region.**
   - **Size:** `60 GB`
   - **Volume Type:** Default (NVMe)
3. Click **Create**
4. Note the volume ID — you'll attach it when creating pods

**Cost:** ~$4.20/month. Delete the volume when the project is done.

#### Step 2: Create the GPU Pod

Do this at the start of each session. Stop (not delete) the pod when done — stopping is free, the volume keeps your data.

1. Go to **GPU Cloud** → **Deploy**
2. Settings:

| Setting | Value | Why |
|---------|-------|-----|
| **GPU Type** | RTX 4090 (24 GB) | Best $/FLOP for 15M param model. ~$0.40/hr community. |
| **GPU Count** | 1 | Single GPU is sufficient |
| **Template** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Needs `devel` for torch.compile + TensorRT. CUDA 12.4 matches latest PyTorch. |
| **Container Disk** | `20 GB` | OS (~8GB) + Python packages (~5GB) + compiled engines (~2GB) + headroom. Bulk data on volume. |
| **Volume Disk** | `0 GB` (use network volume instead) | Pod volume is ephemeral — don't use it. |
| **Network Volume** | Attach `lewm-data` (60GB, created in Step 1) | Mounts at `/workspace/data`. Persists across stop/start. |
| **Expose HTTP Ports** | Leave default (8888 for Jupyter) | Optional — SSH is the primary access method |
| **Cloud Type** | Community Cloud | Cheapest. Use Secure Cloud only if you need guaranteed uptime. |

3. Click **Deploy On-Demand** (or **Spot** for ~30% cheaper if you can tolerate interruptions — fine for Phases 0-2 which are stateless sweeps)

#### Step 3: First-Time Setup (Session 1 Only)

SSH into the pod and run:

```bash
# Save setup script to the network volume (runs once, reused every session)
cat > /workspace/data/setup.sh << 'SETUP'
#!/bin/bash
set -e

# System deps
apt-get update && apt-get install -y -qq libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev ffmpeg zstd > /dev/null 2>&1

# Clone/update repo
cd /workspace
if [ -d "le-wm" ]; then
  cd le-wm && git pull
else
  git clone https://github.com/kingjulio8238/le-harness.git && cd le-wm
fi

# Python deps
pip install -q "stable-worldmodel[train,env]" gdown

# Set data path
export STABLEWM_HOME=/workspace/data
echo 'export STABLEWM_HOME=/workspace/data' >> ~/.bashrc
mkdir -p $STABLEWM_HOME/pusht

# Download dataset (43GB — skip if already on volume)
if [ ! -f "$STABLEWM_HOME/pusht_expert_train.h5" ]; then
  echo "Downloading PushT dataset (~13GB compressed → ~43GB)..."
  gdown 1WrtW2jWfZ8W5CAIfTXIw-bbvu9NMK964 -O /tmp/pusht_expert_train.h5.zst
  zstd -d /tmp/pusht_expert_train.h5.zst -o $STABLEWM_HOME/pusht_expert_train.h5
  rm /tmp/pusht_expert_train.h5.zst
else
  echo "Dataset already on volume."
fi

# Download checkpoint (134MB — skip if already on volume)
if [ ! -f "$STABLEWM_HOME/pusht/lejepa_object.ckpt" ]; then
  echo "Downloading checkpoint..."
  gdown 1CagjbwPOovHlmcvot07eWvq7fGswdYtI -O /tmp/lejepa.tar.zst
  tar --zstd -xvf /tmp/lejepa.tar.zst -C $STABLEWM_HOME/pusht/
  mv $STABLEWM_HOME/pusht/pusht/* $STABLEWM_HOME/pusht/ 2>/dev/null
  rmdir $STABLEWM_HOME/pusht/pusht 2>/dev/null
  rm /tmp/lejepa.tar.zst
else
  echo "Checkpoint already on volume."
fi

echo ""
echo "=== Ready ==="
echo "cd /workspace/le-wm"
echo "python eval.py --config-name=pusht.yaml policy=pusht/lejepa"
SETUP

chmod +x /workspace/data/setup.sh

# Run it
bash /workspace/data/setup.sh
```

First run takes ~20 min (mostly downloading 43GB). Subsequent sessions: `bash /workspace/data/setup.sh` takes ~60 seconds.

#### Step 4: Every Subsequent Session

SSH into the pod and run:

```bash
bash /workspace/data/setup.sh
cd /workspace/le-wm
```

That's it. Volume has your data, setup script installs deps and pulls latest code.

#### Step 5: After Each Session

1. **Save any results to the volume:** `cp -r results/ /workspace/data/results/`
2. **Push code changes to git:** `git add -A && git commit -m "..." && git push`
3. **Stop the pod** (RunPod dashboard → pod → Stop). This is free. The volume persists.
4. Do NOT delete the pod unless you want to change GPU type or region.

#### Volume Layout After Setup

```
/workspace/data/                          # Network volume (60GB, persistent)
├── setup.sh                              # Reusable setup script
├── pusht_expert_train.h5                 # PushT dataset (~43GB)
├── pusht/
│   ├── lejepa_object.ckpt               # Pretrained checkpoint
│   └── lejepa_weights.ckpt
├── results/                              # Saved across sessions
│   ├── phase0_baseline.txt
│   ├── phase1_fidelity.txt
│   ├── phase2_sweeps.txt
│   └── ...
└── checkpoints/                          # Trained models (value fn, policy)
    ├── value_ensemble.pt
    └── fast_policy.pt

/workspace/le-wm/                         # Git repo (on container disk, re-cloned each session)
├── jepa.py
├── eval.py
├── harness/                              # Your harness code
└── ...
```

#### Cost Summary

| Item | Cost |
|------|------|
| Network volume (60GB, ~1-2 months) | ~$4-8 |
| Session 1: Setup + Phase 0 + 1 (2-3 hrs) | ~$1 |
| Session 2: Phase 2 sweeps (8-15 hrs) | ~$3-6 |
| Session 3: Phase 3 + 4 (3-5 hrs) | ~$1-2 |
| Session 4: Phase 5 compilation (3-6 hrs) | ~$1-2 |
| Session 5: Phase 6 DAgger (6-12 hrs) | ~$2-5 |
| Session 6: Phase 7 integration (2-4 hrs) | ~$1-2 |
| **Total** | **~$13-25** |

#### Tips

- **Use `tmux` or `nohup`** for long-running sweeps so you can disconnect from SSH without killing the job. Example: `tmux new -s sweep` → run script → `Ctrl+B, D` to detach → reconnect later with `tmux attach -t sweep`.
- **Use Spot instances** for Phases 0-2 (stateless sweeps). If interrupted, just re-run — no state is lost. Save ~30% on GPU cost.
- **Stop, don't delete** the pod between sessions within the same day. Stopping is instant and free. Starting is faster than creating a new pod.
- **Delete the volume** when the project is done. It costs $4.20/month even when no pod is running.
| **Total** | **~$14-26** |

### RunPod Pod Startup Script

Save this to your volume at `/workspace/data/setup.sh` on first run. On subsequent sessions, just run `bash /workspace/data/setup.sh` — takes ~60 seconds instead of re-downloading everything.

```bash
#!/bin/bash
set -e

# 1. System deps (cached in container disk, re-run if container was rebuilt)
apt-get update && apt-get install -y libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev ffmpeg zstd

# 2. Clone/update repo
cd /workspace
if [ -d "le-wm" ]; then
  cd le-wm && git pull
else
  git clone https://github.com/kingjulio8238/le-harness.git && cd le-wm
fi

# 3. Install Python deps (pip caches, fast on repeat runs)
pip install -q "stable-worldmodel[train,env]" gdown

# 4. Link volume for data persistence
export STABLEWM_HOME=/workspace/data
echo 'export STABLEWM_HOME=/workspace/data' >> ~/.bashrc
mkdir -p $STABLEWM_HOME/pusht

# 5. Download dataset + checkpoint (skip if already on volume — this is the key cost saver)
if [ ! -f "$STABLEWM_HOME/pusht_expert_train.h5" ]; then
  echo "Downloading PushT dataset (~13GB compressed → ~43GB)..."
  gdown 1WrtW2jWfZ8W5CAIfTXIw-bbvu9NMK964 -O /tmp/pusht_expert_train.h5.zst
  zstd -d /tmp/pusht_expert_train.h5.zst -o $STABLEWM_HOME/pusht_expert_train.h5
  rm /tmp/pusht_expert_train.h5.zst
else
  echo "Dataset already on volume, skipping download."
fi

if [ ! -f "$STABLEWM_HOME/pusht/lejepa_object.ckpt" ]; then
  echo "Downloading checkpoint (~134MB)..."
  gdown 1CagjbwPOovHlmcvot07eWvq7fGswdYtI -O /tmp/lejepa.tar.zst
  tar --zstd -xvf /tmp/lejepa.tar.zst -C $STABLEWM_HOME/pusht/
  mv $STABLEWM_HOME/pusht/pusht/* $STABLEWM_HOME/pusht/ 2>/dev/null
  rmdir $STABLEWM_HOME/pusht/pusht 2>/dev/null
  rm /tmp/lejepa.tar.zst
else
  echo "Checkpoint already on volume, skipping download."
fi

echo ""
echo "=== Ready ==="
echo "cd /workspace/le-wm"
echo "python eval.py --config-name=pusht.yaml policy=pusht/lejepa"
```

### Anti-Patterns (Things That Waste GPU Money)

- **Writing code on the pod.** Write locally, push to git, pull on pod. The pod is for running, not editing.
- **Running one eval at a time interactively.** Script the full sweep/session and let it run. Use `nohup` or `tmux` so you can disconnect.
- **Leaving the pod running overnight.** Stop the pod when the session's work is done. The volume keeps your data.
- **Re-downloading the 43GB dataset.** That's why the network volume exists. First session downloads; every subsequent session skips.
- **Using A100/H100.** This is a 15M param model. A 4090 is already overkill on VRAM. You're paying for FLOPS throughput, and the 4090 has plenty.

## Future Directions (Beyond This Project)

The following extend the system beyond the RTX 4090 demonstration and are out of scope for the current plan:

### Near-term: Jetson Onboard Deployment
- **Hardware:** Jetson AGX Orin Developer Kit 64GB (~$2000) — minimum for 10 Hz deliberate planning
- **Work:** Build TRT engines on-device (not portable from x86), thermal testing under sustained load, camera pipeline (V4L2 capture, GPU preprocessing), power mode optimization (MAXN 60W vs 30W default)
- **Expected performance:** 10-20 Hz deliberate, 50+ Hz fast (based on known 4090-to-AGX-Orin ~3-5x scaling factor)
- **Gate:** PushT end-to-end on Jetson at 10+ Hz within 15% of RTX 4090 success rate

### Medium-term: Co-Trained Action Head (Fast-WAM Insight)
Fast-WAM (Yuan et al., 2026) showed that in large (6B param) world action models, the value of video/world prediction comes from **training-time dense supervision**, not test-time imagination — removing future generation at inference costs <1.5% success while giving 4x speedup. At our 15M-param scale, we still need test-time search (the model is too small to amortize dynamics into weights), but the insight suggests an experiment: co-train a small action prediction head alongside LeWM's JEPA loss from the start, rather than post-hoc DAgger distillation. If the co-trained head is strong enough, the planner becomes a fallback rather than the primary action source. This would require modifying LeWM's training pipeline.

### Medium-term: Toward a VLA Backbone
- **Language conditioning:** Add a text encoder (e.g., CLIP text encoder) that maps language goals to the same embedding space as image goals, enabling language-specified tasks.
- **Multi-task generalization:** Train a single goal-conditioned value function across tasks rather than per-task, using diverse robot manipulation datasets (Open X-Embodiment, DROID).
- **Sim-to-real transfer:** Domain randomization during world model training, real-world fine-tuning of the predictor, uncertainty-aware planning that is conservative in unfamiliar states.
- **Cross-embodiment support:** Action space abstraction via tokenization or adapter layers for different robot morphologies.

## Related Work

This project builds directly on:
- **LeWM** (Maes et al., 2026) — The world model at the core. 15M params, JEPA architecture, stable end-to-end training.
- **iCEM** (Pinneri et al., CoRL 2020) — Sample-efficient planning with colored noise and elite retention. 2.7-22x fewer samples than vanilla CEM.
- **TD-MPC2** (Hansen et al., ICLR 2024) — Amortized policy + online planning hybrid. Policy prior warm-starts the planner. Scales from 1M to 317M params.
- **SimDist** (2025) — Simulation distillation: train reward/value models in sim, freeze for deployment. Small 1-layer Transformer reward models running at 50 Hz.
- **Value Summation** (Shao et al., 2022) — Per-step value estimates summed over trajectories outperform trajectory-level and terminal-only scoring for MPC.
- **MBPO** (Janner et al., NeurIPS 2019) — Model error bounds scale quadratically with horizon. Short branched rollouts from real data outperform long rollouts.
- **DAgger** (Ross et al., 2011) — Iterative imitation learning that addresses covariate shift. BC error compounds quadratically; DAgger reduces to linear.
- **DAMPC** (Marquez Julbe et al., IROS 2025) — BC fails on multimodal MPC outputs. Diffusion policy heads capture the full solution distribution.
- **Diff-DAgger** (2024) — Ensemble disagreement conflates multimodality with uncertainty. Use diffusion-based or value-based uncertainty instead.
- **Fast-in-Slow VLA** (2025) — Dual-system VLA: System 2 (VLM) at 7-9 Hz, System 1 at 117-200 Hz. 8-11% improvement over single-system approaches.
- **GR-2** (ByteDance, 2024) — Video-pretrained world model as VLA backbone. 97.7% success on CALVIN, 2x generalization improvement.
- **V-JEPA 2-AC** (Meta, 2025) — JEPA world model for robotics. Plans 16x faster than pixel-space world models. Zero-shot transfer to unseen Franka arms.
- **Fast-WAM** (Yuan et al., 2026) — Shows the value of world action models is training-time video co-supervision, not test-time imagination. At 6B params, skipping future generation at inference costs <1.5% success with 4x speedup (190ms vs 810ms). Validates our Phase 6 amortized policy approach as the small-model equivalent. At 15M params, test-time search is still needed.

## Repository Structure

```
le-harness/
  jepa.py          # LeWM world model (encoder, predictor, rollout, cost)
  module.py        # Transformer blocks, attention, embedders, SIGReg
  train.py         # Training loop
  eval.py          # Evaluation with CEM/Adam planning
  utils.py         # Preprocessing, callbacks
  config/          # Hydra configs for training and eval
  docs/
    PROJECT.md     # This document
    HARNESS.md     # Technical architecture details
  harness/         # LeHarness code (built during phases)
  scripts/         # Sweep, benchmark, and utility scripts
```
