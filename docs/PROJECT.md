# LeWM Planning Harness: Project Overview

## What Is This

LeWM is a 15M-parameter world model that predicts the future in latent space. It's small, fast to train, and its representations encode real physical structure. But out of the box, it plans slowly (1-2 seconds per decision) and scores trajectories with raw embedding distance — a blunt instrument.

This project wraps LeWM in a **planning harness** that makes it plan better and plan faster, turning it from a research artifact into something that can run onboard a robot and make decisions in real time.

The end state: a system where LeWM imagines candidate futures, a learned reward model picks the best one, and a low-level policy executes it — all running on a Jetson at 10-50 Hz.

## Why This Matters

The robotics field is converging on two approaches to robot intelligence:

1. **VLM-backbone VLAs** (RT-2, OpenVLA, Pi0) — Large vision-language models that map observations directly to actions. They understand semantics but don't model physics. They're slow (5-10 Hz on cloud GPUs) and expensive to deploy.

2. **World-model-based planning** (Dreamer, TD-MPC, LeWM) — Small dynamics models that imagine futures and search for good actions. They understand physics but are limited by their planning loop.

The opportunity: **a world model that plans well enough and fast enough can replace the VLM as the backbone of a VLA**. LeWM is the right starting point — 15M params vs 7-55B for VLM-based VLAs, latent-space planning instead of pixel generation, and a representation that already encodes physical quantities.

What's missing is the infrastructure to make it plan intelligently and deploy at real-time speeds. That's what this harness provides.

## What We're Building

Four layers, each building on the last:

### Layer 1 — Faster Planning
Take the existing CEM planner from 300 samples x 30 iterations (45,000 forward passes, ~1-2s) down to 64 samples x 5 iterations (~320 forward passes, ~50-100ms) without losing task success. This is done through:
- iCEM solver (colored noise, elite retention from prior timestep)
- Adaptive early stopping (exit when cost converges)
- TensorRT compilation of the encoder and predictor
- CUDA Graphs to eliminate kernel launch overhead

### Layer 2 — Smarter Trajectory Scoring
Replace the current cost function (MSE between predicted embedding and goal embedding) with a learned reward model that operates on LeWM's latent trajectories. A 1-5M parameter Transformer or MLP that takes a sequence of 192-dim embeddings and outputs a progress score. Trained offline using expert demonstrations and VLM-generated quality labels. This captures task-relevant structure that raw embedding distance misses — intermediate progress, contact events, orientation constraints.

### Layer 3 — Dual-Mode Execution
A small amortized policy (~2-5M params) distilled from the planner handles routine states at 50+ Hz. The full world model + reward model planning loop activates for novel or high-stakes states at 10-20 Hz. A confidence gate decides which mode to use.

### Layer 4 — Offline Improvement Pipeline
A server-side pipeline that continuously improves all three components:
- Runs large-budget CEM to generate expert (state, action) pairs for policy distillation
- Uses a VLM (Qwen-VL-4B) to label trajectory quality for reward model training
- Fine-tunes the world model predictor on task-specific data

## How We Evaluate Success

### Primary Metrics

**Task success rate** — Does the robot complete the task? Measured on PushT (2D manipulation), TwoRoom (navigation), Cube (3D manipulation), and Reacher (continuous control). The baseline is LeWM's current eval: 50 episodes per task with CEM planning.

**Planning latency (ms/decision)** — How long does a single planning step take? Current baseline: ~1000-2500ms on AGX Orin. Target: <100ms.

**Control frequency (Hz)** — How many decisions per second? Current: <1 Hz. Target: 10-20 Hz in deliberate mode, 50+ Hz in fast mode.

### Secondary Metrics

**Planning efficiency curve** — Success rate as a function of CEM sample budget. Shows whether the reward model lets us achieve the same success with fewer samples.

**Anytime performance** — Success rate if we stop planning after k CEM iterations. Validates adaptive early stopping.

**Reward-task correlation** — Does the learned reward model's score actually predict task success? Measured as Pearson r between trajectory reward and binary outcome.

**Representation utility** — Linear probing accuracy for physical quantities (position, angle, velocity) from LeWM's latent space. Inherited from the original paper; we track it to ensure harness modifications don't degrade the representation.

**Amortization gap** — Success rate of the distilled fast policy vs. the full planner. Measures how much we lose by skipping planning.

### What Success Looks Like

| Milestone | Success Criterion |
|-----------|-------------------|
| M1: Faster planning | Same success rate at 10x fewer samples (64 vs 300) |
| M2: Reward model | Higher success rate than MSE cost at equal sample budget |
| M3: Real-time on Jetson | 10+ Hz deliberate planning on AGX Orin with TensorRT |
| M4: Fast policy | >80% of planner success rate at 50+ Hz |
| M5: Integrated system | Dual-mode harness running on Jetson, completing manipulation tasks end-to-end |

### What Failure Looks Like

- Reducing CEM budget tanks success rate, meaning the world model's predictions aren't good enough for efficient planning — fix the model, not just the planner
- Learned reward model doesn't outperform MSE cost — the latent space is already well-structured and a simpler cost is sufficient
- Amortized policy can't capture the planner's behavior — the action distribution is too multimodal for a feedforward network
- TensorRT export breaks the autoregressive rollout loop — need custom ONNX tracing for the sequential prediction

## Phases

Each phase has a hard gate. Do not proceed to the next phase until the gate is passed. If a gate fails, the failure mode tells you what to fix before retrying.

---

### Phase 0: Baseline

Establish ground truth numbers. Everything that follows is measured against these.

**Steps:**
1. Set up eval environment (RunPod GPU or local CUDA machine)
2. Download PushT dataset + pretrained `lejepa` checkpoint to `$STABLEWM_HOME`
3. Run stock eval: `python eval.py --config-name=pusht.yaml policy=pusht/lejepa`
4. Record: success rate (%), mean planning time per step (ms), total eval time (s)
5. Run random baseline: `python eval.py --config-name=pusht.yaml policy=random`
6. Record random success rate as lower bound

**Artifacts:**
- Baseline success rate and planning latency on PushT
- Random baseline success rate
- These two numbers are the reference for every gate that follows

**Gate:** Baseline eval completes successfully and produces a non-trivial success rate above random. If LeWM's pretrained checkpoint doesn't outperform random, the world model itself needs work — stop here and investigate the checkpoint or the eval setup before building anything on top.

---

### Phase 1: Planning Budget Reduction

Determine how much planning compute can be cut without losing performance. This is pure configuration sweeps — no new code beyond a sweep script and logging.

**Steps:**
1. Sweep sample count: run eval at `solver.num_samples=` {300, 128, 64, 32, 16} with 30 iterations each. Record success rate and planning time for each.
2. Sweep iteration count: run eval at `solver.n_steps=` {30, 15, 10, 5, 3} with 300 samples each. Record success rate and planning time for each.
3. Sweep both jointly at the promising intersection points from steps 1-2.
4. Swap CEM for iCEM: run eval with `solver=icem` (available in stable-worldmodel as `ICEMSolver`) at the same sample/iteration grid. Compare against CEM at equal budgets.
5. Log CEM cost at each iteration to identify convergence behavior — this informs whether adaptive early stopping is viable.

**Artifacts:**
- Planning efficiency table: success rate vs. (samples, iterations, solver)
- Cost convergence curves per task
- Identified minimum viable planning budget (samples x iterations) for iCEM

**Gate:** There exists a configuration where success rate is within 5% of baseline AND total forward passes are reduced by at least 5x (e.g., 64 samples x 5 iterations = 320 vs. baseline 9,000). If no such configuration exists, the world model's predictions are too noisy for efficient planning — consider fine-tuning the predictor or increasing the embedding dimension before proceeding.

---

### Phase 2: Adaptive Early Stopping

Build the first piece of harness code. The planner should stop spending compute when it has already converged.

**Steps:**
1. Using the cost convergence data from Phase 1, define a stopping criterion: exit CEM when the relative cost improvement between iterations drops below a threshold (e.g., `|cost[i] - cost[i-1]| / cost[i-1] < epsilon`)
2. Implement adaptive stopping as a wrapper around the solver (does not modify stable-worldmodel internals)
3. Run full eval with adaptive stopping enabled at the best fixed budget from Phase 1
4. Measure: actual iterations used per step (mean, p50, p95), success rate, planning time

**Artifacts:**
- `harness/adaptive_solver.py` — Solver wrapper with early stopping
- Distribution of actual iterations used (most steps should exit early)
- Updated planning latency numbers

**Gate:** Adaptive stopping reduces mean iterations by at least 30% compared to fixed budget while maintaining success rate within 2% of Phase 1's best fixed configuration. If most steps use the full budget (the cost never converges early), the CEM landscape is too flat — this means the cost function lacks gradient signal, which motivates Phase 3.

---

### Phase 3: Learned Reward Model

Replace MSE embedding distance with a learned cost function that captures task-relevant structure.

**Steps:**
1. **Data collection:** Run LeWM's encoder over the expert dataset to extract latent trajectories. For each trajectory, compute ground-truth progress (from environment state, e.g., PushT's `eval_state` distance-to-goal over time). Store as (latent_trajectory, progress_scores) pairs.
2. **Architecture:** Build a small reward model — 1-2 layer Transformer or MLP that takes a sequence of 192-dim LeWM embeddings and outputs a scalar progress score per trajectory. Target size: 1-5M parameters.
3. **Training:** Train on the collected pairs with MSE loss between predicted progress and ground-truth progress. Split 80/20 train/val.
4. **Integration:** Create a new cost model class that wraps the reward model and conforms to the `get_cost()` interface expected by the solvers. The cost is the negative reward score (planner minimizes cost).
5. **Eval:** Run the full planning eval with the reward model as cost function, using the best planning budget from Phase 1-2. Compare success rate against MSE baseline at equal sample budgets. Also compare at reduced budgets — does the reward model let you plan effectively with even fewer samples?

**Artifacts:**
- `harness/reward_model.py` — Reward model architecture and training script
- `harness/reward_cost.py` — Cost model wrapper for solver integration
- Trained reward model checkpoint
- Comparison table: success rate at {32, 64, 128} samples with MSE cost vs. learned reward cost

**Gate:** Learned reward model achieves higher success rate than MSE cost at the same planning budget, OR achieves equal success rate at half the planning budget. If neither holds, the latent space is already well-structured for MSE-based planning and the reward model adds no value — consider whether the reward model needs richer training signal (VLM labels, preference data) or whether to skip this layer and proceed directly to speed optimization.

---

### Phase 4: TensorRT Optimization

Make the existing pipeline fast enough for real-time by compiling the neural network components.

**Steps:**
1. Export LeWM's ViT-tiny encoder to ONNX with fixed input shape (1, 3, 224, 224). Convert to TensorRT FP16 engine. Verify output matches PyTorch within tolerance.
2. Export the predictor (ARPredictor) to ONNX. This is harder due to the autoregressive loop — export the single-step predictor with fixed batch size and sequence length, then call it in a Python loop. Alternatively, use `torch.compile` with the `tensorrt` backend.
3. If a learned reward model exists from Phase 3, export it to TensorRT as well.
4. Benchmark on target hardware:
   - Desktop GPU: measure ms/decision to establish a performance ceiling
   - Jetson AGX Orin (if available): measure ms/decision end-to-end
5. Profile to identify remaining bottlenecks (kernel launch overhead, memory transfers, Python overhead). Apply CUDA Graphs if kernel launches dominate.

**Artifacts:**
- TensorRT engine files for encoder, predictor, and reward model
- `harness/trt_inference.py` — TensorRT inference wrapper
- Benchmark table: ms/decision breakdown by component on each hardware target

**Gate:** Full deliberate planning step (encode + rollout + score + select) completes in <100ms on desktop GPU, projecting to <200ms on AGX Orin (based on known desktop-to-Jetson scaling factors). If TensorRT export fails for the autoregressive predictor, fall back to `torch.compile` or `torch.jit.script` — the gate is about latency, not the specific compilation method.

---

### Phase 5: Policy Distillation

Train a fast reactive policy that handles easy states without invoking the full planner.

**Steps:**
1. **Data generation:** Run the optimized planner (iCEM + reward model + TRT) on a large and diverse set of (start_state, goal_state) pairs from the dataset. For each, record (observation, goal, planned_action). Generate at least 50K pairs.
2. **Policy architecture:** A small feedforward network (2-5M params) that maps (encoded_observation, encoded_goal) → action. Input is LeWM's 192-dim embeddings (encoder runs once, shared with the planner). Output is continuous action.
3. **Training:** Behavioral cloning on the planner's output. MSE loss on actions. Train until validation loss plateaus.
4. **Uncertainty estimation:** Add a lightweight uncertainty signal — either a small ensemble (3-5 copies of the policy head, not the encoder) or MC dropout. High disagreement = invoke the full planner.
5. **Dual-mode harness:** Implement the mode-switching logic. Default to fast policy. Switch to deliberate planning when uncertainty exceeds threshold. Switch back when uncertainty drops.
6. **Eval:** Measure success rate and effective Hz for: (a) planner only, (b) fast policy only, (c) dual-mode harness.

**Artifacts:**
- `harness/fast_policy.py` — Distilled policy network with uncertainty head
- `harness/dual_mode.py` — Mode-switching controller
- Trained fast policy checkpoint
- Comparison: success rate and Hz for planner-only vs. policy-only vs. dual-mode

**Gate:** The dual-mode harness achieves >80% of planner-only success rate while operating at an effective frequency of 20+ Hz (meaning most steps use the fast policy). If the fast policy's success rate is too low (<60% of planner), the planner's action distribution may be too complex for behavioral cloning — consider DAgger (iterative dataset aggregation where the policy acts and the planner corrects) or a diffusion-based policy head.

---

### Phase 6: End-to-End Integration

Bring everything together on target hardware with real sensor input.

**Steps:**
1. Assemble the full pipeline on Jetson AGX Orin: camera capture → image preprocessing → LeWM encoder (TRT) → dual-mode harness (fast policy or planner) → action output
2. End-to-end latency profiling: measure total time from frame capture to action output. Break down by component.
3. Run the standard eval suite (PushT, and optionally TwoRoom/Cube/Reacher) through the integrated system. Compare success rates to Phase 0 baseline.
4. Stress tests: introduce perturbations not seen during training — shifted start positions, rotated goals, novel obstacle configurations. Measure degradation.
5. Compare against published VLA baselines on the same tasks where numbers are available (OpenVLA, Pi0 on similar manipulation benchmarks).

**Artifacts:**
- `harness/pipeline.py` — Full end-to-end inference pipeline
- Jetson deployment scripts and TRT engine files
- Final performance table: success rate, Hz, ms/decision, memory usage
- Comparison to VLA baselines

**Gate:** The integrated system completes PushT episodes end-to-end on Jetson at 10+ Hz with success rate within 15% of Phase 0 desktop baseline. If latency is too high, profile and identify the bottleneck — it's likely either the encoder (consider a smaller/faster vision backbone) or the planner (reduce budget further or lean more on the fast policy). If success rate drops more than 15%, the TRT conversion or the dual-mode switching is introducing errors — debug by running each component in isolation.

## Related Work

This project builds directly on:
- **LeWM** (Maes et al., 2026) — The world model at the core. 15M params, JEPA architecture, stable end-to-end training.
- **iCEM** (Pinneri et al., 2020) — Sample-efficient planning with colored noise and elite retention.
- **SimDist** (2025) — Simulation distillation: train reward/value models in sim, freeze for deployment. Small 1-layer Transformer reward models running at 50 Hz.
- **SARM** (2025) — Stage-aware reward modeling with frozen CLIP encoder + 60M Transformer. Shows small reward models work for trajectory scoring.
- **TD-MPC2** (Hansen et al., 2024) — Demonstrates amortized policy + online planning hybrid. Policy prior warm-starts the planner.
- **GR-2** (ByteDance, 2024) — Video-pretrained world model as VLA backbone. 97.7% success on CALVIN, 2x generalization improvement over non-world-model approaches.

## Repository Structure

```
le-wm/
  jepa.py          # LeWM world model (encoder, predictor, rollout, cost)
  module.py        # Transformer blocks, attention, embedders, SIGReg
  train.py         # Training loop
  eval.py          # Evaluation with CEM/Adam planning
  utils.py         # Preprocessing, callbacks
  config/          # Hydra configs for training and eval
  docs/
    PROJECT.md     # This document
    HARNESS.md     # Technical architecture details
```
