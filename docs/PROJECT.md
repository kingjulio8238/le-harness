# LeWM Planning Harness: Project Overview

## What Is This

LeWM is a 15M-parameter world model that predicts the future in latent space. It's small, fast to train, and its representations encode real physical structure. But out of the box, it plans slowly (1-2 seconds per decision) and scores trajectories with raw embedding distance — a blunt instrument.

This project wraps LeWM in a **planning harness** that makes it plan better and plan faster, turning it from a research artifact into something that can run onboard a robot and make decisions in real time.

The end state: a system where LeWM imagines candidate futures, a learned value function picks the best one, and a low-level policy executes it — all running on a Jetson at 10-50 Hz.

## Why This Matters

The robotics field is converging on two approaches to robot intelligence:

1. **VLM-backbone VLAs** (RT-2, OpenVLA, Pi0) — Large vision-language models that map observations directly to actions. They understand semantics but don't model physics. They're slow (5-10 Hz on cloud GPUs) and expensive to deploy.

2. **World-model-based planning** (Dreamer, TD-MPC, LeWM) — Small dynamics models that imagine futures and search for good actions. They understand physics but are limited by their planning loop.

This project demonstrates the latency and efficiency advantages of small world models for real-time robotic planning. A 15M-param model that plans at 10-50 Hz on a Jetson is a fundamentally different deployment profile than a 7-55B VLM requiring cloud GPUs. This is a stepping stone toward using world models as the planning backbone in a dual-system VLA architecture — the System 1 (fast reactive control) and System 2 (deliberate planning) that systems like GR00T N1 and Fast-in-Slow VLA are converging on.

**What this project is:** An efficient, real-time, image-goal-conditioned planner with a learned value function and dual-mode execution.

**What this project is not (yet):** A full VLA backbone. That would additionally require language conditioning, multi-task generalization, and cross-embodiment support — all valid future directions that build on the infrastructure created here.

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
| M0: Validated model | Prediction error grows gracefully with rollout depth; baseline outperforms random |
| M1: Faster planning | Same success rate at 5x fewer forward passes (iCEM at reduced budget) |
| M2: Value function | Higher success rate than MSE cost at equal budget, OR equal success at half budget |
| M3: Real-time | <100ms per planning step on desktop GPU |
| M4: Fast policy | >80% of planner success rate at 20+ Hz effective frequency |
| M5: Integrated system | Dual-mode harness on Jetson at 10+ Hz within 15% of desktop success rate |

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

### Phase 1: Model Fidelity Audit

Verify that the world model's predictions are reliable enough to support reduced planning budgets. This is the most commonly skipped step in MBRL projects and the most common reason they fail (MBPO, Janner 2019; PETS, Chua 2018).

**Steps:**
1. Run the LeWM encoder over the expert dataset to extract ground-truth latent trajectories (z_1, z_2, ..., z_T) for each episode.
2. For each trajectory, use the predictor to autoregressively generate predicted embeddings from z_1 with the same action sequence. Measure MSE between predicted and actual embeddings at rollout depths 1, 3, 5, and 10.
3. Plot prediction error vs. rollout depth. Identify where error becomes large relative to the inter-state distances in the latent space.
4. Run linear probing on the frozen encoder embeddings for task-relevant physical quantities (position, angle, velocity). Record accuracy as the representation quality baseline.
5. Correlate per-episode prediction error with per-episode task success from Phase 0. Does high prediction error predict failure?

**Artifacts:**
- Prediction error vs. rollout depth plot
- Linear probing accuracy for physical quantities
- Correlation between prediction error and task success

**Gate:** Prediction error at the planning horizon (depth 5) is less than 50% of the mean inter-state distance in the latent space — meaning predicted states are closer to their targets than to random other states. If prediction error is too high, the model's predictions are unreliable at the planning horizon — consider fine-tuning the predictor, reducing the planning horizon, or increasing the embedding dimension before proceeding. Do not try to optimize a planner around an unreliable model.

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

### Phase 7: End-to-End Integration

Bring everything together on target hardware.

**Steps:**
1. Assemble the full pipeline on Jetson AGX Orin in MAXN power mode: camera capture (V4L2 on separate thread) → GPU preprocessing (resize 224x224 + normalize) → LeWM encoder (compiled) → dual-mode harness (fast policy or planner) → action output
2. Build all compiled engines on-device (TRT engines are hardware-specific). Warm up all engines during initialization.
3. End-to-end latency profiling with `nsys`: measure total time from frame capture to action output. Break down by component. Test under sustained load for thermal throttling.
4. Run the standard eval suite (PushT, and optionally TwoRoom/Cube/Reacher) through the integrated system. Compare success rates to Phase 0 baseline.
5. Stress tests: introduce perturbations not seen during training — shifted start positions, rotated goals, novel obstacle configurations. Measure degradation.
6. Report an efficiency comparison: params, FLOPS/decision, memory, latency, and success rate. This positions the system relative to larger approaches (VLM-backbone VLAs) on the metrics where small world models have a structural advantage — not as a head-to-head task comparison (which would be unfair in both directions).

**Artifacts:**
- `harness/pipeline.py` — Full end-to-end inference pipeline
- Jetson deployment scripts, power mode configs, and compiled engine files
- Final performance table: success rate, Hz, ms/decision, memory usage, power draw
- Efficiency comparison against published VLA numbers (params, FLOPS, latency)

**Gate:** The integrated system completes PushT episodes end-to-end on Jetson at 10+ Hz with success rate within 15% of Phase 0 desktop baseline. If latency is too high, profile and identify the bottleneck — it's likely either the encoder (consider a smaller/faster vision backbone) or the planner (reduce budget further or lean more on the fast policy). If success rate drops more than 15%, the compilation or the dual-mode switching is introducing errors — debug by running each component in isolation vs. the desktop PyTorch reference.

## Future Directions (Beyond This Project)

The following would extend this system toward a full VLA backbone but are out of scope for the current plan:

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
  harness/         # Planning harness code (built during phases)
```
