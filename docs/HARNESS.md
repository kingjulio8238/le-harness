# LeWM Planning Harness

Plan for building an optimized planning harness around LeWM for onboard robotic deployment.

## The System

```
┌─────────────────────────────────────────────────┐
│                  JETSON AGX ORIN                 │
│                                                  │
│  Camera → LeWM Encoder (ViT-tiny, 5.5M)         │
│              ↓                                   │
│  LeWM World Model generates N candidate         │
│  trajectories in latent space (~9M params)       │
│              ↓                                   │
│  Value Function scores each trajectory step      │
│  V(z_t, z_goal) summed over rollout (~200-500K)  │
│              ↓                                   │
│  Low-level policy executes best trajectory       │
│              ↓                                   │
│  Robot actuators                                 │
└─────────────────────────────────────────────────┘
```

## Does The Plan Hold Up? Yes, with changes.

The world model (LeWM) is the right size for onboard. 15M params fits easily in Jetson memory (~30MB FP16). The problem is the **planning loop**, not the model.

Current CEM config is way too slow for Jetson. 300 samples x 30 iterations x 5 horizon steps = 45,000 neural net forward passes per decision. On AGX Orin that's ~1-2.5 seconds. On Orin Nano, 4-10 seconds. Unusable.

The fix isn't just making CEM faster — it's restructuring the whole planning loop.

## Revised Plan: What The Harness Actually Needs To Do

### Layer 1: Faster Trajectory Generation (World Model Side)

| Change | Impact | Effort |
|--------|--------|--------|
| iCEM (colored noise + elite retention) | 3-10x fewer samples needed | Swap solver, zero retraining |
| Adaptive early stopping | 2-3x fewer iterations | ~50 lines of code |
| `torch.compile(backend="tensorrt")` for predictor + encoder | 2-3x faster per forward pass | Engineering work |
| CUDA Graphs for single predictor forward pass | 1.5-2x (eliminates kernel launch overhead) | Engineering work |
| Pre-allocated rollout buffers (replace `torch.cat`) | Eliminates per-step memory allocation | Engineering work |
| **Net target**: 64 samples, 5 iterations, compiled | **~50-100ms per decision → 10-20 Hz** | |

**Note on CUDA Graphs:** CUDA Graphs require fixed computation graphs and cannot contain conditional branching. They apply to the single-step predictor call only — the CEM iteration loop and adaptive stopping logic must remain in Python. Do not attempt to graph the outer loop.

### Layer 2: Per-Step Value Function for Trajectory Scoring

This replaces LeWM's current cost function (last-step MSE to goal embedding) with a learned value function.

**What to build:** A per-step value function V(z_t, z_goal) that:
- Takes a single LeWM embedding (dim=192) and the goal embedding (dim=192) as input
- Outputs a scalar progress score for that state relative to the goal
- Trajectory cost = sum of V(z_t, z_goal) over rollout steps, with higher weight on terminal step
- Trained offline using expert demonstrations + synthetically degraded trajectories

**Why per-step instead of trajectory-level:**
- TD-MPC2, Dreamer, and Value Summation (Shao 2022) all show summed per-step values outperform single trajectory scores
- Simpler to train (each (state, goal, progress) triple is an independent example — more data-efficient)
- Naturally handles variable-length horizons
- Avoids causal confusion (trajectory-level models can learn to detect easy initial states rather than evaluating action quality)
- Doubles as a terminal value function, extending effective planning horizon beyond the rollout

**Architecture (based on SimDist + TD-MPC2):**
- Input: concatenation of z_t (192-dim) and z_goal (192-dim) = 384-dim
- 2-layer MLP with hidden dim 256, LayerNorm, Mish activation
- Output: scalar progress score
- Size: ~200-500K parameters
- Latency: <0.1ms on Jetson (negligible compared to world model rollouts)

**Why not 1-5M or 60M params:** The value function operates on 192-dim embeddings that are already semantically rich. PlaNet, Dreamer, and TD-MPC2 all use small MLP heads (<500K params) for reward/value prediction on latent states. SARM's 60M figure is for a model processing raw video frames through its own vision backbone — irrelevant when operating on precomputed embeddings.

**Training data must include failures:** Training only on expert demonstrations is insufficient (Robometer, 2026). Generate synthetic suboptimal trajectories by running CEM with low budgets or random actions. Also include predictor-generated rollouts (not just encoder outputs from real trajectories) to avoid distribution shift between training and planning.

**Reward hacking mitigation:** Train a small ensemble (3-5 copies with different seeds, ~200K each = ~1M total). Use the minimum or mean prediction as the cost signal. The CEM optimizer will exploit any spurious correlations — the ensemble guards against this.

### Layer 3: Low-Level Policy Execution

**Two modes:**

**Fast mode (amortized, 50+ Hz):** A small policy network (~2-5M params) that maps (encoded_observation, encoded_goal) → action. Trained via DAgger (iterative dataset aggregation), not pure behavioral cloning — BC fails when the planner produces multimodal action distributions (DAMPC, IROS 2025). Consider a Gaussian mixture or diffusion policy head if actions are multimodal.

**Deliberate mode (10-20 Hz):** Full world model rollout + value function scoring when the system detects it needs to think harder.

**Mode switching:** Use state-novelty detection (Mahalanobis distance on 192-dim embeddings from training distribution) combined with value function confidence (ensemble disagreement on the value prediction). Do NOT rely solely on policy ensemble disagreement — it conflates multimodality with uncertainty (Diff-DAgger, 2024). Add switching hysteresis (minimum dwell time in each mode) to prevent oscillation. Warm-start the planner with the fast policy's recent actions when switching to deliberate mode.

**Safety:** Enforce action bounds (joint limits, velocity limits) on all outputs from both modes. Add a fallback stop action when both the policy uncertainty and value function uncertainty are high.

### Layer 4: Offline Training Pipeline (Not On Jetson)

This runs on your GPU server to improve all three components:

1. **Run large-budget CEM on server** → collect (state, goal, action, outcome) tuples → **train fast policy via DAgger** (deploy policy, collect failure states, re-plan, add to training set, repeat)
2. **Collect value function training data**: expert trajectories with ground-truth progress, synthetic failures from low-budget CEM, predictor-generated rollouts labeled against actual outcomes. Optionally use VLM (Qwen-VL-4B) for labeling subtle quality dimensions not captured by environment state.
3. **Fine-tune LeWM's predictor** on task-specific data if needed

## Hardware Verdict

All latency estimates assume AGX Orin in MAXN power mode (60W, full 275 TOPS). Default 30W mode gives roughly half performance.

| Component | Params | Jetson AGX Orin Latency | Feasible? |
|-----------|--------|------------------------|-----------|
| LeWM encoder (ViT-tiny) | 5.5M | ~5-8ms (TRT FP16) | Yes |
| LeWM predictor rollout (64 samples, 5 steps) | 9M | ~30-60ms (TRT) | Yes |
| Value function scoring (ensemble of 5) | ~1M total | <0.5ms | Yes |
| Fast policy (amortized) | 2-5M | <2ms | Yes |
| **Full deliberate planning step** | | **~50-100ms** | **10-20 Hz** |
| **Fast policy only** | | **~7-10ms** | **50+ Hz** |
| Total GPU memory | | ~200-400MB FP16 | 8GB Orin Nano works |

Orin Nano (8GB) is viable for the optimized system. AGX Orin gives more headroom and is recommended for development.

## Compilation Strategy

Prefer `torch.compile(backend="tensorrt")` over raw ONNX-to-TensorRT export. It is officially supported on Jetson (JetPack 6.2+), benchmarks show parity with standalone TRT for small transformers, and it avoids the fragile ONNX tracing step. Fall back to explicit ONNX export only if torch.compile fails to hit latency targets.

For the autoregressive predictor: compile the single-step forward pass. The rollout loop stays in Python. This is the same pattern used by TensorRT-LLM and vLLM for autoregressive decoding.

**Additional optimizations for Phase 4+:**
- INT8 post-training quantization (1.5-2x over FP16; collect ~500 representative input samples for calibration)
- Pre-allocate rollout buffers instead of `torch.cat` per step in `jepa.py:94,97`
- Pin host memory for CPU-GPU transfers (action sequences from CEM solver)
- Dedicated CUDA streams: overlap encoder inference with CEM action sampling
- Profile with `nsys` (Nsight Systems) for timeline analysis, `trtexec` for isolated engine benchmarks

## Jetson Deployment Checklist

- [ ] Set `nvpmodel` to MAXN (60W) for benchmarking; document which mode the 10 Hz target assumes
- [ ] Build TRT engines on-device (engines are hardware-specific, do not cross-compile from desktop)
- [ ] Warm up all engines during initialization (first-inference cold-start penalty)
- [ ] Test under sustained load for thermal throttling (Orin throttles if overheated)
- [ ] Camera pipeline: V4L2 capture on separate thread, preprocessing (resize + normalize) on GPU
- [ ] Track engine-to-checkpoint version mapping when models are retrained

## What's NOT Feasible On Jetson

- Running a 4-8B VLM reward model (Robometer, LRM) — must distill offline
- 300 samples x 30 iterations CEM — must reduce to ~64 x 5
- Vanilla PyTorch inference — must use torch.compile or TensorRT
- CUDA Graphs around the full CEM loop (conflicts with adaptive stopping) — graph single predictor call only

## Build Order

1. **Model fidelity audit** (measure prediction error vs. rollout depth — verify the model before optimizing the planner)
2. **iCEM + adaptive stopping** (immediate speedup, validates the approach)
3. **Per-step value function** trained on LeWM embeddings (replaces MSE cost)
4. **torch.compile / TensorRT** for encoder + predictor + INT8 calibration
5. **DAgger-trained fast policy** with novelty-based mode switching
6. **Dual-mode harness** with safety constraints and switching hysteresis
7. **Jetson deployment** with pre-allocated buffers, CUDA Graphs on predictor, thermal testing
