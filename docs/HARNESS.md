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
│  Reward Model scores each trajectory (~60M)      │
│              ↓                                   │
│  Low-level policy executes best trajectory       │
│              ↓                                   │
│  Robot actuators                                 │
└─────────────────────────────────────────────────┘
```

## Does The Plan Hold Up? Yes, with changes.

The world model (LeWM) is the right size for onboard. 15M params fits easily in Jetson memory (~30MB FP16). The problem is the **planning loop**, not the model.

Current CEM config is way too slow for Jetson. 300 samples × 30 iterations × 5 horizon steps = 45,000 neural net forward passes per decision. On AGX Orin that's ~1-2.5 seconds. On Orin Nano, 4-10 seconds. Unusable.

The fix isn't just making CEM faster — it's restructuring the whole planning loop.

## Revised Plan: What The Harness Actually Needs To Do

### Layer 1: Faster Trajectory Generation (World Model Side)

| Change | Impact | Effort |
|--------|--------|--------|
| iCEM (colored noise + elite retention) | 3-10x fewer samples needed | Swap solver, zero retraining |
| Adaptive early stopping | 2-3x fewer iterations | ~50 lines of code |
| TensorRT export of predictor + encoder | 2-3x faster per forward pass | Engineering work |
| CUDA Graphs for rollout loop | 1.5-2x (eliminates kernel launch overhead) | Engineering work |
| **Net target**: 64 samples, 5 iterations, TRT | **~50-100ms per decision → 10-20 Hz** | |

### Layer 2: Reward Model for Trajectory Scoring

This replaces LeWM's current cost function (raw MSE to goal embedding) with something smarter.

**What to build:** A small learned reward model (~60M params, SARM-style) that:
- Takes LeWM's latent trajectory embeddings as input (not raw pixels — the encoder already ran)
- Scores trajectory progress toward the goal (continuous 0→1, not binary)
- Trained offline using expert demonstrations + VLM-generated labels

**Why this matters:** MSE in latent space is a crude cost. A learned reward model can capture:
- Progress even when the straight-line latent path is wrong
- Task-specific success criteria (contact, orientation, clearance)
- Intermediate subgoal satisfaction

**Architecture (based on SimDist + SARM):**
- Input: sequence of LeWM embeddings (dim=192) from a rollout
- 1-2 layer Transformer or MLP over the sequence
- Output: scalar score per trajectory
- Size: ~1-5M params (operates on 192-dim embeddings, not pixels)
- Latency: <1ms on Jetson (tiny compared to world model rollouts)

### Layer 3: Low-Level Policy Execution

**Two modes:**

**Fast mode (amortized, 50+ Hz):** A small feedforward policy network (~2-5M params) trained by distilling the CEM planner's outputs. Handles easy/familiar states with a single forward pass.

**Deliberate mode (10-20 Hz):** Full world model rollout + reward scoring when:
- Policy uncertainty is high
- New/unfamiliar observation
- Approaching critical task phase

The harness switches between modes based on a confidence threshold.

### Layer 4: Offline Training Pipeline (Not On Jetson)

This runs on your GPU server to improve all three components:

1. **Run CEM with many samples on server** → collect (state, best_action) pairs → **distill into fast policy**
2. **Use VLM (Qwen-VL-4B or similar) to label trajectory quality** → **distill into small reward model** that runs on Jetson
3. **Fine-tune LeWM's predictor** on task-specific data if needed

## Hardware Verdict

| Component | Params | Jetson AGX Orin Latency | Feasible? |
|-----------|--------|------------------------|-----------|
| LeWM encoder (ViT-tiny) | 5.5M | ~5-8ms (TRT FP16) | Yes |
| LeWM predictor rollout (64 samples, 5 steps) | 9M | ~30-60ms (TRT) | Yes |
| Reward model scoring | 1-5M | <1ms | Yes |
| Fast policy (amortized) | 2-5M | <2ms | Yes |
| **Full deliberate planning step** | | **~50-100ms** | **10-20 Hz** |
| **Fast policy only** | | **~7-10ms** | **50+ Hz** |
| Total GPU memory | | ~200-400MB FP16 | 8GB Orin Nano works |

Orin Nano (8GB) is viable for the optimized system. AGX Orin gives more headroom but isn't strictly necessary once you've reduced the CEM budget.

## What's NOT Feasible On Jetson

- Running a 4-8B VLM reward model (Robometer, LRM) — must distill offline
- 300 samples × 30 iterations CEM — must reduce to ~64 × 5
- Vanilla PyTorch inference — must use TensorRT

## Build Order

1. **iCEM + adaptive stopping** (immediate speedup, validates the approach)
2. **Small reward model** trained on LeWM embeddings (replaces MSE cost)
3. **TensorRT export** of encoder + predictor
4. **Amortized policy** distilled from improved planner
5. **Dual-mode harness** that switches fast/deliberate
6. **Jetson deployment** with CUDA Graphs
