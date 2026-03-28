# LeHarness: How It Works

## Overview

LeHarness is a 15M-parameter world model (LeWM) that plans in 192-dim latent space using CEM search. It doesn't output actions directly — it **evaluates** candidate actions by simulating their outcomes in latent space, thousands of times per planning step, and picks the best one.

---

## The World Model

The model operates entirely in **192-dim latent space**. It never reconstructs images.

```
obs image (224×224×3)                     goal image (224×224×3)
        ↓                                         ↓
   ViT encoder                                ViT encoder
        ↓                                         ↓
   obs_emb (192)                            goal_emb (192)
        ↓                                         ↓
   predictor(obs_emb, act_emb) → pred_emb (192)   │
        ↓                                         ↓
   cost = MSE(pred_emb, goal_emb)  ←──────────────┘
        ↓
   CEM selects lowest-cost action
        ↓
   action (10) → env
```

### Components

| Component | Input | Output | Size |
|-----------|-------|--------|------|
| ViT encoder | (224, 224, 3) image | (192,) embedding | 12-layer ViT-Tiny |
| Action encoder | (10,) raw action | (192,) embedding | Conv1d + MLP |
| Predictor | (emb, act_emb) | next-state emb (192,) | 6-layer transformer |
| Cost | pred vs goal | scalar MSE | sum of squared diffs |

### Key Dimensions

| Dimension | Value | Notes |
|-----------|-------|-------|
| Embedding (D) | 192 | Core representation dimensionality |
| Action dimension | 10 = frameskip(5) × action(2) | Task-specific (TwoRoom/PushT) |
| History window | 3 steps | Fixed by ARPredictor |
| Predictor depth | 6 blocks | Transformer layers |
| CEM samples | 128 | Trajectory candidates per iteration |
| CEM iterations | 7 (batched) / 15 (sequential) | Refinement steps |
| Planning horizon | 5 steps | Look-ahead depth |

---

## The Planning Loop

The planner runs every ~276ms (3.6 Hz with batched Dream Tree):

```
Camera → obs image (224×224)
                │
                ▼
         ┌─────────────────────────────────────────┐
         │  CEM: try 128 random action sequences   │
         │                                         │
         │  For each candidate [a1, a2, a3, a4, a5]:│
         │    dream step 1: predict(obs, a1) → s1  │
         │    dream step 2: predict(s1, a2)  → s2  │
         │    dream step 3: predict(s2, a3)  → s3  │
         │    dream step 4: predict(s3, a4)  → s4  │
         │    dream step 5: predict(s4, a5)  → s5  │
         │                                         │
         │    cost = ||s5 - goal||²                 │
         │                                         │
         │  Keep top 25, refine, repeat 7x         │
         └──────────────┬──────────────────────────┘
                        │
                        ▼
              Best action: a1  ← only the FIRST action
                        │
                        ▼
                Robot executes a1
                        │
                        ▼
              New camera image → repeat
```

The model dreams **128 × 5 × 7 = 4,480 futures per planning step**. It picks the action sequence whose dream ends closest to the goal, executes only the first action, then replans from the new observation. This is **receding-horizon control** (also called model-predictive control / MPC).

---

## Dream Tree: Parallel Exploration

Flat CEM runs one search. Dream Tree runs **4 independent searches**, then asks: "which search leads to a future that's easiest to plan from?"

```
                    obs
                   / | \ \
                  /  |  \ \
         CEM₁  CEM₂  CEM₃  CEM₄     ← 4 root searches (batched, 1 GPU call)
          ↓      ↓      ↓      ↓
         s1₁    s1₂    s1₃    s1₄     ← 4 different predicted futures
          ↓      ↓      ↓      ↓
         CEM₅  CEM₆  CEM₇  CEM₈     ← "can I plan FROM here?" (batched)
          ↓      ↓      ↓      ↓
        cost₁  cost₂  cost₃  cost₄

         Pick root action with lowest depth cost
```

The tree doesn't just ask "which action gets me closest to the goal?" — it asks "which action puts me in a state where I can *keep making progress*?" This is why it outperforms flat CEM (+35-62% relative improvement).

With batched CEM (N1), the 8 CEM calls collapse into 2 batched GPU operations, running the full tree in 276ms.

---

## Language Conditioning

Goals can come from images or text. Two paths to the same 192-dim goal embedding:

```
Path 1 (Image goal):
  goal_image → ViT encoder → (192,)

Path 2 (CLIP text goal):
  "go to the upper left area" → CLIP ViT-B/32 (frozen) → MLP → (192,)

Path 3 (Coordinate text goal):
  "navigate to (0.43, 0.57)" → parse (x,y) → MLP → (192,)
```

Everything downstream (CEM, Dream Tree, cost function) is identical regardless of how the goal embedding was produced. Text-conditioned planning matches image-conditioned performance (104% on TwoRoom).

---

## What's Task-Specific vs Task-Agnostic

### Task-agnostic (transfers to any task):
- **Planning stack**: CEM, Dream Tree, language conditioning, batched inference
- **Architecture**: ViT encoder, ARPredictor transformer, embedding space design
- **Pipeline code**: `PlanningPipeline`, `DreamTreePlanner`, `LanguageEncoder`

### Task-specific (must retrain for each task):
- **World model weights**: the encoder/predictor learn the dynamics of a specific environment
- **Action encoder**: hardcoded to `frameskip × action_dim` (e.g., 5×2=10 for TwoRoom). Different tasks have different action dimensions.
- **Language projection**: trained on task-specific (caption, embedding) pairs

### To add a new task:
1. Collect demonstration data (real or sim)
2. Train a new LeWM on that data (same architecture, new weights + new action encoder dim)
3. The planning stack transfers unchanged
4. Optionally train a new language projection for text goals

---

## What's Needed for a Real Robot

The current loop is: **sim camera → plan → sim action → sim camera → ...**

For a real robot, replace the sim with real hardware:

```
WHAT EXISTS (sim)                    WHAT'S NEEDED (real robot)
──────────────                       ──────────────────────────
sim.render() → image                 camera.capture() → image
plan(image, goal) → action           plan(image, goal) → action  ← SAME
sim.step(action) → next state        robot.send(action) → wait
                                     camera.capture() → next image
```

The planner itself is **unchanged**. The gaps are:

1. **Camera interface** — feed real 224×224 RGB frames instead of sim renders
2. **Action mapping** — translate the model's action space to actual motor commands (joint velocities, end-effector deltas, etc.)
3. **Timing** — the robot must execute actions at the planner's rate. At 3.6 Hz (276ms per action), this is fine for coarse manipulation and navigation, but too slow for high-precision grasping or dynamic tasks.
4. **Sim-to-real transfer** — the world model was trained on sim images. Real images look different. Options:
   - Train directly on real robot data
   - Domain randomization in sim
   - Fine-tune on a small amount of real data

### The real bottleneck

The architecture is sound. The actual blocker for real robotics is: **the world model only works for tasks it was trained on.** The current model was trained on TwoRoom (2D navigation, 2D actions). It cannot plan for a robot arm because it has never seen robot arm images or predicted robot arm dynamics.

---

## Current Validation Status

### Completed Phases

| Phase | What | Result |
|-------|------|--------|
| D1 | Multi-task validation | PushT 88%, TwoRoom 88% (flat CEM) |
| D3 | Dream Trees | +35-62% over flat CEM on both tasks |
| N1 | Batched CEM | 697ms → 282ms (2.5x speedup, 3.5 Hz) |
| N2 | Language conditioning | Text goals match image goals (104%) |
| Integration | Full stack (text + batched tree) | 276ms / 3.6 Hz, 86% of image baseline |
| S1.5 | Tactical replanning (confidence + drift) | 64% on 50 eps (94% of baseline), conf replans=49, drift replans=9 |

### Failed Experiments (documented learnings)

| Phase | What | Why It Failed |
|-------|------|---------------|
| D2 | Dream Chaining | Interpolation in latent space produces blurred averages, not reachable states |
| D4 | Learned scorers | Hurt tree precision — MSE is the right cost signal, not learned value functions |
| N2 v1 | Linear CLIP projection | CLIP treats all coordinate strings as identical text |

### Blocked

| Phase | What | Blocker |
|-------|------|---------|
| N3 | More tasks (Cube, Reacher) | Datasets unavailable (Google Drive quota, 44GB/22GB) |
| N4 | Jetson deployment | Hardware not procured (~$2K) |

---

## Performance Summary

```
Full stack on TwoRoom (RTX 4090):

  CLIP text + batched Dream Tree
  ├── Latency:      276ms per planning step
  ├── Throughput:    3.6 Hz
  ├── Success rate:  48% (86% of image baseline)
  ├── CEM samples:   128 × 7 iterations = 896 per call
  ├── Tree calls:    2 batched CEM ops (root + depth)
  └── Total dreams:  ~6,300 latent rollouts per planning step

  S1.5 Tactical Planning (confidence + drift replanning)
  ├── Baseline:       68% flat CEM (50 episodes)
  ├── S1.5:           64% (94% of baseline)
  ├── Conf replans:   49 total across 50 episodes
  ├── Drift replans:  9 (threshold=500)
  ├── Mean confidence: 0.81
  └── Drift sweet spot: threshold 500-750
```
