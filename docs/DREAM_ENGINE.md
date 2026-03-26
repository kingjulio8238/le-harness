# LeHarness Dream Engine

The next evolution of LeHarness: from flat CEM search to structured dream exploration.

## What Changes

LeHarness v1 (Phases 0-7) proved that a 15M-param world model can plan at 11 Hz on an RTX 4090. But it uses **flat CEM** — all dreams are the same length, same depth, independently sampled. This limits it to short-horizon tasks (horizon 5) and misses opportunities for structured exploration.

The Dream Engine replaces flat search with:

```
                    ┌─ dream 1a ─── dream 1aa (score: 0.92) ← BEST
        ┌─ dream 1 ┤
        │          └─ dream 1b ─── dream 1ba (score: 0.71)
root ───┤
        │          ┌─ dream 2a ─── dream 2aa (score: 0.88)
        ├─ dream 2 ┤
        │          └─ dream 2b     (pruned: score < threshold)
        │
        └─ dream 3                 (pruned at depth 1)
```

**Dreams** = candidate trajectories in latent space (192-dim JEPA embeddings).
**Dream trees** = branching exploration where promising dreams spawn children.
**Dream chains** = sequences of short dreams stitched together for long-horizon tasks.

## Why the 15M Model is the Right Size for This

Large models (1B+) can afford ~10-100 rollouts per planning step. LeWM at 15M can afford **thousands**. On an RTX 4090, the predictor runs at ~1ms per call (compiled). That means:

- 1,000 rollouts = 1 second
- 10,000 rollouts = 10 seconds (offline planning)
- At 11 Hz real-time: ~100 rollouts per 89ms budget

This budget enables tree search that large models cannot do. The small model's speed is its moat.

---

## Phase D1: Multi-Task Validation (NON-BLOCKING)

**Goal:** Prove the LeHarness pipeline generalizes across tasks.

**Status:** Deferred — Google Drive permissions needed for TwoRoom/Cube/Reacher checkpoints. PushT baseline confirmed at 96% success. D2-D5 do not depend on D1. Do this whenever Drive access is sorted.

**Steps:**
1. Download TwoRoom, Cube, and Reacher datasets + checkpoints from the existing Google Drive.
2. Run `harness/pipeline.py` on each task with the task's pretrained checkpoint. No code changes — just swap the config and policy path.
3. For each task, record: success rate, planning latency, and whether the pipeline works out of the box.
4. If a task fails, diagnose: is it the config (different action dims, horizons)? The compilation (different input shapes)? The CEM budget (different cost landscapes)?
5. Fix any task-specific issues in the pipeline to make it truly task-agnostic.
6. Document the per-task results and any tuning needed.

**Gate:** Pipeline achieves >50% success on at least 3 of 4 tasks without task-specific code changes.

**Estimated cost:** ~$2-4 (one pod session).

---

## Phase D2: Dream Chaining — COMPLETE

**Goal:** Enable long-horizon planning (horizon 20-50+) by chaining short, reliable dreams.

**Status:** Complete. Gate not passed — chaining does not outperform single-horizon on PushT. Root cause identified and partially mitigated.

**Architecture:**

```
Dream Chain for a long-horizon task:

  Observation → [Dream 1: steps 1-5] → subgoal_1
                                         ↓
                            re-encode → [Dream 2: steps 6-10] → subgoal_2
                                                                  ↓
                                                     re-encode → [Dream 3: steps 11-15] → goal
```

Each dream is a standard CEM solve over horizon 5. Between dreams, the predicted endpoint is used as the next dream's starting state.

### D2 Results

| Config | Success Rate | Latency | Hz |
|--------|-------------|---------|-----|
| Single horizon (baseline) | **28%** (14/50) | 93ms | 10.7 |
| Chained 3×5 (interpolated subgoals) | **4%** (2/50) | 267ms | 3.7 |
| Chained 5×5 (interpolated subgoals) | **2%** (1/50) | 436ms | 2.2 |
| Chained 3×5 (no subgoals — all chains target goal) | **22%** (11/50) | 267ms | 3.3 |

### Drift Measurement (5 chains, 20 episodes)

| Metric | Value |
|--------|-------|
| MSE (predicted vs actual embedding) | 0.053 +/- 0.037 |
| MSE max | 0.443 |
| Cosine similarity | 0.976 +/- 0.018 |
| Cosine similarity min | 0.803 |

Drift is moderate — the model predicts direction well (cosine ~0.98) but magnitude drifts, compounding across chains.

### Key Findings

1. **Linear interpolation in latent space is geometrically valid but semantically useless.** The 192-dim embeddings are unnormalized (no L2 constraint) and SIGReg pushes them toward isotropic Gaussian. Convex combinations stay in-distribution mathematically. But interpolated midpoints don't correspond to reachable physical states — a point halfway between "T at position A" and "T at position B" is a blurred average, not a useful waypoint.

2. **Removing interpolated subgoals recovered most performance (4% → 22%).** When all chains target the final goal directly, chaining no longer actively hurts. But it still doesn't help — 22% vs 28% single-horizon, at 3x the latency.

3. **PushT is too short-horizon for chaining to matter.** The task is typically solved in <10 steps. Looking 15-25 steps ahead via chaining provides no benefit when the planning horizon (5 steps) already reaches the goal.

4. **Chaining's value requires a harder task.** On tasks requiring 20+ steps with obstacles or sequenced contacts, the multi-step lookahead from chaining could provide signal that flat CEM misses. PushT doesn't test this.

### Implications for D3

- D3 (Dream Trees) does NOT automatically avoid the interpolation problem — tree nodes still optimize MSE-to-goal through the same latent space.
- However, D3's tree structure provides value through **structured exploration** (branching, pruning), not through longer horizons. This is a different mechanism than chaining.
- D3 should be evaluated on PushT first (to compare against flat CEM at matched compute), then on harder tasks if PushT is too simple.

**Actual cost:** ~$2 (one pod session).

---

## Phase D3: Dream Trees

**Goal:** Replace flat CEM sampling with tree-structured search over latent rollouts. Promising dreams branch into children; unpromising ones are pruned.

**Why this is the core differentiator:** Every world model planner today uses flat sampling (CEM, MPPI, gradient descent). Tree search allocates more compute to promising futures and less to dead ends. At 15M params, we can afford thousands of tree nodes per planning step — something a 6B model cannot do.

**Architecture: CEM-inside-MCTS**

Combine CEM's sampling efficiency at the root with MCTS's structured exploration deeper in the tree. This is a novel hybrid:

```
Root node (current state):
  ├─ CEM round 1: sample 64 action sequences, evaluate, keep top 12
  ├─ CEM round 2: resample around elites, evaluate, keep top 12
  ├─ CEM round 3: final refinement → top 8 "root dreams"
  │
  For each of the top 8 root dreams:
    ├─ Expand: roll out to depth 5, arrive at predicted state z_5
    ├─ At z_5: sample 8 child action sequences (from policy prior or CEM)
    ├─ Evaluate children via rollout to depth 10
    ├─ Keep top 2 children per parent
    │
    For each surviving child:
      ├─ Expand to depth 15
      └─ Score with terminal value estimate
```

**Budget math:** 3 CEM rounds × 64 samples = 192 root evaluations. 8 parents × 8 children = 64 depth-2 evaluations. 16 survivors × 1 final rollout = 16 depth-3 evaluations. Total: ~272 full rollouts. At ~5 predictor calls per rollout = ~1,360 forward passes. Well within the 89ms budget at compiled speed.

**Steps:**
1. **Implement `DreamTree`**: A tree data structure where each node holds (latent_state, action_sequence, cost, depth, children).
2. **Root expansion via CEM**: At the root, run 2-3 CEM iterations to identify promising action regions. Extract the top-K elite action sequences as root children.
3. **Progressive widening**: At each non-root node, use progressive widening (Couetoux 2011) to control branching. New children are added only when `|children| < C × visits^alpha` (alpha ~0.3-0.5). This prevents the tree from growing too wide at depth.
4. **Child evaluation**: Each child is evaluated by rolling out the world model from the parent's predicted latent state. Score = cumulative reward along the rollout + terminal value estimate (Phase D4 value function, or MSE to goal as baseline).
5. **Backpropagation**: After evaluating a leaf, propagate the score up the tree (max or mean of children's scores at each node). Use this to guide future expansion toward high-value branches.
6. **Pruning**: Remove branches whose score is more than 2σ below the best sibling. This frees compute budget for promising branches.
7. **Action selection**: After the tree is built, select the root action sequence whose subtree has the highest mean score.
8. **Benchmark**: Compare DreamTree vs. flat CEM at matched compute budgets. The tree should outperform on tasks requiring multi-step lookahead (avoiding obstacles, sequencing contacts).

**Key references:**
- Sampled MuZero (Hubert et al., 2021) — MCTS with sampled actions for continuous spaces
- Progressive widening (Couetoux et al., 2011) — controls branching in continuous MCTS
- TD-MPC2 (Hansen et al., 2024) — MPPI in latent space, the current SOTA flat planner
- TreeQN (Farquhar et al., 2018) — learned tree-structured planning in latent space

**Gate:** Dream trees outperform flat CEM at equal compute budget on at least one task. Specifically: higher success rate OR same success rate with fewer total forward passes. If trees don't help, the tasks may be too simple for structured search — test on longer-horizon or more complex environments.

**Estimated cost:** ~$4-8 (significant code, multiple eval sessions).

---

## Phase D4: Dream Scoring v2

**Goal:** Fix the reward hacking problem from Phase 4 and build a robust dream scoring system that combines progress, confidence, and diversity.

**Why Phase 4 failed:** The value function was trained only on expert trajectories and then exploited by CEM — the optimizer found latent states with spuriously high value that were out of the training distribution. The fix requires both better training data and architectural constraints.

**Architecture: Multi-Signal Dream Scorer**

```
Dream Score = R_progress(tau)                    # task progress
            - lambda * sigma_ensemble(tau)       # confidence penalty (MOPO-style)
            + alpha * diversity(tau)              # diversity bonus (optional)
```

**Steps:**
1. **Training data v2**: Collect three types of trajectories:
   - Expert demonstrations (high progress, in-distribution)
   - Synthetic failures: CEM with budget 8 samples / 2 iterations (low progress, shows what failure looks like)
   - Predictor-generated rollouts: roll out the world model from real states with random actions (mixed progress, matches the distribution CEM will actually query)
   - This triples the training data diversity and covers the regions CEM explores.

2. **Conservative value training**: Train V(z_t, z_goal) with a CQL-style penalty (Kumar et al., 2020) that pushes down values for out-of-distribution inputs. Alternatively, use DreamerV3's symlog + discrete distribution head — this architecturally bounds the output range, preventing arbitrarily high values.

3. **Ensemble with WARM**: Train 5 value functions independently, then weight-average their parameters (WARM, Rame et al., 2024). This retains only the generalizable features and suppresses memorized exploitable ones — one forward pass, cost of one model, robustness of five.

4. **Uncertainty penalty**: During planning, subtract `lambda × max_std_across_ensemble` from the score (MOPO-style). Trajectories through uncertain regions get penalized, preventing CEM from exploiting model errors.

5. **Diversity scoring (for dream trees)**: When selecting which dreams to expand in the tree, add a DPP-based diversity term (Yuan et al., 2019) that favors dreams with orthogonal latent trajectories. Compute the kernel matrix over dream embeddings and select a diverse subset.

6. **Adaptive rollout truncation**: If cumulative ensemble uncertainty exceeds a threshold during a rollout, truncate and use the value estimate at the truncation point. This prevents error from compounding and being exploited (Infoprop, 2025).

7. **Evaluate**: Replace MSE cost with the multi-signal scorer in both flat CEM and DreamTree. Compare success rates. The scorer should match or exceed MSE on easy tasks and significantly improve on tasks where MSE gets stuck (non-convex paths, obstacles).

**Key references:**
- MOPO (Yu et al., 2020) — uncertainty penalty on rewards
- CQL (Kumar et al., 2020) — conservative Q-learning for OOD inputs
- WARM (Rame et al., 2024) — weight-averaged reward models resist hacking
- DreamerV3 (Hafner et al., 2023) — symlog + discrete distribution heads
- DPP trajectory selection (Yuan et al., 2019) — diversity via determinantal point processes

**Gate:** Multi-signal dream scorer achieves ≥ MSE success rate on PushT AND higher success rate on at least one task where MSE struggles (e.g., long-horizon or obstacle-avoidance variants). If it still underperforms MSE, the latent space is genuinely well-structured for distance-based planning and the scorer adds no value — accept this and move on.

**Estimated cost:** ~$4-8 (data collection, training, evaluation).

---

## Phase D5: Language-Conditioned Dreams

**Goal:** Let engineers specify dream goals with language instead of images.

**Why last:** Language conditioning changes the goal representation but not the dream engine itself. All the tree search, chaining, and scoring infrastructure from D2-D4 works the same — only the goal embedding source changes. Building it last means it benefits from all prior improvements.

**Architecture:**

```
"push the T to the target"
         ↓
  CLIP Text Encoder (frozen, ~150M params)
         ↓
  text_embedding (512-dim)
         ↓
  Linear projection (512 → 192)
         ↓
  goal_embedding (192-dim) ← same space as LeWM image goals
         ↓
  Dream Engine plans toward goal_embedding
```

**Steps:**
1. **Add CLIP text encoder**: Use `openai/clip-vit-base-patch32` (frozen, ~150M params). Extract the text embedding for a task instruction. This runs once per task instruction, not per planning step — negligible latency.
2. **Train a projection layer**: A small linear layer (512 → 192) that maps CLIP text embeddings to LeWM's goal embedding space. Train it by collecting (goal_image, goal_text) pairs from the dataset, encoding both, and minimizing MSE between `projection(CLIP_text(goal_text))` and `LeWM_encoder(goal_image)`.
3. **Integrate with DreamEngine**: Replace `goal_embedding = encoder(goal_image)` with `goal_embedding = projection(CLIP_text(goal_text))` when a text instruction is provided. The rest of the pipeline (CEM, trees, chains, scoring) is unchanged.
4. **Evaluate**: Test with held-out text instructions. Compare success rates when the goal is specified via image vs. text. The text path will likely be slightly worse (lossy projection) but should be within 10%.
5. **Failure mode**: If the CLIP-to-LeWM projection is too lossy (text goal embeddings are far from image goal embeddings), consider fine-tuning the projection with contrastive loss or using a larger CLIP model.

**What this enables:** An engineer can now do:
```python
pipeline = DreamEngine("pusht/lejepa")
action = pipeline.plan(obs_image, goal_text="push the T to the target")
```

**Gate:** Language-conditioned planning achieves ≥80% of image-conditioned success rate on PushT.

**Estimated cost:** ~$2-4 (CLIP is off-the-shelf, projection training is minutes).

---

## On-Pod vs Off-Pod Work

### The Rule

**Off-pod (local, free):** Write code, design architectures, review results, plan experiments. No GPU needed.

**On-pod (RunPod 4090, ~$0.40/hr):** Run evals, collect data, train models, benchmark. GPU time is for execution, not thinking.

### Per-Phase Breakdown

#### D1: Multi-Task Validation

| Off-Pod | On-Pod |
|---------|--------|
| Nothing — this phase is all execution | Download TwoRoom/Cube/Reacher data + checkpoints to volume |
| | Run `final_benchmark.py` on each task |
| | Record results, diagnose any failures |
| Review results, document per-task findings | |

#### D2: Dream Chaining

| Off-Pod | On-Pod |
|---------|--------|
| Design `DreamChainer` class (wrapper around CEM solver) | Run chained vs. single-horizon eval on PushT |
| Implement subgoal interpolation logic (pure Python, no GPU) | Test with extended eval_budget (100-200 steps) |
| Write eval script for long-horizon comparison | Measure drift between predicted and actual re-encoded states |
| Push code to git | |

#### D3: Dream Trees

| Off-Pod | On-Pod |
|---------|--------|
| Design `DreamTree` data structure (nodes, edges, scores) | Benchmark DreamTree vs flat CEM at matched compute |
| Implement CEM-inside-MCTS logic: root expansion, progressive widening, backprop, pruning | Profile per-node latency to verify budget math |
| Implement action selection from completed tree | Test on multiple tasks from D1 |
| Write tree visualization / logging for debugging | |
| This is the most code-heavy phase — most work is off-pod | |

#### D4: Dream Scoring v2

| Off-Pod | On-Pod |
|---------|--------|
| Design multi-signal scorer architecture | Collect training data v2 (expert + failures + predictor rollouts) |
| Implement CQL penalty, WARM weight averaging, symlog heads | Train value ensemble (~minutes) |
| Implement DPP diversity selection | Evaluate scorer in flat CEM and DreamTree |
| Implement adaptive rollout truncation | Compare vs MSE baseline on all tasks |
| Write training script with all three data sources | |

#### D5: Language Conditioning

| Off-Pod | On-Pod |
|---------|--------|
| Design CLIP → LeWM projection architecture | Collect (goal_image, goal_text) pairs from dataset |
| Implement text goal encoding path in pipeline | Train projection layer (~minutes) |
| Write eval script for text vs image goal comparison | Evaluate text-conditioned planning |
| | Compare success rates: text vs image goals |

### Dependency Chain

```
D2: Dream Chaining     ← needs working pipeline (have it, 96% on PushT)
D3: Dream Trees         ← needs D2 (chaining is how trees go deep)
D4: Dream Scoring v2    ← needs D3 (scoring matters most inside trees)
D5: Language Conditioning ← independent, can be done anytime
D1: Multi-Task          ← independent, do when Drive access works
```

D1 is non-blocking. D5 is independent. The critical path is D2 → D3 → D4.

### Pod Sessions (Batched)

| Session | Phase | What to write locally first | Pod work |
|---------|-------|-----------------------------|----------|
| **D-Session 1** | D2 | `DreamChainer`, subgoal interpolation, eval script | Run chained vs single-horizon eval on PushT |
| **D-Session 2** | D3 | `DreamTree` (all logic), tree logging, benchmark script | Profile and benchmark trees vs flat CEM |
| **D-Session 3** | D4 | Scorer architecture, training script, all penalty implementations | Collect data, train, evaluate |
| **D-Session 4** | D5 | CLIP integration, projection layer, text eval script | Train projection, evaluate |
| **D-Session 5** | D1 | Nothing (needs Drive access) | Download data, run benchmarks on 4 tasks |

**Between sessions:** Review results from the previous session. Decide whether the gate passed. Write all code for the next session. Push to git. Only spin up the pod when code is ready to run.

**Anti-patterns (same as v1):**
- Writing code on the pod — write locally, push to git, pull on pod
- Running one eval interactively — script everything, use tmux
- Leaving pod idle — stop when the session's work is done
- Re-downloading data — it's on the network volume from v1

### Pod Startup (Every Session)

```bash
bash /workspace/data/setup.sh
cd /workspace/le-harness
python scripts/patch_icem.py  # apply iCEM fix if not already applied
```

### D-Session 1 (D2: Dream Chaining) — On-Pod Instructions

This is the current next session. Code should be written locally and pushed before SSHing in.

**What should already be pushed to git before this session:**
- `harness/dream_chainer.py` — DreamChainer class
- `scripts/eval_dream_chaining.py` — Eval script comparing chained vs single-horizon
- Any modifications to `harness/pipeline.py` to support chaining

**On-pod (copy-paste):**

```bash
# 1. Setup
bash /workspace/data/setup.sh
cd /workspace/le-harness
python scripts/patch_icem.py
export STABLEWM_HOME=/workspace/data

# 2. Start tmux
tmux new -s d2

# 3. Sanity check — confirm PushT baseline still works
python scripts/final_benchmark.py --policy pusht/lejepa 2>&1 | tail -5

# 4. Run dream chaining eval — single-horizon baseline (horizon 5, receding)
python scripts/eval_dream_chaining.py \
  --policy pusht/lejepa \
  --mode single \
  --eval-budget 100 \
  --num-eval 50 \
  2>&1 | tee /workspace/data/results/d2_single_horizon.log

# 5. Run dream chaining — 3 chains of horizon 5 with subgoal interpolation
python scripts/eval_dream_chaining.py \
  --policy pusht/lejepa \
  --mode chained \
  --num-chains 3 \
  --eval-budget 100 \
  --num-eval 50 \
  2>&1 | tee /workspace/data/results/d2_chained_3x5.log

# 6. Run dream chaining — 5 chains (horizon 25)
python scripts/eval_dream_chaining.py \
  --policy pusht/lejepa \
  --mode chained \
  --num-chains 5 \
  --eval-budget 200 \
  --num-eval 50 \
  2>&1 | tee /workspace/data/results/d2_chained_5x5.log

# 7. Measure drift — how much do predicted endpoints diverge from re-encoded reality?
python scripts/eval_dream_chaining.py \
  --policy pusht/lejepa \
  --mode chained \
  --num-chains 5 \
  --measure-drift \
  --num-eval 20 \
  2>&1 | tee /workspace/data/results/d2_drift.log

# 8. Review results
echo ""
echo "========================================="
echo "  D2: Dream Chaining Results"
echo "========================================="
echo ""
echo "--- Single horizon (baseline) ---"
tail -10 /workspace/data/results/d2_single_horizon.log
echo ""
echo "--- Chained 3x5 ---"
tail -10 /workspace/data/results/d2_chained_3x5.log
echo ""
echo "--- Chained 5x5 ---"
tail -10 /workspace/data/results/d2_chained_5x5.log
echo ""
echo "--- Drift measurement ---"
tail -10 /workspace/data/results/d2_drift.log
echo ""
echo "========================================="
echo "  GATE: Chained > single on tasks needing >10 steps?"
echo "========================================="

# 9. Save and stop
git add -A && git commit -m "D2: dream chaining results" && git push
# Stop pod from dashboard
```

**Detach tmux:** `Ctrl+B, D`. Reconnect: `tmux attach -t d2`.

**After stopping pod (off-pod):** Review the 4 log files. If chaining helps, start writing `DreamTree` locally for D-Session 2. If chaining doesn't help (single-horizon is just as good at extended budgets), the task may be too simple — consider whether to proceed to D3 or first find a harder task.

---

## Summary: Dream Engine Phases

```
D2: Dream Chaining           ← COMPLETE — gate not passed on PushT, subgoal interpolation diagnosed
D3: Dream Trees              ← NEXT — structured search, core differentiator
D4: Dream Scoring v2         ← robust multi-signal scoring, fix reward hacking
D5: Language Conditioning     ← independent, text goals for adoption
D1: Multi-Task Validation    ← non-blocking, do when Drive access works
```

**Critical path:** D2 → D3 → D4. D1 and D5 are independent.

**Total estimated cost:** ~$15-30 across 5 pod sessions on RunPod RTX 4090.

**What the Dream Engine enables that flat CEM cannot:**
- Long-horizon tasks (20-50+ steps) via dream chaining
- Structured exploration via dream trees (allocate compute to promising futures)
- Robust scoring that resists reward hacking
- Language-specified goals
- All of this on a 15M-param model at interactive speeds

**The thesis:** Small world models + smart dream infrastructure > large models + flat search.
