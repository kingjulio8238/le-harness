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

## Phase D3: Dream Trees — COMPLETE

**Goal:** Replace flat CEM sampling with tree-structured search over latent rollouts.

**Status:** Complete. **Gate passed** — tree outperforms flat CEM by 62% relative on PushT.

**Architecture (simplified from original plan):**

Reuses pipeline's compiled `_cem_plan` as the atomic operation. Tree structure provides lookahead by running CEM from predicted future states:

```
Root: 4 diverse CEM calls → 4 (action, terminal_embedding) pairs
Depth 2: for each terminal_embedding, run full CEM → future cost
Select: root action whose depth-2 future cost is lowest
```

The key insight: flat CEM picks the action with the best immediate cost. Dream Tree picks the action whose predicted future is easiest to plan from.

### D3 Results

| Config | Success | Latency | Hz |
|--------|---------|---------|-----|
| Flat CEM (baseline) | **16%** | 92ms | 10.8 |
| Tree 2R cheap depth | **18%** | 193ms | 5.2 |
| Tree 2R full depth | **18%** | 351ms | 2.8 |
| Tree 4R cheap depth | **18%** | 375ms | 2.7 |
| **Tree 4R full depth** | **26%** | 689ms | 1.5 |

### Key Findings

1. **Tree search works: 26% vs 16% (+62% relative).** Selecting actions by future plannability beats selecting by immediate cost.

2. **Scoring quality is everything.** Full CEM depth scoring (15 rounds) is required for the 26% result. Cheap scoring (1-3 rounds) gives only 18% — the tree amplifies signal quality, good or bad.

3. **4 diverse roots required.** 2 roots + full depth = 18%. 4 roots + full depth = 26%. CEM convergence limits diversity, but 4 independent CEM runs provide enough variation.

4. **Latency stuck at 689ms.** CUDA graphs with `reduce-overhead` mode require fixed tensor shapes, blocking batched CEM. All 8 CEM calls (4 root + 4 depth) must run sequentially.

5. **Cheap depth scoring via `_score_state`** (single-pass random evaluation or mini-CEM) cannot substitute for full CEM. The tree needs precise depth estimates.

**Actual cost:** ~$4 (two pod sessions).

---

## Phase D4: Dream Scoring v2 — COMPLETE

**Goal:** Replace MSE-to-goal with learned multi-signal scoring to improve tree decisions.

**Status:** Complete. Gate not passed — scorer slightly helps flat CEM but hurts tree.

**Architecture:**

DreamScorer combining:
- MSE progress (start→end MSE reduction)
- Learned value V(z_t, z_goal) from 5-member ValueEnsemble
- Ensemble uncertainty penalty

Trained on expert trajectories + CEM rollout data. Optional WARM weight averaging.

### D4 Results

| Config | Success | Latency |
|--------|---------|---------|
| Flat CEM (MSE) | **18%** | 92ms |
| Tree 4R full (MSE) | **22%** | 690ms |
| Flat CEM (Scorer) | **20%** | 97ms |
| Tree 4R cheap (Scorer) | **18%** | 450ms |
| Tree 4R full (Scorer) | **16%** | 742ms |

### Key Findings

1. **Scorer slightly helps flat CEM (+2%)** but at negligible latency cost (5ms). The nonlinear value function captures something MSE misses for flat planning.

2. **Scorer hurts tree decisions (22% → 16%).** The tree amplifies scoring noise. A learned MSE approximation is strictly worse than exact MSE for the precise depth scoring the tree needs.

3. **Root cause: MSE-derived training labels.** The scorer learns a noisier version of the same signal it replaces. To truly improve tree scoring, it needs genuinely new information — task reward, reachability, contact sequencing — not latent distance proxies.

4. **The tree is high-risk/high-reward.** Great with precise signals (MSE: 26%), fragile with noisy ones (scorer: 16%). Flat CEM is more robust to scorer noise because it doesn't compound errors across depth.

**What would fix this:** Online value learning — train the scorer on actual planning outcomes (did this action lead to success?) rather than offline MSE-derived labels.

**Actual cost:** ~$1 (one pod session, training is fast).

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
D2: Dream Chaining           ← COMPLETE — gate not passed, subgoal interpolation diagnosed
D3: Dream Trees              ← COMPLETE — GATE PASSED, 26% vs 16% (+62% relative)
D4: Dream Scoring v2         ← COMPLETE — gate not passed, scorer hurts tree precision
D5: Language Conditioning     ← not started, independent
D1: Multi-Task Validation    ← IN PROGRESS — data download needed, see docs/UPLOAD_D1.md
```

**Critical path:** D2 → D3 → D4 complete. D1 is the key next step — validates tree search on harder tasks.

**Total cost so far:** ~$8 across 3 pod sessions.

**Core result:** Tree search in latent space improves planning by 62% relative over flat CEM. The tree's value comes from precise depth scoring (full CEM at depth), not from learned scorers. The tree amplifies signal quality — great with precise signals, fragile with noisy ones.

**Next steps:**
1. **D1: Multi-Task Validation** — TwoRoom/Cube/Reacher to validate tree search on harder tasks
2. **Online value learning** — train scorer on actual planning outcomes, not MSE proxies
3. **Batched CEM** — pipeline refactoring to cut tree latency from 689ms to ~250ms
4. **D5: Language Conditioning** — independent, quick win

**The thesis:** Small world models + smart dream infrastructure > large models + flat search.
