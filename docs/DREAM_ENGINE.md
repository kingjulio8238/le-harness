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

## Phase D1: Multi-Task Validation — COMPLETE

**Goal:** Prove the LeHarness pipeline generalizes across tasks.

**Status:** Complete. Pipeline generalizes to TwoRoom. Dream Tree outperforms flat CEM on TwoRoom, confirming PushT findings.

### D1 Results (TwoRoom)

| Config | Success | Latency | Hz |
|--------|---------|---------|-----|
| Flat CEM (eval.py, 50 parallel envs) | **88%** (44/50) | ~250s total | — |
| Flat CEM (eval_dream_tree.py, sequential) | **46%** (23/50) | 118ms/step | 8.5 |
| **Dream Tree 4R depth-2** (sequential) | **62%** (31/50) | 448ms/step | 2.2 |

### Key Findings

1. **Pipeline generalizes out of the box.** TwoRoom required zero code changes to `eval.py` — just swap the config and policy path. 88% success rate on first run.

2. **Dream Tree improves over flat CEM on TwoRoom: 62% vs 46% (+35% relative).** This confirms the PushT result (26% vs 16%, +62% relative). Tree search in latent space is a general improvement, not task-specific.

3. **`eval_dream_tree.py` required a bug fix for multi-task support.** The script hardcoded `"state"` as the dataset key for setting agent/goal positions. TwoRoom uses `"proprio"` instead. Fixed to use config-driven callables (same mechanism as `eval.py`). Also added `--config-name` argument to support non-PushT configs.

4. **Success rate gap between scripts (88% vs 46% flat CEM).** `eval.py` runs 50 parallel envs via the library's `evaluate_from_dataset`; `eval_dream_tree.py` runs `num_envs=1` sequentially with its own episode loop. The apples-to-apples comparison (same script, same episodes) shows Dream Tree clearly wins.

5. **Cube and Reacher checkpoints available on HF but no datasets.** Only TwoRoom has both checkpoint + dataset. Full 3/4 task gate requires Cube and Reacher datasets to be collected or sourced.

**Gate:** Partial pass — pipeline achieves >50% on 2 of 4 tasks (PushT 88%+, TwoRoom 88%) without task-specific code changes. Cube/Reacher blocked on missing datasets.

**Actual cost:** ~$1 (shared pod session with data setup).

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
D1: Multi-Task Validation    ← COMPLETE — TwoRoom 88% flat / 62% tree, pipeline generalizes
D2: Dream Chaining           ← COMPLETE — gate not passed, subgoal interpolation diagnosed
D3: Dream Trees              ← COMPLETE — GATE PASSED, 26% vs 16% (+62% relative) on PushT
D4: Dream Scoring v2         ← COMPLETE — gate not passed, scorer hurts tree precision
D5: Language Conditioning     ← not started, independent
```

**Critical path:** D1-D4 complete. Tree search validated on two tasks.

**Total cost so far:** ~$9 across 4 pod sessions.

**Core result:** Tree search in latent space improves planning across tasks — +62% relative on PushT (26% vs 16%), +35% relative on TwoRoom (62% vs 46%). The tree's value comes from precise depth scoring (full CEM at depth), not from learned scorers. The tree amplifies signal quality — great with precise signals, fragile with noisy ones.

**The thesis:** Small world models + smart dream infrastructure > large models + flat search.

---

## Next Steps: Making LeHarness Real

Four things separate the current Dream Engine from something engineers would deploy. Each is independent — work on whichever is unblocked.

---

### N1: Batched CEM (Speed) — COMPLETE

**Problem:** Dream Tree runs at 1.5-2.8 Hz. The tree needs 8 sequential CEM calls (4 root + 4 depth). Each CEM is ~89ms compiled, but CUDA graphs with `reduce-overhead` mode require fixed tensor shapes, so the 8 calls can't be batched into a single GPU operation. Total: ~690ms per planning step.

**Fix:** Batched CEM (`_cem_plan_batched`) + reduced CEM iterations. Batching alone gave 1.27x (549ms), but CEM convergence analysis showed iterations 7-15 contribute <1% cost improvement. Combined: 2.5x speedup.

**Results (TwoRoom, RTX 4090):**

| Config | Latency | Hz | Gate |
|--------|---------|-----|------|
| Sequential (n_steps=15, reduce-overhead) | 697ms | 1.4 | baseline |
| Batched (n_steps=15, default) | 549ms | 1.8 | FAIL |
| Batched (n_steps=10, default) | 385ms | 2.6 | FAIL |
| Batched (n_steps=8, default) | 325ms | 3.1 | FAIL |
| **Batched (n_steps=7, default)** | **282ms** | **3.5** | **PASS** |
| Batched (n_steps=5, default) | 213ms | 4.5 | PASS |

**Key insight:** CEM converges very quickly — the cost at iteration 7 is within 1% of iteration 15. The main win isn't batching GPU ops (B=4 barely helps because S=128 already saturates the GPU), it's **reducing wasted iterations**.

**Implementation:**
- `pipeline._cem_plan_batched()`: batched CEM supporting (B, 1, D) inputs
- `DreamTreePlanner(batched=True)`: defaults to `cem_steps=7` (overridable)
- Requires `compile_mode="default"` (CUDA graphs in reduce-overhead mode don't support variable batch sizes)

**Gate:** PASS — 282ms at n_steps=7 (3.5 Hz).

**Actual cost:** ~$2 (one pod session).

---

### N2: Language Conditioning (D5)

**Problem:** LeHarness only accepts goal images. Engineers want to say "push the block to the target" or "navigate to the green room."

**Fix:** Frozen CLIP text encoder → small projection layer (512→192) → LeWM's goal embedding space. Everything downstream (CEM, tree, scoring) is unchanged.

**Architecture:**

```
"navigate to the green room"
         ↓
  CLIP Text Encoder (frozen, ~150M params, runs once per instruction)
         ↓
  text_embedding (512-dim)
         ↓
  Learned linear projection (512 → 192, ~98K params)
         ↓
  goal_embedding (192-dim) ← same space as LeWM image goals
         ↓
  Dream Engine plans toward goal_embedding (unchanged)
```

**Steps:**
1. **Collect (goal_image, goal_text) pairs**: For TwoRoom, generate synthetic text descriptions from goal states: "navigate to position (X, Y)" or "reach the target in the green room." These can be generated programmatically from the dataset's state columns — no manual annotation needed. For PushT (if dataset is re-downloaded): "push the T-block to position (X, Y) at angle θ."
2. **Encode both modalities**: Run CLIP text encoder on goal_text, run LeWM encoder on goal_image. Collect ~5,000-10,000 pairs.
3. **Train projection**: Linear layer (512→192), MSE loss between `projection(CLIP_text(goal_text))` and `LeWM_encoder(goal_image)`. This is a <1 minute training job.
4. **Integrate into pipeline**: Add `plan_from_text(obs_image, goal_text)` method to `PlanningPipeline` that encodes the text via CLIP → projection, then runs the normal planning loop.
5. **Evaluate**: Compare success rates with image goal vs text goal on the same episodes. Text path will be lossier but should be within 80% of image-conditioned performance.

**Blocker:** Need a dataset with both images and text. TwoRoom doesn't have natural language annotations, but synthetic ones ("navigate to (X, Y)") are trivially generated from state data. This is sufficient for proving the mechanism works, even if the language is template-based.

**Off-pod work:**
- Implement `harness/language_encoder.py`: CLIP loading, projection layer, text→goal_emb pipeline
- Write synthetic text generation script for TwoRoom (template-based from state data)
- Write `scripts/train_language_projection.py`
- Write `scripts/eval_language.py` comparing image vs text goals
- Push to git

**On-pod work:**
- `pip install transformers` (for CLIP)
- Generate (goal_image, goal_text) pairs from TwoRoom dataset
- Train projection layer (~1 min)
- Evaluate text-conditioned planning vs image-conditioned
- Run Dream Tree with text goals to verify full stack works

**Gate:** Text-conditioned planning achieves ≥80% of image-conditioned success rate on TwoRoom.

**Estimated cost:** ~$2-4 (one pod session).

---

### N3: More Tasks (Generality)

**Problem:** Dream Tree is validated on 2 tasks (PushT 2D manipulation, TwoRoom navigation). For engineers to trust it, it should work on 3D tasks too.

**Missing data:**
- Cube (OGBench): checkpoint on HF (`kingJulio/le-harness-data/cube_lewm.tar.zst`), dataset is 44GB (too large to download from Drive due to quota limits)
- Reacher (DMControl): checkpoint on HF (`kingJulio/le-harness-data/reacher_lewm.tar.zst`), dataset is 22GB (download failed from Drive)

**Options to unblock:**
1. **Generate datasets yourself**: Run the environments with random/expert policies to collect data. The `stable-worldmodel` package includes environment wrappers. Would require training new LeWM checkpoints on the collected data.
2. **Contact the LeWM authors**: Ask Lucas Maes (lucas.maes@mila.quebec) for direct dataset access or an alternative download link.
3. **Use alternative benchmarks**: Find other JEPA-compatible environments with smaller datasets (e.g., MetaWorld, RoboSuite tasks).
4. **Wait for Drive quota reset**: Google Drive download quotas reset periodically. Try again in 24-48 hours.

**Steps (once data is available):**
1. Upload dataset to HF (`huggingface-cli upload`)
2. Download on pod from HF (datacenter speed)
3. Run flat CEM baseline: `python eval.py --config-name=cube policy=ogb_cube/lewm`
4. Run Dream Tree: `python scripts/eval_dream_tree.py --policy ogb_cube/lewm --config-name=cube --mode both`
5. Compare results

**Off-pod work:**
- Secure dataset access (contact authors, retry Drive, or generate)
- Upload to HF when obtained

**On-pod work:**
- Download from HF, decompress
- Run benchmarks (flat CEM + Dream Tree)
- ~1-2 hours per task

**Gate:** Dream Tree outperforms flat CEM on at least 1 additional task beyond PushT and TwoRoom.

**Estimated cost:** ~$2-4 per task (if data available).

---

### N4: Onboard Deployment (Jetson)

**Problem:** Everything runs on RTX 4090 in the cloud. For a robot, it needs to run on a Jetson AGX Orin strapped to the chassis.

**Hardware needed:** Jetson AGX Orin Developer Kit 64GB (~$2,000). Not currently available.

**Expected performance** (based on known 4090-to-Orin ~3-5x scaling):

| Component | RTX 4090 | AGX Orin (est.) |
|-----------|----------|-----------------|
| Flat CEM planning | 89ms | ~270-450ms |
| Dream Tree (4R full) | 689ms | ~2-3.5s |
| Dream Tree (batched, target) | ~200ms | ~600ms-1s |
| Fast policy (if distilled) | ~5ms | ~15-25ms |

Flat CEM would run at ~2-4 Hz on Orin. Dream Tree would need batched CEM (N1) first, then would run at ~1-1.5 Hz — not real-time for manipulation but usable for slow tasks (assembly, inspection).

**Steps (when hardware is available):**
1. **Install JetPack 6.2+** on the Orin (CUDA 12.x, TensorRT, cuDNN)
2. **Install PyTorch for Jetson** from NVIDIA's pip index
3. **Transfer checkpoints** from HF or pod
4. **Build compiled models on-device**: `torch.compile` engines are not portable from x86. Must recompile on ARM/Orin.
5. **Set power mode**: `nvpmodel` to MAXN (60W) for benchmarking
6. **Benchmark**: Run flat CEM and Dream Tree, measure ms/decision end-to-end
7. **Thermal testing**: Run under sustained load, check for throttling
8. **Camera pipeline**: V4L2 capture on separate thread, GPU preprocessing (resize + normalize)
9. **TensorRT export** (if torch.compile latency is insufficient): Export encoder and predictor to TRT FP16/INT8 engines built on-device

**Off-pod work (before hardware arrives):**
- Document the full deployment pipeline in `docs/JETSON.md`
- Prepare a deployment script that handles: model download, compilation, warmup, benchmark
- Consider INT8 quantization calibration (collect calibration data on pod, transfer to Jetson)

**On-Jetson work:**
- Everything in steps 1-9 above
- Iterate on compilation settings (reduce-overhead may not work on Jetson — test default and max-autotune modes)

**Gate:** Flat CEM runs at ≥2 Hz on AGX Orin. Dream Tree (batched) runs at ≥1 Hz. Both within 15% of RTX 4090 success rates.

**Estimated cost:** ~$2,000 for hardware. No cloud cost.

---

## Next Steps Priority

```
N1: Batched CEM        ← DONE (282ms, 3.5 Hz)
N2: Language (D5)       ← highest impact, unblocked (synthetic text from TwoRoom state data)
N3: More Tasks          ← blocked on dataset access
N4: Jetson              ← blocked on hardware
```

**Recommended order:** N2 next (makes the system usable with language commands). N3 and N4 are nice-to-have when blockers clear.
