
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e?usp=sharing">Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from the [Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e?usp=sharing) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pre-trained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e). Download the checkpoint archive and place the extracted files under `$STABLEWM_HOME/`.

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

To load the object checkpoint via the `stable_worldmodel` API:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

Both functions accept:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

---

## LeHarness: Optimized Planning Infrastructure

LeHarness wraps LeWM in an optimized planning infrastructure that reduces planning latency from **1,310ms to 89ms** (15x speedup) while maintaining 94% success rate on PushT.

### What LeHarness Does

LeHarness makes LeWM plan **faster** without changing the model. It optimizes the CEM planning loop that searches for good actions:

1. **Budget reduction** — Finds the minimum CEM samples × iterations that maintains accuracy (4.7x fewer forward passes)
2. **Compiled inference** — `torch.compile` with CUDA graphs on the predictor transformer (3.6x per-call speedup)
3. **Cached encoding** — Encodes observation and goal images once per planning step, not per CEM iteration
4. **Pre-allocated buffers** — Eliminates tensor allocation in the autoregressive rollout loop

### Quick Start (on RunPod RTX 4090)

```bash
# Setup
pip install "stable-worldmodel[train,env]"
export STABLEWM_HOME=/workspace/data

# Use the pipeline
python -c "
from harness.pipeline import PlanningPipeline
import numpy as np

pipeline = PlanningPipeline('pusht/lejepa', num_samples=128, n_steps=15)
pipeline.warmup()  # one-time compilation (~30s)

# Plan from images
obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
goal = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
action = pipeline.plan(obs, goal)

print(f'Action: {action}')
print(f'Timing: {pipeline.get_timing_summary()}')
"
```

### Reproducing Results

```bash
# Run the final benchmark
python scripts/final_benchmark.py --policy pusht/lejepa

# Run the standard eval (50 episodes)
python eval.py --config-name=pusht policy=pusht/lejepa solver=cem \
    solver.num_samples=128 solver.n_steps=15 solver.topk=25 eval.num_eval=50

# Run the budget sweep (finds optimal CEM config)
python scripts/sweep_budget.py --policy pusht/lejepa --dry-run
```

### Results

| Metric | Baseline | LeHarness | Improvement |
|--------|----------|-----------|-------------|
| Planning latency | 1,310 ms | **89 ms** | **15x** |
| Control frequency | 0.76 Hz | **11.2 Hz** | **15x** |
| Success rate | 98% | **94%** | -4pp |
| Forward passes/step | 45,000 | **9,600** | 4.7x |
| Peak GPU memory | — | **89 MB** | — |
| Model size | 15M | **15M** | unchanged |

### S1.5: Tactical Planning Layer

S1.5 extends LeHarness with closed-loop feedback for VLM orchestration — the planner reports confidence and drift back to the VLM, enabling replanning when the world model's predictions diverge from reality. This is the S1→S2 feedback loop that no other dual-system architecture (GR00T N1, Helix, OpenHelix) provides.

**On-pod results (TwoRoom, 50 episodes):** Baseline flat CEM 68%, S1.5 with replanning 64%. Confidence replans: 49, drift replans: 9, mean confidence: 0.81. Drift threshold sweet spot: 500-750.

See [docs/S1_5.md](docs/S1_5.md) for the full S1.5 specification, VLM interface design, and on-pod validation results.

### Repository Structure

```
le-harness/
  jepa.py                          # LeWM model (encoder, predictor, rollout, cost)
  module.py                        # Transformer blocks, attention, embedders
  eval.py                          # Standard evaluation with CEM planning
  harness/
    pipeline.py                    # End-to-end planning API — returns PlanResult with confidence
    compiled_inference.py          # torch.compile + CUDA graphs + buffers
    plan_result.py                 # Structured return: action + confidence + terminal_embedding
    dream_tree.py                  # Dream Tree planner (CEM-inside-MCTS)
    language_encoder.py            # CLIP/coord text goal encoding
    s15_loop.py                    # S1.5 control loop: VLM → plan → execute → drift → replan
    drift_detector.py              # Prediction vs reality divergence monitoring
    sim_components.py              # SimVLM + SimMotorPolicy for sim evaluation
    goal_adapter.py                # Unified goal ingestion (image/text/VLM embedding)
    protocols.py                   # VLMProtocol + MotorProtocol interfaces
    projections.py                 # VLM→LeWM projection MLPs (SigLIP, T5, Eagle, etc.)
    subgoal_sequencer.py           # Multi-step goal sequencing
    adaptive_solver.py             # Early stopping CEM wrapper
    value_function.py              # Learned value function (Phase 4)
    value_cost.py                  # Value cost model for solver (Phase 4)
  scripts/
    eval_s15_integration.py        # Zero-mock S1.5 integration eval
    eval_dream_tree.py             # Dream Tree vs flat CEM comparison
    eval_combined.py               # N1+N2+D3 all combinations
    train_vlm_projection.py        # Train any VLM→LeWM projection MLP
    final_benchmark.py             # Reproducible end-to-end benchmark
    sweep_budget.py                # Staged CEM budget sweep
    benchmark_latency.py           # Component-level latency profiling
    patch_icem.py                  # Fix for iCEM action bounds
  tests/                           # 124 unit tests for S1.5 components
  config/                          # Hydra configs for training and eval
  docs/
    S1_5.md                        # S1.5 tactical planning spec + validation results
    STATE.md                       # How the world model works end-to-end
    DREAM_ENGINE.md                # Dream Tree, language conditioning, integration results
    PROJECT.md                     # Full project documentation
    VOLUME.md                      # Network volume layout
    HARNESS.md                     # Technical architecture details
```

See [docs/PROJECT.md](docs/PROJECT.md) for the full project documentation including all phase results, design decisions, and future directions.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
