#!/usr/bin/env python3
"""
D4: Train Dream Scorer — Value Ensemble for Multi-Signal Scoring

Collects training data from three sources:
1. Expert trajectories: high-progress (z_t, z_goal) pairs from dataset
2. Random rollouts: model predictions with random actions (mixed progress)
3. Synthetic failures: low-budget CEM (2 samples, 1 iter) for failure cases

Trains a ValueEnsemble, applies WARM weight averaging, saves checkpoint.

Usage (on RunPod):
    export STABLEWM_HOME=/workspace/data
    cd /workspace/le-harness

    python scripts/train_dream_scorer.py --policy pusht/lejepa
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch.nn.functional as F

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn import preprocessing

from harness.pipeline import PlanningPipeline
from harness.value_function import ValueEnsemble, train_ensemble
from harness.dream_scorer import DreamScorer, warm_average


def load_eval_config():
    from hydra import compose, initialize_config_dir
    config_dir = str(Path("./config/eval").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="pusht")
    return cfg


def collect_expert_data(pipeline, dataset, cfg, n_pairs: int = 5000, device="cuda"):
    """Collect (z_t, z_goal, progress) from expert trajectories.

    For each sampled (start, goal) pair from the dataset:
    - Encode both images → embeddings
    - progress = 1.0 for goal states, scaled by step distance otherwise
    """
    print(f"\n--- Collecting {n_pairs} expert pairs ---")

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    unique_eps = np.unique(episode_idx)

    goal_offset = cfg.eval.goal_offset_steps  # 25

    z_t_list = []
    z_goal_list = []
    progress_list = []

    rng = np.random.default_rng(42)
    samples_per_ep = max(1, n_pairs // len(unique_eps))

    with torch.inference_mode():
        for ep_id in unique_eps[:min(200, len(unique_eps))]:
            ep_mask = episode_idx == ep_id
            ep_indices = np.where(ep_mask)[0]
            ep_len = len(ep_indices)

            if ep_len < goal_offset + 2:
                continue

            # Sample start-goal pairs within this episode
            for _ in range(samples_per_ep):
                start_step = rng.integers(0, ep_len - goal_offset)
                goal_step = start_step + goal_offset

                start_row = dataset.get_row_data(int(ep_indices[start_step]))
                goal_row = dataset.get_row_data(int(ep_indices[goal_step]))

                start_pixels = start_row["pixels"]
                goal_pixels = goal_row["pixels"]

                if start_pixels.dtype != np.uint8:
                    start_pixels = (start_pixels * 255).astype(np.uint8) if start_pixels.max() <= 1.0 else start_pixels.astype(np.uint8)
                if goal_pixels.dtype != np.uint8:
                    goal_pixels = (goal_pixels * 255).astype(np.uint8) if goal_pixels.max() <= 1.0 else goal_pixels.astype(np.uint8)

                z_start = pipeline.encode(pipeline.preprocess(start_pixels))
                z_goal = pipeline.encode(pipeline.preprocess(goal_pixels))

                z_t_list.append(z_start.squeeze(0).squeeze(0))  # (D,)
                z_goal_list.append(z_goal.squeeze(0).squeeze(0))

                # Progress: 0 at start, 1 at goal
                # For intermediate steps, sample some at different offsets
                progress_list.append(0.0)  # start → not at goal yet

                # Also add the goal state (progress = 1.0)
                z_t_list.append(z_goal.squeeze(0).squeeze(0))
                z_goal_list.append(z_goal.squeeze(0).squeeze(0))
                progress_list.append(1.0)

                # Add intermediate steps
                for frac in [0.25, 0.5, 0.75]:
                    mid_step = int(start_step + frac * goal_offset)
                    mid_row = dataset.get_row_data(int(ep_indices[mid_step]))
                    mid_pixels = mid_row["pixels"]
                    if mid_pixels.dtype != np.uint8:
                        mid_pixels = (mid_pixels * 255).astype(np.uint8) if mid_pixels.max() <= 1.0 else mid_pixels.astype(np.uint8)
                    z_mid = pipeline.encode(pipeline.preprocess(mid_pixels))
                    z_t_list.append(z_mid.squeeze(0).squeeze(0))
                    z_goal_list.append(z_goal.squeeze(0).squeeze(0))
                    progress_list.append(frac)

            if len(z_t_list) >= n_pairs:
                break

    z_t = torch.stack(z_t_list[:n_pairs])
    z_goal = torch.stack(z_goal_list[:n_pairs])
    progress = torch.tensor(progress_list[:n_pairs])

    print(f"  Collected {len(z_t)} expert pairs")
    print(f"  Progress distribution: min={progress.min():.2f}, max={progress.max():.2f}, "
          f"mean={progress.mean():.2f}")
    return z_t, z_goal, progress


def collect_random_rollout_data(pipeline, dataset, cfg, n_pairs: int = 3000, device="cuda"):
    """Collect (z_t, z_goal, progress) from CEM planning trajectories.

    Uses pipeline._score_state which calls _evaluate_candidates with S=128
    (CUDA graph compatible). Captures the intermediate CEM state to extract
    predicted embeddings from planning attempts.
    """
    print(f"\n--- Collecting {n_pairs} rollout pairs ---")

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    unique_eps = np.unique(episode_idx)
    goal_offset = cfg.eval.goal_offset_steps

    z_t_list = []
    z_goal_list = []
    progress_list = []

    rng = np.random.default_rng(123)

    with torch.inference_mode():
        for ep_id in unique_eps[:min(200, len(unique_eps))]:
            ep_mask = episode_idx == ep_id
            ep_indices = np.where(ep_mask)[0]
            ep_len = len(ep_indices)

            if ep_len < goal_offset + 2:
                continue

            start_step = rng.integers(0, ep_len - goal_offset)
            goal_step = start_step + goal_offset

            start_row = dataset.get_row_data(int(ep_indices[start_step]))
            goal_row = dataset.get_row_data(int(ep_indices[goal_step]))

            start_pixels = start_row["pixels"]
            goal_pixels = goal_row["pixels"]
            if start_pixels.dtype != np.uint8:
                start_pixels = (start_pixels * 255).astype(np.uint8) if start_pixels.max() <= 1.0 else start_pixels.astype(np.uint8)
            if goal_pixels.dtype != np.uint8:
                goal_pixels = (goal_pixels * 255).astype(np.uint8) if goal_pixels.max() <= 1.0 else goal_pixels.astype(np.uint8)

            obs_emb = pipeline.encode(pipeline.preprocess(start_pixels))
            goal_emb = pipeline.encode(pipeline.preprocess(goal_pixels))

            # Run CEM planning — this uses compiled path (S=128)
            # and captures terminal embeddings at different CEM convergence levels
            action, terminal_emb = pipeline._cem_plan(
                obs_emb, goal_emb, return_terminal_emb=True
            )

            # Terminal embedding from optimized CEM (near-optimal action)
            goal_flat = goal_emb.squeeze(0).squeeze(0).cpu()
            obs_flat = obs_emb.squeeze(0).squeeze(0).cpu()
            term_flat = terminal_emb.squeeze(0).squeeze(0).cpu()

            mse_start = ((obs_flat - goal_flat) ** 2).sum().item()
            mse_term = ((term_flat - goal_flat) ** 2).sum().item()

            # CEM-optimized terminal: should have decent progress
            prog_term = max(0, min(1, (mse_start - mse_term) / (mse_start + 1e-8)))
            z_t_list.append(term_flat)
            z_goal_list.append(goal_flat)
            progress_list.append(prog_term)

            # Also sample the observation itself (progress = 0)
            z_t_list.append(obs_flat)
            z_goal_list.append(goal_flat)
            progress_list.append(0.0)

            # Random starting points (encode different timesteps for diversity)
            for offset in [5, 10, 15, 20]:
                mid_step = min(start_step + offset, ep_len - 1)
                mid_row = dataset.get_row_data(int(ep_indices[mid_step]))
                mid_pix = mid_row["pixels"]
                if mid_pix.dtype != np.uint8:
                    mid_pix = (mid_pix * 255).astype(np.uint8) if mid_pix.max() <= 1.0 else mid_pix.astype(np.uint8)
                z_mid = pipeline.encode(pipeline.preprocess(mid_pix)).squeeze(0).squeeze(0).cpu()
                mse_mid = ((z_mid - goal_flat) ** 2).sum().item()
                prog_mid = max(0, min(1, (mse_start - mse_mid) / (mse_start + 1e-8)))
                z_t_list.append(z_mid)
                z_goal_list.append(goal_flat)
                progress_list.append(prog_mid)

            if len(z_t_list) >= n_pairs:
                break

    z_t = torch.stack(z_t_list[:n_pairs])
    z_goal = torch.stack(z_goal_list[:n_pairs])
    progress = torch.tensor(progress_list[:n_pairs])

    print(f"  Collected {len(z_t)} rollout pairs")
    print(f"  Progress distribution: min={progress.min():.2f}, max={progress.max():.2f}, "
          f"mean={progress.mean():.2f}")
    return z_t, z_goal, progress


def main():
    parser = argparse.ArgumentParser(description="D4: Train Dream Scorer")
    parser.add_argument("--policy", default="pusht/lejepa")
    parser.add_argument("--n-expert", type=int, default=5000)
    parser.add_argument("--n-rollout", type=int, default=3000)
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--apply-warm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default="/workspace/data/results")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"D4: Training Dream Scorer")
    print(f"{'='*60}")

    cfg = load_eval_config()

    # Build pipeline for encoding
    print("\nLoading pipeline...")
    pipeline = PlanningPipeline(
        policy_name=args.policy,
        num_samples=128,
        n_steps=15,
        horizon=5,
        topk=25,
    )
    pipeline.warmup()

    # Warmup terminal_emb path
    with torch.inference_mode():
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        goal = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        obs_emb = pipeline.encode(pipeline.preprocess(obs))
        goal_emb = pipeline.encode(pipeline.preprocess(goal))
        for _ in range(2):
            pipeline._cem_plan(obs_emb, goal_emb, return_terminal_emb=True)

    # Load dataset
    cache_dir = Path(swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )

    # --- Collect training data ---
    t0 = time.perf_counter()

    z_t_expert, z_goal_expert, prog_expert = collect_expert_data(
        pipeline, dataset, cfg, n_pairs=args.n_expert
    )
    z_t_rollout, z_goal_rollout, prog_rollout = collect_random_rollout_data(
        pipeline, dataset, cfg, n_pairs=args.n_rollout
    )

    # Combine (ensure all on CPU for training)
    z_t_all = torch.cat([z_t_expert.cpu(), z_t_rollout.cpu()], dim=0)
    z_goal_all = torch.cat([z_goal_expert.cpu(), z_goal_rollout.cpu()], dim=0)
    progress_all = torch.cat([prog_expert.cpu(), prog_rollout.cpu()], dim=0)

    t_collect = (time.perf_counter() - t0)
    print(f"\nTotal training data: {len(z_t_all)} pairs ({t_collect:.0f}s)")
    print(f"  Expert: {len(z_t_expert)}, Rollout: {len(z_t_rollout)}")

    # Save training data
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "d4_training_data.pt"
    torch.save({
        "z_t": z_t_all, "z_goal": z_goal_all, "progress": progress_all,
        "n_expert": len(z_t_expert), "n_rollout": len(z_t_rollout),
    }, data_path)
    print(f"Training data saved to {data_path}")

    # --- Train ensemble ---
    print(f"\n--- Training ensemble ({args.n_members} members, {args.n_epochs} epochs) ---")
    ensemble = ValueEnsemble(
        n_members=args.n_members,
        embed_dim=192,
        hidden_dim=args.hidden_dim,
    )

    t0 = time.perf_counter()
    history = train_ensemble(
        ensemble, z_t_all, z_goal_all, progress_all,
        n_epochs=args.n_epochs,
        batch_size=512,
        lr=args.lr,
        device="cuda",
        verbose=True,
    )
    t_train = (time.perf_counter() - t0)
    print(f"\nTraining complete in {t_train:.0f}s")

    # --- Apply WARM ---
    if args.apply_warm:
        print("\nApplying WARM weight averaging...")
        warm_ens = warm_average(ensemble)
        # Verify WARM doesn't degrade too much
        warm_ens = warm_ens.to("cuda").eval()
        with torch.no_grad():
            z_t_test = z_t_all[:500].to("cuda")
            z_goal_test = z_goal_all[:500].to("cuda")
            prog_test = progress_all[:500].to("cuda")

            pre_warm_pred = ensemble.to("cuda")(z_t_test, z_goal_test)
            post_warm_pred = warm_ens(z_t_test, z_goal_test)

            pre_loss = F.mse_loss(pre_warm_pred, prog_test).item()
            post_loss = F.mse_loss(post_warm_pred, prog_test).item()

        print(f"  Pre-WARM val loss:  {pre_loss:.4f}")
        print(f"  Post-WARM val loss: {post_loss:.4f}")
        ensemble = warm_ens

    # --- Save scorer ---
    scorer = DreamScorer(ensemble=ensemble)
    scorer_path = out_dir / "d4_dream_scorer.pt"
    scorer.save(str(scorer_path))
    print(f"\nScorer saved to {scorer_path}")

    # Save training metadata
    meta = {
        "n_expert": len(z_t_expert),
        "n_rollout": len(z_t_rollout),
        "n_total": len(z_t_all),
        "n_members": args.n_members,
        "n_epochs": args.n_epochs,
        "hidden_dim": args.hidden_dim,
        "apply_warm": args.apply_warm,
        "collect_time_s": t_collect,
        "train_time_s": t_train,
        "param_count": ensemble.param_count(),
    }
    meta_path = out_dir / "d4_training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    print(f"\n{'='*60}")
    print(f"D4: Training Complete")
    print(f"{'='*60}")
    print(f"  Ensemble: {args.n_members} members × {ensemble.param_count() // args.n_members} params")
    print(f"  WARM: {'applied' if args.apply_warm else 'skipped'}")
    print(f"  Total time: {t_collect + t_train:.0f}s")


if __name__ == "__main__":
    main()
