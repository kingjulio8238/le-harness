#!/usr/bin/env python3
"""
D2: Evaluate Dream Chaining vs Single-Horizon Planning

Compares standard receding-horizon CEM (single) vs chained CEM (chained)
on PushT with extended eval budgets. Optionally measures latent drift
between predicted chain endpoints and re-encoded actual observations.

Usage (on RunPod):
    export STABLEWM_HOME=/workspace/data
    cd /workspace/le-harness

    # Single-horizon baseline (100-step episodes)
    python scripts/eval_dream_chaining.py --policy pusht/lejepa --mode single --eval-budget 100

    # Chained 3x5 (100-step episodes)
    python scripts/eval_dream_chaining.py --policy pusht/lejepa --mode chained --num-chains 3 --eval-budget 100

    # Chained 5x5 with drift measurement
    python scripts/eval_dream_chaining.py --policy pusht/lejepa --mode chained --num-chains 5 --eval-budget 200 --measure-drift
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from harness.pipeline import PlanningPipeline
from harness.dream_chainer import DreamChainer


def load_eval_config():
    """Load the PushT eval config manually (without Hydra decorator)."""
    from hydra import compose, initialize_config_dir

    config_dir = str(Path("./config/eval").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="pusht")
    return cfg


def run_eval(args):
    """Run the evaluation."""
    print(f"\n{'='*60}")
    print(f"D2: Dream Chaining Evaluation")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Num chains: {args.num_chains}")
    print(f"Eval budget: {args.eval_budget} steps")
    print(f"Num eval: {args.num_eval} episodes")
    print(f"Measure drift: {args.measure_drift}")
    print(f"Policy: {args.policy}")

    # Load config for dataset/environment setup
    cfg = load_eval_config()

    # Build pipeline
    print("\nLoading and compiling model...")
    pipeline = PlanningPipeline(
        policy_name=args.policy,
        num_samples=128,
        n_steps=15,
        horizon=5,
        topk=25,
    )
    pipeline.warmup()

    # Build chainer (only used in chained mode)
    chainer = None
    if args.mode == "chained":
        chainer = DreamChainer(pipeline, num_chains=args.num_chains)
        print(f"DreamChainer initialized: {args.num_chains} chains × horizon 5")

    # Image transforms
    img_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])

    # Load dataset for episode sampling
    cache_dir = Path(swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )

    # Action scaler
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # Create world environment
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True, throw_on_missing=False)
    world_cfg["max_episode_steps"] = 2 * args.eval_budget
    world_cfg["num_envs"] = 1  # run one episode at a time for chaining control
    world = swm.World(**world_cfg, image_shape=(224, 224))

    # Sample episodes from dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    unique_eps = np.unique(episode_idx)

    episode_len = []
    for ep_id in unique_eps:
        episode_len.append(int(np.max(step_idx[episode_idx == ep_id]) + 1))
    episode_len = np.array(episode_len)

    goal_offset = cfg.eval.goal_offset_steps
    max_start_idx = episode_len - goal_offset - 1
    max_start_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(unique_eps)}
    max_start_per_row = np.array(
        [max_start_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(len(valid_indices) - 1, size=args.num_eval, replace=False)
    sample_indices = np.sort(valid_indices[sample_indices])

    eval_episodes = dataset.get_row_data(sample_indices)[col_name]
    eval_start_steps = dataset.get_row_data(sample_indices)["step_idx"]

    # Run evaluation
    print(f"\nRunning {args.num_eval} episodes...")
    successes = []
    planning_times = []
    drift_records = []

    for ep_i in range(args.num_eval):
        ep_id = int(eval_episodes[ep_i])
        start_step = int(eval_start_steps[ep_i])

        # Get start state and goal state from dataset
        ep_mask = episode_idx == ep_id
        ep_indices = np.where(ep_mask)[0]

        start_row = dataset.get_row_data(int(ep_indices[start_step]))
        goal_step = min(start_step + goal_offset, len(ep_indices) - 1)
        goal_row = dataset.get_row_data(int(ep_indices[goal_step]))

        # Get images
        start_pixels = start_row["pixels"]
        goal_pixels = goal_row["pixels"]

        if isinstance(start_pixels, np.ndarray) and start_pixels.dtype != np.uint8:
            start_pixels = (start_pixels * 255).astype(np.uint8) if start_pixels.max() <= 1.0 else start_pixels.astype(np.uint8)
        if isinstance(goal_pixels, np.ndarray) and goal_pixels.dtype != np.uint8:
            goal_pixels = (goal_pixels * 255).astype(np.uint8) if goal_pixels.max() <= 1.0 else goal_pixels.astype(np.uint8)

        # Reset environment, then set state via unwrapped env
        world.envs.reset()
        unwrapped_env = world.envs.envs[0].unwrapped
        if "state" in start_row:
            unwrapped_env._set_state(state=start_row["state"])
        if "state" in goal_row:
            unwrapped_env._set_goal_state(goal_state=goal_row["state"])

        # Run episode
        episode_success = False
        obs_image = start_pixels
        raw_action_dim = process["action"].scale_.shape[0] if "action" in process else 2
        action_block = pipeline._action_dim // raw_action_dim

        env_step = 0
        while env_step < args.eval_budget:
            t0 = time.perf_counter()

            predicted_terminal_emb = None
            if args.mode == "single":
                # Standard receding-horizon CEM
                raw_action = pipeline.plan(obs_image, goal_pixels)
            else:
                # Dream chaining
                if args.measure_drift:
                    raw_action, predicted_terminal_emb = chainer.plan(
                        obs_image, goal_pixels, measure_drift=True
                    )
                else:
                    raw_action = chainer.plan(obs_image, goal_pixels)

            planning_times.append((time.perf_counter() - t0) * 1000)

            # Reshape (action_dim,) → (action_block, raw_action_dim) for frameskip
            sub_actions = raw_action.reshape(action_block, raw_action_dim)

            # Execute each sub-action
            for sub_action in sub_actions:
                if env_step >= args.eval_budget:
                    break

                # Inverse-transform action if needed
                if "action" in process:
                    sub_action = process["action"].inverse_transform(
                        sub_action.reshape(1, -1)
                    ).squeeze()

                # Step environment
                try:
                    obs_dict, reward, terminated, truncated, info = world.envs.step(
                        np.array([sub_action])
                    )
                    env_step += 1
                    # Get next observation image via render
                    obs_image = world.envs.render()[0]

                    if isinstance(terminated, (list, np.ndarray)):
                        terminated = bool(terminated[0])
                    if isinstance(info, (list, tuple)):
                        step_info = info[0] if info else {}
                    elif isinstance(info, dict):
                        step_info = info
                    else:
                        step_info = {}

                    # PushT: terminated=True means goal reached (from eval_state).
                    # Other envs may use info["is_success"] instead.
                    if terminated or step_info.get("is_success", False):
                        episode_success = True
                        break
                except Exception as e:
                    print(f"  Episode {ep_i} step {env_step}: env error: {e}")
                    break

            # Measure drift: compare predicted terminal embedding to actual
            if args.measure_drift and predicted_terminal_emb is not None and not episode_success:
                with torch.inference_mode():
                    actual_tensor = pipeline.preprocess(obs_image)
                    actual_emb = pipeline.encode(actual_tensor)  # (1, 1, D)
                    mse = torch.nn.functional.mse_loss(
                        predicted_terminal_emb, actual_emb
                    ).item()
                    cos_sim = torch.nn.functional.cosine_similarity(
                        predicted_terminal_emb.flatten(),
                        actual_emb.flatten(), dim=0
                    ).item()
                    drift_records.append({
                        "episode": ep_i, "env_step": env_step,
                        "mse": mse, "cosine_sim": cos_sim,
                    })

            if episode_success:
                break

        successes.append(episode_success)

        if (ep_i + 1) % 10 == 0 or ep_i == 0:
            sr_so_far = np.mean(successes) * 100
            mean_ms = np.mean(planning_times[-args.eval_budget:]) if planning_times else 0
            print(f"  Episode {ep_i+1}/{args.num_eval}: "
                  f"success_rate={sr_so_far:.1f}%, "
                  f"mean_planning={mean_ms:.0f}ms")

    # Compute final metrics
    success_rate = np.mean(successes) * 100
    mean_planning_ms = np.mean(planning_times) if planning_times else 0
    effective_hz = 1000 / mean_planning_ms if mean_planning_ms > 0 else 0

    results = {
        "mode": args.mode,
        "num_chains": args.num_chains if args.mode == "chained" else 1,
        "eval_budget": args.eval_budget,
        "num_eval": args.num_eval,
        "success_rate": float(success_rate),
        "mean_planning_ms": float(mean_planning_ms),
        "effective_hz": float(effective_hz),
        "num_successes": int(sum(successes)),
        "seed": args.seed,
    }

    # Add chainer timing if available
    if chainer is not None:
        results["chainer_timing"] = chainer.get_timing_summary()

    # Add drift metrics
    if drift_records:
        mses = [d["mse"] for d in drift_records]
        cosines = [d["cosine_sim"] for d in drift_records]
        results["drift"] = {
            "num_samples": len(drift_records),
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "mse_median": float(np.median(mses)),
            "mse_max": float(np.max(mses)),
            "cosine_sim_mean": float(np.mean(cosines)),
            "cosine_sim_std": float(np.std(cosines)),
            "cosine_sim_min": float(np.min(cosines)),
        }

    # Print summary
    print(f"\n{'='*60}")
    print(f"D2 Results: {args.mode}" + (f" ({args.num_chains} chains)" if args.mode == "chained" else ""))
    print(f"{'='*60}")
    print(f"Success rate: {success_rate:.1f}% ({int(sum(successes))}/{args.num_eval})")
    print(f"Planning latency: {mean_planning_ms:.0f} ms/step ({effective_hz:.1f} Hz)")
    print(f"Eval budget: {args.eval_budget} steps/episode")

    if chainer is not None:
        ts = chainer.get_timing_summary()
        if ts:
            print(f"Chain timing: {ts.get('chain_ms', {}).get('mean', 0):.0f} ms total "
                  f"({ts.get('num_chains', 0)} chains)")
            if "per_chain_mean_ms" in ts:
                for i, ms in enumerate(ts["per_chain_mean_ms"]):
                    print(f"  Chain {i+1}: {ms:.0f} ms")

    # Print drift metrics
    if drift_records:
        d = results["drift"]
        print(f"\nDrift (predicted vs actual embedding after chain 1):")
        print(f"  MSE:        {d['mse_mean']:.6f} +/- {d['mse_std']:.6f} "
              f"(median: {d['mse_median']:.6f}, max: {d['mse_max']:.6f})")
        print(f"  Cosine sim: {d['cosine_sim_mean']:.4f} +/- {d['cosine_sim_std']:.4f} "
              f"(min: {d['cosine_sim_min']:.4f})")
        print(f"  Samples:    {d['num_samples']}")

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.mode}"
    if args.mode == "chained":
        suffix += f"_{args.num_chains}x5"
    suffix += f"_h{args.eval_budget}"
    if args.measure_drift:
        suffix += "_drift"

    out_path = out_dir / f"d2_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="D2: Dream Chaining Evaluation")
    parser.add_argument("--policy", default="pusht/lejepa")
    parser.add_argument("--mode", choices=["single", "chained"], default="single")
    parser.add_argument("--num-chains", type=int, default=3)
    parser.add_argument("--eval-budget", type=int, default=100)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--measure-drift", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="/workspace/data/results")
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()
