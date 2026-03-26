#!/usr/bin/env python3
"""
D3: Evaluate Dream Tree vs Flat CEM

Compares DreamTree (CEM-inside-MCTS) against flat CEM on PushT,
measuring success rate, latency, and forward pass count at matched
compute budgets.

Usage (on RunPod):
    export STABLEWM_HOME=/workspace/data
    cd /workspace/le-harness

    # Flat CEM baseline (same as D2 single-horizon)
    python scripts/eval_dream_tree.py --policy pusht/lejepa --mode flat

    # Dream Tree
    python scripts/eval_dream_tree.py --policy pusht/lejepa --mode tree

    # Both modes back-to-back
    python scripts/eval_dream_tree.py --policy pusht/lejepa --mode both
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
from harness.dream_tree import DreamTreePlanner


def load_eval_config():
    from hydra import compose, initialize_config_dir
    config_dir = str(Path("./config/eval").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="pusht")
    return cfg


def setup_eval_env(cfg, args):
    """Set up dataset, world env, action scaler, and episode sampling."""
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

    # World environment
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True, throw_on_missing=False)
    world_cfg["max_episode_steps"] = 2 * args.eval_budget
    world_cfg["num_envs"] = 1
    world = swm.World(**world_cfg, image_shape=(224, 224))

    # Sample episodes
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

    return dataset, process, world, episode_idx, col_name, eval_episodes, eval_start_steps, goal_offset


def run_episodes(planner_fn, dataset, process, world, episode_idx, col_name,
                 eval_episodes, eval_start_steps, goal_offset, args):
    """Run evaluation episodes with a given planner function.

    Args:
        planner_fn: callable(obs_image, goal_image) -> action (action_dim,)
    """
    pipeline = planner_fn.__self__ if hasattr(planner_fn, '__self__') else None
    raw_action_dim = process["action"].scale_.shape[0] if "action" in process else 2

    # Infer action_block from the planner
    if hasattr(planner_fn, '__self__'):
        obj = planner_fn.__self__
        if hasattr(obj, '_action_dim'):
            action_block = obj._action_dim // raw_action_dim
        elif hasattr(obj, 'pipeline'):
            action_block = obj.pipeline._action_dim // raw_action_dim
        else:
            action_block = 5
    else:
        action_block = 5

    successes = []
    planning_times = []

    for ep_i in range(args.num_eval):
        ep_id = int(eval_episodes[ep_i])
        start_step = int(eval_start_steps[ep_i])

        ep_mask = episode_idx == ep_id
        ep_indices = np.where(ep_mask)[0]

        start_row = dataset.get_row_data(int(ep_indices[start_step]))
        goal_step = min(start_step + goal_offset, len(ep_indices) - 1)
        goal_row = dataset.get_row_data(int(ep_indices[goal_step]))

        start_pixels = start_row["pixels"]
        goal_pixels = goal_row["pixels"]

        if isinstance(start_pixels, np.ndarray) and start_pixels.dtype != np.uint8:
            start_pixels = (start_pixels * 255).astype(np.uint8) if start_pixels.max() <= 1.0 else start_pixels.astype(np.uint8)
        if isinstance(goal_pixels, np.ndarray) and goal_pixels.dtype != np.uint8:
            goal_pixels = (goal_pixels * 255).astype(np.uint8) if goal_pixels.max() <= 1.0 else goal_pixels.astype(np.uint8)

        # Reset environment
        world.envs.reset()
        unwrapped_env = world.envs.envs[0].unwrapped
        if "state" in start_row:
            unwrapped_env._set_state(state=start_row["state"])
        if "state" in goal_row:
            unwrapped_env._set_goal_state(goal_state=goal_row["state"])

        episode_success = False
        obs_image = start_pixels

        env_step = 0
        while env_step < args.eval_budget:
            t0 = time.perf_counter()
            raw_action = planner_fn(obs_image, goal_pixels)
            planning_times.append((time.perf_counter() - t0) * 1000)

            # Reshape and execute sub-actions (frameskip)
            sub_actions = raw_action.reshape(action_block, raw_action_dim)

            for sub_action in sub_actions:
                if env_step >= args.eval_budget:
                    break

                if "action" in process:
                    sub_action = process["action"].inverse_transform(
                        sub_action.reshape(1, -1)
                    ).squeeze()

                try:
                    obs_dict, reward, terminated, truncated, info = world.envs.step(
                        np.array([sub_action])
                    )
                    env_step += 1
                    obs_image = world.envs.render()[0]

                    if isinstance(terminated, (list, np.ndarray)):
                        terminated = bool(terminated[0])
                    if isinstance(info, (list, tuple)):
                        step_info = info[0] if info else {}
                    elif isinstance(info, dict):
                        step_info = info
                    else:
                        step_info = {}

                    if terminated or step_info.get("is_success", False):
                        episode_success = True
                        break
                except Exception as e:
                    print(f"  Episode {ep_i} step {env_step}: env error: {e}")
                    break

            if episode_success:
                break

        successes.append(episode_success)

        if (ep_i + 1) % 10 == 0 or ep_i == 0:
            sr = np.mean(successes) * 100
            mean_ms = np.mean(planning_times[-args.eval_budget:]) if planning_times else 0
            print(f"  Episode {ep_i+1}/{args.num_eval}: "
                  f"success_rate={sr:.1f}%, mean_planning={mean_ms:.0f}ms")

    return successes, planning_times


def run_eval(args):
    print(f"\n{'='*60}")
    print(f"D3: Dream Tree Evaluation")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Eval budget: {args.eval_budget} steps")
    print(f"Num eval: {args.num_eval} episodes")
    print(f"Policy: {args.policy}")

    cfg = load_eval_config()

    # Build pipeline (shared by both planners)
    print("\nLoading and compiling model...")
    pipeline = PlanningPipeline(
        policy_name=args.policy,
        num_samples=128,
        n_steps=15,
        horizon=5,
        topk=25,
    )
    pipeline.warmup()

    # Set up eval environment
    (dataset, process, world, episode_idx, col_name,
     eval_episodes, eval_start_steps, goal_offset) = setup_eval_env(cfg, args)

    results = {}

    # --- Flat CEM ---
    if args.mode in ("flat", "both"):
        print(f"\n--- Flat CEM (baseline) ---")
        successes, planning_times = run_episodes(
            pipeline.plan, dataset, process, world, episode_idx, col_name,
            eval_episodes, eval_start_steps, goal_offset, args,
        )
        sr = np.mean(successes) * 100
        mean_ms = np.mean(planning_times) if planning_times else 0
        results["flat"] = {
            "success_rate": float(sr),
            "num_successes": int(sum(successes)),
            "mean_planning_ms": float(mean_ms),
            "effective_hz": 1000 / mean_ms if mean_ms > 0 else 0,
        }
        print(f"\nFlat CEM: {sr:.1f}% ({int(sum(successes))}/{args.num_eval}), "
              f"{mean_ms:.0f}ms/step")

    # --- Dream Tree ---
    if args.mode in ("tree", "both"):
        print(f"\n--- Dream Tree (roots={args.num_roots}, depth={args.max_depth}) ---")

        # Warmup the return_terminal_emb path
        print("Warming up tree planner (first call triggers recompilation)...")
        tree_planner = DreamTreePlanner(
            pipeline,
            num_roots=args.num_roots,
            max_depth=args.max_depth,
        )
        dummy_obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_goal = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tree_planner.plan(dummy_obs, dummy_goal)
        tree_planner.reset_timing()
        print("Tree planner ready.")

        successes, planning_times = run_episodes(
            tree_planner.plan, dataset, process, world, episode_idx, col_name,
            eval_episodes, eval_start_steps, goal_offset, args,
        )
        sr = np.mean(successes) * 100
        mean_ms = np.mean(planning_times) if planning_times else 0
        ts = tree_planner.get_timing_summary()
        results["tree"] = {
            "success_rate": float(sr),
            "num_successes": int(sum(successes)),
            "mean_planning_ms": float(mean_ms),
            "effective_hz": 1000 / mean_ms if mean_ms > 0 else 0,
            "tree_timing": ts,
            "config": {
                "num_roots": args.num_roots,
                "max_depth": args.max_depth,
            },
        }
        print(f"\nDream Tree: {sr:.1f}% ({int(sum(successes))}/{args.num_eval}), "
              f"{mean_ms:.0f}ms/step")
        if ts:
            print(f"  Root: {ts.get('root_ms', {}).get('mean', 0):.0f}ms, "
                  f"Expansion: {ts.get('expansion_ms', {}).get('mean', 0):.0f}ms")
            print(f"  CEM calls/step: {ts.get('total_cem_calls', 0):.0f}, "
                  f"Nodes: {ts.get('total_nodes', 0):.0f}")

    # --- Comparison ---
    if args.mode == "both" and "flat" in results and "tree" in results:
        print(f"\n{'='*60}")
        print(f"D3 Comparison: Dream Tree vs Flat CEM")
        print(f"{'='*60}")
        flat = results["flat"]
        tree = results["tree"]
        print(f"  {'Metric':<25} {'Flat CEM':>12} {'Dream Tree':>12}")
        print(f"  {'-'*49}")
        print(f"  {'Success rate':<25} {flat['success_rate']:>11.1f}% {tree['success_rate']:>11.1f}%")
        print(f"  {'Planning latency':<25} {flat['mean_planning_ms']:>10.0f}ms {tree['mean_planning_ms']:>10.0f}ms")
        print(f"  {'Control frequency':<25} {flat['effective_hz']:>10.1f}Hz {tree['effective_hz']:>10.1f}Hz")

    # Save results
    results["eval_budget"] = args.eval_budget
    results["num_eval"] = args.num_eval
    results["seed"] = args.seed

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"d3_{args.mode}_h{args.eval_budget}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="D3: Dream Tree Evaluation")
    parser.add_argument("--policy", default="pusht/lejepa")
    parser.add_argument("--mode", choices=["flat", "tree", "both"], default="both")
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="/workspace/data/results")

    # Tree config
    parser.add_argument("--num-roots", type=int, default=4,
                        help="Number of diverse root CEM candidates to expand")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="Tree depth (1=root only, 2=one expansion, 3=two expansions)")

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
