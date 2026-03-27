#!/usr/bin/env python3
"""
Integration eval: N1 (batched CEM) + N2 (language) + D3 (dream tree) together.

Tests all combinations on the same TwoRoom episodes:
  1. image + flat CEM (baseline)
  2. image + batched tree
  3. coord text + flat CEM
  4. coord text + batched tree
  5. clip text + flat CEM
  6. clip text + batched tree

Gate: text + batched tree achieves ≥80% of image + flat CEM.

Usage:
    export STABLEWM_HOME=/workspace/data
    export MUJOCO_GL=egl
    python scripts/eval_combined.py --policy tworoom/lewm --config-name tworoom
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import stable_worldmodel as swm
import h5py
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing

from harness.pipeline import PlanningPipeline
from harness.dream_tree import DreamTreePlanner

import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_dream_tree import load_eval_config, setup_eval_env


def get_region_description(x_norm, y_norm):
    if x_norm < 0.25: h = "far left"
    elif x_norm < 0.45: h = "left"
    elif x_norm < 0.55: h = "center"
    elif x_norm < 0.75: h = "right"
    else: h = "far right"
    if y_norm < 0.25: v = "top"
    elif y_norm < 0.45: v = "upper"
    elif y_norm < 0.55: v = "middle"
    elif y_norm < 0.75: v = "lower"
    else: v = "bottom"
    return f"go to the {v} {h} area"


def run_episodes_combo(planner_fn, goal_mode, pipeline, dataset, process, world,
                       episode_idx, col_name, eval_episodes, eval_start_steps,
                       goal_offset, pos_target_all, args, cfg):
    """Run eval episodes with a given planner + goal mode."""
    raw_action_dim = process["action"].scale_.shape[0]

    # Determine action_block
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

        # Goal position for text modes
        goal_global_idx = int(ep_indices[goal_step])
        goal_pos = pos_target_all[goal_global_idx]
        x_norm, y_norm = goal_pos[0] / 224.0, goal_pos[1] / 224.0

        # Set goal based on mode
        if goal_mode == "image":
            # For tree planner: goal is set inside plan() call
            pass
        elif goal_mode == "coord":
            goal_text = f"navigate to ({x_norm:.2f}, {y_norm:.2f})"
            pipeline.set_goal_text(goal_text)
        elif goal_mode == "clip":
            goal_text = get_region_description(x_norm, y_norm)
            pipeline.set_goal_text(goal_text)

        # Reset environment
        world.envs.reset()
        unwrapped_env = world.envs.envs[0].unwrapped

        callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True) if cfg.eval.get("callables") else []
        for spec in callables:
            method_name = spec["method"]
            if not hasattr(unwrapped_env, method_name):
                continue
            method = getattr(unwrapped_env, method_name)
            prepared_args = {}
            for arg_name, arg_data in spec.get("args", {}).items():
                value_key = arg_data.get("value", None)
                if value_key is None:
                    continue
                if value_key.startswith("goal_"):
                    col = value_key[5:]
                    if col in goal_row:
                        prepared_args[arg_name] = goal_row[col]
                else:
                    if value_key in start_row:
                        prepared_args[arg_name] = start_row[value_key]
            if prepared_args:
                method(**prepared_args)

        episode_success = False
        obs_image = start_pixels

        env_step = 0
        while env_step < args.eval_budget:
            t0 = time.perf_counter()

            if goal_mode == "image":
                raw_action = planner_fn(obs_image, goal_pixels)
            else:
                # Text goal already set via set_goal_text above
                # For tree planner: need to handle differently
                if isinstance(planner_fn.__self__, DreamTreePlanner):
                    # Tree planner calls pipeline internally, goal_emb already cached
                    raw_action = planner_fn(obs_image, goal_pixels)
                    # ^ tree.plan() re-encodes goal_pixels, but we want text goal
                    # Need to use pipeline's cached _goal_emb instead
                else:
                    raw_action = pipeline.plan(obs_image, record_timing=True)

            planning_times.append((time.perf_counter() - t0) * 1000)

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
            mean_ms = np.mean(planning_times[-args.eval_budget:])
            print(f"  Episode {ep_i+1}/{args.num_eval}: sr={sr:.1f}%, latency={mean_ms:.0f}ms")

    return successes, planning_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="tworoom/lewm")
    parser.add_argument("--config-name", default="tworoom")
    parser.add_argument("--projection-path", default="/workspace/data/language_projection_v4.pt")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="/workspace/data/results")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Combined Integration Eval: N1 + N2 + D3")
    print(f"{'='*60}")

    cfg = load_eval_config(args.config_name)

    # Build pipeline with default compile mode (required for batching)
    print("\nLoading pipeline (compile_mode=default for batching)...")
    pipeline = PlanningPipeline(
        policy_name=args.policy,
        num_samples=128,
        n_steps=15,
        horizon=5,
        topk=25,
        compile_mode="default",
    )
    pipeline.warmup()

    # Set up eval
    (dataset, process, world, episode_idx, col_name,
     eval_episodes, eval_start_steps, goal_offset) = setup_eval_env(cfg, args)

    # Load positions for text goals
    cache_dir = Path(swm.data.utils.get_cache_dir())
    h5_path = cache_dir / f"{cfg.eval.dataset_name}.h5"
    with h5py.File(h5_path, "r") as f:
        pos_target_all = f["pos_target"][:]

    results = {}

    # --- Configs to test ---
    configs = [
        ("image_flat",        "image", False),
        ("image_tree_batch",  "image", True),
        ("coord_flat",        "coord", False),
        ("coord_tree_batch",  "coord", True),
        ("clip_flat",         "clip",  False),
        ("clip_tree_batch",   "clip",  True),
    ]

    for config_name, goal_mode, use_tree in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config_name} (goal={goal_mode}, tree={'batched' if use_tree else 'no'})")
        print(f"{'='*60}")

        # Load language encoder if needed
        if goal_mode in ("coord", "clip"):
            pipeline.load_language_encoder(args.projection_path, mode=goal_mode)

        if use_tree:
            tree = DreamTreePlanner(
                pipeline,
                num_roots=4,
                max_depth=2,
                cheap_depth=False,
                batched=True,
                # cem_steps=7 is the default for batched=True
            )
            # Warmup tree
            dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            tree.plan(dummy, dummy)
            tree.reset_timing()
            planner_fn = tree.plan
        else:
            planner_fn = pipeline.plan

        # For text + tree: we need to handle goal setting specially
        # The tree's plan() re-encodes goal pixels, overwriting our text goal_emb
        # Fix: for text+tree, pre-set the goal and modify tree to use cached emb
        if goal_mode != "image" and use_tree:
            # Monkey-patch: make tree.plan use pre-set goal_emb
            original_plan = tree.plan

            def make_text_tree_plan(tree_obj, pipeline_obj, goal_mode_val):
                @torch.inference_mode()
                def text_tree_plan(obs_image_np, goal_image_np):
                    t_start = time.perf_counter()

                    obs_tensor = pipeline_obj.preprocess(obs_image_np)
                    obs_emb = pipeline_obj.encode(obs_tensor)
                    # Use pre-cached goal_emb from set_goal_text() instead of encoding goal_image
                    goal_emb = pipeline_obj._goal_emb

                    orig_n_steps = None
                    if tree_obj.cem_steps is not None:
                        orig_n_steps = pipeline_obj.n_steps
                        pipeline_obj.n_steps = tree_obj.cem_steps

                    action = tree_obj._plan_batched(obs_emb, goal_emb)

                    if orig_n_steps is not None:
                        pipeline_obj.n_steps = orig_n_steps

                    t_total = (time.perf_counter() - t_start) * 1000
                    tree_obj.timing["total_ms"].append(t_total)
                    return action
                return text_tree_plan

            planner_fn = make_text_tree_plan(tree, pipeline, goal_mode)
            # Bind __self__ for action_block detection
            planner_fn.__self__ = tree

        successes, planning_times = run_episodes_combo(
            planner_fn, goal_mode, pipeline, dataset, process, world,
            episode_idx, col_name, eval_episodes, eval_start_steps,
            goal_offset, pos_target_all, args, cfg,
        )

        sr = np.mean(successes) * 100
        mean_ms = np.mean(planning_times)
        results[config_name] = {
            "success_rate": float(sr),
            "num_successes": int(sum(successes)),
            "num_eval": args.num_eval,
            "mean_planning_ms": float(mean_ms),
            "effective_hz": 1000 / mean_ms if mean_ms > 0 else 0,
            "goal_mode": goal_mode,
            "tree": use_tree,
        }
        print(f"\n  {config_name}: {sr:.1f}% ({sum(successes)}/{args.num_eval}), {mean_ms:.0f}ms ({1000/mean_ms:.1f} Hz)")

    # Summary
    print(f"\n{'='*60}")
    print(f"COMBINED INTEGRATION EVAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25s} {'Success':>8s} {'Latency':>10s} {'Hz':>6s}")
    print(f"{'-'*55}")
    for name, r in results.items():
        print(f"{name:<25s} {r['success_rate']:>7.1f}% {r['mean_planning_ms']:>8.0f}ms {r['effective_hz']:>5.1f}")

    # Gate checks
    baseline_sr = results.get("image_flat", {}).get("success_rate", 0)
    print(f"\nGate checks (vs image_flat baseline = {baseline_sr:.1f}%):")
    for name, r in results.items():
        if name == "image_flat":
            continue
        ratio = r["success_rate"] / baseline_sr * 100 if baseline_sr > 0 else 0
        gate = "PASS" if ratio >= 80 else "FAIL"
        print(f"  {name}: {ratio:.0f}% of baseline — {gate}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "combined_integration_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
