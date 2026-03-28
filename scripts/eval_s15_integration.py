#!/usr/bin/env python3
"""
S1.5 Integration Evaluation — ZERO MOCKS

Runs the full S1.5 stack with real components on TwoRoom:
  - Real PlanningPipeline (compiled model on GPU)
  - Real SimVLM (dataset-grounded goals, alternative goals on replan)
  - Real SimMotorPolicy (world.envs.step + world.envs.render)
  - Real DriftDetector (comparing real predicted vs actual embeddings)

Compares two modes on the same episodes:
  1. Baseline: standard receding-horizon CEM (no feedback)
  2. S1.5: full control loop with confidence replanning + drift detection

Usage (on-pod with GPU):
    python scripts/eval_s15_integration.py --config-name tworoom
    python scripts/eval_s15_integration.py --num-eval 50 --drift-threshold 0.5
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.pipeline import PlanningPipeline
from harness.plan_result import PlanResult
from harness.s15_loop import S15ControlLoop
from harness.sim_components import SimVLM, SimMotorPolicy
from harness.dream_tree import DreamTreePlanner


def load_eval_config(config_name):
    from hydra import compose, initialize_config_dir
    config_dir = str(Path("./config/eval").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    return cfg


def setup_eval_env(cfg, args):
    """Set up dataset, world env, action scaler, and episode sampling.
    Follows eval_dream_tree.py patterns exactly."""
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
    # frameskip=5 means each planning step = 5 env steps
    world_cfg["max_episode_steps"] = 5 * args.eval_budget + 10
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

    return dataset, process, world, cfg, episode_idx, col_name, \
        eval_episodes, eval_start_steps, goal_offset


class TreePlanResultWrapper:
    """Wraps DreamTreePlanner to return PlanResult (for S15ControlLoop compatibility).

    DreamTreePlanner.plan() returns np.ndarray. This wrapper runs the tree,
    then also calls pipeline.plan() to get confidence signals, but uses
    the tree's action instead of the pipeline's.
    """

    def __init__(self, pipeline, tree_planner):
        self.pipeline = pipeline
        self.tree = tree_planner
        self._cached_goal_image = None
        # Forward pipeline attributes needed by the control loop
        self.device = pipeline.device
        self._action_dim = pipeline._action_dim

    def preprocess(self, img):
        return self.pipeline.preprocess(img)

    def encode(self, tensor):
        return self.pipeline.encode(tensor)

    def set_goal(self, goal_image_np):
        self._cached_goal_image = goal_image_np
        self.pipeline.set_goal(goal_image_np)

    def set_goal_embedding(self, emb):
        self._cached_goal_image = None  # embedding-only, no image for tree
        self.pipeline.set_goal_embedding(emb)

    def plan(self, obs_image_np, goal_image_np=None, record_timing=True):
        """Run tree planning and wrap result as PlanResult with confidence."""
        # Get confidence signals from pipeline (fast — already compiled)
        pipeline_result = self.pipeline.plan(obs_image_np, goal_image_np, record_timing)

        # Resolve goal image: explicit arg > cached from set_goal()
        goal_img = goal_image_np if goal_image_np is not None else self._cached_goal_image

        # Get action from tree planner (better action selection)
        if goal_img is not None:
            tree_action = self.tree.plan(obs_image_np, goal_img)
        else:
            # No goal image available (embedding-only goal) — fall back to pipeline
            return pipeline_result

        # Return tree's action with pipeline's confidence signals
        return PlanResult(
            action=tree_action,
            planning_cost=pipeline_result.planning_cost,
            confidence=pipeline_result.confidence,
            terminal_embedding=pipeline_result.terminal_embedding,
            planability=pipeline_result.planability,
            planning_ms=pipeline_result.planning_ms,
            replan_threshold=pipeline_result.replan_threshold,
        )


def run_baseline_episode(pipeline, motor, start_pixels, goal_pixels, max_steps):
    """Run baseline episode using SimMotorPolicy (no replanning, no drift)."""
    pipeline.set_goal(goal_pixels)
    obs = start_pixels
    planning_times = []

    for step in range(max_steps):
        t0 = time.perf_counter()
        result = pipeline.plan(obs)
        planning_times.append((time.perf_counter() - t0) * 1000)

        obs = motor.execute(result.action)

        if motor.is_success:
            return True, step + 1, np.mean(planning_times)
        if motor.is_done:
            # Truncated (hit max_episode_steps) — not a success
            break

    return False, max_steps, np.mean(planning_times) if planning_times else 0.0


def run_s15_episode(pipeline, motor, start_pixels, goal_pixels,
                    dataset, episode_indices, goal_step_idx, start_step, args):
    """Run S1.5 episode with real SimVLM + SimMotorPolicy (zero mocks)."""
    vlm = SimVLM(
        pipeline=pipeline,
        goal_image=goal_pixels,
        dataset=dataset,
        episode_indices=episode_indices,
        goal_step=goal_step_idx,
        start_step=start_step,
        replan_offset=args.replan_offset,
        replan_strategy=args.replan_strategy,
    )

    loop = S15ControlLoop(
        pipeline=pipeline,
        vlm=vlm,
        motor=motor,
        drift_threshold=args.drift_threshold,
        drift_window=5,
        max_replans_per_episode=args.max_replans,
    )

    stats = loop.run_episode(
        initial_obs=start_pixels,
        max_steps=args.eval_budget,
    )

    return stats


def main():
    parser = argparse.ArgumentParser(description="S1.5 Integration Eval (zero mocks)")
    parser.add_argument("--policy", default="tworoom/lewm")
    parser.add_argument("--config-name", default="tworoom")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--eval-budget", type=int, default=100)
    parser.add_argument("--drift-threshold", type=float, default=999999.0,
                        help="MSE threshold for drift escalation (default: disabled, confidence-only)")
    parser.add_argument("--max-replans", type=int, default=10)
    parser.add_argument("--replan-offset", type=int, default=5)
    parser.add_argument("--replan-strategy", default="waypoint",
                        choices=["nearby", "waypoint", "persist"],
                        help="SimVLM replan strategy")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=15)
    parser.add_argument("--cost-scale", type=float, default=200.0,
                        help="Normalizer for confidence: confidence = 1 - cost/cost_scale")
    parser.add_argument("--replan-threshold", type=float, default=0.05,
                        help="Confidence below which needs_replan triggers")
    parser.add_argument("--tree", action="store_true",
                        help="Use DreamTreePlanner instead of flat CEM")
    parser.add_argument("--num-roots", type=int, default=4,
                        help="Dream Tree root candidates (only with --tree)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/s15_integration_eval.json")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    cfg = load_eval_config(args.config_name)
    dataset, process, world, cfg, episode_idx, col_name, \
        eval_episodes, eval_start_steps, goal_offset = setup_eval_env(cfg, args)

    # Build pipeline
    print("Building pipeline...")
    pipeline = PlanningPipeline(
        args.policy,
        num_samples=args.num_samples,
        n_steps=args.n_steps,
        cost_scale=args.cost_scale,
        replan_threshold=args.replan_threshold,
    )
    pipeline.warmup()

    # Optionally wrap pipeline with Dream Tree
    planner = pipeline
    if args.tree:
        tree = DreamTreePlanner(
            pipeline, num_roots=args.num_roots, max_depth=2,
            batched=True, cem_steps=7,
        )
        # Warmup tree
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tree.plan(dummy, dummy)
        tree.reset_timing()
        planner = TreePlanResultWrapper(pipeline, tree)
        print(f"Using Dream Tree ({args.num_roots} roots, batched)")

    # Build motor policy (shared across all episodes, reset per episode)
    motor = SimMotorPolicy(world, process, pipeline._action_dim)

    # --- Baseline episodes ---
    print(f"\n=== Baseline (receding-horizon CEM, SimMotorPolicy) ===")
    baseline_results = []
    for i in range(args.num_eval):
        ep_id = int(eval_episodes[i])
        start_step = int(eval_start_steps[i])

        start_pix, goal_pix, ep_indices, goal_step_idx = motor.reset_env(
            cfg, dataset, episode_idx, col_name, ep_id, start_step, goal_offset
        )

        success, steps, mean_ms = run_baseline_episode(
            planner, motor, start_pix, goal_pix, args.eval_budget
        )
        baseline_results.append({"success": success, "steps": steps, "mean_ms": float(mean_ms)})
        status = "OK" if success else "FAIL"
        print(f"  Episode {i+1}/{args.num_eval}: {status} ({steps} steps, {mean_ms:.0f}ms/step)")

    baseline_rate = sum(r["success"] for r in baseline_results) / args.num_eval * 100

    # --- S1.5 episodes (same episodes, re-reset) ---
    print(f"\n=== S1.5 (SimVLM + DriftDetector, zero mocks) ===")
    s15_results = []
    for i in range(args.num_eval):
        ep_id = int(eval_episodes[i])
        start_step = int(eval_start_steps[i])

        start_pix, goal_pix, ep_indices, goal_step_idx = motor.reset_env(
            cfg, dataset, episode_idx, col_name, ep_id, start_step, goal_offset
        )

        stats = run_s15_episode(
            planner, motor, start_pix, goal_pix,
            dataset, ep_indices, goal_step_idx, start_step, args
        )
        s15_results.append(stats)
        status = "OK" if stats.success else "FAIL"
        replans = f"R={stats.total_replans}" if stats.total_replans > 0 else ""
        print(f"  Episode {i+1}/{args.num_eval}: {status} ({stats.steps} steps) {replans}")

    s15_rate = sum(s.success for s in s15_results) / args.num_eval * 100

    # Summary
    print(f"\n{'='*50}")
    print(f"  Baseline: {baseline_rate:.0f}% ({sum(r['success'] for r in baseline_results)}/{args.num_eval})")
    print(f"  S1.5:     {s15_rate:.0f}% ({sum(s.success for s in s15_results)}/{args.num_eval})")
    print(f"  S1.5 replans (confidence): {sum(s.replans_confidence for s in s15_results)}")
    print(f"  S1.5 replans (drift):      {sum(s.replans_drift for s in s15_results)}")
    print(f"  S1.5 drift events:         {sum(s.drift_events for s in s15_results)}")
    if s15_results:
        print(f"  S1.5 mean confidence:      {np.mean([s.mean_confidence for s in s15_results]):.3f}")
    print(f"{'='*50}")

    # Save results
    drift_mses = [s.mean_drift_mse for s in s15_results if s.drift_mses]
    output = {
        "baseline": {
            "success_rate": baseline_rate,
            "num_eval": args.num_eval,
            "results": baseline_results,
        },
        "s15": {
            "success_rate": s15_rate,
            "num_eval": args.num_eval,
            "total_replans_confidence": sum(s.replans_confidence for s in s15_results),
            "total_replans_drift": sum(s.replans_drift for s in s15_results),
            "total_drift_events": sum(s.drift_events for s in s15_results),
            "mean_confidence": float(np.mean([s.mean_confidence for s in s15_results])),
            "mean_planning_cost": float(np.mean([s.mean_planning_cost for s in s15_results])),
            "mean_drift_mse": float(np.mean(drift_mses)) if drift_mses else 0.0,
        },
        "config": {
            "policy": args.policy,
            "num_samples": args.num_samples,
            "n_steps": args.n_steps,
            "drift_threshold": args.drift_threshold,
            "max_replans": args.max_replans,
            "replan_offset": args.replan_offset,
            "replan_strategy": args.replan_strategy,
            "eval_budget": args.eval_budget,
            "cost_scale": args.cost_scale,
            "replan_threshold": args.replan_threshold,
            "tree": args.tree,
            "num_roots": args.num_roots if args.tree else None,
            "seed": args.seed,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
