#!/usr/bin/env python3
"""
D4: Evaluate Dream Scorer vs MSE in Dream Tree

Compares DreamTree with multi-signal scorer vs MSE-to-goal baseline.
Tests both cheap depth (1-3 rounds) and full depth (15 rounds) modes.

Usage (on RunPod):
    export STABLEWM_HOME=/workspace/data
    cd /workspace/le-harness

    # First train the scorer:
    python scripts/train_dream_scorer.py --policy pusht/lejepa

    # Then benchmark:
    python scripts/eval_dream_scorer.py --policy pusht/lejepa \
        --scorer-path /workspace/data/results/d4_dream_scorer.pt
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

from harness.pipeline import PlanningPipeline
from harness.dream_tree import DreamTreePlanner
from harness.dream_scorer import DreamScorer

# Reuse eval infrastructure from eval_dream_tree
from scripts.eval_dream_tree import setup_eval_env, run_episodes, load_eval_config


def run_eval(args):
    print(f"\n{'='*60}")
    print(f"D4: Dream Scorer Evaluation")
    print(f"{'='*60}")
    print(f"Scorer: {args.scorer_path or 'MSE baseline'}")
    print(f"Num roots: {args.num_roots}")
    print(f"Max depth: {args.max_depth}")
    print(f"Eval budget: {args.eval_budget} steps")
    print(f"Num eval: {args.num_eval} episodes")

    cfg = load_eval_config()

    # Build pipeline
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

    # Set up eval environment
    (dataset, process, world, episode_idx, col_name,
     eval_episodes, eval_start_steps, goal_offset) = setup_eval_env(cfg, args)

    results = {}

    # --- MSE baseline (flat CEM + tree) ---
    print(f"\n--- Flat CEM (MSE baseline) ---")
    pipeline.scorer = None
    successes_flat, times_flat = run_episodes(
        pipeline.plan, dataset, process, world, episode_idx, col_name,
        eval_episodes, eval_start_steps, goal_offset, args,
    )
    sr_flat = np.mean(successes_flat) * 100
    results["flat_mse"] = {
        "success_rate": float(sr_flat),
        "num_successes": int(sum(successes_flat)),
        "mean_planning_ms": float(np.mean(times_flat)),
    }
    print(f"Flat CEM (MSE): {sr_flat:.1f}%")

    print(f"\n--- Dream Tree (MSE, {args.num_roots}R full depth) ---")
    pipeline.scorer = None
    tree_mse = DreamTreePlanner(pipeline, num_roots=args.num_roots, max_depth=args.max_depth, cheap_depth=False)
    tree_mse.plan(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                  np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    tree_mse.reset_timing()

    successes_tree_mse, times_tree_mse = run_episodes(
        tree_mse.plan, dataset, process, world, episode_idx, col_name,
        eval_episodes, eval_start_steps, goal_offset, args,
    )
    sr_tree_mse = np.mean(successes_tree_mse) * 100
    results["tree_mse"] = {
        "success_rate": float(sr_tree_mse),
        "num_successes": int(sum(successes_tree_mse)),
        "mean_planning_ms": float(np.mean(times_tree_mse)),
        "tree_timing": tree_mse.get_timing_summary(),
    }
    print(f"Tree (MSE): {sr_tree_mse:.1f}%, {np.mean(times_tree_mse):.0f}ms")

    # --- Scorer-based (if provided) ---
    if args.scorer_path:
        scorer = DreamScorer.from_checkpoint(args.scorer_path)
        pipeline.scorer = scorer

        print(f"\n--- Flat CEM (Dream Scorer) ---")
        successes_flat_s, times_flat_s = run_episodes(
            pipeline.plan, dataset, process, world, episode_idx, col_name,
            eval_episodes, eval_start_steps, goal_offset, args,
        )
        sr_flat_s = np.mean(successes_flat_s) * 100
        results["flat_scorer"] = {
            "success_rate": float(sr_flat_s),
            "num_successes": int(sum(successes_flat_s)),
            "mean_planning_ms": float(np.mean(times_flat_s)),
        }
        print(f"Flat CEM (Scorer): {sr_flat_s:.1f}%")

        print(f"\n--- Dream Tree (Scorer, {args.num_roots}R cheap depth) ---")
        tree_scorer = DreamTreePlanner(pipeline, num_roots=args.num_roots, max_depth=args.max_depth, cheap_depth=True)
        tree_scorer.plan(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                         np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        tree_scorer.reset_timing()

        successes_tree_s, times_tree_s = run_episodes(
            tree_scorer.plan, dataset, process, world, episode_idx, col_name,
            eval_episodes, eval_start_steps, goal_offset, args,
        )
        sr_tree_s = np.mean(successes_tree_s) * 100
        results["tree_scorer_cheap"] = {
            "success_rate": float(sr_tree_s),
            "num_successes": int(sum(successes_tree_s)),
            "mean_planning_ms": float(np.mean(times_tree_s)),
            "tree_timing": tree_scorer.get_timing_summary(),
        }
        print(f"Tree (Scorer, cheap): {sr_tree_s:.1f}%, {np.mean(times_tree_s):.0f}ms")

        # Also test tree with scorer + full depth
        print(f"\n--- Dream Tree (Scorer, {args.num_roots}R full depth) ---")
        tree_scorer_full = DreamTreePlanner(pipeline, num_roots=args.num_roots, max_depth=args.max_depth, cheap_depth=False)
        tree_scorer_full.plan(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                              np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        tree_scorer_full.reset_timing()

        successes_tree_sf, times_tree_sf = run_episodes(
            tree_scorer_full.plan, dataset, process, world, episode_idx, col_name,
            eval_episodes, eval_start_steps, goal_offset, args,
        )
        sr_tree_sf = np.mean(successes_tree_sf) * 100
        results["tree_scorer_full"] = {
            "success_rate": float(sr_tree_sf),
            "num_successes": int(sum(successes_tree_sf)),
            "mean_planning_ms": float(np.mean(times_tree_sf)),
            "tree_timing": tree_scorer_full.get_timing_summary(),
        }
        print(f"Tree (Scorer, full): {sr_tree_sf:.1f}%, {np.mean(times_tree_sf):.0f}ms")

        pipeline.scorer = None  # reset

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"D4 Results Summary")
    print(f"{'='*60}")
    for key, val in results.items():
        sr = val["success_rate"]
        ms = val["mean_planning_ms"]
        print(f"  {key:<25} {sr:>5.1f}%  {ms:>6.0f}ms")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "d4_scorer_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="D4: Dream Scorer Evaluation")
    parser.add_argument("--policy", default="pusht/lejepa")
    parser.add_argument("--scorer-path", default=None,
                        help="Path to trained scorer checkpoint")
    parser.add_argument("--num-roots", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="/workspace/data/results")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
