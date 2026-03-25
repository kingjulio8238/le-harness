#!/usr/bin/env python3
"""
Phase 2 (Step 5): Log per-iteration CEM/iCEM cost convergence.

Instruments the solver's get_cost() to capture per-iteration costs,
then runs a small eval (5 episodes) on selected configs. Outputs
JSON with per-step, per-iteration cost curves.

Run AFTER sweep_budget.py completes. Use the top 2-3 configs from
the sweep results.

Usage (on RunPod):
    export STABLEWM_HOME=/workspace/data
    cd /workspace/le-harness

    # Log convergence for a specific config
    python scripts/log_convergence.py \
        --solver cem --num-samples 64 --n-steps 10 \
        --policy pusht/lejepa

    # Log convergence for iCEM
    python scripts/log_convergence.py \
        --solver icem --num-samples 64 --n-steps 10 \
        --policy pusht/lejepa

Output: /workspace/data/results/phase2_convergence_{solver}_{samples}x{iters}.json
"""

import argparse
import json
import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


def run_instrumented_eval(solver_name, num_samples, n_steps, policy, num_eval=5, seed=42):
    """Run eval with instrumented cost logging."""

    # Load config via Hydra compose API
    from hydra import compose, initialize_config_dir

    config_dir = str(Path("./config/eval").resolve())

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        overrides = [
            f"policy={policy}",
            f"solver={solver_name}",
            f"solver.num_samples={num_samples}",
            f"solver.n_steps={n_steps}",
            f"solver.topk={max(1, min(num_samples // 5, 30))}",
            f"eval.num_eval={num_eval}",
            f"seed={seed}",
        ]
        if solver_name == "icem":
            n_elite_keep = max(1, min(5, num_samples // 10))
            overrides.append(f"solver.n_elite_keep={n_elite_keep}")

        cfg = compose(config_name="pusht", overrides=overrides)

    # Load model
    print(f"Loading model: {policy}")
    model = swm.policy.AutoCostModel(cfg.policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    # Set up transforms and dataset (same as eval.py)
    transform = {
        "pixels": transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]),
        "goal": transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]),
    }

    dataset_path = Path(cfg.get("cache_dir", None) or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # Instrument the solver's model.get_cost to log per-iteration costs
    iteration_costs = []  # Will be populated by the monkey-patch
    call_counter = [0]

    original_get_cost = model.get_cost

    def instrumented_get_cost(info_dict, candidates):
        costs = original_get_cost(info_dict, candidates)
        # costs shape: (batch, num_samples)
        best_cost = costs.min(dim=1).values.mean().item()
        mean_cost = costs.mean().item()
        iteration_costs.append({
            "call": call_counter[0],
            "best_cost": best_cost,
            "mean_cost": mean_cost,
        })
        call_counter[0] += 1
        return costs

    model.get_cost = instrumented_get_cost

    # Create solver and policy
    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    wm_policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )

    # Create world
    cfg_world = dict(cfg.world)
    cfg_world["max_episode_steps"] = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg_world, image_shape=(224, 224))

    # Sample episodes
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    ep_indices = np.unique(episode_idx)

    episode_len = []
    for ep_id in ep_indices:
        episode_len.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    episode_len = np.array(episode_len)

    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    g = np.random.default_rng(seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    world.set_policy(wm_policy)

    # Clear cost log and run
    iteration_costs.clear()
    call_counter[0] = 0

    print(f"Running {num_eval} episodes with {solver_name} "
          f"(samples={num_samples}, iters={n_steps})...")
    start = time.time()

    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
    )

    elapsed = time.time() - start

    # Organize costs by planning step
    # Each planning step calls get_cost n_steps times (one per CEM iteration)
    costs_per_planning_step = []
    for i in range(0, len(iteration_costs), n_steps):
        step_costs = iteration_costs[i:i + n_steps]
        if len(step_costs) == n_steps:
            costs_per_planning_step.append({
                "iteration": list(range(n_steps)),
                "best_cost": [c["best_cost"] for c in step_costs],
                "mean_cost": [c["mean_cost"] for c in step_costs],
            })

    # Compute convergence statistics across all planning steps
    convergence_stats = {"iteration": list(range(n_steps)), "best_cost_mean": [], "best_cost_std": []}
    for iter_idx in range(n_steps):
        costs_at_iter = [s["best_cost"][iter_idx] for s in costs_per_planning_step
                         if iter_idx < len(s["best_cost"])]
        if costs_at_iter:
            convergence_stats["best_cost_mean"].append(float(np.mean(costs_at_iter)))
            convergence_stats["best_cost_std"].append(float(np.std(costs_at_iter)))
        else:
            convergence_stats["best_cost_mean"].append(None)
            convergence_stats["best_cost_std"].append(None)

    # Print convergence summary
    print(f"\nConvergence summary ({len(costs_per_planning_step)} planning steps):")
    print(f"{'Iter':>4s} {'Best Cost (mean)':>16s} {'Std':>10s} {'Δ%':>8s}")
    print("-" * 42)
    prev = None
    for i, (mean_c, std_c) in enumerate(zip(
        convergence_stats["best_cost_mean"],
        convergence_stats["best_cost_std"]
    )):
        if mean_c is None:
            continue
        delta = ""
        if prev is not None and prev > 0:
            delta = f"{(mean_c - prev) / prev * 100:+.1f}%"
        print(f"{i:>4d} {mean_c:>16.6f} {std_c:>10.6f} {delta:>8s}")
        prev = mean_c

    # Identify where convergence plateaus (< 1% improvement)
    plateau_iter = n_steps
    for i in range(1, len(convergence_stats["best_cost_mean"])):
        curr = convergence_stats["best_cost_mean"][i]
        prev_c = convergence_stats["best_cost_mean"][i - 1]
        if curr is not None and prev_c is not None and prev_c > 0:
            improvement = abs(curr - prev_c) / prev_c
            if improvement < 0.01:  # < 1% improvement
                plateau_iter = i
                break

    print(f"\nPlateau detected at iteration {plateau_iter} (<1% improvement)")
    print(f"Suggestion: adaptive stopping at ~{plateau_iter} iterations "
          f"(vs {n_steps} configured)")

    return {
        "config": {
            "solver": solver_name,
            "num_samples": num_samples,
            "n_steps": n_steps,
            "topk": max(1, min(num_samples // 5, 30)),
            "policy": policy,
            "num_eval": num_eval,
        },
        "metrics": {
            "success_rate": float(metrics.get("success_rate", 0)),
            "eval_time_s": round(elapsed, 1),
            "total_planning_steps": len(costs_per_planning_step),
            "total_get_cost_calls": len(iteration_costs),
        },
        "convergence_stats": convergence_stats,
        "plateau_iteration": plateau_iter,
        "per_step_costs": costs_per_planning_step[:10],  # first 10 steps as examples
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Step 5: CEM Cost Convergence Logging")
    parser.add_argument("--solver", default="cem", choices=["cem", "icem"])
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--policy", default="pusht/lejepa")
    parser.add_argument("--num-eval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="/workspace/data/results")
    args = parser.parse_args()

    result = run_instrumented_eval(
        args.solver, args.num_samples, args.n_steps,
        args.policy, args.num_eval, args.seed,
    )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"phase2_convergence_{args.solver}_{args.num_samples}x{args.n_steps}.json"
    out_path = out_dir / filename

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
