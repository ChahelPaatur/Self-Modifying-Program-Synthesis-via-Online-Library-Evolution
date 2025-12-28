#!/usr/bin/env python3
"""
Spacecraft navigation ablation benchmark.

Runs multiple random seeds over controller/safety variants and produces:
  - paper_assets/spacecraft_nav_results.json
  - paper_assets/figures/fig_spacecraft_ablation_success.pdf/png
  - paper_assets/figures/fig_spacecraft_ablation_fuel.pdf/png

This is a research benchmark (not flight software validation).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List

import numpy as np

from spacecraft_nav_autonomy import EpisodeConfig, AutonomyConfig, run_episode_cfg


def run_variant(name: str, cfg: AutonomyConfig, seeds: List[int]) -> Dict:
    rows = []
    for s in seeds:
        ep = EpisodeConfig(seed=s)
        m = run_episode_cfg(ep=ep, cfg=cfg, verbose=False)
        m["seed"] = s
        rows.append(m)
    success = float(np.mean([r["success"] for r in rows]))
    fuel = [r["fuel_used"] for r in rows]
    steps = [r["steps"] for r in rows]
    return {
        "name": name,
        "cfg": asdict(cfg),
        "n": len(rows),
        "success_rate": success,
        "fuel_mean": float(np.mean(fuel)),
        "fuel_std": float(np.std(fuel)),
        "steps_mean": float(np.mean(steps)),
        "steps_std": float(np.std(steps)),
        "rows": rows,
    }


def plot_results(results: List[Dict], fig_dir: str):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    names = [r["name"] for r in results]
    succ = [100.0 * r["success_rate"] for r in results]

    plt.figure(figsize=(7.2, 3.8))
    plt.bar(names, succ, color="#4C78A8", edgecolor="white")
    plt.ylabel("Success rate (%)")
    plt.ylim(0, 100)
    plt.title("Spacecraft Rendezvous: Ablation Success Rate (20 seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_spacecraft_ablation_success.pdf"))
    plt.savefig(os.path.join(fig_dir, "fig_spacecraft_ablation_success.png"))
    plt.close()

    # Fuel boxplots
    fuel_sets = [np.array([row["fuel_used"] for row in r["rows"]]) for r in results]
    plt.figure(figsize=(7.2, 3.8))
    plt.boxplot(fuel_sets, labels=names, showfliers=False)
    plt.ylabel("Fuel used (delta-v proxy)")
    plt.title("Spacecraft Rendezvous: Fuel Usage Distribution (20 seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_spacecraft_ablation_fuel.pdf"))
    plt.savefig(os.path.join(fig_dir, "fig_spacecraft_ablation_fuel.png"))
    plt.close()


def main():
    os.makedirs("paper_assets/figures", exist_ok=True)
    seeds = list(range(20))

    variants = []

    # Main: LQR + docking PD + shield
    variants.append(("lqr+dockpd+shield", AutonomyConfig(controller="lqr", docking_mode=True, safety_shield=True)))

    # Ablations
    variants.append(("lqr+dockpd", AutonomyConfig(controller="lqr", docking_mode=True, safety_shield=False)))
    variants.append(("lqr+shield", AutonomyConfig(controller="lqr", docking_mode=False, safety_shield=True)))
    variants.append(("lqr_only", AutonomyConfig(controller="lqr", docking_mode=False, safety_shield=False)))
    variants.append(("mpc+dockpd+shield", AutonomyConfig(controller="mpc", docking_mode=True, safety_shield=True)))

    results = [run_variant(name, cfg, seeds) for name, cfg in variants]

    out = {"seeds": seeds, "results": results}
    with open("paper_assets/spacecraft_nav_results.json", "w") as f:
        json.dump(out, f, indent=2)

    plot_results(results, "paper_assets/figures")
    print("Benchmark complete")
    print("  paper_assets/spacecraft_nav_results.json")


if __name__ == "__main__":
    main()


