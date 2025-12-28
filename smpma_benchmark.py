#!/usr/bin/env python3
"""
Benchmark SMPMA on ARC-AGI evaluation challenges and generate paper-quality figures.

Outputs:
  - JSON results: paper_assets/smpma_eval_results.json
  - Figures:       paper_assets/figures/*.pdf and *.png
  - Paper section: paper_assets/FIGURES_SECTION.md
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from smpma_agi import SMPMA, Program


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def grids_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return len(b) == 0
    if len(a[0]) != len(b[0]):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def score_similarity(a: List[List[int]], b: List[List[int]]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return 0.0
    h, w = len(a), len(a[0])
    tot = h * w
    same = 0
    for i in range(h):
        rowa = a[i]
        rowb = b[i]
        for j in range(w):
            if rowa[j] == rowb[j]:
                same += 1
    return same / tot if tot else 0.0


@dataclass
class TaskResult:
    task_id: str
    train_score: float
    strict_solved: bool
    program_len: int
    program_str: str
    seconds: float
    n_train: int
    n_test: int


def solve_task_with_program(smpma: SMPMA, task_id: str, task: Dict[str, Any], solution_outputs: Optional[List[List[List[int]]]]):
    train_pairs: List[Tuple[List[List[int]], List[List[int]]]] = []
    for ex in task["train"]:
        train_pairs.append((ex["input"], ex["output"]))

    t0 = time.time()
    program, train_score = smpma.solve_task(train_pairs, task_id)
    dt = time.time() - t0

    prog: Optional[Program] = program
    prog_str = str(prog) if prog else "None"
    prog_len = len(prog.operations) if prog else 0

    strict = False
    if prog and solution_outputs is not None:
        preds = []
        for ex in task["test"]:
            preds.append(prog.apply(ex["input"]))
        # Strict: all test cases correct.
        if len(preds) == len(solution_outputs):
            strict = all(grids_equal(p, gt) for p, gt in zip(preds, solution_outputs))

    return TaskResult(
        task_id=task_id,
        train_score=float(train_score) if train_score is not None else 0.0,
        strict_solved=bool(strict),
        program_len=int(prog_len),
        program_str=prog_str,
        seconds=float(dt),
        n_train=len(task["train"]),
        n_test=len(task["test"]),
    )


def make_figures(results: List[TaskResult], out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib import failed; skipping figure generation:", e)
        return

    os.makedirs(out_dir, exist_ok=True)

    # Global style for paper
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    scores = [r.train_score for r in results]
    times = [r.seconds for r in results]
    lens = [r.program_len for r in results]
    solved = [1 if r.strict_solved else 0 for r in results]

    # Fig 1: Train-score distribution
    plt.figure(figsize=(6.2, 3.6))
    plt.hist(scores, bins=20, color="#4C78A8", alpha=0.9, edgecolor="white")
    plt.title("SMPMA: Train Similarity Score Distribution (Evaluation Set)")
    plt.xlabel("Average train similarity")
    plt.ylabel("Number of tasks")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_train_score_hist.pdf"))
    plt.savefig(os.path.join(out_dir, "fig_train_score_hist.png"))
    plt.close()

    # Fig 2: Strict accuracy
    acc = sum(solved) / max(1, len(solved))
    plt.figure(figsize=(4.8, 3.4))
    plt.bar(["Strict solved"], [acc * 100.0], color="#F58518", edgecolor="white")
    plt.ylim(0, 100)
    plt.title("SMPMA Strict Accuracy (Evaluation Set)")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_strict_accuracy.pdf"))
    plt.savefig(os.path.join(out_dir, "fig_strict_accuracy.png"))
    plt.close()

    # Fig 3: Runtime distribution
    plt.figure(figsize=(6.2, 3.6))
    plt.hist(times, bins=20, color="#54A24B", alpha=0.9, edgecolor="white")
    plt.title("Per-Task Runtime Distribution (seconds)")
    plt.xlabel("Seconds")
    plt.ylabel("Number of tasks")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_runtime_hist.pdf"))
    plt.savefig(os.path.join(out_dir, "fig_runtime_hist.png"))
    plt.close()

    # Fig 4: Program length distribution
    plt.figure(figsize=(6.2, 3.6))
    plt.hist(lens, bins=range(0, max(lens) + 2), color="#B279A2", alpha=0.9, edgecolor="white", align="left")
    plt.title("Program Length Distribution")
    plt.xlabel("Number of operations")
    plt.ylabel("Number of tasks")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_program_len_hist.pdf"))
    plt.savefig(os.path.join(out_dir, "fig_program_len_hist.png"))
    plt.close()


def write_figures_section(fig_md_path: str):
    # This is intentionally plain and paper-friendly (no emojis, no hype).
    content = """## Figures

### Figure 1: Train-score distribution
File: `paper_assets/figures/fig_train_score_hist.pdf` (and `.png`)

**Caption.** Distribution of average train-set similarity scores obtained by SMPMA on the ARC-AGI evaluation challenges. Higher scores indicate better fit on training pairs but do not necessarily imply strict generalization to test cases.

### Figure 2: Strict accuracy
File: `paper_assets/figures/fig_strict_accuracy.pdf` (and `.png`)

**Caption.** Strict task accuracy on the ARC-AGI evaluation challenges, where a task is counted as solved only if the synthesized program produces correct outputs for all test cases in that task.

### Figure 3: Runtime distribution
File: `paper_assets/figures/fig_runtime_hist.pdf` (and `.png`)

**Caption.** Distribution of per-task runtimes for SMPMA (seconds per task) on the ARC-AGI evaluation challenges under the current hyperparameters.

### Figure 4: Program length distribution
File: `paper_assets/figures/fig_program_len_hist.pdf` (and `.png`)

**Caption.** Distribution of synthesized program lengths (number of operations) across evaluation tasks.
"""
    with open(fig_md_path, "w") as f:
        f.write(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="arc-prize-2025", help="Path to ARC dataset folder")
    ap.add_argument("--limit", type=int, default=120, help="Max evaluation tasks to run")
    ap.add_argument("--quiet", action="store_true", help="Reduce solver verbosity (recommended for benchmarks)")
    ap.add_argument("--ablations", action="store_true", help="Run lightweight ablations and plot comparison")
    ap.add_argument("--ablation_limit", type=int, default=30, help="Tasks to use for ablations (kept small)")
    args = ap.parse_args()

    base = args.data
    eval_ch_path = os.path.join(base, "arc-agi_evaluation_challenges.json")
    eval_sol_path = os.path.join(base, "arc-agi_evaluation_solutions.json")

    challenges = load_json(eval_ch_path)
    solutions = load_json(eval_sol_path)

    import smpma_agi as sm

    def run_one(tag: str) -> Dict[str, Any]:
        if args.quiet:
            sm.VERBOSE = False
        smpma = SMPMA()
        results: List[TaskResult] = []
        t_start = time.time()
        for idx, (task_id, task) in enumerate(list(challenges.items())[: args.limit], start=1):
            sol = solutions.get(task_id)
            sol_outputs = sol if isinstance(sol, list) else None
            r = solve_task_with_program(smpma, task_id, task, sol_outputs)
            results.append(r)
            if idx % 10 == 0:
                print(f"[{tag}] Progress: {idx}/{min(args.limit, len(challenges))} tasks")
        total = time.time() - t_start
        strict_acc = sum(1 for r in results if r.strict_solved) / max(1, len(results))
        return {
            "tag": tag,
            "n_tasks": len(results),
            "strict_accuracy": strict_acc,
            "total_seconds": total,
            "avg_seconds_per_task": total / max(1, len(results)),
            "results": [r.__dict__ for r in results],
        }

    # Main run
    payload = run_one("main")
    out_json = "paper_assets/smpma_eval_results.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    fig_dir = "paper_assets/figures"
    make_figures([TaskResult(**r) for r in payload["results"]], fig_dir)
    write_figures_section("paper_assets/FIGURES_SECTION.md")

    # Ablations
    if args.ablations:
        # Keep ablations cheap and interpretable: toggle one mechanism at a time.
        saved = {
            "ENABLE_MACRO_TEMPLATES": sm.ENABLE_MACRO_TEMPLATES,
            "ENABLE_OP_PRIORITIZATION": sm.ENABLE_OP_PRIORITIZATION,
            "SCORING_MODE": sm.SCORING_MODE,
            "MINAVG_ALPHA": sm.MINAVG_ALPHA,
        }

        variants = []

        # Baseline (current)
        variants.append(("baseline", dict(saved)))

        # No macros
        variants.append(("no_macros", {**saved, "ENABLE_MACRO_TEMPLATES": False}))

        # No prioritization
        variants.append(("no_prioritization", {**saved, "ENABLE_OP_PRIORITIZATION": False}))

        # Avg-only scoring
        variants.append(("avg_scoring", {**saved, "SCORING_MODE": "avg"}))

        ablation_payloads = []
        for name, cfg in variants:
            sm.ENABLE_MACRO_TEMPLATES = cfg["ENABLE_MACRO_TEMPLATES"]
            sm.ENABLE_OP_PRIORITIZATION = cfg["ENABLE_OP_PRIORITIZATION"]
            sm.SCORING_MODE = cfg["SCORING_MODE"]
            sm.MINAVG_ALPHA = cfg["MINAVG_ALPHA"]

            # Run on a small prefix for speed
            old_limit = args.limit
            args.limit = args.ablation_limit
            ablation_payloads.append(run_one(name))
            args.limit = old_limit

        # Restore globals
        sm.ENABLE_MACRO_TEMPLATES = saved["ENABLE_MACRO_TEMPLATES"]
        sm.ENABLE_OP_PRIORITIZATION = saved["ENABLE_OP_PRIORITIZATION"]
        sm.SCORING_MODE = saved["SCORING_MODE"]
        sm.MINAVG_ALPHA = saved["MINAVG_ALPHA"]

        out_ablation = "paper_assets/smpma_ablations.json"
        with open(out_ablation, "w") as f:
            json.dump({"ablations": ablation_payloads}, f, indent=2)

        # Figure: ablation strict accuracy
        try:
            import matplotlib.pyplot as plt
            plt.rcParams.update({
                "figure.dpi": 140,
                "savefig.dpi": 300,
                "font.size": 11,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "axes.spines.top": False,
                "axes.spines.right": False,
            })
            names = [p["tag"] for p in ablation_payloads]
            accs = [100.0 * p["strict_accuracy"] for p in ablation_payloads]
            plt.figure(figsize=(6.4, 3.6))
            plt.bar(names, accs, color="#4C78A8", edgecolor="white")
            plt.ylabel("Strict accuracy (%)")
            plt.title("Ablation: Strict Accuracy (Evaluation Subset)")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "fig_ablation_accuracy.pdf"))
            plt.savefig(os.path.join(fig_dir, "fig_ablation_accuracy.png"))
            plt.close()
        except Exception as e:
            print("Ablation plot failed:", e)

        print(f"Ablations JSON: {out_ablation}")

    print("Benchmark complete")
    print(f"  Results JSON: {out_json}")
    print(f"  Figures dir:  {fig_dir}")
    print("  Paper section: paper_assets/FIGURES_SECTION.md")


if __name__ == "__main__":
    main()


