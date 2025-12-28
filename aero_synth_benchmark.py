#!/usr/bin/env python3
"""
Aerospace-motivated synthetic benchmark suite for abstraction + composition.

This does not "prove AGI" (nothing does), but it provides:
  - a controlled, out-of-distribution evaluation target relative to ARC-AGI grids
  - tasks analogous to aerospace perception/planning primitives:
      - change detection (frame differencing)
      - debris/target extraction (component selection)
      - track-and-align (translation inference)
      - fuse overlays (compose a delta back onto a reference canvas)

Outputs:
  - paper_assets/aero_results.json
  - paper_assets/figures/fig_aero_accuracy.pdf/png
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from smpma_agi import SMPMA


def zeros(h: int, w: int) -> List[List[int]]:
    return [[0] * w for _ in range(h)]


def place_rect(grid: List[List[int]], top: int, left: int, h: int, w: int, color: int):
    H, W = len(grid), len(grid[0])
    for i in range(top, top + h):
        for j in range(left, left + w):
            if 0 <= i < H and 0 <= j < W:
                grid[i][j] = color


def translate_grid(grid: List[List[int]], dr: int, dc: int) -> List[List[int]]:
    H, W = len(grid), len(grid[0])
    out = zeros(H, W)
    for i in range(H):
        for j in range(W):
            v = grid[i][j]
            if v == 0:
                continue
            ni, nj = i + dr, j + dc
            if 0 <= ni < H and 0 <= nj < W:
                out[ni][nj] = v
    return out


def overlay(obj: List[List[int]], canvas: List[List[int]]) -> List[List[int]]:
    H, W = len(canvas), len(canvas[0])
    out = [row[:] for row in canvas]
    for i in range(H):
        for j in range(W):
            if obj[i][j] != 0:
                out[i][j] = obj[i][j]
    return out


def make_task_track_and_overlay(rng: random.Random, dr: int, dc: int, target_color: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Input: a frame with a target + clutter.
    Output: target moved by a translation, overlaid onto the original canvas.
    """
    H = rng.randint(10, 16)
    W = rng.randint(10, 16)
    canvas = zeros(H, W)

    # clutter
    for _ in range(rng.randint(2, 5)):
        c = rng.randint(1, 4)
        place_rect(canvas, rng.randint(0, H - 2), rng.randint(0, W - 2), 1, 1, c)

    # Keep target color fixed per task so the rule is consistent and discoverable.
    th = rng.randint(2, 4)
    tw = rng.randint(2, 4)
    top = rng.randint(0, H - th)
    left = rng.randint(0, W - tw)
    place_rect(canvas, top, left, th, tw, target_color)

    target_only = zeros(H, W)
    place_rect(target_only, top, left, th, tw, target_color)
    moved = translate_grid(target_only, dr, dc)
    out = overlay(moved, canvas)
    return canvas, out


def make_dataset(n_tasks: int, seed: int = 0) -> Dict[str, Dict[str, Any]]:
    rng = random.Random(seed)
    tasks: Dict[str, Dict[str, Any]] = {}
    for t in range(n_tasks):
        tid = f"aero_{t:04d}"
        # Sample a single (dr, dc) per task; all examples share the same motion model.
        dr = rng.choice([-2, -1, 1, 2])
        dc = rng.choice([-2, -1, 1, 2])
        target_color = 9  # fixed across tasks for this benchmark variant

        # 3 train, 1 test
        train = []
        for _ in range(3):
            inp, out = make_task_track_and_overlay(rng, dr=dr, dc=dc, target_color=target_color)
            train.append({"input": inp, "output": out})
        test_inp, test_out = make_task_track_and_overlay(rng, dr=dr, dc=dc, target_color=target_color)
        tasks[tid] = {"train": train, "test": [{"input": test_inp, "output": test_out}]}
    return tasks


def grids_equal(a: List[List[int]], b: List[List[int]]) -> bool:
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


def main():
    os.makedirs("paper_assets/figures", exist_ok=True)
    tasks = make_dataset(n_tasks=120, seed=42)

    # Quiet mode for benchmarks
    import smpma_agi as sm
    sm.VERBOSE = False
    smpma = SMPMA()

    solved = 0
    results = []
    for idx, (tid, task) in enumerate(tasks.items(), start=1):
        examples = [(ex["input"], ex["output"]) for ex in task["train"]]
        program, _ = smpma.solve_task(examples, tid)
        pred = program.apply(task["test"][0]["input"]) if program else task["test"][0]["input"]
        ok = grids_equal(pred, task["test"][0]["output"])
        solved += 1 if ok else 0
        results.append({"task_id": tid, "solved": ok, "program": str(program) if program else "None"})
        if idx % 20 == 0:
            print(f"Progress: {idx}/120")

    acc = solved / 120.0
    out_json = {"n_tasks": 120, "strict_accuracy": acc, "results": results}
    with open("paper_assets/aero_results.json", "w") as f:
        json.dump(out_json, f, indent=2)

    # figure
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
        plt.figure(figsize=(4.8, 3.4))
        plt.bar(["AeroSynth"], [acc * 100.0], color="#54A24B", edgecolor="white")
        plt.ylim(0, 100)
        plt.ylabel("Strict accuracy (%)")
        plt.title("AeroSynth: Track-and-Overlay Accuracy")
        plt.tight_layout()
        plt.savefig("paper_assets/figures/fig_aero_accuracy.pdf")
        plt.savefig("paper_assets/figures/fig_aero_accuracy.png")
        plt.close()
    except Exception as e:
        print("Figure generation failed:", e)

    print("AeroSynth complete")
    print(f"  strict_accuracy: {acc:.3f}")
    print("  paper_assets/aero_results.json")


if __name__ == "__main__":
    main()


