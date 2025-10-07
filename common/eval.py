from typing import Dict, Any, List, Tuple
import pandas as pd
from .arc_io import Grid


def accuracy(preds: List[Grid], targets: List[Grid]) -> float:
    if not preds:
        return 0.0
    correct = sum(1 for p, t in zip(preds, targets) if p == t)
    return correct / len(preds)


def write_results_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def pad_or_trim(grid: Grid, target_shape: Tuple[int, int]) -> Grid:
    th, tw = target_shape
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == th and w == tw:
        return grid
    # simple zero-pad/crop to match shapes for baselines
    new = [[0 for _ in range(tw)] for _ in range(th)]
    for i in range(min(h, th)):
        for j in range(min(w, tw)):
            new[i][j] = grid[i][j]
    return new
