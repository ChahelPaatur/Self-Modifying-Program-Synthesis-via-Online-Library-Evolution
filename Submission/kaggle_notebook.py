"""
ARC Prize 2025 - Kaggle Submission Notebook
Copy this entire file into a Kaggle notebook and run it
"""

# ============================================================================
# CELL 1: Imports and Setup
# ============================================================================

import json
import sys
from typing import List, Dict, Tuple, Optional, Set, Callable
from collections import Counter, defaultdict, deque
import itertools

Grid = List[List[int]]

# ============================================================================
# CELL 2: Advanced Operations Library
# ============================================================================

def upscale_by_repetition(grid: Grid, factor: int) -> Grid:
    """Upscale by repeating each cell factor×factor times"""
    if not grid or factor <= 0:
        return grid
    result = []
    for row in grid:
        expanded_row = []
        for cell in row:
            expanded_row.extend([cell] * factor)
        for _ in range(factor):
            result.append(expanded_row[:])
    return result

def downscale_by_sampling(grid: Grid, factor: int) -> Grid:
    """Downscale by sampling every factor-th cell"""
    if not grid or factor <= 0:
        return grid
    h, w = len(grid), len(grid[0])
    return [[grid[i][j] for j in range(0, w, factor)] for i in range(0, h, factor)]

def repeat_pattern_n_times(grid: Grid, n: int, axis: str = 'horizontal') -> Grid:
    """Repeat pattern n times along axis"""
    if not grid or n <= 0:
        return grid
    if axis == 'horizontal':
        return [row * n for row in grid]
    else:
        return grid * n

def swap_two_colors(grid: Grid, c1: int, c2: int) -> Grid:
    """Swap two specific colors"""
    result = []
    for row in grid:
        new_row = []
        for cell in row:
            if cell == c1:
                new_row.append(c2)
            elif cell == c2:
                new_row.append(c1)
            else:
                new_row.append(cell)
        result.append(new_row)
    return result

def stack_grids_horizontally(g1: Grid, g2: Grid) -> Grid:
    """Stack two grids horizontally"""
    if not g1:
        return g2
    if not g2:
        return g1
    return [r1 + r2 for r1, r2 in zip(g1, g2)]

def stack_grids_vertically(g1: Grid, g2: Grid) -> Grid:
    """Stack two grids vertically"""
    return g1 + g2

def hollow_out(grid: Grid, bg: int = 0) -> Grid:
    """Keep only the border of objects"""
    if not grid or len(grid) < 3:
        return grid
    result = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r][c] != bg:
                if all(grid[r+dr][c+dc] == grid[r][c] 
                       for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]):
                    result[r][c] = bg
    return result

def fill_holes(grid: Grid, bg: int = 0) -> Grid:
    """Fill enclosed regions with surrounding color"""
    if not grid:
        return grid
    result = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if result[r][c] == bg:
                neighbors = [grid[r+dr][c+dc] for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]]
                non_bg = [x for x in neighbors if x != bg]
                if len(non_bg) == 4 and len(set(non_bg)) == 1:
                    result[r][c] = non_bg[0]
    return result

# ============================================================================
# CELL 3: Core Hybrid Solver
# ============================================================================

class Operation:
    def __init__(self, name: str, func: Callable, params: Optional[Dict] = None):
        self.name = name
        self.func = func
        self.params = params or {}
    
    def apply(self, grid: Grid) -> Optional[Grid]:
        try:
            if self.params:
                return self.func(grid, **self.params)
            else:
                return self.func(grid)
        except:
            return None

def identity(g: Grid) -> Grid:
    return [row[:] for row in g]

def rotate_90(g: Grid) -> Grid:
    if not g:
        return g
    return [list(row) for row in zip(*g[::-1])]

def rotate_180(g: Grid) -> Grid:
    return [row[::-1] for row in g[::-1]]

def rotate_270(g: Grid) -> Grid:
    if not g:
        return g
    return [list(row) for row in zip(*g)][::-1]

def flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]

def flip_v(g: Grid) -> Grid:
    return g[::-1]

def transpose(g: Grid) -> Grid:
    if not g:
        return g
    return [list(row) for row in zip(*g)]

def replace_specific_color(g: Grid, old: int, new: int) -> Grid:
    return [[new if c == old else c for c in row] for row in g]

def apply_gravity_down(g: Grid, bg: int = 0) -> Grid:
    if not g:
        return g
    h, w = len(g), len(g[0])
    result = [[bg] * w for _ in range(h)]
    for c in range(w):
        stack = [g[r][c] for r in range(h) if g[r][c] != bg]
        for i, val in enumerate(stack):
            result[h - 1 - i][c] = val
    return result

def build_operation_library() -> List[Operation]:
    """Build operation library"""
    ops = []
    
    # Geometric
    ops.append(Operation("identity", identity))
    ops.append(Operation("rotate_90", rotate_90))
    ops.append(Operation("rotate_180", rotate_180))
    ops.append(Operation("rotate_270", rotate_270))
    ops.append(Operation("flip_h", flip_h))
    ops.append(Operation("flip_v", flip_v))
    ops.append(Operation("transpose", transpose))
    
    # Tiling
    for n in [2, 3, 4, 5]:
        ops.append(Operation(f"tile_h_{n}", repeat_pattern_n_times, {"n": n, "axis": "horizontal"}))
        ops.append(Operation(f"tile_v_{n}", repeat_pattern_n_times, {"n": n, "axis": "vertical"}))
    
    # Scaling
    for factor in [2, 3, 4]:
        ops.append(Operation(f"upscale_{factor}x", upscale_by_repetition, {"factor": factor}))
        ops.append(Operation(f"downscale_{factor}x", downscale_by_sampling, {"factor": factor}))
    
    # Colors
    for c1, c2 in [(1, 2), (1, 3), (2, 3), (0, 1), (0, 2), (0, 3)]:
        ops.append(Operation(f"swap_{c1}_{c2}", swap_two_colors, {"c1": c1, "c2": c2}))
    
    for old_c in range(10):
        for new_c in range(10):
            if old_c != new_c and old_c <= 3 and new_c <= 3:
                ops.append(Operation(f"replace_{old_c}_to_{new_c}", replace_specific_color, {"old": old_c, "new": new_c}))
    
    # Spatial
    ops.append(Operation("hollow_out", hollow_out))
    ops.append(Operation("fill_holes", fill_holes))
    ops.append(Operation("gravity_down", apply_gravity_down))
    
    return ops

# ============================================================================
# CELL 4: Solver Class
# ============================================================================

class HybridSuperSolver:
    def __init__(self):
        self.operations = build_operation_library()
    
    def _grids_equal(self, g1: Grid, g2: Grid) -> bool:
        return g1 == g2
    
    def _find_match(self, inp: Grid, out: Grid, depth: int = 1) -> Optional[List[Operation]]:
        """Find program of given depth"""
        if depth == 1:
            for op in self.operations:
                result = op.apply(inp)
                if result is not None and self._grids_equal(result, out):
                    return [op]
        elif depth == 2:
            for op1 in self.operations[:40]:
                temp = op1.apply(inp)
                if temp is None:
                    continue
                for op2 in self.operations[:40]:
                    result = op2.apply(temp)
                    if result is not None and self._grids_equal(result, out):
                        return [op1, op2]
        return None
    
    def predict(self, task) -> List[Grid]:
        train = task.get('train', [])
        test = task.get('test', [])
        
        if not train or not test:
            return [t['input'] for t in test]
        
        train_pairs = [(ex['input'], ex['output']) for ex in train]
        first_in, first_out = train_pairs[0]
        
        # Try depth 1
        program = self._find_match(first_in, first_out, depth=1)
        
        # Try depth 2 if needed
        if program is None:
            program = self._find_match(first_in, first_out, depth=2)
        
        if program is None:
            return [t['input'] for t in test]
        
        # Verify on all train
        for inp, out in train_pairs[1:]:
            result = inp
            for op in program:
                result = op.apply(result)
                if result is None:
                    break
            if result is None or not self._grids_equal(result, out):
                return [t['input'] for t in test]
        
        # Apply to test
        preds = []
        for t in test:
            result = t['input']
            try:
                for op in program:
                    result = op.apply(result)
                    if result is None:
                        result = t['input']
                        break
                preds.append(result)
            except:
                preds.append(t['input'])
        
        return preds

# ============================================================================
# CELL 5: Generate Submission
# ============================================================================

print("Loading test data...")
test_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
with open(test_path, 'r') as f:
    test_challenges = json.load(f)

print(f"Found {len(test_challenges)} test tasks")

solver = HybridSuperSolver()
submission = {}

for i, (task_id, task_data) in enumerate(test_challenges.items()):
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(test_challenges)} tasks...")
    
    task = {
        'train': task_data.get('train', []),
        'test': task_data.get('test', [])
    }
    
    try:
        predictions = solver.predict(task)
        task_predictions = []
        for pred in predictions:
            task_predictions.append({
                'attempt_1': pred,
                'attempt_2': pred
            })
        submission[task_id] = task_predictions
    except:
        task_predictions = []
        for test_item in task['test']:
            inp = test_item['input']
            task_predictions.append({
                'attempt_1': inp,
                'attempt_2': inp
            })
        submission[task_id] = task_predictions

with open('submission.json', 'w') as f:
    json.dump(submission, f)

print(f"✅ Submission complete! Total tasks: {len(submission)}")

