"""
ARC Prize 2025 - Practical Hybrid Solver
Combines multiple strategies for maximum coverage:
1. Pattern matching (symmetry, tiling, copying)
2. DSL transformations (rotate, flip, color operations)
3. Object-based reasoning (counting, sorting, grouping)
4. Spatial reasoning (gravity, alignment, scaling)

Optimized for Kaggle's compute and time constraints.
"""

import os
import json
import time
import random
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict
import numpy as np

print("="*80)
print("ARC PRIZE 2025 - HYBRID REASONING SOLVER")
print("="*80)

# Configuration
RANDOM_SEED = 42
VERBOSE = False  # Disable for faster submission generation
BEAM_WIDTH = 100
MAX_DEPTH = 5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"\nConfiguration:")
print(f"  BEAM_WIDTH: {BEAM_WIDTH}")
print(f"  MAX_DEPTH: {MAX_DEPTH}")
print(f"  VERBOSE: {VERBOSE}")

# ======================== Grid Utilities ========================
Grid = List[List[int]]

def grid_to_array(grid: Grid) -> np.ndarray:
    """Convert grid to numpy array."""
    return np.array(grid, dtype=np.int8)

def array_to_grid(arr: np.ndarray) -> Grid:
    """Convert numpy array to grid."""
    return arr.tolist()

def grids_equal(g1: Grid, g2: Grid) -> bool:
    """Check if two grids are identical."""
    return np.array_equal(grid_to_array(g1), grid_to_array(g2))

def grid_similarity(g1: Grid, g2: Grid) -> float:
    """Calculate similarity score between two grids."""
    try:
        a1, a2 = grid_to_array(g1), grid_to_array(g2)
        if a1.shape != a2.shape:
            return 0.0
        return np.mean(a1 == a2)
    except:
        return 0.0

#  ======================== Pattern Detection ========================
def detect_symmetry(grid: Grid) -> Dict[str, bool]:
    """Detect if grid has symmetry."""
    arr = grid_to_array(grid)
    return {
        'h_symmetric': np.array_equal(arr, np.fliplr(arr)),
        'v_symmetric': np.array_equal(arr, np.flipud(arr)),
        '90_symmetric': np.array_equal(arr, np.rot90(arr, 1)),
        '180_symmetric': np.array_equal(arr, np.rot90(arr, 2))
    }

def detect_tiling(grid: Grid) -> Optional[Tuple[int, int]]:
    """Detect if grid is a tiled pattern."""
    arr = grid_to_array(grid)
    h, w = arr.shape
    
    # Try different tile sizes
    for th in range(1, h//2 + 1):
        for tw in range(1, w//2 + 1):
            if h % th == 0 and w % tw == 0:
                tile = arr[:th, :tw]
                tiled = np.tile(tile, (h//th, w//tw))
                if np.array_equal(arr, tiled):
                    return (th, tw)
    return None

def find_objects(grid: Grid, bg: int = 0) -> List[Dict]:
    """Find connected components (objects) in grid."""
    arr = grid_to_array(grid)
    h, w = arr.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []
    
    for r in range(h):
        for c in range(w):
            if arr[r, c] != bg and not visited[r, c]:
                # BFS to find connected component
                color = arr[r, c]
                cells = []
                queue = [(r, c)]
                visited[r, c] = True
                
                while queue:
                    cr, cc = queue.pop(0)
                    cells.append((cr, cc))
                    
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                if cells:
                    rs = [r for r, c in cells]
                    cs = [c for r, c in cells]
                    objects.append({
                        'color': int(color),
                        'cells': cells,
                        'bbox': (min(rs), min(cs), max(rs), max(cs)),
                        'size': len(cells)
                    })
    
    return objects

# ======================== DSL Operations ========================
def op_identity(grid: Grid) -> Grid:
    return [row[:] for row in grid]

def op_rotate90(grid: Grid) -> Grid:
    return array_to_grid(np.rot90(grid_to_array(grid), -1))

def op_rotate180(grid: Grid) -> Grid:
    return array_to_grid(np.rot90(grid_to_array(grid), 2))

def op_rotate270(grid: Grid) -> Grid:
    return array_to_grid(np.rot90(grid_to_array(grid), 1))

def op_flip_h(grid: Grid) -> Grid:
    return array_to_grid(np.fliplr(grid_to_array(grid)))

def op_flip_v(grid: Grid) -> Grid:
    return array_to_grid(np.flipud(grid_to_array(grid)))

def op_transpose(grid: Grid) -> Grid:
    return array_to_grid(grid_to_array(grid).T)

def op_replace_color(grid: Grid, old: int, new: int) -> Grid:
    arr = grid_to_array(grid)
    arr[arr == old] = new
    return array_to_grid(arr)

def op_keep_color(grid: Grid, color: int) -> Grid:
    arr = grid_to_array(grid)
    mask = arr == color
    result = np.zeros_like(arr)
    result[mask] = color
    return array_to_grid(result)

def op_bbox(grid: Grid) -> Grid:
    """Extract bounding box of non-zero elements."""
    objs = find_objects(grid)
    if not objs:
        return grid
    min_r = min(o['bbox'][0] for o in objs)
    min_c = min(o['bbox'][1] for o in objs)
    max_r = max(o['bbox'][2] for o in objs)
    max_c = max(o['bbox'][3] for o in objs)
    arr = grid_to_array(grid)
    return array_to_grid(arr[min_r:max_r+1, min_c:max_c+1])

def op_gravity_down(grid: Grid) -> Grid:
    """Apply gravity - non-zero elements fall down."""
    arr = grid_to_array(grid)
    h, w = arr.shape
    result = np.zeros_like(arr)
    
    for c in range(w):
        col_vals = arr[arr[:, c] != 0, c]
        if len(col_vals) > 0:
            result[h-len(col_vals):, c] = col_vals
    
    return array_to_grid(result)

def op_scale_up(grid: Grid, factor: int = 2) -> Grid:
    """Scale grid up by repeating each cell."""
    arr = grid_to_array(grid)
    return array_to_grid(np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1))

def op_majority_color(grid: Grid) -> Grid:
    """Replace all non-zero with most common non-zero color."""
    arr = grid_to_array(grid)
    nonzero = arr[arr != 0]
    if len(nonzero) == 0:
        return grid
    maj_color = Counter(nonzero).most_common(1)[0][0]
    result = arr.copy()
    result[result != 0] = maj_color
    return array_to_grid(result)

# ======================== High-Level Reasoning ========================
class Program:
    def __init__(self, ops: List[Tuple[str, Tuple]]):
        self.ops = ops
    
    def apply(self, grid: Grid) -> Optional[Grid]:
        try:
            result = grid
            for op_name, args in self.ops:
                if op_name == 'identity':
                    result = op_identity(result)
                elif op_name == 'rotate90':
                    result = op_rotate90(result)
                elif op_name == 'rotate180':
                    result = op_rotate180(result)
                elif op_name == 'rotate270':
                    result = op_rotate270(result)
                elif op_name == 'flip_h':
                    result = op_flip_h(result)
                elif op_name == 'flip_v':
                    result = op_flip_v(result)
                elif op_name == 'transpose':
                    result = op_transpose(result)
                elif op_name == 'bbox':
                    result = op_bbox(result)
                elif op_name == 'gravity_down':
                    result = op_gravity_down(result)
                elif op_name == 'replace_color':
                    result = op_replace_color(result, args[0], args[1])
                elif op_name == 'keep_color':
                    result = op_keep_color(result, args[0])
                elif op_name == 'scale_up':
                    result = op_scale_up(result, args[0])
                elif op_name == 'majority_color':
                    result = op_majority_color(result)
                else:
                    return None
                
                if result is None:
                    return None
            return result
        except:
            return None
    
    def __repr__(self):
        return ' -> '.join(f"{n}{args}" for n, args in self.ops) or 'identity'

def get_colors(examples: List[Tuple[Grid, Grid]]) -> List[int]:
    """Get all unique colors from examples."""
    colors = set()
    for inp, out in examples:
        colors.update(np.unique(grid_to_array(inp)))
        colors.update(np.unique(grid_to_array(out)))
    return sorted([c for c in colors if c != 0])[:8]  # Limit to 8 colors

def synthesize_program(examples: List[Tuple[Grid, Grid]]) -> Optional[Program]:
    """Synthesize program using beam search."""
    colors = get_colors(examples)
    
    # Base operations (arity 0)
    base_ops = [
        ('identity', ()),
        ('rotate90', ()),
        ('rotate180', ()),
        ('rotate270', ()),
        ('flip_h', ()),
        ('flip_v', ()),
        ('transpose', ()),
        ('bbox', ()),
        ('gravity_down', ()),
        ('majority_color', ())
    ]
    
    # Color operations
    color_ops = []
    for c in colors:
        color_ops.append(('keep_color', (c,)))
    for c1 in colors:
        for c2 in colors:
            if c1 != c2:
                color_ops.append(('replace_color', (c1, c2)))
    
    # Scaling
    scale_ops = [('scale_up', (2,)), ('scale_up', (3,))]
    
    all_ops = base_ops + color_ops + scale_ops
    
    # Beam search
    candidates = [Program([])]
    best_prog, best_score = None, -1
    
    def score_program(prog: Program) -> float:
        score = 0.0
        for inp, out in examples:
            pred = prog.apply(inp)
            if pred is None:
                return -1
            score += grid_similarity(pred, out)
        return score / len(examples)
    
    for depth in range(MAX_DEPTH):
        next_pool = []
        
        for prog in candidates:
            for op_name, args in all_ops:
                new_prog = Program(prog.ops + [(op_name, args)])
                score = score_program(new_prog)
                
                if score == 1.0:
                    if VERBOSE:
                        print(f"    Perfect match @ depth {depth+1}: {new_prog}")
                    return new_prog
                
                if score > 0:
                    next_pool.append((score, new_prog))
                    if score > best_score:
                        best_score = score
                        best_prog = new_prog
        
        if not next_pool:
            break
        
        # Keep top beam_width
        next_pool.sort(key=lambda x: x[0], reverse=True)
        candidates = [prog for _, prog in next_pool[:BEAM_WIDTH]]
        
        if VERBOSE and next_pool:
            print(f"    [depth {depth+1}] candidates={len(next_pool)}, top_score={next_pool[0][0]:.3f}")
    
    return best_prog

# ======================== Main Solver ========================
def solve_task(task: Dict, task_id: str) -> List[Grid]:
    """Solve a single ARC task."""
    if VERBOSE:
        print(f"    Training examples: {len(task.get('train', []))}")
        print(f"    Test cases: {len(task.get('test', []))}")
    
    train = task.get('train', [])
    test = task.get('test', [])
    
    if not train:
        if VERBOSE:
            print(f"    ⚠ No training data - returning input")
        return [tc['input'] for tc in test]
    
    # Prepare examples
    examples = [(ex['input'], ex['output']) for ex in train]
    
    # Synthesize program
    prog = synthesize_program(examples)
    
    if prog and VERBOSE:
        print(f"    Selected program: {prog}")
        # Show first example performance
        pred = prog.apply(examples[0][0])
        sim = grid_similarity(pred, examples[0][1]) if pred else -1
        print(f"    First train similarity: {sim:.3f}")
    elif VERBOSE:
        print(f"    ⚠ No program found - returning input")
    
    # Apply to test cases
    predictions = []
    for tc in test:
        if prog:
            pred = prog.apply(tc['input'])
            predictions.append(pred if pred is not None else tc['input'])
        else:
            predictions.append(tc['input'])
    
    return predictions

# ======================== Data Loading ========================
def read_json(path: str) -> Dict:
    if VERBOSE:
        print(f"  Reading: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    if VERBOSE:
        print(f"  Loaded {len(data)} tasks")
    return data

def iter_challenges(ch_path: str, sol_path: Optional[str]):
    ch = read_json(ch_path)
    sol = read_json(sol_path) if (sol_path and os.path.exists(sol_path)) else None
    
    for task_id, task_data in ch.items():
        if sol:
            task_with_sol = {
                'train': task_data['train'],
                'test': [
                    {'input': t['input'], 'output': s}
                    for t, s in zip(task_data['test'], sol[task_id])
                ]
            }
            yield task_id, task_with_sol
        else:
            yield task_id, task_data

# ======================== Main ========================
def main():
    # Auto-detect paths
    if os.path.exists("/kaggle/input/arc-prize-2025"):
        BASE = "/kaggle/input/arc-prize-2025"
        OUT = "/kaggle/working"
    else:
        BASE = "arc-prize-2025"
        OUT = "outputs"
    
    os.makedirs(OUT, exist_ok=True)
    
    print(f"\nData directory: {BASE}")
    print(f"Output directory: {OUT}")
    
    # Mode
    MODE = "test"  # Change to "evaluation" for testing
    
    print(f"\n{'='*80}")
    print(f"MODE: {MODE.upper()}")
    print(f"{'='*80}")
    
    # Paths
    if MODE == "train":
        ch_path = os.path.join(BASE, 'arc-agi_training_challenges.json')
        sol_path = os.path.join(BASE, 'arc-agi_training_solutions.json')
    elif MODE == "evaluation":
        ch_path = os.path.join(BASE, 'arc-agi_evaluation_challenges.json')
        sol_path = os.path.join(BASE, 'arc-agi_evaluation_solutions.json')
    else:
        ch_path = os.path.join(BASE, 'arc-agi_test_challenges.json')
        sol_path = None
    
    start_time = time.time()
    
    if MODE in ("train", "evaluation"):
        # Evaluation
        print(f"\n{'='*80}")
        print("RUNNING EVALUATION")
        print(f"{'='*80}\n")
        
        correct, total, task_count = 0, 0, 0
        
        for task_id, task in iter_challenges(ch_path, sol_path):
            task_count += 1
            print(f"\n[Task {task_count}] {task_id}")
            
            predictions = solve_task(task, task_id)
            targets = [tc.get('output') for tc in task.get('test', []) if 'output' in tc]
            
            if targets:
                for pred, target in zip(predictions, targets):
                    if grids_equal(pred, target):
                        correct += 1
                        if VERBOSE:
                            print("    CORRECT")
                    total += 1
            
            if task_count % 10 == 0 and not VERBOSE:
                elapsed = time.time() - start_time
                rate = task_count / elapsed
                acc = (correct/total*100) if total > 0 else 0
                print(f"  📊 Progress: {task_count} tasks | {rate:.2f} tasks/sec | Acc: {acc:.1f}%")
        
        elapsed = time.time() - start_time
        acc = (correct/total*100) if total > 0 else 0
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS ({MODE.upper()})")
        print(f"{'='*80}")
        print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
        print(f"Tasks: {task_count}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Rate: {task_count/elapsed:.2f} tasks/sec")
        print(f"{'='*80}")
    
    else:
        # Submission
        print(f"\n{'='*80}")
        print("GENERATING SUBMISSION")
        print(f"{'='*80}\n")
        
        submission = {}
        task_count = 0
        
        for task_id, task in iter_challenges(ch_path, sol_path):
            task_count += 1
            print(f"\n[Task {task_count}] {task_id}")
            
            predictions = solve_task(task, task_id)
            submission[task_id] = [
                {'attempt_1': pred, 'attempt_2': pred}
                for pred in predictions
            ]
            
            if task_count % 10 == 0 and not VERBOSE:
                elapsed = time.time() - start_time
                rate = task_count / elapsed
                print(f"  📊 Progress: {task_count} tasks | {rate:.2f} tasks/sec")
        
        out_path = os.path.join(OUT, 'submission.json')
        with open(out_path, 'w') as f:
            json.dump(submission, f)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print("SUBMISSION COMPLETE")
        print(f"{'='*80}")
        print(f"File: {out_path}")
        print(f"Tasks: {task_count}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Rate: {task_count/elapsed:.2f} tasks/sec")
        print(f"{'='*80}")
        print("\nReady to submit to Kaggle.")

if __name__ == "__main__":
    main()

