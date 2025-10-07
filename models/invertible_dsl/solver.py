"""
Invertible DSL Solver:
- Small set of invertible operations (geometric + color)
- Short program search (depth 1-3)
- Cycle-consistency scoring: program(inv_program(output)) ≈ output
"""
from typing import List, Dict, Callable, Tuple, Optional
import itertools

Grid = List[List[int]]


# ============ Invertible Operations ============

def identity(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def rotate_90(grid: Grid) -> Grid:
    if not grid:
        return grid
    return [list(row) for row in zip(*grid[::-1])]


def rotate_270(grid: Grid) -> Grid:
    if not grid:
        return grid
    return [list(row) for row in zip(*grid)][::-1]


def rotate_180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]


def flip_h(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def flip_v(grid: Grid) -> Grid:
    return grid[::-1]


def transpose_grid(grid: Grid) -> Grid:
    if not grid:
        return grid
    return [list(row) for row in zip(*grid)]


# Color operations (parameterized, need inverse map)
def swap_colors(grid: Grid, a: int, b: int) -> Grid:
    """Swap two colors"""
    result = []
    for row in grid:
        new_row = []
        for c in row:
            if c == a:
                new_row.append(b)
            elif c == b:
                new_row.append(a)
            else:
                new_row.append(c)
        result.append(new_row)
    return result


def replace_color(grid: Grid, old: int, new: int) -> Grid:
    """Replace all old with new"""
    return [[new if c == old else c for c in row] for row in grid]


# Inverses
INVERSES = {
    'identity': 'identity',
    'rotate_90': 'rotate_270',
    'rotate_270': 'rotate_90',
    'rotate_180': 'rotate_180',
    'flip_h': 'flip_h',
    'flip_v': 'flip_v',
    'transpose': 'transpose',
}

OPS = {
    'identity': identity,
    'rotate_90': rotate_90,
    'rotate_270': rotate_270,
    'rotate_180': rotate_180,
    'flip_h': flip_h,
    'flip_v': flip_v,
    'transpose': transpose_grid,
}


def grid_equal(g1: Grid, g2: Grid) -> bool:
    return g1 == g2


def get_colors(grid: Grid) -> set:
    return {c for row in grid for c in row}


def apply_sequence(grid: Grid, ops_seq: List[str]) -> Grid:
    """Apply sequence of operation names"""
    result = grid
    for op_name in ops_seq:
        result = OPS[op_name](result)
    return result


def inverse_sequence(ops_seq: List[str]) -> List[str]:
    """Reverse and invert each op"""
    return [INVERSES[op] for op in reversed(ops_seq)]


def score_cycle_consistency(inp: Grid, out: Grid, program: List[str]) -> float:
    """
    Score = how well inv_program(program(inp)) matches inp
    and program(inv_program(out)) matches out
    """
    try:
        forward = apply_sequence(inp, program)
        inv_prog = inverse_sequence(program)
        backward = apply_sequence(forward, inv_prog)
        
        # Count matching cells
        if not backward or not inp:
            return 0.0
        if len(backward) != len(inp) or len(backward[0]) != len(inp[0]):
            return 0.0
        
        total = len(inp) * len(inp[0])
        matches = sum(1 for i in range(len(inp)) for j in range(len(inp[0])) 
                     if backward[i][j] == inp[i][j])
        return matches / total if total > 0 else 0.0
    except:
        return 0.0


def search_programs(inp: Grid, out: Grid, max_depth: int = 3) -> Optional[List[str]]:
    """Search for short program that transforms inp -> out"""
    # Depth 1
    for op in OPS.keys():
        result = OPS[op](inp)
        if grid_equal(result, out):
            return [op]
    
    # Depth 2
    if max_depth >= 2:
        for op1 in OPS.keys():
            temp = OPS[op1](inp)
            for op2 in OPS.keys():
                result = OPS[op2](temp)
                if grid_equal(result, out):
                    return [op1, op2]
    
    # Depth 3
    if max_depth >= 3:
        for op1 in OPS.keys():
            temp1 = OPS[op1](inp)
            for op2 in OPS.keys():
                temp2 = OPS[op2](temp1)
                for op3 in OPS.keys():
                    result = OPS[op3](temp2)
                    if grid_equal(result, out):
                        return [op1, op2, op3]
    
    return None


def find_consistent_program(train_pairs: List[Tuple[Grid, Grid]], max_depth: int = 3) -> Optional[List[str]]:
    """Find program that works on all train pairs"""
    if not train_pairs:
        return None
    
    # Search on first pair
    first_in, first_out = train_pairs[0]
    candidates = []
    
    # Generate candidates from first pair
    for depth in range(1, max_depth + 1):
        if depth == 1:
            for op in OPS.keys():
                result = OPS[op](first_in)
                if grid_equal(result, first_out):
                    candidates.append([op])
        elif depth == 2:
            for op1 in OPS.keys():
                temp = OPS[op1](first_in)
                for op2 in OPS.keys():
                    result = OPS[op2](temp)
                    if grid_equal(result, first_out):
                        candidates.append([op1, op2])
        elif depth == 3:
            for op1 in OPS.keys():
                temp1 = OPS[op1](first_in)
                for op2 in OPS.keys():
                    temp2 = OPS[op2](temp1)
                    for op3 in OPS.keys():
                        result = OPS[op3](temp2)
                        if grid_equal(result, first_out):
                            candidates.append([op1, op2, op3])
    
    # Filter by remaining pairs
    for prog in candidates:
        works = True
        for inp, out in train_pairs[1:]:
            result = apply_sequence(inp, prog)
            if not grid_equal(result, out):
                works = False
                break
        if works:
            return prog
    
    return None


class InvertibleDSL:
    def __init__(self):
        self.max_depth = 3
    
    def predict(self, task) -> List[Grid]:
        train = task.get('train', [])
        test = task.get('test', [])
        
        if not train:
            return [t['input'] for t in test]
        
        # Extract train pairs
        train_pairs = [(ex['input'], ex['output']) for ex in train]
        
        # Find consistent program
        program = find_consistent_program(train_pairs, self.max_depth)
        
        if program is None:
            # Fallback: return input
            return [t['input'] for t in test]
        
        # Apply program to test inputs
        preds = []
        for t in test:
            try:
                result = apply_sequence(t['input'], program)
                preds.append(result)
            except:
                preds.append(t['input'])
        
        return preds