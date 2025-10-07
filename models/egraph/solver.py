"""
E-Graph-Style Solver:
- Generate multiple program candidates
- Apply rewrite rules to canonicalize/deduplicate
- Score by MDL (minimum description length)
- Pick simplest program consistent with train examples
"""
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import itertools

Grid = List[List[int]]


# ============ Basic Operations ============

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


OPS = {
    'id': identity,
    'r90': rotate_90,
    'r180': rotate_180,
    'r270': rotate_270,
    'fh': flip_h,
    'fv': flip_v,
    'tr': transpose,
}


# ============ Program Representation ============

class Program:
    def __init__(self, ops: List[str]):
        self.ops = ops
    
    def __repr__(self):
        return f"Prog({' -> '.join(self.ops)})"
    
    def __hash__(self):
        return hash(tuple(self.ops))
    
    def __eq__(self, other):
        return isinstance(other, Program) and self.ops == other.ops
    
    def apply(self, grid: Grid) -> Grid:
        result = grid
        for op in self.ops:
            result = OPS[op](result)
        return result
    
    def mdl_score(self) -> int:
        """Minimum description length: number of ops"""
        return len(self.ops)


# ============ Rewrite Rules ============

REWRITE_RULES = {
    # Identity elimination
    ('id',): [],
    ('id', 'id'): [],
    # Rotation simplifications
    ('r90', 'r90', 'r90', 'r90'): [],
    ('r90', 'r90'): ['r180'],
    ('r90', 'r90', 'r90'): ['r270'],
    ('r180', 'r180'): [],
    ('r270', 'r90'): [],
    ('r90', 'r270'): [],
    # Flip simplifications
    ('fh', 'fh'): [],
    ('fv', 'fv'): [],
    # Transpose
    ('tr', 'tr'): [],
}


def apply_rewrites(prog: Program) -> Program:
    """Apply rewrite rules to simplify program"""
    ops = prog.ops[:]
    changed = True
    max_iter = 10
    iteration = 0
    
    while changed and iteration < max_iter:
        changed = False
        iteration += 1
        
        # Try to match rewrite rules
        for pattern, replacement in REWRITE_RULES.items():
            pattern_len = len(pattern)
            i = 0
            while i <= len(ops) - pattern_len:
                if tuple(ops[i:i+pattern_len]) == pattern:
                    ops = ops[:i] + list(replacement) + ops[i+pattern_len:]
                    changed = True
                    break
                i += 1
            if changed:
                break
    
    return Program(ops)


def canonicalize_program(prog: Program) -> Program:
    """Apply rewrites to get canonical form"""
    return apply_rewrites(prog)


# ============ Program Search ============

def generate_programs(max_depth: int = 3) -> List[Program]:
    """Generate all programs up to max_depth"""
    programs = []
    
    for depth in range(1, max_depth + 1):
        for ops_tuple in itertools.product(OPS.keys(), repeat=depth):
            prog = Program(list(ops_tuple))
            canonical = canonicalize_program(prog)
            programs.append(canonical)
    
    # Deduplicate
    seen = set()
    unique = []
    for p in programs:
        key = tuple(p.ops)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    return unique


def grid_equal(g1: Grid, g2: Grid) -> bool:
    return g1 == g2


def find_best_program(train_pairs: List[Tuple[Grid, Grid]], max_depth: int = 3) -> Optional[Program]:
    """Find simplest program (by MDL) that fits all train pairs"""
    candidates = generate_programs(max_depth)
    
    # Filter programs that work on all train pairs
    valid = []
    for prog in candidates:
        works = True
        for inp, out in train_pairs:
            try:
                result = prog.apply(inp)
                if not grid_equal(result, out):
                    works = False
                    break
            except:
                works = False
                break
        if works:
            valid.append(prog)
    
    if not valid:
        return None
    
    # Sort by MDL score (shortest first)
    valid.sort(key=lambda p: p.mdl_score())
    return valid[0]


# ============ Solver ============

class EGraphSolverCore:
    def __init__(self):
        self.max_depth = 3
    
    def predict(self, task) -> List[Grid]:
        train = task.get('train', [])
        test = task.get('test', [])
        
        if not train:
            return [t['input'] for t in test]
        
        # Extract train pairs
        train_pairs = [(ex['input'], ex['output']) for ex in train]
        
        # Find best program
        best_prog = find_best_program(train_pairs, self.max_depth)
        
        if best_prog is None:
            # Fallback: return input
            return [t['input'] for t in test]
        
        # Apply to test inputs
        preds = []
        for t in test:
            try:
                result = best_prog.apply(t['input'])
                preds.append(result)
            except:
                preds.append(t['input'])
        
        return preds