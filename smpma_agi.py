#!/usr/bin/env python3
"""
SMPMA: Self-Modifying Program Synthesis with Meta-Learned Abstractions

A novel AGI architecture that learns to expand its own capability space
by discovering and integrating abstractions from successful solutions.

Author: Research Implementation
Date: December 2025
"""

import json
import random
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Dict, Set, Any, Optional
from copy import deepcopy
import time
import inspect

print("=" * 80)
print("SMPMA: SELF-MODIFYING PROGRAM SYNTHESIS WITH META-LEARNED ABSTRACTIONS")
print("=" * 80)

# Configuration (kept global so experiments are reproducible without argparse)
RANDOM_SEED = 42
BEAM_WIDTH = 50
MAX_DEPTH = 4
UTILITY_THRESHOLD = 0.3
MIN_FREQUENCY = 2
COMPRESSION_WEIGHT = 0.5
VERBOSE = True

# Mining / search heuristics
MINE_EVERY_N_TASKS = 5
MINE_MIN_SCORE = 0.70
COMPLEXITY_PENALTY = 0.005  # subtract penalty * program_length from similarity

# Performance
ENABLE_INCREMENTAL_EVAL = True
MAX_OPS_PER_EXPANSION = 250  # prioritize a subset of operations per task for search focus

# Feature toggles (used by ablations/benchmarks)
ENABLE_MACRO_TEMPLATES = True
ENABLE_OP_PRIORITIZATION = True

# Scoring (reduce train overfit)
# - "avg": mean similarity across train examples
# - "minavg": blend of min and mean similarity (robust to overfitting a subset)
SCORING_MODE = "minavg"
MINAVG_ALPHA = 0.65  # weight for min similarity in the blend (rest is mean)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==============================================================================
# PART 1: TYPE SYSTEM
# ==============================================================================

class Type:
    """Type system for programs"""
    GRID = "Grid"
    INT = "Int"
    COLOR = "Color"
    BOOL = "Bool"
    OBJECT = "Object"

# ==============================================================================
# PART 2: PRIMITIVE OPERATIONS
# ==============================================================================

def identity(grid):
    return [row[:] for row in grid]

def flip_v(grid):
    return grid[::-1]

def flip_h(grid):
    return [row[::-1] for row in grid]

def rotate90(grid):
    h, w = len(grid), len(grid[0]) if grid else 0
    return [[grid[h-1-j][i] for j in range(h)] for i in range(w)]

def rotate180(grid):
    return flip_v(flip_h(grid))

def transpose(grid):
    if not grid: return grid
    h, w = len(grid), len(grid[0])
    return [[grid[j][i] for j in range(h)] for i in range(w)]

def replace_color(grid, old_color, new_color):
    return [[new_color if cell == old_color else cell for cell in row] for row in grid]

def keep_color(grid, color):
    return [[cell if cell == color else 0 for cell in row] for row in grid]


def keep_nonzero(grid):
    """Keep all non-zero cells (sets zeros to zero; identity for non-zero)."""
    return [[cell if cell != 0 else 0 for cell in row] for row in grid]

def remove_color(grid, color):
    return [[0 if cell == color else cell for cell in row] for row in grid]

def bbox(grid):
    """Extract bounding box of non-zero elements"""
    if not grid: return grid
    h, w = len(grid), len(grid[0])
    
    min_r, max_r = h, -1
    min_c, max_c = w, -1
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] != 0:
                min_r = min(min_r, i)
                max_r = max(max_r, i)
                min_c = min(min_c, j)
                max_c = max(max_c, j)
    
    if max_r < 0:
        return grid
    
    return [row[min_c:max_c+1] for row in grid[min_r:max_r+1]]

def scale_up(grid, factor):
    """Scale grid by repeating cells"""
    if not grid or factor < 1: return grid
    result = []
    for row in grid:
        new_row = []
        for cell in row:
            new_row.extend([cell] * factor)
        for _ in range(factor):
            result.append(new_row[:])
    return result

def majority_color(grid):
    """Fill grid with most common non-zero color"""
    if not grid: return grid
    flat = [cell for row in grid for cell in row if cell != 0]
    if not flat: return grid
    
    counter = Counter(flat)
    maj_color = counter.most_common(1)[0][0]
    
    h, w = len(grid), len(grid[0])
    return [[maj_color] * w for _ in range(h)]

def gravity_down(grid):
    """Apply gravity - non-zero cells fall down"""
    if not grid: return grid
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    
    for j in range(w):
        non_zero = [grid[i][j] for i in range(h) if grid[i][j] != 0]
        for i, val in enumerate(non_zero):
            result[h - len(non_zero) + i][j] = val
    
    return result


def _neighbors4(i: int, j: int):
    return ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))


def keep_largest_component(grid):
    """
    Keep only the largest 4-connected non-zero component (all colors treated as foreground).
    Returns a grid of the same size.
    """
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    best = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 or seen[i][j]:
                continue
            comp = []
            stack = [(i, j)]
            seen[i][j] = True
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for nx, ny in _neighbors4(x, y):
                    if 0 <= nx < h and 0 <= ny < w and (not seen[nx][ny]) and grid[nx][ny] != 0:
                        seen[nx][ny] = True
                        stack.append((nx, ny))
            if len(comp) > len(best):
                best = comp
    out = [[0] * w for _ in range(h)]
    for x, y in best:
        out[x][y] = grid[x][y]
    return out


def center_bbox(grid):
    """
    Translate all non-zero cells so the overall bounding box is centered in the grid.
    Keeps grid size; out-of-bounds cells are clipped.
    """
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    min_r, max_r = h, -1
    min_c, max_c = w, -1
    for i in range(h):
        for j in range(w):
            if grid[i][j] != 0:
                min_r = min(min_r, i)
                max_r = max(max_r, i)
                min_c = min(min_c, j)
                max_c = max(max_c, j)
    if max_r < 0:
        return grid
    box_h = max_r - min_r + 1
    box_w = max_c - min_c + 1
    target_min_r = (h - box_h) // 2
    target_min_c = (w - box_w) // 2
    dr = target_min_r - min_r
    dc = target_min_c - min_c
    out = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            v = grid[i][j]
            if v == 0:
                continue
            ni, nj = i + dr, j + dc
            if 0 <= ni < h and 0 <= nj < w:
                out[ni][nj] = v
    return out


def keep_largest_component_of_color(grid, color):
    """Keep the largest 4-connected component of a specific color."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    best = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] != color or seen[i][j]:
                continue
            comp = []
            stack = [(i, j)]
            seen[i][j] = True
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for nx, ny in _neighbors4(x, y):
                    if 0 <= nx < h and 0 <= ny < w and (not seen[nx][ny]) and grid[nx][ny] == color:
                        seen[nx][ny] = True
                        stack.append((nx, ny))
            if len(comp) > len(best):
                best = comp
    out = [[0] * w for _ in range(h)]
    for x, y in best:
        out[x][y] = color
    return out


def translate(grid, dr, dc):
    """Translate all non-zero cells by (dr, dc) within bounds, preserving size."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    out = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            v = grid[i][j]
            if v == 0:
                continue
            ni, nj = i + dr, j + dc
            if 0 <= ni < h and 0 <= nj < w:
                out[ni][nj] = v
    return out


def move_bbox_to_topleft(grid):
    """Translate non-zero bounding box to the top-left corner."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    min_r, max_r = h, -1
    min_c, max_c = w, -1
    for i in range(h):
        for j in range(w):
            if grid[i][j] != 0:
                min_r = min(min_r, i)
                max_r = max(max_r, i)
                min_c = min(min_c, j)
                max_c = max(max_c, j)
    if max_r < 0:
        return grid
    return translate(grid, -min_r, -min_c)


def move_bbox_to_bottomright(grid):
    """Translate non-zero bounding box to the bottom-right corner."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    min_r, max_r = h, -1
    min_c, max_c = w, -1
    for i in range(h):
        for j in range(w):
            if grid[i][j] != 0:
                min_r = min(min_r, i)
                max_r = max(max_r, i)
                min_c = min(min_c, j)
                max_c = max(max_c, j)
    if max_r < 0:
        return grid
    dr = (h - 1) - max_r
    dc = (w - 1) - max_c
    return translate(grid, dr, dc)


def overlay_current_on_input(current, original):
    """
    Overlay: keep original background, but overwrite with non-zero cells from current.
    Output size matches original.
    """
    if not original:
        return current
    h, w = len(original), len(original[0])
    out = [row[:] for row in original]
    if not current or len(current) != h or len(current[0]) != w:
        return out
    for i in range(h):
        for j in range(w):
            v = current[i][j]
            if v != 0:
                out[i][j] = v
    return out


def overlay_input_on_current(current, original):
    """
    Overlay: keep current background, but overwrite with non-zero cells from original.
    Output size matches current.
    """
    if not current:
        return original
    h, w = len(current), len(current[0])
    out = [row[:] for row in current]
    if not original or len(original) != h or len(original[0]) != w:
        return out
    for i in range(h):
        for j in range(w):
            v = original[i][j]
            if v != 0:
                out[i][j] = v
    return out


def xor_nonzero(current, original):
    """
    Return non-zero cells where current and original differ.
    This is useful for extracting a 'delta' object after transforming the input.
    """
    if not original:
        return current
    h, w = len(original), len(original[0])
    if not current or len(current) != h or len(current[0]) != w:
        return [[0] * w for _ in range(h)]
    out = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            a = current[i][j]
            b = original[i][j]
            if a != b and a != 0:
                out[i][j] = a
    return out


def keep_current_where_input_zero(current, original):
    """Keep current's non-zero cells only where original is zero."""
    if not original:
        return current
    h, w = len(original), len(original[0])
    if not current or len(current) != h or len(current[0]) != w:
        return [[0] * w for _ in range(h)]
    out = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if original[i][j] == 0 and current[i][j] != 0:
                out[i][j] = current[i][j]
    return out


def keep_current_where_input_nonzero(current, original):
    """Keep current's non-zero cells only where original is non-zero."""
    if not original:
        return current
    h, w = len(original), len(original[0])
    if not current or len(current) != h or len(current[0]) != w:
        return [[0] * w for _ in range(h)]
    out = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if original[i][j] != 0 and current[i][j] != 0:
                out[i][j] = current[i][j]
    return out


def obj_from_canvas_largest_component(canvas):
    """Extract object: largest component of the canvas (same-size mask with original colors)."""
    return keep_largest_component(canvas)


def obj_from_canvas_largest_component_color(canvas, color):
    """Extract object: largest component of a specific color from the canvas."""
    return keep_largest_component_of_color(canvas, color)


def obj_bbox(obj):
    """Transform object: crop to its bounding box."""
    return bbox(obj)


def obj_translate(obj, dr, dc):
    """Transform object: translate within its own grid (same-size)."""
    return translate(obj, dr, dc)


def paste_obj_on_canvas(canvas, obj):
    """Paste: overwrite canvas with non-zero obj cells."""
    if obj is None:
        return canvas
    # overlay_current_on_input expects (current, original)
    return overlay_current_on_input(obj, canvas)


def paste_obj_where_canvas_zero(canvas, obj):
    """Paste object only where canvas is zero."""
    if not canvas:
        return obj
    h, w = len(canvas), len(canvas[0])
    if not obj or len(obj) != h or len(obj[0]) != w:
        return canvas
    out = [row[:] for row in canvas]
    for i in range(h):
        for j in range(w):
            if canvas[i][j] == 0 and obj[i][j] != 0:
                out[i][j] = obj[i][j]
    return out


def paste_obj_where_canvas_nonzero(canvas, obj):
    """Paste object only where canvas is non-zero."""
    if not canvas:
        return obj
    h, w = len(canvas), len(canvas[0])
    if not obj or len(obj) != h or len(obj[0]) != w:
        return canvas
    out = [row[:] for row in canvas]
    for i in range(h):
        for j in range(w):
            if canvas[i][j] != 0 and obj[i][j] != 0:
                out[i][j] = obj[i][j]
    return out

# ==============================================================================
# PART 3: PROGRAM REPRESENTATION
# ==============================================================================

@dataclass
class Operation:
    """A single operation in a program"""
    name: str
    func: Callable
    args: List[Any] = field(default_factory=list)
    return_type: str = Type.GRID
    param_types: List[str] = field(default_factory=list)
    n_inputs: int = 1
    
    def apply(self, *inputs):
        """Apply this operation. Failures fall back to the first input."""
        try:
            ins = inputs[: self.n_inputs] if self.n_inputs > 0 else ()
            if self.args:
                return self.func(*ins, *self.args)
            return self.func(*ins)
        except Exception:
            return inputs[0] if inputs else None
    
    def __str__(self):
        args_str = ', '.join(map(str, self.args))
        return f"{self.name}({args_str})" if self.args else f"{self.name}()"
    
    def __hash__(self):
        return hash((self.name, tuple(self.args)))
    
    def __eq__(self, other):
        return self.name == other.name and self.args == other.args

@dataclass
class Program:
    """A sequence of operations"""
    operations: List[Operation] = field(default_factory=list)
    
    def apply(self, grid):
        """
        Execute program on an input grid.

        Scratchpad execution model:
          - canvas: current working grid
          - obj: an auxiliary grid extracted from / derived from the canvas
          - original: immutable copy of the input for 2-input operators

        Most operators transform the canvas. Operators with names starting with:
          - obj_from_canvas_* : set/update obj from the current canvas
          - obj_*             : transform obj (if obj is None, uses canvas as fallback)
          - paste_obj_*       : write obj back onto the canvas
          - swap_canvas_obj   : swap canvas and obj
        """
        original = grid
        canvas = grid
        obj = None
        for op in self.operations:
            name = op.name

            # Obj extraction from canvas
            if name.startswith("obj_from_canvas_"):
                obj = op.apply(canvas)
                continue

            # Obj-only ops
            if name.startswith("obj_"):
                base = obj if obj is not None else canvas
                obj = op.apply(base)
                continue

            # Paste obj to canvas
            if name.startswith("paste_obj_"):
                if obj is None:
                    continue
                canvas = op.apply(canvas, obj)
                if canvas is None:
                    canvas = grid
                continue

            if name == "swap_canvas_obj":
                if obj is None:
                    obj = canvas
                canvas, obj = obj, canvas
                continue

            # Default: canvas transforms. Always provide (canvas, original) to support 2-input ops.
            canvas = op.apply(canvas, original)
            if canvas is None:
                return grid
        return canvas
    
    def __str__(self):
        return " -> ".join(str(op) for op in self.operations)
    
    def __len__(self):
        return len(self.operations)
    
    def __hash__(self):
        return hash(tuple(self.operations))
    
    def sub_programs(self, min_len=2, max_len=4):
        """Extract all sub-programs of given length range"""
        subs = []
        n = len(self.operations)
        for length in range(min_len, min(max_len + 1, n + 1)):
            for i in range(n - length + 1):
                subs.append(Program(operations=self.operations[i:i+length]))
        return subs

    def signature(self) -> Tuple[Tuple[str, Tuple[Any, ...]], ...]:
        """Stable, hashable signature for mining and deduplication."""
        return tuple((op.name, tuple(op.args)) for op in self.operations)

    def kind_signature(self) -> Tuple[str, ...]:
        """Kind-only signature that ignores concrete arguments (used for macro mining)."""
        kinds: List[str] = []
        for op in self.operations:
            if op.name.startswith("replace_"):
                kinds.append("replace")
            elif op.name.startswith("keep_"):
                kinds.append("keep")
            elif op.name.startswith("remove_"):
                kinds.append("remove")
            elif op.name.startswith("scale_"):
                kinds.append("scale")
            elif op.name.startswith("translate_"):
                kinds.append("translate")
            elif op.name.startswith("obj_from_canvas_"):
                kinds.append("obj_from_canvas")
            elif op.name.startswith("obj_"):
                kinds.append("obj")
            elif op.name.startswith("paste_obj_"):
                kinds.append("paste_obj")
            else:
                kinds.append(op.name)
        return tuple(kinds)

# ==============================================================================
# PART 4: ABSTRACTION
# ==============================================================================

@dataclass
class Abstraction:
    """A learned abstraction (new operation)"""
    name: str
    parameters: List[Tuple[str, str]]  # (name, type) pairs
    body: Program
    utility: float = 0.0
    tasks_used: Set[str] = field(default_factory=set)
    frequency: int = 0
    compression_ratio: float = 1.0
    
    def to_operation(self, *args) -> Operation:
        """Convert to an executable macro-operation (currently parameter-free)."""
        def execute(grid):
            result = grid
            for op in self.body.operations:
                result = op.apply(result)
                if result is None:
                    return grid
            return result
        
        return Operation(
            name=self.name,
            func=execute,
            args=list(args),
            return_type=Type.GRID
        )
    
    def __str__(self):
        params_str = ', '.join(f"{name}:{typ}" for name, typ in self.parameters)
        return f"{self.name}({params_str}) = {self.body}"
    
    def __hash__(self):
        return hash((self.name, str(self.body)))


@dataclass
class MacroTemplate:
    """
    A mined macro template that can be instantiated into concrete operations for a task.

    This is the key mechanism that makes mining useful on ARC: patterns repeat across tasks,
    but color parameters typically differ. Templates capture the structure, and instantiation
    uses task-specific color evidence.
    """
    name: str
    kinds: Tuple[str, ...]  # e.g. ("replace", "replace") or ("keep", "flip_h")
    frequency: int
    tasks_used: Set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.kinds)


@dataclass(frozen=True)
class ParamStep:
    """A single step in a parameterized macro template."""
    kind: str  # e.g. replace/keep/remove/scale/flip_h/bbox/...
    args: Tuple[Any, ...] = ()  # each entry is either ('var', idx) or an int constant


@dataclass
class ParamMacroTemplate:
    """
    Parameterized macro template with shared variables across steps.

    Example:
      replace(var0 -> var2) ; replace(var1 -> var2)
    captures "map multiple colors into one" independent of which colors those are.
    """
    name: str
    steps: Tuple[ParamStep, ...]
    frequency: int
    tasks_used: Set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.steps)

# ==============================================================================
# PART 5: DSL (DOMAIN-SPECIFIC LANGUAGE)
# ==============================================================================

class DSL:
    """Dynamic Domain-Specific Language with learned abstractions"""
    
    def __init__(self):
        self.primitives = self._initialize_primitives()
        self.abstractions: Dict[str, Abstraction] = {}
        self.macros: Dict[str, MacroTemplate] = {}
        self.param_macros: Dict[str, ParamMacroTemplate] = {}
        self.task_history: List[Dict] = []
        
        if VERBOSE:
            print(f"\nInitialized DSL with {len(self.primitives)} primitives")
    
    def _initialize_primitives(self) -> List[Operation]:
        """Create initial set of primitive operations"""
        ops = [
            Operation("identity", identity, [], Type.GRID),
            Operation("flip_v", flip_v, [], Type.GRID),
            Operation("flip_h", flip_h, [], Type.GRID),
            Operation("rotate90", rotate90, [], Type.GRID),
            Operation("rotate180", rotate180, [], Type.GRID),
            Operation("transpose", transpose, [], Type.GRID),
            Operation("bbox", bbox, [], Type.GRID),
            Operation("majority_color", majority_color, [], Type.GRID),
            Operation("gravity_down", gravity_down, [], Type.GRID),
            Operation("keep_largest_component", keep_largest_component, [], Type.GRID),
            Operation("center_bbox", center_bbox, [], Type.GRID),
            Operation("keep_nonzero", keep_nonzero, [], Type.GRID),
            Operation("move_bbox_to_topleft", move_bbox_to_topleft, [], Type.GRID),
            Operation("move_bbox_to_bottomright", move_bbox_to_bottomright, [], Type.GRID),
            Operation("overlay_current_on_input", overlay_current_on_input, [], Type.GRID, n_inputs=2),
            Operation("overlay_input_on_current", overlay_input_on_current, [], Type.GRID, n_inputs=2),
            Operation("xor_nonzero", xor_nonzero, [], Type.GRID, n_inputs=2),
            Operation("keep_current_where_input_zero", keep_current_where_input_zero, [], Type.GRID, n_inputs=2),
            Operation("keep_current_where_input_nonzero", keep_current_where_input_nonzero, [], Type.GRID, n_inputs=2),
            # Scratchpad: object extraction / transform / paste
            Operation("obj_from_canvas_largest_component", obj_from_canvas_largest_component, [], Type.GRID, n_inputs=1),
            Operation("obj_bbox", obj_bbox, [], Type.GRID, n_inputs=1),
            Operation("paste_obj_on_canvas", paste_obj_on_canvas, [], Type.GRID, n_inputs=2),
            Operation("paste_obj_where_canvas_zero", paste_obj_where_canvas_zero, [], Type.GRID, n_inputs=2),
            Operation("paste_obj_where_canvas_nonzero", paste_obj_where_canvas_nonzero, [], Type.GRID, n_inputs=2),
            Operation("swap_canvas_obj", lambda a: a, [], Type.GRID, n_inputs=1),
        ]
        
        # Add parameterized operations
        for c1 in range(10):
            for c2 in range(10):
                if c1 != c2:
                    ops.append(Operation(f"replace_{c1}_to_{c2}", replace_color, [c1, c2], Type.GRID))
        
        for c in range(10):
            ops.append(Operation(f"keep_{c}", keep_color, [c], Type.GRID))
            ops.append(Operation(f"remove_{c}", remove_color, [c], Type.GRID))
            ops.append(Operation(f"keep_largest_comp_{c}", keep_largest_component_of_color, [c], Type.GRID))
            ops.append(Operation(f"obj_from_canvas_largest_comp_{c}", obj_from_canvas_largest_component_color, [c], Type.GRID, n_inputs=1))
        
        for f in [2, 3]:
            ops.append(Operation(f"scale_{f}x", scale_up, [f], Type.GRID))

        # Small translations (kept intentionally small for tractable search)
        for dr in (-2, -1, 1, 2):
            for dc in (-2, -1, 1, 2):
                ops.append(Operation(f"translate_{dr}_{dc}", translate, [dr, dc], Type.GRID))
                ops.append(Operation(f"obj_translate_{dr}_{dc}", obj_translate, [dr, dc], Type.GRID))
        
        return ops
    
    def all_operations(self) -> List[Operation]:
        """Get all available operations (primitives + abstractions)"""
        ops = self.primitives[:]
        for _, abstraction in self.abstractions.items():
            ops.append(abstraction.to_operation())
        return ops

    def all_macros(self) -> List[MacroTemplate]:
        return list(self.macros.values())

    def all_param_macros(self) -> List[ParamMacroTemplate]:
        return list(self.param_macros.values())
    
    def add_abstraction(self, abstraction: Abstraction):
        """Add a new abstraction to the DSL"""
        # Avoid accidental name collisions.
        if abstraction.name in self.abstractions:
            suffix = 2
            while f"{abstraction.name}_{suffix}" in self.abstractions:
                suffix += 1
            abstraction.name = f"{abstraction.name}_{suffix}"
        self.abstractions[abstraction.name] = abstraction
        if VERBOSE:
            print(f"  Added abstraction: {abstraction.name} (utility={abstraction.utility:.3f})")

    def add_macro(self, macro: MacroTemplate):
        if macro.name in self.macros:
            suffix = 2
            while f"{macro.name}_{suffix}" in self.macros:
                suffix += 1
            macro.name = f"{macro.name}_{suffix}"
        self.macros[macro.name] = macro
        if VERBOSE:
            print(f"  Added macro: {macro.name} kinds={macro.kinds} freq={macro.frequency}")

    def add_param_macro(self, macro: ParamMacroTemplate):
        if macro.name in self.param_macros:
            suffix = 2
            while f"{macro.name}_{suffix}" in self.param_macros:
                suffix += 1
            macro.name = f"{macro.name}_{suffix}"
        self.param_macros[macro.name] = macro
        if VERBOSE:
            step_desc = "; ".join([f"{s.kind}{s.args}" for s in macro.steps])
            print(f"  Added param-macro: {macro.name} steps={step_desc} freq={macro.frequency}")
    
    def prune_abstractions(self, min_utility=UTILITY_THRESHOLD):
        """Remove low-utility abstractions"""
        to_remove = [name for name, abs_obj in self.abstractions.items()
                     if abs_obj.utility < min_utility]
        for name in to_remove:
            del self.abstractions[name]
        if VERBOSE and to_remove:
            print(f"  Pruned {len(to_remove)} low-utility abstractions")
    
    def summary(self):
        """Print DSL statistics"""
        print(f"\n{'='*80}")
        print("DSL SUMMARY")
        print(f"{'='*80}")
        print(f"Primitives: {len(self.primitives)}")
        print(f"Learned Abstractions: {len(self.abstractions)}")
        print(f"Learned Macros: {len(self.macros)}")
        print(f"Learned Param-Macros: {len(self.param_macros)}")
        print(f"Total Operations: {len(self.all_operations())}")
        
        if self.abstractions:
            print(f"\nTop Abstractions:")
            sorted_abs = sorted(self.abstractions.values(), 
                               key=lambda a: a.utility, reverse=True)
            for abs in sorted_abs[:10]:
                print(f"  • {abs.name}: utility={abs.utility:.3f}, freq={abs.frequency}, tasks={len(abs.tasks_used)}")

        if self.macros:
            print("\nTop Macros:")
            sorted_macros = sorted(self.macros.values(), key=lambda m: (m.frequency, len(m.tasks_used)), reverse=True)
            for m in sorted_macros[:10]:
                print(f"  • {m.name}: kinds={m.kinds}, freq={m.frequency}, tasks={len(m.tasks_used)}")

        if self.param_macros:
            print("\nTop Param-Macros:")
            sorted_pm = sorted(self.param_macros.values(), key=lambda m: (m.frequency, len(m.tasks_used)), reverse=True)
            for m in sorted_pm[:10]:
                print(f"  • {m.name}: freq={m.frequency}, tasks={len(m.tasks_used)} steps={len(m.steps)}")

# ==============================================================================
# PART 6: PROGRAM SYNTHESIZER
# ==============================================================================

class ProgramSynthesizer:
    """Synthesize programs using beam search"""
    
    def __init__(self, dsl: DSL, beam_width=BEAM_WIDTH, max_depth=MAX_DEPTH):
        self.dsl = dsl
        self.beam_width = beam_width
        self.max_depth = max_depth
        self._eval_cache: Dict[Tuple[Tuple[Tuple[int, ...], ...], Program], float] = {}
    
    def synthesize(self, task_examples: List[Tuple], task_id: str) -> Tuple[Optional[Program], float]:
        """
        Synthesize a program from input-output examples
        
        Args:
            task_examples: List of (input_grid, output_grid) pairs
            task_id: Unique identifier for this task
        
        Returns:
            (best_program, best_score)
        """
        if not task_examples:
            return None, 0.0
        
        # Precompute inputs/outputs and hashable keys for caching.
        inputs = [inp for (inp, _) in task_examples]
        targets = [out for (_, out) in task_examples]

        # Build a prioritized operation list for this task (still uses the DSL, but focuses the search).
        ops = self._prioritized_ops(task_examples) if ENABLE_OP_PRIORITIZATION else self.dsl.all_operations()
        macros = self.dsl.all_macros() if ENABLE_MACRO_TEMPLATES else []
        p_macros = self.dsl.all_param_macros() if ENABLE_MACRO_TEMPLATES else []

        # Beam stores: (program, score, outputs_for_each_example)
        empty = Program()
        empty_outputs = inputs[:]  # identity
        empty_score = self._evaluate_from_outputs(empty, empty_outputs, targets)
        beam = [(empty, empty_score, empty_outputs)]
        best_program, best_score, _ = beam[0]
        
        for depth in range(1, self.max_depth + 1):
            candidates = []
            
            # Expand each program in beam
            for program, _, outputs in beam:
                # Primitive operations
                for op in ops:
                    new_program = Program(operations=program.operations + [op])
                    if ENABLE_INCREMENTAL_EVAL:
                        new_outputs = [op.apply(g) for g in outputs]
                        new_score = self._evaluate_from_outputs(new_program, new_outputs, targets)
                    else:
                        new_score = self._evaluate(new_program, task_examples)
                        new_outputs = None
                    candidates.append((new_program, new_score, new_outputs))

                # Macro templates (instantiate to small set of concrete sequences)
                for macro in macros:
                    instantiations = self._instantiate_macro(macro, task_examples, max_instantiations=12)
                    for macro_ops in instantiations:
                        new_program = Program(operations=program.operations + macro_ops)
                        if ENABLE_INCREMENTAL_EVAL:
                            new_outputs = outputs
                            for mop in macro_ops:
                                new_outputs = [mop.apply(g) for g in new_outputs]
                            new_score = self._evaluate_from_outputs(new_program, new_outputs, targets)
                        else:
                            new_score = self._evaluate(new_program, task_examples)
                            new_outputs = None
                        candidates.append((new_program, new_score, new_outputs))

                # Parameterized macro templates (variable binding per task)
                for pm in p_macros:
                    instantiations = self._instantiate_param_macro(pm, task_examples, max_instantiations=12)
                    for macro_ops in instantiations:
                        new_program = Program(operations=program.operations + macro_ops)
                        if ENABLE_INCREMENTAL_EVAL:
                            new_outputs = outputs
                            for mop in macro_ops:
                                new_outputs = [mop.apply(g) for g in new_outputs]
                            new_score = self._evaluate_from_outputs(new_program, new_outputs, targets)
                        else:
                            new_score = self._evaluate(new_program, task_examples)
                            new_outputs = None
                        candidates.append((new_program, new_score, new_outputs))
            
            # Keep top-K candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.beam_width]
            
            # Track best
            if beam and beam[0][1] > best_score:
                best_program, best_score, _ = beam[0]
                
                # Early stopping should only trigger on essentially perfect training fit.
                # ARC-style tasks often require deeper composition even when pixel similarity is already high.
                if best_score >= 0.999:
                    if VERBOSE:
                        print(f"    [depth {depth}] Found perfect-fit solution: {best_program} (score={best_score:.3f})")
                    break
        
        return best_program, best_score

    def _prioritized_ops(self, task_examples: List[Tuple]) -> List[Operation]:
        """
        Prioritize a subset of operations using cheap, task-local evidence.
        This improves beam efficiency: most ARC tasks only need a few relevant operations.
        """
        all_ops = self.dsl.all_operations()

        # Add a few task-conditioned translate operations inferred from bbox shifts.
        # This avoids blowing up the global DSL while still allowing useful moves.
        inferred_translates: List[Operation] = []
        try:
            shifts = Counter()
            for inp, out in task_examples:
                if not inp or not out:
                    continue
                # only compare if same shape (most translate tasks keep shape)
                if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                    continue
                h, w = len(inp), len(inp[0])
                # bbox on inp
                def bb(g):
                    mr, Mr = h, -1
                    mc, Mc = w, -1
                    for i in range(h):
                        for j in range(w):
                            if g[i][j] != 0:
                                mr = min(mr, i); Mr = max(Mr, i)
                                mc = min(mc, j); Mc = max(Mc, j)
                    return (mr, Mr, mc, Mc)
                ib = bb(inp)
                ob = bb(out)
                if ib[1] < 0 or ob[1] < 0:
                    continue
                # align bbox top-left as a translation hypothesis
                dr = ob[0] - ib[0]
                dc = ob[2] - ib[2]
                shifts[(dr, dc)] += 1
                # also align bbox bottom-right
                dr2 = ob[1] - ib[1]
                dc2 = ob[3] - ib[3]
                shifts[(dr2, dc2)] += 1
            for (dr, dc), _ in shifts.most_common(6):
                if dr == 0 and dc == 0:
                    continue
                inferred_translates.append(Operation(f"translate_{dr}_{dc}", translate, [dr, dc], Type.GRID))
        except Exception:
            inferred_translates = []

        all_ops = all_ops + inferred_translates
        if len(all_ops) <= MAX_OPS_PER_EXPANSION:
            return all_ops

        # Evidence: color mappings observed in train diffs.
        mapping_counts = Counter()
        colors_in = Counter()
        colors_out = Counter()
        for inp, out in task_examples:
            for row in inp:
                colors_in.update([c for c in row if c != 0])
            for row in out:
                colors_out.update([c for c in row if c != 0])
            if inp and out and len(inp) == len(out) and len(inp[0]) == len(out[0]):
                for i in range(len(inp)):
                    for j in range(len(inp[0])):
                        a, b = inp[i][j], out[i][j]
                        if a != b:
                            mapping_counts[(a, b)] += 1

        likely_replaces = set()
        for (a, b), _ in mapping_counts.most_common(20):
            if a != b:
                likely_replaces.add((a, b))

        likely_colors = set([c for c, _ in colors_in.most_common(5)] + [c for c, _ in colors_out.most_common(5)])

        def op_priority(op: Operation) -> float:
            # Prefer structural ops early
            if op.name in {"bbox", "flip_h", "flip_v", "rotate90", "rotate180", "transpose", "gravity_down", "keep_largest_component", "center_bbox", "keep_nonzero"}:
                return 3.0
            if op.name in {"move_bbox_to_topleft", "move_bbox_to_bottomright"}:
                return 2.8
            if op.name.startswith("keep_largest_comp_"):
                return 2.2
            if op.name.startswith("translate_"):
                return 2.0
            if op.name in {"overlay_current_on_input", "overlay_input_on_current", "xor_nonzero", "keep_current_where_input_zero", "keep_current_where_input_nonzero"}:
                return 2.6
            if op.name.startswith("obj_from_canvas_") or op.name.startswith("paste_obj_") or op.name.startswith("obj_") or op.name == "swap_canvas_obj":
                return 2.7
            if op.name.startswith("replace_") and len(op.args) == 2 and tuple(op.args) in likely_replaces:
                return 5.0
            if op.name.startswith("keep_") and len(op.args) == 1 and op.args[0] in likely_colors:
                return 4.0
            if op.name.startswith("remove_") and len(op.args) == 1 and op.args[0] in likely_colors:
                return 2.5
            if op.name.startswith("scale_"):
                return 1.5
            return 0.1

        scored = [(op_priority(op), op) for op in all_ops]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [op for _, op in scored[:MAX_OPS_PER_EXPANSION]]

    def _instantiate_macro(self, macro: MacroTemplate, task_examples: List[Tuple], max_instantiations: int = 10) -> List[List[Operation]]:
        """
        Instantiate a macro template into concrete operations for the current task.
        Currently supports:
          - 'replace' kind: choose top color mappings from train diffs
          - other kinds: map directly to existing primitive ops without args (best-effort)
        """
        # Color mapping evidence
        mapping_counts = Counter()
        colors_in = Counter()
        colors_out = Counter()
        for inp, out in task_examples:
            for row in inp:
                colors_in.update([c for c in row if c != 0])
            for row in out:
                colors_out.update([c for c in row if c != 0])
            if inp and out and len(inp) == len(out) and len(inp[0]) == len(out[0]):
                for i in range(len(inp)):
                    for j in range(len(inp[0])):
                        a, b = inp[i][j], out[i][j]
                        if a != b:
                            mapping_counts[(a, b)] += 1
        top_maps = [m for (m, _) in mapping_counts.most_common(12) if m[0] != m[1]]
        top_in = [c for (c, _) in colors_in.most_common(8)]
        top_out = [c for (c, _) in colors_out.most_common(8)]

        # Primitive lookup
        name_to_op = {op.name: op for op in self.dsl.all_operations()}

        instantiations: List[List[Operation]] = []

        # Build candidate lists per macro slot
        slot_candidates: List[List[Operation]] = []
        for kind in macro.kinds:
            if kind == "replace":
                ops = []
                for (a, b) in top_maps[:8]:
                    op_name = f"replace_{a}_to_{b}"
                    if op_name in name_to_op:
                        ops.append(name_to_op[op_name])
                slot_candidates.append(ops if ops else [])
            elif kind == "keep":
                ops = []
                for c in (top_out + top_in)[:8]:
                    op_name = f"keep_{c}"
                    if op_name in name_to_op:
                        ops.append(name_to_op[op_name])
                slot_candidates.append(ops if ops else [])
            elif kind == "remove":
                ops = []
                for c in (top_in + top_out)[:8]:
                    op_name = f"remove_{c}"
                    if op_name in name_to_op:
                        ops.append(name_to_op[op_name])
                slot_candidates.append(ops if ops else [])
            elif kind == "scale":
                ops = []
                for nm in ("scale_2x", "scale_3x"):
                    if nm in name_to_op:
                        ops.append(name_to_op[nm])
                slot_candidates.append(ops if ops else [])
            else:
                # try exact name kinds like flip_h, bbox, etc.
                if kind in name_to_op:
                    slot_candidates.append([name_to_op[kind]])
                else:
                    slot_candidates.append([])

        # Cartesian product with early cutoff
        def dfs(idx: int, cur: List[Operation]):
            if len(instantiations) >= max_instantiations:
                return
            if idx == len(slot_candidates):
                instantiations.append(cur[:])
                return
            cands = slot_candidates[idx]
            if not cands:
                return
            for op in cands:
                cur.append(op)
                dfs(idx + 1, cur)
                cur.pop()

        dfs(0, [])
        return instantiations

    def _instantiate_param_macro(self, pm: ParamMacroTemplate, task_examples: List[Tuple], max_instantiations: int = 10) -> List[List[Operation]]:
        """
        Instantiate a parameterized macro by binding its variables to candidate colors.
        Strategy:
          - Build candidate color pools from input/output palettes and observed diffs.
          - Enumerate a limited set of variable assignments and materialize the ops.
        """
        mapping_counts = Counter()
        colors_in = Counter()
        colors_out = Counter()
        for inp, out in task_examples:
            for row in inp:
                colors_in.update([c for c in row if c != 0])
            for row in out:
                colors_out.update([c for c in row if c != 0])
            if inp and out and len(inp) == len(out) and len(inp[0]) == len(out[0]):
                for i in range(len(inp)):
                    for j in range(len(inp[0])):
                        a, b = inp[i][j], out[i][j]
                        if a != b:
                            mapping_counts[(a, b)] += 1

        top_maps = [m for (m, _) in mapping_counts.most_common(16) if m[0] != m[1]]
        top_in = [c for (c, _) in colors_in.most_common(8)]
        top_out = [c for (c, _) in colors_out.most_common(8)]
        all_cands = list(dict.fromkeys(top_out + top_in + [a for a, _ in top_maps] + [b for _, b in top_maps]))
        if not all_cands:
            all_cands = list(range(10))

        # Determine number of variables
        var_ids = set()
        for step in pm.steps:
            for a in step.args:
                if isinstance(a, tuple) and len(a) == 2 and a[0] == "var":
                    var_ids.add(a[1])
        if not var_ids:
            return []
        nvars = max(var_ids) + 1

        # Primitive lookup
        name_to_op = {op.name: op for op in self.dsl.all_operations()}

        # Candidate assignment lists per var (keep small)
        per_var = []
        for vid in range(nvars):
            per_var.append(all_cands[:6])

        scored_instantiations: List[Tuple[float, List[Operation]]] = []

        def score_ops(ops: List[Operation]) -> float:
            # Evaluate this short sequence on train examples and return mean similarity.
            total = 0.0
            for inp, out in task_examples:
                g = inp
                for op in ops:
                    g = op.apply(g, inp)
                total += self._similarity(g, out)
            return total / max(1, len(task_examples))

        def materialize(assignment: List[int]) -> Optional[List[Operation]]:
            ops: List[Operation] = []
            for step in pm.steps:
                if step.kind == "replace":
                    src = assignment[step.args[0][1]] if isinstance(step.args[0], tuple) else step.args[0]
                    dst = assignment[step.args[1][1]] if isinstance(step.args[1], tuple) else step.args[1]
                    nm = f"replace_{src}_to_{dst}"
                    if nm not in name_to_op:
                        return None
                    ops.append(name_to_op[nm])
                elif step.kind == "keep":
                    c = assignment[step.args[0][1]] if isinstance(step.args[0], tuple) else step.args[0]
                    nm = f"keep_{c}"
                    if nm not in name_to_op:
                        return None
                    ops.append(name_to_op[nm])
                elif step.kind == "remove":
                    c = assignment[step.args[0][1]] if isinstance(step.args[0], tuple) else step.args[0]
                    nm = f"remove_{c}"
                    if nm not in name_to_op:
                        return None
                    ops.append(name_to_op[nm])
                elif step.kind == "scale":
                    f = step.args[0]
                    nm = f"scale_{f}x"
                    if nm not in name_to_op:
                        return None
                    ops.append(name_to_op[nm])
                elif step.kind == "translate":
                    dr, dc = step.args
                    nm = f"translate_{dr}_{dc}"
                    if nm not in name_to_op:
                        return None
                    ops.append(name_to_op[nm])
                elif step.kind == "obj_from_canvas":
                    # stored as concrete op args, so use name_to_op lookup by matching args
                    # support: obj_from_canvas_largest_component and obj_from_canvas_largest_comp_{c}
                    if len(step.args) == 0:
                        nm = "obj_from_canvas_largest_component"
                    else:
                        # assume (color,)
                        c = step.args[0]
                        nm = f"obj_from_canvas_largest_comp_{c}"
                    if nm not in name_to_op:
                        return None
                    ops.append(name_to_op[nm])
                elif step.kind == "obj_bbox":
                    if "obj_bbox" not in name_to_op:
                        return None
                    ops.append(name_to_op["obj_bbox"])
                elif step.kind == "obj_translate":
                    dr, dc = step.args
                    nm = f"obj_translate_{dr}_{dc}"
                    if nm not in name_to_op:
                        return None
                    ops.append(name_to_op[nm])
                elif step.kind == "paste_obj":
                    # args = (op_name,)
                    nm = step.args[0] if step.args else "paste_obj_on_canvas"
                    if nm not in name_to_op:
                        return None
                    ops.append(name_to_op[nm])
                elif step.kind == "swap_canvas_obj":
                    if "swap_canvas_obj" not in name_to_op:
                        return None
                    ops.append(name_to_op["swap_canvas_obj"])
                else:
                    # structural op kind is the exact primitive name
                    if step.kind not in name_to_op:
                        return None
                    ops.append(name_to_op[step.kind])
            return ops

        # Enumerate a limited number of assignments; prefer observed mappings
        seeds = []
        for (a, b) in top_maps[:10]:
            seeds.append((a, b))
        # simple heuristic: for 2 variables, seed with mapping pairs
        if nvars == 2 and seeds:
            for a, b in seeds:
                ops = materialize([a, b])
                if ops:
                    scored_instantiations.append((score_ops(ops), ops))

        # Generic cartesian product with early cutoff
        seen_assignments = 0

        def dfs(vid: int, cur: List[int]):
            nonlocal seen_assignments
            if seen_assignments >= 250:
                return
            if vid == nvars:
                seen_assignments += 1
                ops = materialize(cur)
                if ops:
                    scored_instantiations.append((score_ops(ops), ops))
                return
            for c in per_var[vid]:
                cur.append(c)
                dfs(vid + 1, cur)
                cur.pop()

        dfs(0, [])
        scored_instantiations.sort(key=lambda x: x[0], reverse=True)
        return [ops for _, ops in scored_instantiations[:max_instantiations]]
    
    def _evaluate(self, program: Program, examples: List[Tuple]) -> float:
        """Evaluate program on examples"""
        # NOTE: identity must be evaluated across all examples, not just the first.
        
        total_similarity = 0.0
        min_similarity = 1.0
        for input_grid, output_grid in examples:
            try:
                predicted = program.apply(input_grid)
                similarity = self._similarity(predicted, output_grid)
                total_similarity += similarity
                min_similarity = min(min_similarity, similarity)
            except Exception:
                total_similarity += 0.0
                min_similarity = 0.0
        
        mean_sim = total_similarity / len(examples)
        if SCORING_MODE == "avg":
            base = mean_sim
        else:
            base = MINAVG_ALPHA * min_similarity + (1.0 - MINAVG_ALPHA) * mean_sim
        return max(0.0, base - COMPLEXITY_PENALTY * len(program))

    def _evaluate_from_outputs(self, program: Program, outputs: List, targets: List) -> float:
        """Evaluate a program given precomputed outputs (incremental evaluation)."""
        total_similarity = 0.0
        min_similarity = 1.0
        for pred, tgt in zip(outputs, targets):
            sim = self._similarity(pred, tgt)
            total_similarity += sim
            min_similarity = min(min_similarity, sim)
        mean_sim = total_similarity / max(1, len(targets))
        if SCORING_MODE == "avg":
            base = mean_sim
        else:
            base = MINAVG_ALPHA * min_similarity + (1.0 - MINAVG_ALPHA) * mean_sim
        return max(0.0, base - COMPLEXITY_PENALTY * len(program))
    
    def _similarity(self, grid1, grid2) -> float:
        """
        Compute similarity between two grids.

        Uses a blend of:
          - full-grid accuracy (background-dominated)
          - foreground-focused accuracy (emphasizes changed/non-zero regions)
        This reduces the tendency to prefer near-identity programs when outputs differ
        only in a small region (common in ARC-style tasks).
        """
        if not grid1 or not grid2:
            return 0.0
        
        h1, w1 = len(grid1), len(grid1[0]) if grid1 else 0
        h2, w2 = len(grid2), len(grid2[0]) if grid2 else 0
        
        if h1 != h2 or w1 != w2:
            return 0.0
        
        matches = 0
        fg_matches = 0
        fg_total = 0
        total = h1 * w1
        for i in range(h1):
            r1 = grid1[i]
            r2 = grid2[i]
            for j in range(w1):
                a = r1[j]
                b = r2[j]
                if a == b:
                    matches += 1
                if a != 0 or b != 0:
                    fg_total += 1
                    if a == b:
                        fg_matches += 1

        full = matches / total if total > 0 else 0.0
        fg = (fg_matches / fg_total) if fg_total > 0 else full
        return 0.5 * full + 0.5 * fg

# ==============================================================================
# PART 7: ABSTRACTION MINER
# ==============================================================================

class AbstractionMiner:
    """Mine reusable abstractions from successful programs"""
    
    def __init__(self, min_frequency=MIN_FREQUENCY):
        self.min_frequency = min_frequency
        self.sub_program_counts = Counter()
        self.sub_program_tasks = defaultdict(set)
        self.sub_program_repr: Dict[Tuple[Tuple[str, Tuple[Any, ...]], ...], Program] = {}
        self.kind_counts = Counter()
        self.kind_tasks = defaultdict(set)
        self.param_counts = Counter()
        self.param_tasks = defaultdict(set)
    
    def mine(self, programs: List[Tuple[Program, float, str]], task_id: str) -> List[Abstraction]:
        """
        Mine abstractions from successful programs
        
        Args:
            programs: List of (program, score, task_id) tuples
            task_id: Current task identifier
        
        Returns:
            List of discovered abstractions
        """
        # Extract sub-programs from high-scoring programs
        for program, score, tid in programs:
            if score < MINE_MIN_SCORE:
                continue
            
            for sub_prog in program.sub_programs(min_len=2, max_len=4):
                sig = sub_prog.signature()
                self.sub_program_counts[sig] += 1
                self.sub_program_tasks[sig].add(tid)
                if sig not in self.sub_program_repr:
                    self.sub_program_repr[sig] = sub_prog

            # Macro mining on kind-only sequences (captures cross-task structure)
            kinds = program.kind_signature()
            for sub_len in (2, 3):
                for i in range(0, max(0, len(kinds) - sub_len + 1)):
                    ks = tuple(kinds[i:i + sub_len])
                    self.kind_counts[ks] += 1
                    self.kind_tasks[ks].add(tid)

            # Parameterized macro mining (captures shared-argument structure)
            for sub_prog in program.sub_programs(min_len=2, max_len=4):
                pm = self._generalize_to_param_macro(sub_prog)
                if pm is None:
                    continue
                key = pm.steps
                self.param_counts[key] += 1
                self.param_tasks[key].add(tid)
        
        # Find frequent sub-programs that appear across multiple tasks
        abstractions = []
        for sig, count in self.sub_program_counts.items():
            if count < self.min_frequency:
                continue
            tasks_used = self.sub_program_tasks[sig]
            if len(tasks_used) < 2:
                continue

            sub_prog = self.sub_program_repr.get(sig)
            if not sub_prog:
                continue

            # Name: stable and readable.
            op_names = "_".join(op.name for op in sub_prog.operations[:3])
            name = f"abs_{op_names}_{len(sig)}"
            abstraction = Abstraction(
                name=name,
                parameters=[],
                body=sub_prog,
                frequency=count,
                compression_ratio=max(1, len(sub_prog)),
                tasks_used=tasks_used,
            )
            abstractions.append(abstraction)
        
        if VERBOSE and abstractions:
            print(f"\n  Mined {len(abstractions)} abstraction candidates from task {task_id}")
        
        return abstractions

    def mine_macros(self) -> List[MacroTemplate]:
        macros: List[MacroTemplate] = []
        for ks, count in self.kind_counts.items():
            if count < self.min_frequency:
                continue
            tasks = self.kind_tasks[ks]
            if len(tasks) < 2:
                continue
            name = "macro_" + "_".join(ks)
            macros.append(MacroTemplate(name=name, kinds=ks, frequency=count, tasks_used=tasks))
        return macros

    def mine_param_macros(self) -> List[ParamMacroTemplate]:
        macros: List[ParamMacroTemplate] = []
        for steps, count in self.param_counts.items():
            if count < self.min_frequency:
                continue
            tasks = self.param_tasks[steps]
            if len(tasks) < 2:
                continue
            # Prefer short templates; long ones are brittle.
            if not (2 <= len(steps) <= 4):
                continue
            name = "pmacro_" + "_".join([s.kind for s in steps])
            macros.append(ParamMacroTemplate(name=name, steps=steps, frequency=count, tasks_used=tasks))
        return macros

    def _generalize_to_param_macro(self, prog: Program) -> Optional[ParamMacroTemplate]:
        """
        Convert a concrete sub-program into a parameterized template by replacing
        concrete colors with variables, preserving equality constraints.
        """
        var_map: Dict[Tuple[str, int, int], int] = {}
        next_var = 0

        steps: List[ParamStep] = []
        for op in prog.operations:
            # Scratchpad families
            if op.name.startswith("obj_from_canvas_"):
                # keep arguments as constants (e.g., color) for now
                steps.append(ParamStep(kind="obj_from_canvas", args=tuple(op.args)))
                continue
            if op.name.startswith("obj_translate_") and len(op.args) == 2:
                steps.append(ParamStep(kind="obj_translate", args=(op.args[0], op.args[1])))
                continue
            if op.name == "obj_bbox":
                steps.append(ParamStep(kind="obj_bbox", args=()))
                continue
            if op.name.startswith("paste_obj_") or op.name == "paste_obj_on_canvas":
                steps.append(ParamStep(kind="paste_obj", args=(op.name,)))
                continue
            if op.name == "swap_canvas_obj":
                steps.append(ParamStep(kind="swap_canvas_obj", args=()))
                continue

            # Replace-family
            if op.name.startswith("replace_") and len(op.args) == 2:
                a, b = op.args[0], op.args[1]
                # key by value only (so shared constants become shared variables)
                for val, pos in ((a, 0), (b, 1)):
                    k = ("color", val, 0)
                    if k not in var_map:
                        var_map[k] = next_var
                        next_var += 1
                va = ("var", var_map[("color", a, 0)])
                vb = ("var", var_map[("color", b, 0)])
                steps.append(ParamStep(kind="replace", args=(va, vb)))
                continue

            # Keep/remove-family
            if op.name.startswith("keep_") and len(op.args) == 1:
                c = op.args[0]
                k = ("color", c, 0)
                if k not in var_map:
                    var_map[k] = next_var
                    next_var += 1
                vc = ("var", var_map[k])
                steps.append(ParamStep(kind="keep", args=(vc,)))
                continue

            if op.name.startswith("remove_") and len(op.args) == 1:
                c = op.args[0]
                k = ("color", c, 0)
                if k not in var_map:
                    var_map[k] = next_var
                    next_var += 1
                vc = ("var", var_map[k])
                steps.append(ParamStep(kind="remove", args=(vc,)))
                continue

            # Scale-family
            if op.name.startswith("scale_") and len(op.args) == 1:
                f = op.args[0]
                steps.append(ParamStep(kind="scale", args=(f,)))
                continue

            # Exact structural ops
            if op.name in {"flip_h", "flip_v", "rotate90", "rotate180", "transpose", "bbox", "gravity_down",
                           "keep_largest_component", "center_bbox", "keep_nonzero", "move_bbox_to_topleft", "move_bbox_to_bottomright"}:
                steps.append(ParamStep(kind=op.name, args=()))
                continue

            # Translate family
            if op.name.startswith("translate_") and len(op.args) == 2:
                steps.append(ParamStep(kind="translate", args=(op.args[0], op.args[1])))
                continue

            # Unsupported op type for parameterized template
            return None

        if not steps:
            return None
        # name/frequency/tasks are assigned later by miner
        return ParamMacroTemplate(name="pmacro", steps=tuple(steps), frequency=1, tasks_used=set())

# ==============================================================================
# PART 8: META-LEARNER
# ==============================================================================

class MetaLearner:
    """Meta-learn abstraction utilities across tasks"""
    
    def __init__(self, dsl: DSL):
        self.dsl = dsl
        self.task_solutions: Dict[str, Tuple[Program, float]] = {}
    
    def compute_utility(self, abstraction: Abstraction) -> float:
        """
        Compute cross-task utility of an abstraction
        
        Utility = frequency × cross_task_usage × compression_benefit
        """
        frequency_score = min(abstraction.frequency / 10.0, 1.0)
        cross_task_score = len(abstraction.tasks_used) / max(len(self.task_solutions), 1)
        compression_score = 1.0 / abstraction.compression_ratio
        
        utility = (frequency_score * 0.3 + 
                  cross_task_score * 0.5 + 
                  compression_score * 0.2)
        
        return utility
    
    def update(self, task_id: str, program: Program, score: float):
        """Update with new task solution"""
        self.task_solutions[task_id] = (program, score)
    
    def meta_learn(self, new_abstractions: List[Abstraction]):
        """Meta-learn: decide which abstractions to add to DSL"""
        for abstraction in new_abstractions:
            utility = self.compute_utility(abstraction)
            abstraction.utility = utility
            
            if utility > UTILITY_THRESHOLD:
                self.dsl.add_abstraction(abstraction)
        
        # Prune low-utility abstractions
        self.dsl.prune_abstractions()

# ==============================================================================
# PART 9: SMPMA SYSTEM
# ==============================================================================

class SMPMA:
    """
    Self-Modifying Program Synthesis with Meta-Learned Abstractions
    
    Main system that combines synthesis, mining, and meta-learning.
    """
    
    def __init__(self):
        self.dsl = DSL()
        self.synthesizer = ProgramSynthesizer(self.dsl)
        self.miner = AbstractionMiner()
        self.meta_learner = MetaLearner(self.dsl)
        self.task_count = 0
        
        print("\nSMPMA system initialized")
        print(f"  Beam width: {BEAM_WIDTH}")
        print(f"  Max depth: {MAX_DEPTH}")
        print(f"  Utility threshold: {UTILITY_THRESHOLD}")
    
    def solve_task(self, task_examples: List[Tuple], task_id: str) -> Tuple[Program, float]:
        """
        Solve a task using SMPMA's three-stage process:
        1. Synthesize program with current DSL
        2. Mine abstractions from successful solution
        3. Meta-learn abstraction utilities and update DSL
        """
        self.task_count += 1
        
        if VERBOSE:
            print(f"\n[Task {self.task_count}] {task_id}")
            print(f"  Examples: {len(task_examples)}")
        
        # Stage 1: Synthesize
        start = time.time()
        program, score = self.synthesizer.synthesize(task_examples, task_id)
        synth_time = time.time() - start
        
        if VERBOSE:
            if program:
                print(f"  Solution: {program}")
                print(f"  Score: {score:.3f}")
                print(f"  Time: {synth_time:.2f}s")
            else:
                print("  No solution found")
        
        # Always record the outcome for benchmarking and mining context.
        if program:
            self.meta_learner.update(task_id, program, score)

        # If the solution is weak, skip mining but keep the result.
        if not program or score < MINE_MIN_SCORE:
            return program, score
        
        # Stage 2: Mine abstractions (periodically)
        if self.task_count % MINE_EVERY_N_TASKS == 0:
            recent_programs = [(p, s, tid) for tid, (p, s) in 
                             list(self.meta_learner.task_solutions.items())[-20:]]
            recent_programs.append((program, score, task_id))
            
            new_abstractions = self.miner.mine(recent_programs, task_id)
            
            # Stage 3: Meta-learn
            if new_abstractions:
                self.meta_learner.meta_learn(new_abstractions)

            # Promote mined macro templates
            for macro in self.miner.mine_macros()[:8]:
                # Heuristic gate: only add short macros that are likely to be useful.
                if 2 <= len(macro) <= 3:
                    self.dsl.add_macro(macro)

            # Promote mined parameterized macro templates
            for pm in self.miner.mine_param_macros()[:12]:
                if 2 <= len(pm) <= 4:
                    self.dsl.add_param_macro(pm)
        
        return program, score
    
    def evaluate(self, tasks: Dict[str, List[Tuple]]) -> Dict:
        """Evaluate on multiple tasks"""
        results = {
            "total_tasks": len(tasks),
            "solved": 0,
            "avg_score": 0.0,
            "abstraction_count": 0
        }
        
        total_score = 0.0
        
        for task_id, examples in tasks.items():
            program, score = self.solve_task(examples, task_id)
            if score > 0.95:
                results["solved"] += 1
            total_score += score
        
        results["avg_score"] = total_score / len(tasks)
        results["abstraction_count"] = len(self.dsl.abstractions)
        
        return results
    
    def summary(self):
        """Print system summary"""
        print(f"\n{'='*80}")
        print("SMPMA SUMMARY")
        print(f"{'='*80}")
        print(f"Tasks processed: {self.task_count}")
        print(f"Solutions found: {len(self.meta_learner.task_solutions)}")
        self.dsl.summary()

# ==============================================================================
# PART 10: MAIN & TESTING
# ==============================================================================

def generate_synthetic_task():
    """Generate a synthetic reasoning task"""
    h, w = random.randint(3, 8), random.randint(3, 8)
    
    # Create random input
    input_grid = [[random.randint(0, 3) for _ in range(w)] for _ in range(h)]
    
    # Apply random transformation
    transforms = [flip_v, flip_h, rotate90, transpose, bbox]
    transform = random.choice(transforms)
    output_grid = transform(input_grid)
    
    return [(input_grid, output_grid)]

def main():
    """Main testing function"""
    print(f"\n{'='*80}")
    print("RUNNING SMPMA ON SYNTHETIC TASKS")
    print(f"{'='*80}")
    
    # Initialize system
    smpma = SMPMA()
    
    # Generate synthetic tasks
    num_tasks = 50
    print(f"\nGenerating {num_tasks} synthetic tasks...")
    
    synthetic_tasks = {}
    for i in range(num_tasks):
        task_id = f"synthetic_{i:03d}"
        synthetic_tasks[task_id] = generate_synthetic_task()
    
    # Evaluate
    print(f"\nEvaluating SMPMA...")
    results = smpma.evaluate(synthetic_tasks)
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Solved (>95% accuracy): {results['solved']} ({results['solved']/results['total_tasks']*100:.1f}%)")
    print(f"Average score: {results['avg_score']:.3f}")
    print(f"Learned abstractions: {results['abstraction_count']}")
    
    # System summary
    smpma.summary()
    
    print(f"\n{'='*80}")
    print("SMPMA demonstration complete")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

