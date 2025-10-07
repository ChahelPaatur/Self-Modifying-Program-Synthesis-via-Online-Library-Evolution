"""
Canonicalizer Solver:
- Full D4 (dihedral group) transformations
- Learn color mapping from train pairs
- Apply consistent transform + color map to test
- Enhanced with flood fill and color operations
"""
from typing import List, Dict, Tuple, Optional, Set
import itertools
from collections import Counter
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from common.grid_ops import replace_color, flood_fill, get_most_common_color

Grid = List[List[int]]


def rotate_90(grid: Grid) -> Grid:
    """Rotate 90° clockwise"""
    if not grid:
        return grid
    return [list(row) for row in zip(*grid[::-1])]


def rotate_180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]


def rotate_270(grid: Grid) -> Grid:
    if not grid:
        return grid
    return [list(row) for row in zip(*grid)][::-1]


def flip_h(grid: Grid) -> Grid:
    """Horizontal flip"""
    return [row[::-1] for row in grid]


def flip_v(grid: Grid) -> Grid:
    """Vertical flip"""
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    if not grid:
        return grid
    return [list(row) for row in zip(*grid)]


# All 8 D4 transformations
D4_TRANSFORMS = {
    'identity': lambda g: [row[:] for row in g],
    'rot90': rotate_90,
    'rot180': rotate_180,
    'rot270': rotate_270,
    'flip_h': flip_h,
    'flip_v': flip_v,
    'transpose': transpose,
    'anti_transpose': lambda g: flip_v(transpose(g)),
}


def get_colors(grid: Grid) -> Set[int]:
    return {c for row in grid for c in row}


def get_unique_colors(grid: Grid) -> Set[int]:
    """Alias for get_colors for compatibility"""
    return get_colors(grid)


def apply_color_map(grid: Grid, mapping: Dict[int, int]) -> Grid:
    return [[mapping.get(c, c) for c in row] for row in grid]


def grid_to_tuple(grid: Grid) -> tuple:
    """Hashable representation"""
    return tuple(tuple(row) for row in grid)


def grids_same_shape(g1: Grid, g2: Grid) -> bool:
    if not g1 or not g2:
        return len(g1) == len(g2)
    return len(g1) == len(g2) and len(g1[0]) == len(g2[0])


def find_color_mapping(src: Grid, tgt: Grid) -> Optional[Dict[int, int]]:
    """Find color bijection src -> tgt if grids have same structure"""
    if not grids_same_shape(src, tgt):
        return None
    
    mapping = {}
    for i in range(len(src)):
        for j in range(len(src[0])):
            s, t = src[i][j], tgt[i][j]
            if s in mapping:
                if mapping[s] != t:
                    return None  # conflict
            else:
                mapping[s] = t
    return mapping


def find_transform_and_color_map(inp: Grid, out: Grid) -> Optional[Tuple[str, Dict[int, int]]]:
    """Find (transform_name, color_map) such that color_map(transform(inp)) == out"""
    for name, tfn in D4_TRANSFORMS.items():
        t_inp = tfn(inp)
        cmap = find_color_mapping(t_inp, out)
        if cmap is not None:
            # verify
            result = apply_color_map(t_inp, cmap)
            if grid_to_tuple(result) == grid_to_tuple(out):
                return (name, cmap)
    return None


def merge_color_maps(maps: List[Dict[int, int]]) -> Optional[Dict[int, int]]:
    """Merge multiple color maps; return None if conflicts"""
    merged = {}
    for m in maps:
        for k, v in m.items():
            if k in merged:
                if merged[k] != v:
                    return None
            merged[k] = v
    return merged


class Canonicalizer:
    def predict(self, task) -> List[Grid]:
        train = task.get('train', [])
        test = task.get('test', [])
        
        if not train:
            # No training examples; return input as-is
            return [t['input'] for t in test]
        
        # Try to find consistent (transform, color_map) across all train pairs
        # First: find transform candidates from first train pair
        first_in = train[0]['input']
        first_out = train[0]['output']
        
        candidates = []
        for name, tfn in D4_TRANSFORMS.items():
            t_inp = tfn(first_in)
            cmap = find_color_mapping(t_inp, first_out)
            if cmap is not None:
                candidates.append((name, cmap))
        
        if not candidates:
            # Fallback: try direct pattern matching or return input
            return [t['input'] for t in test]
        
        # Filter candidates that work on all train pairs
        valid = []
        for name, cmap in candidates:
            tfn = D4_TRANSFORMS[name]
            works = True
            color_maps = [cmap]
            
            for ex in train[1:]:
                t_inp = tfn(ex['input'])
                ex_cmap = find_color_mapping(t_inp, ex['output'])
                if ex_cmap is None:
                    works = False
                    break
                color_maps.append(ex_cmap)
            
            if works:
                # Try to merge color maps
                merged = merge_color_maps(color_maps)
                if merged is not None:
                    valid.append((name, merged))
                else:
                    # Use the first color map if merging fails
                    valid.append((name, cmap))
        
        if not valid:
            # No consistent transform found; fallback strategies
            return self._fallback_predict(train, test)
        
        # Use the first valid transform
        best_name, best_cmap = valid[0]
        tfn = D4_TRANSFORMS[best_name]
        
        preds = []
        for t in test:
            transformed = tfn(t['input'])
            result = apply_color_map(transformed, best_cmap)
            preds.append(result)
        
        return preds
    
    def _fallback_predict(self, train, test) -> List[Grid]:
        """Fallback: simple heuristics when no transform fits"""
        # Strategy 1: Check for simple color replacement
        if train:
            first_in = train[0]['input']
            first_out = train[0]['output']
            
            if grids_same_shape(first_in, first_out):
                # Try color-only transformations
                in_colors = get_unique_colors(first_in)
                out_colors = get_unique_colors(first_out)
                
                # If same number of colors, might be simple recoloring
                if len(in_colors) == len(out_colors):
                    color_map = {}
                    for ex in train:
                        ex_map = find_color_mapping(ex['input'], ex['output'])
                        if ex_map:
                            for k, v in ex_map.items():
                                if k in color_map and color_map[k] != v:
                                    break
                                color_map[k] = v
                        else:
                            break
                    else:
                        # Consistent color map found
                        return [apply_color_map(t['input'], color_map) for t in test]
        
        # Strategy 2: if all train outputs have same shape, use that shape
        out_shapes = [(len(ex['output']), len(ex['output'][0]) if ex['output'] else 0) 
                      for ex in train]
        
        if len(set(out_shapes)) == 1:
            h, w = out_shapes[0]
            preds = []
            for t in test:
                inp = t['input']
                # Pad or crop to target shape
                new = [[0 for _ in range(w)] for _ in range(h)]
                for i in range(min(len(inp), h)):
                    for j in range(min(len(inp[0]) if inp else 0, w)):
                        new[i][j] = inp[i][j]
                preds.append(new)
            return preds
        
        # Strategy 3: return input as-is
        return [t['input'] for t in test]
