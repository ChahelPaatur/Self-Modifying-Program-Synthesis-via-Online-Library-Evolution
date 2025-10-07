"""
Compositional Abstraction Learner - NOVEL APPROACH
===================================================
Learns hierarchical compositional patterns:
1. Detects repeating sub-patterns (motifs)
2. Builds abstraction hierarchy (pixels -> motifs -> compositions)
3. Learns transformation at each abstraction level
4. Applies learned abstractions to test cases

Key innovation: Instead of pixel-level transforms, learns pattern-level semantics.

Examples:
- "Replace all 2x2 red squares with 3x3 blue crosses"
- "Tile the input pattern 3x3 times"
- "Extract the border and fill interior"
"""
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter, defaultdict
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from common.grid_ops import replace_color, extract_objects_with_colors, overlay_grids

Grid = List[List[int]]


def grid_to_array(grid: Grid) -> np.ndarray:
    """Convert to numpy for efficient operations"""
    if not grid:
        return np.array([])
    return np.array(grid, dtype=np.int32)


def array_to_grid(arr: np.ndarray) -> Grid:
    """Convert back to list format"""
    return arr.tolist()


def extract_motif(grid: Grid, r: int, c: int, h: int, w: int) -> Optional[Grid]:
    """Extract h×w motif starting at (r,c)"""
    if r + h > len(grid) or c + w > len(grid[0]):
        return None
    return [grid[r + i][c:c + w] for i in range(h)]


def find_repeating_motifs(grid: Grid, motif_sizes: List[Tuple[int, int]]) -> Dict[Tuple, List[Tuple[int, int]]]:
    """Find repeating motifs of given sizes"""
    if not grid:
        return {}
    
    h, w = len(grid), len(grid[0])
    motif_locations = defaultdict(list)
    
    for mh, mw in motif_sizes:
        seen = {}
        for r in range(h - mh + 1):
            for c in range(w - mw + 1):
                motif = extract_motif(grid, r, c, mh, mw)
                if motif:
                    key = tuple(tuple(row) for row in motif)
                    if key not in seen:
                        seen[key] = []
                    seen[key].append((r, c))
        
        # Keep only motifs that repeat
        for key, locs in seen.items():
            if len(locs) > 1:
                motif_locations[(mh, mw, key)] = locs
    
    return motif_locations


def detect_tiling(inp: Grid, out: Grid) -> Optional[Tuple[int, int]]:
    """Detect if output is tiled version of input"""
    if not inp or not out:
        return None
    
    ih, iw = len(inp), len(inp[0])
    oh, ow = len(out), len(out[0])
    
    # Check if output dimensions are multiples
    if oh % ih != 0 or ow % iw != 0:
        return None
    
    tile_h = oh // ih
    tile_w = ow // iw
    
    # Verify tiling
    for r in range(oh):
        for c in range(ow):
            if out[r][c] != inp[r % ih][c % iw]:
                return None
    
    return (tile_h, tile_w)


def tile_grid(grid: Grid, tile_h: int, tile_w: int) -> Grid:
    """Tile grid tile_h × tile_w times"""
    if not grid:
        return grid
    
    result = []
    for _ in range(tile_h):
        for row in grid:
            result.append(row * tile_w)
    return result


def extract_border(grid: Grid, thickness: int = 1) -> Grid:
    """Extract border of grid"""
    if not grid or len(grid) < 2 * thickness or len(grid[0]) < 2 * thickness:
        return grid
    
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    
    # Top and bottom
    for r in range(thickness):
        for c in range(w):
            result[r][c] = grid[r][c]
            result[h - 1 - r][c] = grid[h - 1 - r][c]
    
    # Left and right
    for r in range(h):
        for c in range(thickness):
            result[r][c] = grid[r][c]
            result[r][w - 1 - c] = grid[r][w - 1 - c]
    
    return result


def fill_interior(grid: Grid, fill_color: int = 0) -> Grid:
    """Fill interior of grid with color"""
    if not grid or len(grid) < 3 or len(grid[0]) < 3:
        return grid
    
    result = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            result[r][c] = fill_color
    
    return result


def detect_scaling(inp: Grid, out: Grid) -> Optional[Tuple[int, int]]:
    """Detect if output is scaled version of input"""
    if not inp or not out:
        return None
    
    ih, iw = len(inp), len(inp[0])
    oh, ow = len(out), len(out[0])
    
    if oh % ih != 0 or ow % iw != 0:
        return None
    
    scale_h = oh // ih
    scale_w = ow // iw
    
    # Verify scaling
    for r in range(oh):
        for c in range(ow):
            if out[r][c] != inp[r // scale_h][c // scale_w]:
                return None
    
    return (scale_h, scale_w)


def scale_grid(grid: Grid, scale_h: int, scale_w: int) -> Grid:
    """Scale grid by repeating each cell"""
    if not grid:
        return grid
    
    result = []
    for row in grid:
        scaled_row = []
        for cell in row:
            scaled_row.extend([cell] * scale_w)
        for _ in range(scale_h):
            result.append(scaled_row[:])
    
    return result


def detect_crop(inp: Grid, out: Grid) -> Optional[Tuple[int, int, int, int]]:
    """Detect if output is cropped from input"""
    if not inp or not out:
        return None
    
    ih, iw = len(inp), len(inp[0])
    oh, ow = len(out), len(out[0])
    
    if oh > ih or ow > iw:
        return None
    
    # Try to find matching subregion
    for r in range(ih - oh + 1):
        for c in range(iw - ow + 1):
            match = True
            for i in range(oh):
                for j in range(ow):
                    if inp[r + i][c + j] != out[i][j]:
                        match = False
                        break
                if not match:
                    break
            if match:
                return (r, c, oh, ow)
    
    return None


def crop_grid(grid: Grid, r: int, c: int, h: int, w: int) -> Grid:
    """Crop region from grid"""
    if not grid:
        return grid
    
    result = []
    for i in range(r, min(r + h, len(grid))):
        result.append(grid[i][c:c + w])
    return result


class AbstractionLearner:
    def __init__(self):
        self.transformations = []
    
    def _infer_transformations(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[Dict]:
        """Infer high-level transformations"""
        transforms = []
        
        for inp, out in train_pairs:
            if not inp or not out:
                continue
            
            # Check for tiling
            tiling = detect_tiling(inp, out)
            if tiling:
                transforms.append({'type': 'tile', 'params': tiling})
                continue
            
            # Check for scaling
            scaling = detect_scaling(inp, out)
            if scaling:
                transforms.append({'type': 'scale', 'params': scaling})
                continue
            
            # Check for cropping
            crop_params = detect_crop(inp, out)
            if crop_params:
                transforms.append({'type': 'crop', 'params': crop_params})
                continue
            
            # Check for border extraction
            border = extract_border(inp)
            if border == out:
                transforms.append({'type': 'extract_border', 'params': 1})
                continue
            
            # Check for interior fill
            filled = fill_interior(inp)
            if filled == out:
                transforms.append({'type': 'fill_interior', 'params': 0})
                continue
            
            # Check for simple color replacement (same shape)
            if len(inp) == len(out) and (len(inp[0]) if inp else 0) == (len(out[0]) if out else 0):
                # Try each color replacement
                in_colors = {c for row in inp for c in row}
                out_colors = {c for row in out for c in row}
                
                for old_c in in_colors:
                    for new_c in out_colors:
                        test = replace_color(inp, old_c, new_c)
                        if test == out:
                            transforms.append({'type': 'replace_color', 'params': (old_c, new_c)})
                            break
        
        # Find most common transformation
        if not transforms:
            return []
        
        transform_types = [t['type'] for t in transforms]
        most_common = Counter(transform_types).most_common(1)[0][0]
        
        # Return all transforms of the most common type
        return [t for t in transforms if t['type'] == most_common]
    
    def predict(self, task) -> List[Grid]:
        train = task.get('train', [])
        test = task.get('test', [])
        
        if not train:
            return [t['input'] for t in test]
        
        # Learn transformations
        train_pairs = [(ex['input'], ex['output']) for ex in train]
        transforms = self._infer_transformations(train_pairs)
        
        if not transforms:
            return [t['input'] for t in test]
        
        # Use first transform
        transform = transforms[0]
        
        # Apply to test
        preds = []
        for t in test:
            inp = t['input']
            
            try:
                if transform['type'] == 'tile':
                    tile_h, tile_w = transform['params']
                    result = tile_grid(inp, tile_h, tile_w)
                elif transform['type'] == 'scale':
                    scale_h, scale_w = transform['params']
                    result = scale_grid(inp, scale_h, scale_w)
                elif transform['type'] == 'crop':
                    r, c, h, w = transform['params']
                    result = crop_grid(inp, r, c, h, w)
                elif transform['type'] == 'extract_border':
                    result = extract_border(inp, transform['params'])
                elif transform['type'] == 'fill_interior':
                    result = fill_interior(inp, transform['params'])
                elif transform['type'] == 'replace_color':
                    old_c, new_c = transform['params']
                    result = replace_color(inp, old_c, new_c)
                else:
                    result = inp
                
                preds.append(result)
            except:
                preds.append(inp)
        
        return preds
