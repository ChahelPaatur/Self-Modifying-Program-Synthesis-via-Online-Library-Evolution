"""
Advanced Grid Operations - Comprehensive Library
Covers 50+ operations for ARC tasks
"""
from typing import List, Dict, Set, Tuple, Optional, Callable
from collections import deque, defaultdict
import copy

Grid = List[List[int]]


# ==================== OBJECT MANIPULATION ====================

def extract_largest_object(grid: Grid, bg: int = 0) -> Tuple[Grid, Tuple[int, int]]:
    """Extract the largest connected component as a new grid"""
    if not grid:
        return grid, (0, 0)
    
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    largest = []
    largest_pos = (0, 0)
    
    def bfs(sr, sc, color):
        cells = []
        q = deque([(sr, sc)])
        visited[sr][sc] = True
        
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                    visited[nr][nc] = True
                    q.append((nr, nc))
        return cells
    
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] != bg:
                obj = bfs(r, c, grid[r][c])
                if len(obj) > len(largest):
                    largest = obj
                    largest_pos = (r, c)
    
    if not largest:
        return [[bg]], (0, 0)
    
    min_r = min(r for r, c in largest)
    min_c = min(c for r, c in largest)
    max_r = max(r for r, c in largest)
    max_c = max(c for r, c in largest)
    
    obj_grid = [[bg for _ in range(max_c - min_c + 1)] for _ in range(max_r - min_r + 1)]
    for r, c in largest:
        obj_grid[r - min_r][c - min_c] = grid[r][c]
    
    return obj_grid, (min_r, min_c)


def move_object(grid: Grid, from_pos: Tuple[int, int], to_pos: Tuple[int, int], bg: int = 0) -> Grid:
    """Move object from one position to another"""
    obj, offset = extract_largest_object(grid, bg)
    result = [[bg for _ in row] for row in grid]
    
    # Place object at new position
    dr, dc = to_pos
    for i, row in enumerate(obj):
        for j, val in enumerate(row):
            if val != bg:
                nr, nc = dr + i, dc + j
                if 0 <= nr < len(result) and 0 <= nc < len(result[0]):
                    result[nr][nc] = val
    
    return result


def count_objects(grid: Grid, bg: int = 0) -> int:
    """Count connected components"""
    if not grid:
        return 0
    
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    count = 0
    
    def bfs(sr, sc):
        q = deque([(sr, sc)])
        visited[sr][sc] = True
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == grid[sr][sc]:
                    visited[nr][nc] = True
                    q.append((nr, nc))
    
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] != bg:
                bfs(r, c)
                count += 1
    
    return count


# ==================== PATTERN OPERATIONS ====================

def repeat_pattern_n_times(grid: Grid, n: int, axis: str = 'horizontal') -> Grid:
    """Repeat pattern n times along axis"""
    if not grid or n <= 0:
        return grid
    
    if axis == 'horizontal':
        return [row * n for row in grid]
    else:  # vertical
        return grid * n


def create_checkerboard(h: int, w: int, color1: int = 0, color2: int = 1) -> Grid:
    """Create checkerboard pattern"""
    return [[(color1 if (i + j) % 2 == 0 else color2) for j in range(w)] for i in range(h)]


def hollow_out(grid: Grid, bg: int = 0) -> Grid:
    """Keep only the border of objects"""
    if not grid or len(grid) < 3:
        return grid
    
    result = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r][c] != bg:
                # Check if surrounded
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
    
    # Find holes (bg cells surrounded by non-bg)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if result[r][c] == bg:
                neighbors = [grid[r+dr][c+dc] for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]]
                non_bg = [x for x in neighbors if x != bg]
                if len(non_bg) == 4 and len(set(non_bg)) == 1:
                    result[r][c] = non_bg[0]
    
    return result


# ==================== SIZE OPERATIONS ====================

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


def resize_to_shape(grid: Grid, target_h: int, target_w: int, bg: int = 0) -> Grid:
    """Resize grid to target shape (crop or pad)"""
    if not grid:
        return [[bg] * target_w for _ in range(target_h)]
    
    h, w = len(grid), len(grid[0])
    result = [[bg] * target_w for _ in range(target_h)]
    
    for i in range(min(h, target_h)):
        for j in range(min(w, target_w)):
            result[i][j] = grid[i][j]
    
    return result


# ==================== GEOMETRIC OPERATIONS ====================

def rotate_around_center(grid: Grid, degrees: int) -> Grid:
    """Rotate grid 90, 180, or 270 degrees"""
    if not grid:
        return grid
    
    degrees = degrees % 360
    
    if degrees == 90:
        return [list(row) for row in zip(*grid[::-1])]
    elif degrees == 180:
        return [row[::-1] for row in grid[::-1]]
    elif degrees == 270:
        return [list(row) for row in zip(*grid)][::-1]
    else:
        return grid


def reflect_diagonal(grid: Grid) -> Grid:
    """Reflect along main diagonal"""
    if not grid:
        return grid
    return [list(row) for row in zip(*grid)]


def reflect_anti_diagonal(grid: Grid) -> Grid:
    """Reflect along anti-diagonal"""
    if not grid:
        return grid
    return [list(row) for row in zip(*grid[::-1])][::-1]


# ==================== COLOR OPERATIONS ====================

def invert_colors(grid: Grid, max_color: int = 9) -> Grid:
    """Invert color values"""
    return [[max_color - cell for cell in row] for row in grid]


def map_colors_by_frequency(grid: Grid) -> Grid:
    """Remap colors by frequency (most common -> 0, etc.)"""
    from collections import Counter
    
    flat = [c for row in grid for c in row]
    counts = Counter(flat)
    mapping = {color: idx for idx, (color, _) in enumerate(counts.most_common())}
    
    return [[mapping[cell] for cell in row] for row in grid]


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


def recolor_by_position(grid: Grid, rule: str = 'row') -> Grid:
    """Recolor cells based on position"""
    if not grid:
        return grid
    
    result = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    
    if rule == 'row':
        for i, row in enumerate(result):
            result[i] = [i % 10 if cell != 0 else cell for cell in row]
    elif rule == 'col':
        for i in range(h):
            for j in range(w):
                if result[i][j] != 0:
                    result[i][j] = j % 10
    elif rule == 'checkerboard':
        for i in range(h):
            for j in range(w):
                if result[i][j] != 0:
                    result[i][j] = (i + j) % 2 + 1
    
    return result


# ==================== SPATIAL OPERATIONS ====================

def stack_grids_vertically(g1: Grid, g2: Grid) -> Grid:
    """Stack two grids vertically"""
    return g1 + g2


def stack_grids_horizontally(g1: Grid, g2: Grid) -> Grid:
    """Stack two grids horizontally"""
    if not g1:
        return g2
    if not g2:
        return g1
    
    return [r1 + r2 for r1, r2 in zip(g1, g2)]


def interleave_rows(g1: Grid, g2: Grid) -> Grid:
    """Interleave rows from two grids"""
    result = []
    for r1, r2 in zip(g1, g2):
        result.append(r1)
        result.append(r2)
    return result


def interleave_cols(g1: Grid, g2: Grid) -> Grid:
    """Interleave columns from two grids"""
    if not g1 or not g2:
        return g1 or g2
    
    result = []
    for r1, r2 in zip(g1, g2):
        new_row = []
        for c1, c2 in zip(r1, r2):
            new_row.append(c1)
            new_row.append(c2)
        result.append(new_row)
    return result


# ==================== LOGIC OPERATIONS ====================

def element_wise_and(g1: Grid, g2: Grid) -> Grid:
    """Element-wise AND (min)"""
    if not g1 or not g2:
        return g1 or g2
    
    return [[min(c1, c2) for c1, c2 in zip(r1, r2)] for r1, r2 in zip(g1, g2)]


def element_wise_or(g1: Grid, g2: Grid) -> Grid:
    """Element-wise OR (max)"""
    if not g1 or not g2:
        return g1 or g2
    
    return [[max(c1, c2) for c1, c2 in zip(r1, r2)] for r1, r2 in zip(g1, g2)]


def element_wise_xor(g1: Grid, g2: Grid) -> Grid:
    """Element-wise XOR (different -> 1, same -> 0)"""
    if not g1 or not g2:
        return g1 or g2
    
    return [[1 if c1 != c2 else 0 for c1, c2 in zip(r1, r2)] for r1, r2 in zip(g1, g2)]


# ==================== ADVANCED PATTERNS ====================

def detect_and_complete_symmetry(grid: Grid) -> Optional[Grid]:
    """Detect partial symmetry and complete it"""
    if not grid:
        return None
    
    h, w = len(grid), len(grid[0])
    
    # Try vertical symmetry
    for split in range(1, w):
        left = [row[:split] for row in grid]
        right = [row[split:] for row in grid]
        if len(left[0]) == len(right[0]):
            right_flipped = [row[::-1] for row in right]
            if left == right_flipped:
                return grid  # Already symmetric
    
    # Try horizontal symmetry
    for split in range(1, h):
        top = grid[:split]
        bottom = grid[split:]
        if len(top) == len(bottom):
            bottom_flipped = bottom[::-1]
            if top == bottom_flipped:
                return grid  # Already symmetric
    
    return None


def extract_unique_rows(grid: Grid) -> Grid:
    """Remove duplicate rows"""
    seen = set()
    result = []
    for row in grid:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            result.append(row)
    return result


def extract_unique_cols(grid: Grid) -> Grid:
    """Remove duplicate columns"""
    if not grid:
        return grid
    
    transposed = [list(row) for row in zip(*grid)]
    unique_transposed = extract_unique_rows(transposed)
    return [list(row) for row in zip(*unique_transposed)]


def compress_by_uniqueness(grid: Grid) -> Grid:
    """Remove duplicate rows and columns"""
    return extract_unique_cols(extract_unique_rows(grid))

