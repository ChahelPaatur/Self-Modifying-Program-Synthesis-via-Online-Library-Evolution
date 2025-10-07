"""
Common grid operations shared across solvers
"""
from typing import List, Dict, Set, Tuple, Optional
from collections import deque

Grid = List[List[int]]


def get_unique_colors(grid: Grid) -> Set[int]:
    """Get all unique colors in grid"""
    return {c for row in grid for c in row}


def flood_fill(grid: Grid, r: int, c: int, new_color: int, target_color: Optional[int] = None) -> Grid:
    """Flood fill starting from (r,c)"""
    if not grid or r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]):
        return grid
    
    result = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    
    if target_color is None:
        target_color = grid[r][c]
    
    if target_color == new_color:
        return result
    
    queue = deque([(r, c)])
    visited = {(r, c)}
    
    while queue:
        cr, cc = queue.popleft()
        result[cr][cc] = new_color
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if grid[nr][nc] == target_color:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    
    return result


def replace_color(grid: Grid, old_color: int, new_color: int) -> Grid:
    """Replace all instances of old_color with new_color"""
    return [[new_color if c == old_color else c for c in row] for row in grid]


def find_connected_components(grid: Grid, background: int = 0) -> List[Set[Tuple[int, int]]]:
    """Find all connected components (objects)"""
    if not grid:
        return []
    
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    components = []
    
    def bfs(start_r, start_c, color):
        cells = set()
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        
        while queue:
            r, c = queue.popleft()
            cells.add((r, c))
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc]:
                    if grid[nr][nc] == color:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
        
        return cells
    
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] != background:
                component = bfs(r, c, grid[r][c])
                components.append(component)
    
    return components


def pad_grid(grid: Grid, pad_h: int, pad_w: int, fill_color: int = 0) -> Grid:
    """Pad grid with fill_color"""
    if not grid:
        return [[fill_color] * pad_w for _ in range(pad_h)]
    
    h, w = len(grid), len(grid[0])
    result = [[fill_color] * (w + 2 * pad_w) for _ in range(pad_h)]
    
    for row in grid:
        result.append([fill_color] * pad_w + row + [fill_color] * pad_w)
    
    for _ in range(pad_h):
        result.append([fill_color] * (w + 2 * pad_w))
    
    return result


def overlay_grids(base: Grid, overlay: Grid, offset_r: int = 0, offset_c: int = 0, 
                  transparent_color: Optional[int] = None) -> Grid:
    """Overlay one grid onto another"""
    if not base:
        return overlay
    if not overlay:
        return base
    
    result = [row[:] for row in base]
    h, w = len(result), len(result[0])
    
    for r, row in enumerate(overlay):
        for c, val in enumerate(row):
            target_r = r + offset_r
            target_c = c + offset_c
            
            if 0 <= target_r < h and 0 <= target_c < w:
                if transparent_color is None or val != transparent_color:
                    result[target_r][target_c] = val
    
    return result


def extract_objects_with_colors(grid: Grid, background: int = 0) -> Dict[int, List[Set[Tuple[int, int]]]]:
    """Group connected components by color"""
    if not grid:
        return {}
    
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    color_objects = {}
    
    def bfs(start_r, start_c, color):
        cells = set()
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        
        while queue:
            r, c = queue.popleft()
            cells.add((r, c))
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc]:
                    if grid[nr][nc] == color:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
        
        return cells
    
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] != background:
                color = grid[r][c]
                cells = bfs(r, c, color)
                if color not in color_objects:
                    color_objects[color] = []
                color_objects[color].append(cells)
    
    return color_objects


def count_colors(grid: Grid) -> Dict[int, int]:
    """Count occurrences of each color"""
    counts = {}
    for row in grid:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    return counts


def get_most_common_color(grid: Grid, exclude: Optional[Set[int]] = None) -> Optional[int]:
    """Get most common color, optionally excluding some"""
    counts = count_colors(grid)
    if exclude:
        counts = {c: cnt for c, cnt in counts.items() if c not in exclude}
    
    if not counts:
        return None
    
    return max(counts.items(), key=lambda x: x[1])[0]

