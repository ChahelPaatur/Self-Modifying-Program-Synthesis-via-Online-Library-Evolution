"""
Object-Level Operations
=======================
Operations for detecting, extracting, and manipulating connected components (objects).

These operations are crucial for tasks involving:
- Object detection and counting
- Object movement (translation, rotation)
- Object sorting (by size, color, position)
- Object composition and decomposition
"""
from typing import List, Dict, Tuple, Set, Optional
from collections import deque

Grid = List[List[int]]


# ==================== OBJECT DETECTION ====================

def find_connected_components(grid: Grid, bg_color: int = 0, connectivity: int = 4) -> List[Set[Tuple[int, int]]]:
    """
    Find all connected components (objects) in grid.
    
    Args:
        grid: Input grid
        bg_color: Background color to ignore
        connectivity: 4 or 8 (neighbors)
    
    Returns:
        List of sets, each containing (row, col) coordinates of an object
    """
    if not grid:
        return []
    
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    components = []
    
    def bfs(start_r, start_c, color):
        """BFS to find connected component"""
        component = set()
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        
        while queue:
            r, c = queue.popleft()
            component.add((r, c))
            
            # Get neighbors based on connectivity
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            if connectivity == 8:
                neighbors += [(r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
            
            for nr, nc in neighbors:
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc]:
                    if grid[nr][nc] == color:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
        
        return component
    
    # Find all components
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] != bg_color:
                comp = bfs(r, c, grid[r][c])
                if comp:
                    components.append(comp)
    
    return components


def extract_object(grid: Grid, component: Set[Tuple[int, int]], bg_color: int = 0) -> Grid:
    """Extract object as minimal bounding box grid"""
    if not component:
        return [[bg_color]]
    
    rows = [r for r, c in component]
    cols = [c for r, c in component]
    
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    
    result = [[bg_color] * w for _ in range(h)]
    for r, c in component:
        result[r - min_r][c - min_c] = grid[r][c]
    
    return result


def get_object_color(grid: Grid, component: Set[Tuple[int, int]]) -> int:
    """Get the color of an object"""
    if not component:
        return 0
    r, c = next(iter(component))
    return grid[r][c]


def get_object_size(component: Set[Tuple[int, int]]) -> int:
    """Get size (number of cells) of object"""
    return len(component)


def get_object_bounds(component: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Get bounding box (min_r, min_c, max_r, max_c)"""
    if not component:
        return (0, 0, 0, 0)
    rows = [r for r, c in component]
    cols = [c for r, c in component]
    return (min(rows), min(cols), max(rows), max(cols))


# ==================== OBJECT MANIPULATION ====================

def place_object(grid: Grid, component: Set[Tuple[int, int]], offset_r: int, offset_c: int, source_grid: Grid) -> Grid:
    """Place object at new position with offset"""
    result = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    
    for r, c in component:
        new_r, new_c = r + offset_r, c + offset_c
        if 0 <= new_r < h and 0 <= new_c < w:
            result[new_r][new_c] = source_grid[r][c]
    
    return result


def move_object(grid: Grid, component: Set[Tuple[int, int]], delta_r: int, delta_c: int, bg_color: int = 0) -> Grid:
    """Move object by delta"""
    result = [row[:] for row in grid]
    
    # Clear original position
    for r, c in component:
        result[r][c] = bg_color
    
    # Place at new position
    h, w = len(grid), len(grid[0])
    for r, c in component:
        new_r, new_c = r + delta_r, c + delta_c
        if 0 <= new_r < h and 0 <= new_c < w:
            result[new_r][new_c] = grid[r][c]
    
    return result


def delete_object(grid: Grid, component: Set[Tuple[int, int]], bg_color: int = 0) -> Grid:
    """Delete object from grid"""
    result = [row[:] for row in grid]
    for r, c in component:
        result[r][c] = bg_color
    
    return result


def scale_object(grid: Grid, component: Set[Tuple[int, int]], scale: int, bg_color: int = 0) -> Grid:
    """Scale object by integer factor"""
    if scale <= 0:
        return grid
    
    # Extract object
    obj = extract_object(grid, component, bg_color)
    
    # Scale it
    h, w = len(obj), len(obj[0])
    scaled = []
    for r in range(h):
        for _ in range(scale):
            row = []
            for c in range(w):
                row.extend([obj[r][c]] * scale)
            scaled.append(row)
    
    return scaled


def rotate_object_90(grid: Grid, component: Set[Tuple[int, int]], bg_color: int = 0) -> Grid:
    """Rotate object 90 degrees clockwise"""
    obj = extract_object(grid, component, bg_color)
    rotated = [list(row) for row in zip(*obj[::-1])] if obj else obj
    return rotated


# ==================== OBJECT SORTING & FILTERING ====================

def sort_objects_by_size(components: List[Set[Tuple[int, int]]]) -> List[Set[Tuple[int, int]]]:
    """Sort objects by size (largest first)"""
    return sorted(components, key=lambda c: len(c), reverse=True)


def sort_objects_by_position(components: List[Set[Tuple[int, int]]]) -> List[Set[Tuple[int, int]]]:
    """Sort objects by top-left position (top to bottom, left to right)"""
    def get_top_left(comp):
        if not comp:
            return (float('inf'), float('inf'))
        rows = [r for r, c in comp]
        cols = [c for r, c in comp]
        return (min(rows), min(cols))
    
    return sorted(components, key=get_top_left)


def filter_objects_by_size(components: List[Set[Tuple[int, int]]], min_size: int = 1, max_size: Optional[int] = None) -> List[Set[Tuple[int, int]]]:
    """Filter objects by size range"""
    result = [c for c in components if len(c) >= min_size]
    if max_size is not None:
        result = [c for c in result if len(c) <= max_size]
    return result


def filter_objects_by_color(grid: Grid, components: List[Set[Tuple[int, int]]], color: int) -> List[Set[Tuple[int, int]]]:
    """Filter objects by color"""
    return [c for c in components if get_object_color(grid, c) == color]


# ==================== HIGH-LEVEL OBJECT OPERATIONS ====================

def keep_largest_object(grid: Grid, bg_color: int = 0) -> Grid:
    """Keep only the largest object"""
    components = find_connected_components(grid, bg_color)
    if not components:
        return grid
    
    largest = max(components, key=len)
    result = [[bg_color] * len(grid[0]) for _ in range(len(grid))]
    
    for r, c in largest:
        result[r][c] = grid[r][c]
    
    return result


def count_objects(grid: Grid, bg_color: int = 0) -> int:
    """Count number of objects"""
    return len(find_connected_components(grid, bg_color))


def extract_all_objects(grid: Grid, bg_color: int = 0) -> List[Grid]:
    """Extract all objects as separate grids"""
    components = find_connected_components(grid, bg_color)
    return [extract_object(grid, comp, bg_color) for comp in components]


def align_objects_vertically(grid: Grid, bg_color: int = 0, spacing: int = 1) -> Grid:
    """Stack all objects vertically with spacing"""
    components = find_connected_components(grid, bg_color)
    if not components:
        return grid
    
    # Extract all objects
    objects = [extract_object(grid, comp, bg_color) for comp in components]
    
    # Calculate total height and max width
    total_h = sum(len(obj) for obj in objects) + spacing * (len(objects) - 1)
    max_w = max(len(obj[0]) for obj in objects)
    
    # Create result grid
    result = [[bg_color] * max_w for _ in range(total_h)]
    
    # Place objects
    current_r = 0
    for obj in objects:
        h, w = len(obj), len(obj[0])
        for r in range(h):
            for c in range(w):
                result[current_r + r][c] = obj[r][c]
        current_r += h + spacing
    
    return result


def align_objects_horizontally(grid: Grid, bg_color: int = 0, spacing: int = 1) -> Grid:
    """Stack all objects horizontally with spacing"""
    components = find_connected_components(grid, bg_color)
    if not components:
        return grid
    
    # Extract all objects
    objects = [extract_object(grid, comp, bg_color) for comp in components]
    
    # Calculate total width and max height
    total_w = sum(len(obj[0]) for obj in objects) + spacing * (len(objects) - 1)
    max_h = max(len(obj) for obj in objects)
    
    # Create result grid
    result = [[bg_color] * total_w for _ in range(max_h)]
    
    # Place objects
    current_c = 0
    for obj in objects:
        h, w = len(obj), len(obj[0])
        for r in range(h):
            for c in range(w):
                result[r][current_c + c] = obj[r][c]
        current_c += w + spacing
    
    return result


def duplicate_objects(grid: Grid, count: int = 2, bg_color: int = 0) -> Grid:
    """Duplicate all objects horizontally"""
    components = find_connected_components(grid, bg_color)
    if not components or count < 1:
        return grid
    
    h, w = len(grid), len(grid[0])
    result = [[bg_color] * (w * count) for _ in range(h)]
    
    for i in range(count):
        offset = w * i
        for r in range(h):
            for c in range(w):
                result[r][offset + c] = grid[r][c]
    
    return result


def center_objects(grid: Grid, bg_color: int = 0) -> Grid:
    """Center all objects in the grid"""
    components = find_connected_components(grid, bg_color)
    if not components:
        return grid
    
    h, w = len(grid), len(grid[0])
    result = [[bg_color] * w for _ in range(h)]
    
    for comp in components:
        # Get bounds
        min_r, min_c, max_r, max_c = get_object_bounds(comp)
        
        # Calculate center offset
        obj_h = max_r - min_r + 1
        obj_w = max_c - min_c + 1
        
        offset_r = (h - obj_h) // 2 - min_r
        offset_c = (w - obj_w) // 2 - min_c
        
        # Place centered
        for r, c in comp:
            new_r, new_c = r + offset_r, c + offset_c
            if 0 <= new_r < h and 0 <= new_c < w:
                result[new_r][new_c] = grid[r][c]
    
    return result
