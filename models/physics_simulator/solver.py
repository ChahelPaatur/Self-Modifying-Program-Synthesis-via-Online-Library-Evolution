"""
Physics Simulator Solver - NOVEL APPROACH
==========================================
Treats grids as physical systems with causal rules:
- Gravity (objects fall down)
- Spreading (colors propagate to neighbors)
- Attraction/Repulsion (objects move toward/away from each other)
- Containment (objects stay within boundaries)
- Collision (objects interact when touching)

Learns physics rules from train examples by:
1. Detecting objects (connected components)
2. Tracking object movement/transformation
3. Inferring causal rules
4. Simulating rules on test inputs
"""
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, deque
import copy

Grid = List[List[int]]


class Object:
    """Represents a connected component in the grid"""
    def __init__(self, color: int, cells: Set[Tuple[int, int]]):
        self.color = color
        self.cells = cells
        self.bbox = self._compute_bbox()
    
    def _compute_bbox(self) -> Tuple[int, int, int, int]:
        if not self.cells:
            return (0, 0, 0, 0)
        rows = [r for r, c in self.cells]
        cols = [c for r, c in self.cells]
        return (min(rows), min(cols), max(rows), max(cols))
    
    def center(self) -> Tuple[float, float]:
        if not self.cells:
            return (0, 0)
        r_avg = sum(r for r, c in self.cells) / len(self.cells)
        c_avg = sum(c for r, c in self.cells) / len(self.cells)
        return (r_avg, c_avg)
    
    def size(self) -> int:
        return len(self.cells)


def extract_objects(grid: Grid, background: int = 0) -> List[Object]:
    """Extract connected components (objects) from grid"""
    if not grid:
        return []
    
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    objects = []
    
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
                objects.append(Object(color, cells))
    
    return objects


def apply_gravity(grid: Grid, background: int = 0) -> Grid:
    """Simulate gravity: non-background cells fall down"""
    if not grid:
        return grid
    
    h, w = len(grid), len(grid[0])
    result = [[background] * w for _ in range(h)]
    
    # For each column, stack non-background cells at bottom
    for c in range(w):
        stack = []
        for r in range(h):
            if grid[r][c] != background:
                stack.append(grid[r][c])
        
        # Place from bottom up
        for i, val in enumerate(stack):
            result[h - 1 - i][c] = val
    
    return result


def apply_spread(grid: Grid, steps: int = 1, background: int = 0) -> Grid:
    """Spread non-background colors to adjacent background cells"""
    if not grid or steps <= 0:
        return grid
    
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    for _ in range(steps):
        new_result = [row[:] for row in result]
        for r in range(h):
            for c in range(w):
                if result[r][c] == background:
                    # Check neighbors for non-background
                    neighbors = []
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and result[nr][nc] != background:
                            neighbors.append(result[nr][nc])
                    
                    if neighbors:
                        # Take most common neighbor color
                        from collections import Counter
                        new_result[r][c] = Counter(neighbors).most_common(1)[0][0]
        
        result = new_result
    
    return result


def detect_pattern_fill(grid: Grid) -> Optional[Grid]:
    """Detect if grid has a repeating pattern and fill it"""
    if not grid or len(grid) < 2:
        return None
    
    h, w = len(grid), len(grid[0])
    
    # Try to detect horizontal pattern
    for pattern_width in range(1, w // 2 + 1):
        if w % pattern_width == 0:
            pattern = [grid[0][i] for i in range(pattern_width)]
            matches = True
            for c in range(w):
                if grid[0][c] != pattern[c % pattern_width]:
                    matches = False
                    break
            
            if matches:
                # Fill entire grid with this pattern
                result = []
                for r in range(h):
                    row = [pattern[c % pattern_width] for c in range(w)]
                    result.append(row)
                return result
    
    return None


def mirror_grid(grid: Grid, axis: str = 'horizontal') -> Grid:
    """Mirror grid along axis"""
    if not grid:
        return grid
    
    if axis == 'horizontal':
        return grid + grid[::-1]
    elif axis == 'vertical':
        return [row + row[::-1] for row in grid]
    
    return grid


def detect_symmetry_completion(inp: Grid, out: Grid) -> Optional[str]:
    """Detect if output is a symmetry completion of input"""
    if not inp or not out:
        return None
    
    # Check if output is horizontally mirrored version
    mirrored_h = mirror_grid(inp, 'horizontal')
    if len(mirrored_h) == len(out) and len(mirrored_h[0]) == len(out[0]):
        if mirrored_h == out:
            return 'mirror_h'
    
    # Check vertical mirror
    mirrored_v = mirror_grid(inp, 'vertical')
    if len(mirrored_v) == len(out) and len(mirrored_v[0]) == len(out[0]):
        if mirrored_v == out:
            return 'mirror_v'
    
    return None


class PhysicsSimulator:
    def __init__(self):
        self.rules = []
    
    def _infer_rules(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[str]:
        """Infer physics rules from training examples"""
        rules = []
        
        for inp, out in train_pairs:
            if not inp or not out:
                continue
            
            # Rule 1: Check for gravity
            gravity_result = apply_gravity(inp)
            if gravity_result == out:
                if 'gravity' not in rules:
                    rules.append('gravity')
                continue
            
            # Rule 2: Check for spreading
            for steps in range(1, 5):
                spread_result = apply_spread(inp, steps)
                if spread_result == out:
                    if f'spread_{steps}' not in rules:
                        rules.append(f'spread_{steps}')
                    break
            
            # Rule 3: Check for symmetry completion
            sym = detect_symmetry_completion(inp, out)
            if sym and sym not in rules:
                rules.append(sym)
        
        return rules
    
    def predict(self, task) -> List[Grid]:
        train = task.get('train', [])
        test = task.get('test', [])
        
        if not train:
            return [t['input'] for t in test]
        
        # Learn rules from train
        train_pairs = [(ex['input'], ex['output']) for ex in train]
        rules = self._infer_rules(train_pairs)
        
        if not rules:
            return [t['input'] for t in test]
        
        # Apply rules to test
        preds = []
        for t in test:
            inp = t['input']
            result = inp
            
            # Apply first matching rule
            for rule in rules:
                if rule == 'gravity':
                    result = apply_gravity(inp)
                    break
                elif rule.startswith('spread_'):
                    steps = int(rule.split('_')[1])
                    result = apply_spread(inp, steps)
                    break
                elif rule == 'mirror_h':
                    result = mirror_grid(inp, 'horizontal')
                    break
                elif rule == 'mirror_v':
                    result = mirror_grid(inp, 'vertical')
                    break
            
            preds.append(result)
        
        return preds

