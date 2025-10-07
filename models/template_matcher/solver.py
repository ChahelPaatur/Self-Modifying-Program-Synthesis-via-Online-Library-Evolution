"""
Template Matcher - Pattern Recognition Solver
==============================================
Recognizes common ARC-AGI patterns and applies pre-defined solutions.

Common patterns from ARC analysis:
1. Grid completion (fill missing parts)
2. Pattern extension (continue sequence)
3. Symmetry completion
4. Color mapping puzzles
5. Tiling/repetition detection
6. Border/frame operations
7. Interior/exterior separation
8. Counting and arithmetic
9. Sorting by properties
10. Copy with modifications

Target: +10-15% by recognizing these patterns
"""
from typing import List, Dict, Tuple, Optional, Callable
from collections import Counter
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

Grid = List[List[int]]


# ==================== TEMPLATE DETECTION ====================

class Template:
    """Base class for pattern templates"""
    def __init__(self, name: str):
        self.name = name
    
    def matches(self, task: Dict) -> bool:
        """Check if this template matches the task"""
        raise NotImplementedError
    
    def solve(self, task: Dict) -> Optional[List[Grid]]:
        """Apply template solution"""
        raise NotImplementedError


class SymmetryCompletionTemplate(Template):
    """Complete partial symmetry in grid"""
    def __init__(self):
        super().__init__("symmetry_completion")
    
    def matches(self, task: Dict) -> bool:
        train = task.get('train', [])
        if not train:
            return False
        
        # Check if outputs show completed symmetry
        for ex in train:
            inp, out = ex['input'], ex['output']
            if not self._is_symmetric(out) and self._has_partial_symmetry(inp):
                return True
        return False
    
    def _is_symmetric(self, grid: Grid) -> bool:
        """Check if grid is symmetric"""
        if not grid:
            return False
        h = len(grid)
        # Check horizontal symmetry
        for r in range(h // 2):
            if grid[r] != grid[h - 1 - r]:
                return False
        return True
    
    def _has_partial_symmetry(self, grid: Grid) -> bool:
        """Check if grid has partial symmetry"""
        # Simplified check
        return True
    
    def solve(self, task: Dict) -> Optional[List[Grid]]:
        test = task.get('test', [])
        preds = []
        
        for t in test:
            inp = t['input']
            # Complete symmetry (horizontal)
            h = len(inp)
            result = [row[:] for row in inp]
            for r in range(h // 2):
                result[h - 1 - r] = result[r][:]
            preds.append(result)
        
        return preds


class TilingDetectionTemplate(Template):
    """Detect when output is tiled version of input"""
    def __init__(self):
        super().__init__("tiling_detection")
    
    def matches(self, task: Dict) -> bool:
        train = task.get('train', [])
        if not train:
            return False
        
        # Check if all outputs are exact integer multiples of inputs
        for ex in train:
            inp, out = ex['input'], ex['output']
            ih, iw = len(inp), len(inp[0]) if inp else 0
            oh, ow = len(out), len(out[0]) if out else 0
            
            if oh % ih != 0 or ow % iw != 0:
                return False
            
            # Verify it's actual tiling
            if not self._is_tiled(inp, out):
                return False
        
        return True
    
    def _is_tiled(self, inp: Grid, out: Grid) -> bool:
        """Check if out is tiled version of inp"""
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        
        for r in range(oh):
            for c in range(ow):
                if out[r][c] != inp[r % ih][c % iw]:
                    return False
        return True
    
    def solve(self, task: Dict) -> Optional[List[Grid]]:
        train = task.get('train', [])
        test = task.get('test', [])
        
        if not train:
            return None
        
        # Detect tiling factor from first example
        ex = train[0]
        inp, out = ex['input'], ex['output']
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        
        tile_h = oh // ih if ih > 0 else 1
        tile_w = ow // iw if iw > 0 else 1
        
        preds = []
        for t in test:
            inp = t['input']
            ih, iw = len(inp), len(inp[0]) if inp else 0
            
            # Tile the input
            result = []
            for _ in range(tile_h):
                for row in inp:
                    result.append(row * tile_w)
            preds.append(result)
        
        return preds


class ColorMappingTemplate(Template):
    """Simple color remapping (most common pattern for same-shape tasks)"""
    def __init__(self):
        super().__init__("color_mapping")
    
    def matches(self, task: Dict) -> bool:
        train = task.get('train', [])
        if not train:
            return False
        
        # Check if all train examples are same shape with simple color changes
        for ex in train:
            inp, out = ex['input'], ex['output']
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return False
        
        return True
    
    def solve(self, task: Dict) -> Optional[List[Grid]]:
        train = task.get('train', [])
        test = task.get('test', [])
        
        if not train:
            return None
        
        # Try to find consistent color mapping
        color_map = {}
        
        for ex in train:
            inp, out = ex['input'], ex['output']
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    in_c = inp[r][c]
                    out_c = out[r][c]
                    
                    if in_c in color_map:
                        if color_map[in_c] != out_c:
                            # Inconsistent mapping - not a simple color map
                            return None
                    else:
                        color_map[in_c] = out_c
        
        # Apply mapping to test
        preds = []
        for t in test:
            inp = t['input']
            result = [[color_map.get(cell, cell) for cell in row] for row in inp]
            preds.append(result)
        
        return preds


class BorderExtractionTemplate(Template):
    """Extract border/frame from grid"""
    def __init__(self):
        super().__init__("border_extraction")
    
    def matches(self, task: Dict) -> bool:
        train = task.get('train', [])
        if not train:
            return False
        
        # Check if outputs are borders of inputs
        for ex in train:
            inp, out = ex['input'], ex['output']
            if not self._is_border_extraction(inp, out):
                return False
        
        return True
    
    def _is_border_extraction(self, inp: Grid, out: Grid) -> bool:
        """Check if out is border of inp"""
        if not inp or not out:
            return False
        
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return False
        
        h, w = len(inp), len(inp[0])
        
        # Check if interior is cleared
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if out[r][c] != 0 and out[r][c] == inp[r][c]:
                    return False
        
        return True
    
    def solve(self, task: Dict) -> Optional[List[Grid]]:
        test = task.get('test', [])
        preds = []
        
        for t in test:
            inp = t['input']
            h, w = len(inp), len(inp[0]) if inp else 0
            result = [[0] * w for _ in range(h)]
            
            # Copy border
            for r in range(h):
                result[r][0] = inp[r][0]
                result[r][w - 1] = inp[r][w - 1]
            for c in range(w):
                result[0][c] = inp[0][c]
                result[h - 1][c] = inp[h - 1][c]
            
            preds.append(result)
        
        return preds


class PatternFillTemplate(Template):
    """Fill grid with repeating pattern"""
    def __init__(self):
        super().__init__("pattern_fill")
    
    def matches(self, task: Dict) -> bool:
        train = task.get('train', [])
        if not train:
            return False
        
        # Check if outputs show repeating patterns
        for ex in train:
            out = ex['output']
            if self._has_repeating_pattern(out):
                return True
        
        return False
    
    def _has_repeating_pattern(self, grid: Grid) -> bool:
        """Detect if grid has repeating pattern"""
        if not grid or len(grid) < 2:
            return False
        
        # Check for row repetition
        if len(set(tuple(row) for row in grid)) < len(grid) / 2:
            return True
        
        return False
    
    def solve(self, task: Dict) -> Optional[List[Grid]]:
        # Simplified - just return input
        test = task.get('test', [])
        return [t['input'] for t in test]


class DownscaleTemplate(Template):
    """Detect downscaling by factor"""
    def __init__(self):
        super().__init__("downscale")
    
    def matches(self, task: Dict) -> bool:
        train = task.get('train', [])
        if not train:
            return False
        
        for ex in train:
            inp, out = ex['input'], ex['output']
            ih, iw = len(inp), len(inp[0]) if inp else 0
            oh, ow = len(out), len(out[0]) if out else 0
            
            # Check if output is smaller
            if oh >= ih or ow >= iw:
                return False
            
            # Check if it's integer factor
            if ih % oh != 0 or iw % ow != 0:
                return False
        
        return True
    
    def solve(self, task: Dict) -> Optional[List[Grid]]:
        train = task.get('train', [])
        test = task.get('test', [])
        
        if not train:
            return None
        
        # Detect scaling factor
        ex = train[0]
        inp, out = ex['input'], ex['output']
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        
        scale_h = ih // oh if oh > 0 else 1
        scale_w = iw // ow if ow > 0 else 1
        
        preds = []
        for t in test:
            inp = t['input']
            ih, iw = len(inp), len(inp[0]) if inp else 0
            
            # Downscale
            result = []
            for r in range(0, ih, scale_h):
                row = []
                for c in range(0, iw, scale_w):
                    # Take first cell of block
                    row.append(inp[r][c])
                result.append(row)
            preds.append(result)
        
        return preds


class FlipTemplate(Template):
    """Detect horizontal/vertical flips"""
    def __init__(self):
        super().__init__("flip")
    
    def matches(self, task: Dict) -> bool:
        train = task.get('train', [])
        if not train:
            return False
        
        for ex in train:
            inp, out = ex['input'], ex['output']
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return False
            
            # Check if it's a flip
            if not (self._is_hflip(inp, out) or self._is_vflip(inp, out)):
                return False
        
        return True
    
    def _is_hflip(self, inp: Grid, out: Grid) -> bool:
        """Check horizontal flip"""
        return all(inp[r] == out[r][::-1] for r in range(len(inp)))
    
    def _is_vflip(self, inp: Grid, out: Grid) -> bool:
        """Check vertical flip"""
        return inp[::-1] == out
    
    def solve(self, task: Dict) -> Optional[List[Grid]]:
        train = task.get('train', [])
        test = task.get('test', [])
        
        if not train:
            return None
        
        # Detect flip type
        ex = train[0]
        inp, out = ex['input'], ex['output']
        
        is_hflip = self._is_hflip(inp, out)
        
        preds = []
        for t in test:
            inp = t['input']
            if is_hflip:
                result = [row[::-1] for row in inp]
            else:
                result = inp[::-1]
            preds.append(result)
        
        return preds


class TransposeTemplate(Template):
    """Detect transpose/rotation"""
    def __init__(self):
        super().__init__("transpose")
    
    def matches(self, task: Dict) -> bool:
        train = task.get('train', [])
        if not train:
            return False
        
        for ex in train:
            inp, out = ex['input'], ex['output']
            ih, iw = len(inp), len(inp[0]) if inp else 0
            oh, ow = len(out), len(out[0]) if out else 0
            
            # Check if dimensions are swapped
            if ih != ow or iw != oh:
                return False
        
        return True
    
    def solve(self, task: Dict) -> Optional[List[Grid]]:
        test = task.get('test', [])
        preds = []
        
        for t in test:
            inp = t['input']
            h, w = len(inp), len(inp[0]) if inp else 0
            
            # Transpose
            result = [[inp[r][c] for r in range(h)] for c in range(w)]
            preds.append(result)
        
        return preds


class MostCommonColorTemplate(Template):
    """Replace all with most common color"""
    def __init__(self):
        super().__init__("most_common_color")
    
    def matches(self, task: Dict) -> bool:
        train = task.get('train', [])
        if not train:
            return False
        
        # Check if output is uniform color
        for ex in train:
            out = ex['output']
            if not self._is_uniform(out):
                return False
        
        return True
    
    def _is_uniform(self, grid: Grid) -> bool:
        """Check if grid has single color"""
        if not grid:
            return False
        colors = {cell for row in grid for cell in row}
        return len(colors) == 1
    
    def solve(self, task: Dict) -> Optional[List[Grid]]:
        test = task.get('test', [])
        preds = []
        
        for t in test:
            inp = t['input']
            h, w = len(inp), len(inp[0]) if inp else 0
            
            # Get most common color
            counter = Counter(cell for row in inp for cell in row)
            most_common = counter.most_common(1)[0][0] if counter else 0
            
            # Fill with it
            result = [[most_common] * w for _ in range(h)]
            preds.append(result)
        
        return preds


# ==================== TEMPLATE MATCHER SOLVER ====================

class TemplateMatcher:
    """Solver using template matching"""
    
    def __init__(self):
        # Initialize all templates in order of specificity
        self.templates = [
            ColorMappingTemplate(),  # Most common - try first
            FlipTemplate(),
            TransposeTemplate(),
            TilingDetectionTemplate(),
            DownscaleTemplate(),
            SymmetryCompletionTemplate(),
            BorderExtractionTemplate(),
            MostCommonColorTemplate(),
            PatternFillTemplate(),
        ]
        
        print(f"🎯 Template Matcher initialized with {len(self.templates)} templates")
    
    def predict(self, task) -> List[Grid]:
        """Try each template in order"""
        test = task.get('test', [])
        
        if not test:
            return []
        
        # Try each template
        for template in self.templates:
            try:
                if template.matches(task):
                    result = template.solve(task)
                    if result is not None and len(result) == len(test):
                        # Verify on train if possible
                        if self._verify_on_train(template, task):
                            return result
            except:
                continue
        
        # Fallback: return input
        return [t['input'] for t in test]
    
    def _verify_on_train(self, template: Template, task: Dict) -> bool:
        """Verify template works on training examples"""
        train = task.get('train', [])
        if not train:
            return True
        
        try:
            # Create mini-task with train as test
            mini_task = {
                'train': train,
                'test': [{'input': ex['input']} for ex in train]
            }
            
            preds = template.solve(mini_task)
            if preds is None:
                return False
            
            # Check if predictions match
            for i, ex in enumerate(train):
                if i < len(preds) and preds[i] != ex['output']:
                    return False
            
            return True
        except:
            return False
