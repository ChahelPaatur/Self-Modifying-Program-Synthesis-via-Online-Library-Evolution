"""
Object-Oriented Solver
======================
Specialized solver for tasks involving object detection and manipulation.

Strategies:
1. Detect objects in input
2. Infer transformation on objects (move, scale, rotate, filter, sort)
3. Apply to test inputs

Target: Tasks with clear object structures (20-30% of ARC)
"""
from typing import List, Dict, Tuple, Optional, Set, Callable
from collections import Counter
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from common.object_ops import *

Grid = List[List[int]]


class ObjectSolver:
    """Solver using object-level operations"""
    
    def __init__(self):
        # Build object operation library
        self.operations = self._build_operations()
        print(f"🎯 Object Solver initialized with {len(self.operations)} object operations")
    
    def _build_operations(self) -> List[Callable]:
        """Build list of object operations to try"""
        ops = []
        
        # Object filtering
        ops.append(('keep_largest', lambda g: keep_largest_object(g, bg_color=0)))
        ops.append(('keep_largest_bg1', lambda g: keep_largest_object(g, bg_color=1)))
        
        # Object alignment
        ops.append(('align_vertical', lambda g: align_objects_vertically(g, bg_color=0)))
        ops.append(('align_horizontal', lambda g: align_objects_horizontally(g, bg_color=0)))
        
        # Object centering
        ops.append(('center', lambda g: center_objects(g, bg_color=0)))
        
        # Object duplication
        for count in [2, 3, 4]:
            ops.append((f'duplicate_{count}x', lambda g, c=count: duplicate_objects(g, count=c, bg_color=0)))
        
        return ops
    
    def predict(self, task) -> List[Grid]:
        """Predict using object operations"""
        train = task.get('train', [])
        test = task.get('test', [])
        
        if not train or not test:
            return [t['input'] for t in test]
        
        # Try to find object-based transformation
        for op_name, op_func in self.operations:
            if self._test_operation(train, op_func):
                # Apply to test
                preds = []
                for t in test:
                    try:
                        result = op_func(t['input'])
                        preds.append(result if result else t['input'])
                    except:
                        preds.append(t['input'])
                return preds
        
        # Try compositional object operations
        result = self._try_compositional(train, test)
        if result:
            return result
        
        # Fallback
        return [t['input'] for t in test]
    
    def _test_operation(self, train: List[Dict], op_func: Callable) -> bool:
        """Test if operation works on all training examples"""
        try:
            for ex in train:
                result = op_func(ex['input'])
                if result != ex['output']:
                    return False
            return True
        except:
            return False
    
    def _try_compositional(self, train: List[Dict], test: List[Dict]) -> Optional[List[Grid]]:
        """Try combinations of object operations"""
        # Try pairs of operations
        for op1_name, op1_func in self.operations[:10]:
            for op2_name, op2_func in self.operations[:10]:
                try:
                    # Test on training set
                    works = True
                    for ex in train:
                        result = op1_func(ex['input'])
                        result = op2_func(result)
                        if result != ex['output']:
                            works = False
                            break
                    
                    if works:
                        # Apply to test
                        preds = []
                        for t in test:
                            result = op1_func(t['input'])
                            result = op2_func(result)
                            preds.append(result)
                        return preds
                except:
                    continue
        
        return None
