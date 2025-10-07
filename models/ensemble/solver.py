"""
Ensemble Meta-Solver - NOVEL APPROACH
=====================================
Combines predictions from multiple solvers using:
1. Consensus voting (most common prediction)
2. Confidence-weighted combination
3. Train-set validation scoring
4. Adaptive solver selection per task

Innovation: Different tasks need different approaches.
Learn which solver works best for each task type.
"""
from typing import List, Dict, Tuple, Optional
from collections import Counter
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.canonicalizer.solver import Canonicalizer
from models.invertible_dsl.solver import InvertibleDSL
from models.egraph.solver import EGraphSolverCore
from models.physics_simulator.solver import PhysicsSimulator
from models.abstraction_learner.solver import AbstractionLearner

Grid = List[List[int]]


def grid_hash(grid: Grid) -> tuple:
    """Create hashable representation of grid"""
    return tuple(tuple(row) for row in grid)


def grids_equal(g1: Grid, g2: Grid) -> bool:
    """Check if two grids are equal"""
    return g1 == g2


class EnsembleSolver:
    def __init__(self):
        self.solvers = {
            'canonicalizer': Canonicalizer(),
            'invertible_dsl': InvertibleDSL(),
            'egraph': EGraphSolverCore(),
            'physics': PhysicsSimulator(),
            'abstraction': AbstractionLearner(),
        }
    
    def _evaluate_solver_on_train(self, solver_name: str, task: Dict) -> float:
        """Score a solver on training examples"""
        solver = self.solvers[solver_name]
        train = task.get('train', [])
        
        if not train:
            return 0.0
        
        # Create mini-task from train and test on it
        correct = 0
        total = 0
        
        for i, test_ex in enumerate(train):
            # Use all other examples as "train"
            mini_train = train[:i] + train[i+1:]
            if not mini_train:
                continue
            
            mini_task = {
                'train': mini_train,
                'test': [{'input': test_ex['input']}]
            }
            
            try:
                preds = solver.predict(mini_task)
                if preds and grids_equal(preds[0], test_ex['output']):
                    correct += 1
                total += 1
            except:
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def predict(self, task) -> List[Grid]:
        train = task.get('train', [])
        test = task.get('test', [])
        
        if not test:
            return []
        
        # Score each solver on train set
        solver_scores = {}
        for name in self.solvers.keys():
            try:
                score = self._evaluate_solver_on_train(name, task)
                solver_scores[name] = score
            except:
                solver_scores[name] = 0.0
        
        # Get predictions from all solvers
        all_predictions = {}
        for name, solver in self.solvers.items():
            try:
                preds = solver.predict(task)
                all_predictions[name] = preds
            except:
                all_predictions[name] = [t['input'] for t in test]
        
        # Combine predictions
        final_preds = []
        for test_idx in range(len(test)):
            # Collect all predictions for this test case
            candidate_preds = []
            weights = []
            
            for name, preds in all_predictions.items():
                if test_idx < len(preds):
                    candidate_preds.append(preds[test_idx])
                    weights.append(solver_scores.get(name, 0.0))
            
            if not candidate_preds:
                final_preds.append(test[test_idx]['input'])
                continue
            
            # Strategy 1: If one solver has perfect train score, use it
            max_score = max(weights) if weights else 0.0
            if max_score >= 0.99:
                best_idx = weights.index(max_score)
                final_preds.append(candidate_preds[best_idx])
                continue
            
            # Strategy 2: Majority vote among unique predictions
            pred_counter = Counter()
            for pred, weight in zip(candidate_preds, weights):
                pred_hash = grid_hash(pred)
                pred_counter[pred_hash] += 1 + weight  # Vote + weight
            
            if pred_counter:
                best_hash = pred_counter.most_common(1)[0][0]
                # Find original prediction
                for pred in candidate_preds:
                    if grid_hash(pred) == best_hash:
                        final_preds.append(pred)
                        break
            else:
                final_preds.append(test[test_idx]['input'])
        
        return final_preds
