"""
SUPER ENSEMBLE - Ultimate Meta-Solver v2.0
===========================================
Combines ALL 8 solvers with advanced voting, confidence scoring, and smart selection.

Features:
- Weighted voting based on solver track record
- Per-task-type solver specialization
- Confidence-based prediction selection
- Fallback chains for robustness

Target: 85%+ by leveraging complementary strengths
"""
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.canonicalizer.solver import Canonicalizer
from models.invertible_dsl.solver import InvertibleDSL
from models.egraph.solver import EGraphSolverCore
from models.physics_simulator.solver import PhysicsSimulator
from models.abstraction_learner.solver import AbstractionLearner
from models.hybrid_super.solver import HybridSuperSolver
from models.hybrid_ultra.solver import HybridUltraSolver

Grid = List[List[int]]


def grid_hash(grid: Grid) -> tuple:
    """Hashable grid representation"""
    return tuple(tuple(row) for row in grid)


def grids_equal(g1: Grid, g2: Grid) -> bool:
    return g1 == g2


def grid_similarity(g1: Grid, g2: Grid) -> float:
    """Compute similarity between grids (0.0 to 1.0)"""
    if not g1 or not g2:
        return 0.0
    
    if len(g1) != len(g2) or len(g1[0]) != len(g2[0]):
        return 0.0
    
    h, w = len(g1), len(g1[0])
    matches = sum(1 for r in range(h) for c in range(w) if g1[r][c] == g2[r][c])
    total = h * w
    
    return matches / total if total > 0 else 0.0


class TaskAnalyzer:
    """Analyze task characteristics to route to best solver"""
    
    @staticmethod
    def analyze(task) -> Dict:
        """Extract task features"""
        train = task.get('train', [])
        if not train:
            return {'type': 'unknown'}
        
        first = train[0]
        inp, out = first['input'], first['output']
        
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        
        features = {
            'input_shape': (ih, iw),
            'output_shape': (oh, ow),
            'same_shape': (ih == oh and iw == ow),
            'expansion': (oh > ih or ow > iw),
            'reduction': (oh < ih or ow < iw),
            'size_ratio': (oh / ih if ih > 0 else 1, ow / iw if iw > 0 else 1),
            'num_train': len(train),
        }
        
        # Classify task type
        if features['same_shape']:
            features['type'] = 'same_shape'
        elif features['expansion']:
            h_ratio, w_ratio = features['size_ratio']
            if h_ratio == int(h_ratio) and w_ratio == int(w_ratio):
                features['type'] = 'tiling_or_scaling'
            else:
                features['type'] = 'complex_expansion'
        elif features['reduction']:
            features['type'] = 'reduction'
        else:
            features['type'] = 'complex'
        
        return features


class SuperEnsemble:
    """Advanced ensemble combining all solvers"""
    
    def __init__(self):
        print("🚀 Initializing Super Ensemble with 7 solvers...")
        
        # Initialize all solvers
        self.solvers = {
            'ultra': HybridUltraSolver(),
            'hybrid': HybridSuperSolver(),
            'canonicalizer': Canonicalizer(),
            'abstraction': AbstractionLearner(),
            'physics': PhysicsSimulator(),
            'invertible': InvertibleDSL(),
            'egraph': EGraphSolverCore(),
        }
        
        # Solver weights (learned from training performance)
        self.base_weights = {
            'ultra': 2.5,  # Newest, most operations
            'hybrid': 2.0,  # Strong performer
            'canonicalizer': 1.5,
            'abstraction': 1.5,
            'physics': 1.0,
            'invertible': 0.8,
            'egraph': 0.8,
        }
        
        # Task-type specific solver preferences
        self.type_preferences = {
            'same_shape': ['ultra', 'hybrid', 'canonicalizer', 'physics'],
            'tiling_or_scaling': ['ultra', 'hybrid', 'abstraction', 'canonicalizer'],
            'reduction': ['abstraction', 'ultra', 'hybrid'],
            'complex': ['ultra', 'hybrid', 'canonicalizer'],
            'unknown': ['ultra', 'hybrid', 'abstraction'],
        }
        
        print(f"✅ Super Ensemble ready with {len(self.solvers)} solvers")
    
    def _evaluate_solver_on_train(self, solver_name: str, task: Dict) -> Tuple[float, int]:
        """Evaluate solver on training set, return (accuracy, num_correct)"""
        solver = self.solvers[solver_name]
        train = task.get('train', [])
        
        if not train or len(train) < 2:
            return 0.0, 0
        
        correct = 0
        total = 0
        
        # Leave-one-out cross-validation on train set
        for i in range(len(train)):
            mini_train = train[:i] + train[i+1:]
            test_example = train[i]
            
            mini_task = {
                'train': mini_train,
                'test': [{'input': test_example['input']}]
            }
            
            try:
                preds = solver.predict(mini_task)
                if preds and grids_equal(preds[0], test_example['output']):
                    correct += 1
                total += 1
            except:
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, correct
    
    def _get_solver_priority(self, task_features: Dict) -> List[str]:
        """Get prioritized list of solvers for this task type"""
        task_type = task_features.get('type', 'unknown')
        preferred = self.type_preferences.get(task_type, list(self.solvers.keys()))
        
        # Add remaining solvers
        all_solvers = list(self.solvers.keys())
        for s in all_solvers:
            if s not in preferred:
                preferred.append(s)
        
        return preferred
    
    def predict(self, task) -> List[Grid]:
        """Predict using ensemble"""
        train = task.get('train', [])
        test = task.get('test', [])
        
        if not test:
            return []
        
        # Analyze task
        features = TaskAnalyzer.analyze(task)
        
        # Get solver priority
        solver_priority = self._get_solver_priority(features)
        
        # Evaluate solvers on training set
        solver_scores = {}
        for name in solver_priority[:5]:  # Top 5 priority solvers
            try:
                accuracy, _ = self._evaluate_solver_on_train(name, task)
                solver_scores[name] = accuracy
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
            # Collect predictions for this test case
            candidate_preds = []
            weights = []
            
            for solver_name, preds in all_predictions.items():
                if test_idx < len(preds):
                    pred = preds[test_idx]
                    
                    # Weight = base_weight * train_accuracy
                    base_weight = self.base_weights.get(solver_name, 1.0)
                    train_accuracy = solver_scores.get(solver_name, 0.0)
                    
                    # Boost if solver had perfect train accuracy
                    if train_accuracy >= 0.99:
                        weight = base_weight * 10.0
                    else:
                        weight = base_weight * (1.0 + train_accuracy)
                    
                    candidate_preds.append(pred)
                    weights.append(weight)
            
            if not candidate_preds:
                final_preds.append(test[test_idx]['input'])
                continue
            
            # Strategy 1: If any solver has perfect train score, use it
            max_weight = max(weights) if weights else 0
            if max_weight > 10:  # Perfect score
                best_idx = weights.index(max_weight)
                final_preds.append(candidate_preds[best_idx])
                continue
            
            # Strategy 2: Weighted voting
            pred_votes = defaultdict(float)
            for pred, weight in zip(candidate_preds, weights):
                pred_hash = grid_hash(pred)
                pred_votes[pred_hash] += weight
            
            if pred_votes:
                best_hash = max(pred_votes.items(), key=lambda x: x[1])[0]
                # Find original prediction
                for pred in candidate_preds:
                    if grid_hash(pred) == best_hash:
                        final_preds.append(pred)
                        break
            else:
                final_preds.append(test[test_idx]['input'])
        
        return final_preds