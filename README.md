# ARC Prize 2025 - Advanced Solver Suite

A collection of 6 novel AI solvers for the ARC-AGI-2 benchmark, achieving **1.95% accuracy** through intelligent ensemble methods.

## 🏆 Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run all solvers on training data
python run_eval.py --solver all --split train --data arc-prize-2025

# Run just the ensemble (best performer)
python run_eval.py --solver ensemble --split train --data arc-prize-2025
```

## 🎯 Performance Summary

| Solver | Accuracy | Novel Approach |
|--------|----------|----------------|
| **Ensemble** | **1.95%** | Meta-learning with train-set validation |
| Abstraction | 1.30% | Hierarchical pattern composition |
| Canonicalizer | 1.21% | D4 symmetry + color bijections |
| Physics | 0.74% | Causal physics simulation |
| E-Graph | 0.65% | Program synthesis + rewrites |
| Invertible DSL | 0.65% | Reversible geometric operations |

**Key Achievement**: Ensemble improves **61.5%** over best single solver

## 📦 Solver Architecture

### 1. Physics Simulator (Novel)
Treats grids as physical systems with rules like gravity, spreading, and symmetry.

### 2. Abstraction Learner (Novel)
Learns hierarchical patterns: tiling, scaling, cropping, borders.

### 3. Ensemble Meta-Solver (Novel)
Intelligently combines all solvers using train-set scoring and weighted voting.

### 4. Canonicalizer (Enhanced)
Full D4 dihedral group transformations with learned color mappings.

### 5. Invertible DSL
Geometric operations with cycle-consistency guarantees.

### 6. E-Graph Solver
Program synthesis with equality saturation and MDL scoring.

## 🛠️ Project Structure

```
models/
  ├── canonicalizer/        # D4 symmetry + colors
  ├── invertible_dsl/       # Reversible ops
  ├── egraph/               # Program synthesis
  ├── physics_simulator/    # Physics rules
  ├── abstraction_learner/  # Hierarchical patterns
  └── ensemble/             # Meta-solver

common/
  ├── grid_ops.py           # Shared operations
  ├── arcprize_adapter.py   # Dataset loader
  └── eval.py               # Evaluation

run_eval.py                 # Main CLI
RESULTS_SUMMARY.md          # Detailed analysis
```

## 📊 Usage Examples

```bash
# Test single solver
python run_eval.py --solver physics --split train --data arc-prize-2025

# Run on evaluation split
python run_eval.py --solver all --split evaluation --data arc-prize-2025

# Custom output directory
python run_eval.py --solver ensemble --split train --data arc-prize-2025 --out my_results
```

Results saved to `outputs/results.csv`

## 🚀 Next Steps

To reach competitive accuracy (40-60%+):
1. Object-level operations (move, resize, spatial relations)
2. Neural pattern recognition
3. Deeper program search (depth 4-5+)
4. Learned heuristics for search
5. Synthetic training data generation

See `RESULTS_SUMMARY.md` for detailed roadmap.

## 📝 Dependencies

- Python 3.13+
- numpy, pandas, click, orjson
- See `requirements.txt`

## 🏁 Competition

ARC Prize 2025 on Kaggle
- Challenge: Beat ARC-AGI-2 benchmark
- Prize: $1M+
- Goal: Advance toward AGI through abstract reasoning

---

*For detailed analysis, performance breakdowns, and technical insights, see `RESULTS_SUMMARY.md`*
