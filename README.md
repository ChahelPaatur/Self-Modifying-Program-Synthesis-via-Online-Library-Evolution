# Self-Modifying Program Synthesis via Online Library Evolution

> **A mechanistic approach to few-shot abstract reasoning with cross-domain validation**  
> Independent Research Project, 2024-2025 Academic Year  
> Chahel Paatur | John C. Kimball High School

---

## Abstract

SMPMA (Self-Modifying Program Synthesis with Adaptive Library Evolution) is a neuro-symbolic system that expands its capability space during inference through online library evolution. The system learns reusable program abstractions from successful solutions and promotes high-utility patterns into an active Domain-Specific Language (DSL), enabling compositional reasoning on few-shot tasks.

**Key Results:**
- **2.5% strict accuracy** on ARC-AGI evaluation (47 learned abstractions, 12 macro templates)
- **62% strict accuracy** on AeroSynth controlled benchmark (25× improvement, validates mechanism)
- **100% rendezvous success** on spacecraft autonomy with 79% fuel reduction (cross-domain validation)

---

## Research Contributions

### 1. Mechanistic Architecture
Three-stage cognitive loop with typed beam-search synthesis, template-aware abstraction mining (fixed sequences + parameterized templates), and cross-task utility estimation with promotion/pruning logic.

### 2. Dual-Input Compositional Primitives
Extended DSL with two-grid operations (`overlay`, `xor_nonzero`, conditional masking) enabling "compare A to B and act accordingly" patterns.

### 3. Object-Centric Scratchpad Model
Three-register execution (`canvas`, `obj`, `original`) for structured multi-step object manipulation.

### 4. Cross-Domain Validation
Spacecraft rendezvous/docking with Clohessy-Wiltshire dynamics, Kalman filtering, LQR/MPC control, and safety monitoring—demonstrating abstraction over verifiable structure.

---

## Repository Structure

```
.
├── RESEARCH_PAPER.md              # Main research paper (MLA format, 3,400+ words)
├── smpma_agi.py                   # Core SMPMA implementation (1,803 lines)
├── smpma_benchmark.py             # ARC-AGI evaluation pipeline
├── aero_synth_benchmark.py        # Controlled synthetic benchmark
├── spacecraft_nav_autonomy.py     # Spacecraft rendezvous/docking simulation
├── spacecraft_nav_benchmark.py    # Spacecraft ablation study runner
├── brain_arc_solver.py            # Baseline DSL solver (comparison)
├── paper_assets/
│   ├── figures/                   # Publication-quality figures (PDF/PNG, 300 DPI)
│   │   ├── fig_spacecraft_architecture_detailed.png
│   │   ├── fig_results_comparison.png
│   │   ├── fig_arc_metrics_summary.png
│   │   └── ... (additional benchmark figures)
│   └── spacecraft_nav_results.json  # Complete ablation data (20 seeds × 5 variants)
└── README.md                      # This file
```

---

## Installation

### Requirements
- Python 3.9+
- NumPy, SciPy (numerical computation)
- Matplotlib (visualization)

### Setup
```bash
git clone https://github.com/chahelpaatur/Self-Modifying-Program-Synthesis-via-Online-Library-Evolution.git
cd arc2025-smpma
pip install -r requirements.txt  # If requirements.txt provided
```

### Quick Start
```bash
# Run SMPMA on ARC-AGI evaluation set (first 10 tasks)
python smpma_agi.py --mode evaluation --data arc-prize-2025 --max-tasks 10

# Run controlled AeroSynth benchmark
python aero_synth_benchmark.py

# Run spacecraft ablation study (5 seeds)
python spacecraft_nav_benchmark.py --seeds 5
```

---

## Empirical Findings

### What Works
1. **Mechanism validation:** Online library evolution improves search efficiency on compositional tasks (AeroSynth: 62%)
2. **Interpretability:** All learned abstractions are explicit programs with type checking
3. **Cross-domain transfer:** Abstraction over verifiable structure works in safety-critical autonomy (spacecraft: 100% success)

### Limitations
1. **ARC performance:** 2.5% strict accuracy far below state-of-the-art (35-45%)
2. **Inductive bias mismatch:** ARC requires primitives (counting, recursion) not in initial DSL
3. **Controlled benchmarks:** AeroSynth is synthetic; real-world performance on diverse tasks unknown

### Honest Assessment
This work contributes a **mechanism for capability space expansion** with validated cross-domain applicability. It does **not** claim to solve ARC-AGI. The emphasis is on research trajectory: investigating how systems can learn to expand their operational repertoire while maintaining interpretability and verifiability.

---

## Theoretical Analysis

### Search Space Reduction
For fixed DSL with *N* operations and depth *D*, search space is *O(N^D)*. With *K* learned abstractions of average length *L*, effective depth reduces to *D/L*, giving *O((N+K)^(D/L))* < *O(N^D)* when *L* > 1.

### Sample Efficiency
SMPMA requires *O(log M)* tasks to learn an abstraction appearing in *O(M)* tasks, vs *O(M)* for non-transfer methods (cross-task evidence aggregation).

### Library Dynamics
On ARC training (400 tasks): library grows rapidly (0 → 35 abstractions in first 100 tasks), then stabilizes (35 → 47 over remaining 300) as high-utility abstractions dominate.

---

## Figures

All figures are publication-quality (300 DPI, PDF/PNG) and available in `paper_assets/figures/`:

- **Figure 1:** SMPMA architecture (three-stage cognitive loop)
- **Figure 2:** Spacecraft autonomy architecture (compositional abstraction over verifiable components)
- **Figure 3:** ARC-AGI metrics summary (4-panel: accuracy, train score, runtime, program length)
- **Figure 4:** Combined results (4-panel: ablations + spacecraft success/fuel)

---

## Future Directions

1. **Neural-guided search:** Learned heuristics to prioritize promising program candidates
2. **Hierarchical abstractions:** Meta-abstractions enabling multi-level compositional reasoning
3. **Inductive logic programming:** First-order logic for counting, recursion, higher-order operations
4. **Multi-modal extension:** Images, text, hybrid reasoning domains

---

## Related Work

**Program Synthesis:**
- DreamCoder (Ellis et al., 2021): Wake-sleep Bayesian program learning (requires offline training)
- AlphaCode (Li et al., 2022): Transformer-based with massive pre-training
- FlashFill (Gulwani, 2011): String processing from many examples

**Meta-Learning:**
- MAML (Finn et al., 2017): Gradient-based fast adaptation
- Neural Module Networks (Andreas et al., 2016): Learned module composition

**Gap:** Most systems require large-scale pre-training or don't adapt DSL online during inference.

---

## Author

**Chahel Paatur**  
Email: chahelpaatur@gmail.com  
Institution: John C. Kimball High School, Tracy, California  
Research Period: 2024-2025 Academic Year

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

This research was conducted independently as part of high school independent study. Thanks to the ARC-AGI community for the benchmark and to open-source spacecraft dynamics resources for reference implementations.

---

## Suitable For

- **University admissions portfolios:** Demonstrates independent research capability and intellectual maturity
- **Conference submissions:** AAAI Student Track, NeurIPS Program Synthesis Workshop, IJCAI Student Poster
- **Research internships:** Shows mechanism design, empirical validation, cross-domain thinking
- **Science competitions:** Intel ISEF, Regeneron STS, Google Science Fair

---

## Notes on Research Philosophy

This work emphasizes **mechanism validation** over **benchmark performance**. The primary contribution is not beating state-of-the-art ARC scores (35-45%), but rather:

1. Demonstrating a concrete mechanism for online capability space expansion
2. Validating interpretability and compositional guarantees
3. Showing cross-domain applicability to safety-critical systems

The research trajectory is toward systems that **learn to expand their own operational repertoire** while maintaining verifiability—a fundamental challenge for AGI.

---


