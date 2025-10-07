import os
import json
import click
from typing import List, Dict, Any

from common.arc_io import iter_tasks
from common.arcprize_adapter import iter_aggregated
from common.eval import write_results_csv

from models.canonicalizer.solver import Canonicalizer
from models.invertible_dsl.solver import InvertibleDSL
from models.egraph.solver import EGraphSolverCore
from models.physics_simulator.solver import PhysicsSimulator
from models.abstraction_learner.solver import AbstractionLearner
from models.ensemble.solver import EnsembleSolver
from models.hybrid_super.solver import HybridSuperSolver
from models.hybrid_ultra.solver import HybridUltraSolver
from models.super_ensemble.solver import SuperEnsemble
from models.template_matcher.solver import TemplateMatcher
from models.object_solver.solver import ObjectSolver

SOLVERS = {
    'canonicalizer': Canonicalizer,
    'invertible_dsl': InvertibleDSL,
    'egraph': EGraphSolverCore,
    'physics': PhysicsSimulator,
    'abstraction': AbstractionLearner,
    'ensemble': EnsembleSolver,
    'hybrid': HybridSuperSolver,
    'ultra': HybridUltraSolver,
    'super': SuperEnsemble,
    'template': TemplateMatcher,
    'objects': ObjectSolver,
}


@click.command()
@click.option('--solver', type=click.Choice(['canonicalizer', 'invertible_dsl', 'egraph', 'physics', 'abstraction', 'ensemble', 'hybrid', 'ultra', 'super', 'template', 'objects', 'best', 'all']), default='all')
@click.option('--data', 'data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True)
@click.option('--split', type=click.Choice(['dir', 'train', 'evaluation', 'test']), default='dir', help='dir=per-file JSON directory; or aggregated ARC Prize splits')
@click.option('--out', 'out_dir', type=click.Path(file_okay=False, dir_okay=True), default='outputs')
def main(solver: str, data_dir: str, split: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    if solver == 'all':
        solvers = ['canonicalizer', 'invertible_dsl', 'egraph', 'physics', 'abstraction', 'ensemble', 'hybrid', 'ultra', 'super', 'template', 'objects']
    elif solver == 'best':
        solvers = ['ultra', 'super', 'template', 'objects']  # Best 4
    else:
        solvers = [solver]
    rows: List[Dict[str, Any]] = []

    for solver_name in solvers:
        cls = SOLVERS[solver_name]
        inst = cls()
        solved = 0
        total = 0
        if split == 'dir':
            iterator = iter_tasks(data_dir)
        else:
            # aggregated format under data_dir
            if split == 'train':
                ch = os.path.join(data_dir, 'arc-agi_training_challenges.json')
                sol = os.path.join(data_dir, 'arc-agi_training_solutions.json')
            elif split == 'evaluation':
                ch = os.path.join(data_dir, 'arc-agi_evaluation_challenges.json')
                sol = os.path.join(data_dir, 'arc-agi_evaluation_solutions.json')
            else:  # test (no solutions)
                ch = os.path.join(data_dir, 'arc-agi_test_challenges.json')
                sol = None
            if sol and os.path.exists(sol):
                iterator = iter_aggregated(ch, sol)
            else:
                # fall back to only challenges; predictions will be evaluated as 0 correct
                def _iter_only():
                    import orjson
                    with open(ch, 'rb') as f:
                        data = orjson.loads(f.read())
                    for k, ex in data.items():
                        yield k, ex
                iterator = _iter_only()

        for fname, task in iterator:
            preds = inst.predict(task)
            targets = [x['output'] for x in task.get('test', []) if 'output' in x]
            correct = 0
            if targets:
                correct = sum(1 for p, t in zip(preds, targets) if p == t)
                total += len(targets)
                solved += correct
            rows.append({
                'solver': solver_name,
                'task': fname,
                'num_test': len(task.get('test', [])),
                'num_correct': correct
            })
        acc = (solved / total) if total else 0.0
        print(f"{solver_name}: accuracy={acc:.3f} ({solved}/{total})")

    out_csv = os.path.join(out_dir, 'results.csv')
    write_results_csv(rows, out_csv)
    print(f"Wrote {out_csv}")


if __name__ == '__main__':
    main()
