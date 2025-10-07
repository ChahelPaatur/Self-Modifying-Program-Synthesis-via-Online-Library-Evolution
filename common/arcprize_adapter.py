import orjson
from typing import Dict, Any, Iterator, Tuple, List


def iter_aggregated(path_challenges: str, path_solutions: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    with open(path_challenges, 'rb') as f:
        ch = orjson.loads(f.read())
    with open(path_solutions, 'rb') as f:
        sol = orjson.loads(f.read())
    for k, ex in ch.items():
        task = {"train": ex["train"], "test": ex["test"]}
        # Attach targets from solutions into test entries when available
        solutions = sol.get(k, [])
        test = task.get("test", [])
        if isinstance(solutions, list) and len(solutions) == len(test):
            for i, t in enumerate(test):
                t["output"] = solutions[i]
        yield k, task
