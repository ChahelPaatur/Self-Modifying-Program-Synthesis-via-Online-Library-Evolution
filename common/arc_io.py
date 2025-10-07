import os
import orjson
from typing import List, Dict, Any, Tuple

Grid = List[List[int]]
Example = Dict[str, Grid]
Task = Dict[str, List[Example]]


def load_task(path: str) -> Task:
    with open(path, 'rb') as f:
        return orjson.loads(f.read())


def iter_tasks(data_dir: str):
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith('.json'):
            continue
        yield name, load_task(os.path.join(data_dir, name))


def grid_equal(a: Grid, b: Grid) -> bool:
    return a == b


def summarize_task(task: Task) -> Tuple[int, Tuple[int, int]]:
    train = task.get('train', [])
    if not train:
        return 0, (0, 0)
    h = len(train[0]['input'])
    w = len(train[0]['input'][0]) if h > 0 else 0
    return len(train), (h, w)
