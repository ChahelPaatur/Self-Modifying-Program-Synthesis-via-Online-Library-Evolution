# Kaggle Notebook: ARC Prize 2025 – Simple DSL + Guided Search Submission
# - Reads /kaggle/input/arc-prize-2025/*
# - For each test task, synthesizes a short program from train pairs
# - Writes /kaggle/working/submission.json
#
# No extra dependencies. Uses only Python stdlib (json, os).

import os, json, random, time
from collections import Counter
from typing import List, Dict, Tuple, Optional

print("="*80)
print("ARC PRIZE 2025 - DSL PROGRAM SYNTHESIS SOLVER")
print("="*80)

# ---------------------------- Config ----------------------------
RANDOM_SEED = 1337
BEAM_WIDTH = 80     # lower this if you hit timeouts
MAX_DEPTH  = 4      # lower this if you hit timeouts
VERBOSE    = True   # set to False to reduce output

random.seed(RANDOM_SEED)

print(f"\nConfiguration:")
print(f"  BEAM_WIDTH: {BEAM_WIDTH}")
print(f"  MAX_DEPTH: {MAX_DEPTH}")
print(f"  VERBOSE: {VERBOSE}")
print(f"  RANDOM_SEED: {RANDOM_SEED}")

# ---------------------------- Types -----------------------------
Grid = List[List[int]]

# -------------------------- Perception --------------------------
def find_connected_components(grid: Grid, bg: int = 0) -> List[Dict]:
    if not grid: return []
    h, w = len(grid), len(grid[0])
    seen = [[False]*w for _ in range(h)]
    objects = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and not seen[r][c]:
                color = grid[r][c]
                cells = []
                q = [(r,c)]
                seen[r][c] = True
                while q:
                    r0,c0 = q.pop(0)
                    cells.append((r0,c0))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        rn,cn = r0+dr, c0+dc
                        if 0<=rn<h and 0<=cn<w and not seen[rn][cn] and grid[rn][cn]==color:
                            seen[rn][cn]=True
                            q.append((rn,cn))
                min_r = min(r for r,c in cells); max_r = max(r for r,c in cells)
                min_c = min(c for r,c in cells); max_c = max(c for r,c in cells)
                objects.append({
                    'cells': cells,
                    'color': color,
                    'bbox': (min_r,min_c,max_r,max_c),
                    'center': ((min_r+max_r)/2, (min_c+max_c)/2),
                    'size': len(cells),
                })
    return objects

# ----------------------------- DSL ------------------------------
class Program:
    def __init__(self, steps: List[Tuple[str, Tuple]]):
        self.steps=steps
    def apply(self, g: Grid) -> Optional[Grid]:
        res=[row[:] for row in g]
        for name,args in self.steps:
            res = DSL.apply(name, res, args)
            if res is None: return None
        return res
    def __repr__(self): return ' -> '.join(f"{n}{args}" for n,args in self.steps) or 'identity()'

class DSL:
    OPS  : Dict[str, Tuple] = {}     # name -> (arity, func)
    MACRO: Dict[str, callable] = {}  # name -> (g)->Grid

    @classmethod
    def register(cls, name: str, arity: int, func):
        cls.OPS[name]=(arity,func)

    @classmethod
    def register_macro(cls, name: str, func):
        cls.MACRO[name]=func

    @classmethod
    def apply(cls, name: str, g: Grid, args: Tuple) -> Optional[Grid]:
        try:
            if name in cls.OPS:
                arity, func = cls.OPS[name]
                if arity == 0: return func(g)
                if arity == 1: return func(g, args[0])
                if arity == 2: return func(g, args[0], args[1])
                return None
            elif name in cls.MACRO:
                return cls.MACRO[name](g)
        except Exception:
            return None
        return None

# ------------- Primitive ops (grid transforms/color ops) -------------
def op_identity(g: Grid) -> Grid:
    return [row[:] for row in g]

def op_rotate90(g: Grid) -> Grid:
    return [list(row) for row in zip(*g[::-1])] if g else g

def op_rotate180(g: Grid) -> Grid:
    return [row[::-1] for row in g[::-1]] if g else g

def op_rotate270(g: Grid) -> Grid:
    return [list(row) for row in zip(*g)][::-1] if g else g

def op_flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]

def op_flip_v(g: Grid) -> Grid:
    return g[::-1]

def op_transpose(g: Grid) -> Grid:
    return [list(row) for row in zip(*g)] if g else g

def op_keep_color(g: Grid, color:int) -> Grid:
    return [[cell if cell==color else 0 for cell in row] for row in g]

def op_replace_color(g: Grid, a:int, b:int) -> Grid:
    return [[(b if cell==a else cell) for cell in row] for row in g]

def op_bbox(g: Grid) -> Grid:
    objs = find_connected_components(g)
    if not objs: return g
    min_r=min(o['bbox'][0] for o in objs); max_r=max(o['bbox'][2] for o in objs)
    min_c=min(o['bbox'][1] for o in objs); max_c=max(o['bbox'][3] for o in objs)
    return [row[min_c:max_c+1] for row in g[min_r:max_r+1]]

def op_gravity_down(g: Grid) -> Grid:
    if not g: return g
    h,w=len(g),len(g[0])
    out=[[0]*w for _ in range(h)]
    for c in range(w):
        col=[g[r][c] for r in range(h) if g[r][c]!=0]
        for i,val in enumerate(reversed(col)):
            out[h-1-i][c]=val
    return out

# Register primitives
print("\nRegistering DSL operations...")
DSL.register('identity', 0, op_identity)
DSL.register('rotate90', 0, op_rotate90)
DSL.register('rotate180', 0, op_rotate180)
DSL.register('rotate270', 0, op_rotate270)
DSL.register('flip_h', 0, op_flip_h)
DSL.register('flip_v', 0, op_flip_v)
DSL.register('transpose', 0, op_transpose)
DSL.register('bbox', 0, op_bbox)
DSL.register('gravity_down', 0, op_gravity_down)
DSL.register('keep_color', 1, op_keep_color)
DSL.register('replace_color', 2, op_replace_color)
print(f"  Registered {len(DSL.OPS)} primitive operations")

# ---------------------------- Macros -----------------------------
def macro_row_color_consolidation(g: Grid) -> Grid:
    if not g: return g
    h,w=len(g),len(g[0])
    out=[row[:] for row in g]
    for r in range(h):
        row_colors=[c for c in g[r] if c!=0]
        if not row_colors: continue
        first = next((c for c in g[r] if c!=0), row_colors[0])
        for c in range(w):
            if out[r][c]!=0: out[r][c]=first
    return out

DSL.register_macro('row_color_consolidation', macro_row_color_consolidation)
print(f"  Registered {len(DSL.MACRO)} macro operations")

# ------------------------ Scoring / Search ------------------------
def grid_similarity(a: Grid, b: Grid) -> float:
    if not a or not b: return 0.0
    if len(a)!=len(b) or len(a[0])!=len(b[0]): return 0.0
    h,w=len(a),len(a[0])
    match=sum(1 for r in range(h) for c in range(w) if a[r][c]==b[r][c])
    return match/(h*w)

def suggest_colors(examples: List[Tuple[Grid,Grid]]) -> List[int]:
    colors=set()
    for inp,out in examples:
        for row in inp+out: colors.update(row)
    return [c for c in sorted(colors) if c!=0][:6]

def synthesize_program(examples: List[Tuple[Grid,Grid]], beam:int=BEAM_WIDTH, depth:int=MAX_DEPTH) -> Optional[Program]:
    color_hints=suggest_colors(examples)
    base_ops = ['identity','rotate90','rotate180','rotate270','flip_h','flip_v','transpose',
                'bbox','gravity_down','row_color_consolidation']
    candidates=[Program([])]

    def expand(prog: Program) -> List[Program]:
        progs=[]
        # arity-0 ops
        for name in base_ops:
            progs.append(Program(prog.steps+[(name,())]))
        # arity-1 ops
        for col in color_hints:
            progs.append(Program(prog.steps+[('keep_color',(col,))]))
        # arity-2 ops
        for a in color_hints:
            for b in color_hints:
                if a!=b:
                    progs.append(Program(prog.steps+[('replace_color',(a,b))]))
        return progs

    def score_prog(p: Program) -> float:
        s=0.0
        for inp,out in examples:
            pred=p.apply(inp)
            if pred is None: return -1
            s+=grid_similarity(pred,out)
        return s/len(examples)

    best=None; best_score=-1
    for d in range(depth):
        next_pool=[]
        for prog in candidates:
            for q in expand(prog):
                s=score_prog(q)
                if s==1.0:
                    if VERBOSE: print(f"    Found perfect match @ depth {d+1}: {q}")
                    return q
                if s>0: next_pool.append((s,q))
        if not next_pool: break
        next_pool.sort(key=lambda x:x[0], reverse=True)
        candidates=[q for _,q in next_pool[:beam]]
        if next_pool[0][0]>best_score: best_score=next_pool[0][0]; best=next_pool[0][1]
        if VERBOSE: print(f"    [depth {d+1}] candidates={len(next_pool)}, top_score={best_score:.3f}")
    return best

# ----------------------------- Orchestration -----------------------------
def solve_task(task: Dict) -> List[Grid]:
    train=task.get('train',[]); test=task.get('test',[])
    if not train:
        if VERBOSE: print("    ⚠ No training examples, returning input as-is")
        return [t['input'] for t in test]
    examples=[(ex['input'], ex['output']) for ex in train]
    if VERBOSE: print(f"    Training examples: {len(examples)}")
    best = synthesize_program(examples)
    if VERBOSE:
        print(f"    Selected program: {best}")
        # Show first example debug
        inp, out = examples[0]
        pred = best.apply(inp) if best else None
        sim = grid_similarity(pred, out) if pred else -1
        print(f"    First train similarity: {sim:.3f}")
    preds=[]
    for t in test:
        pred = best.apply(t['input']) if best else t['input']
        preds.append(pred if pred else t['input'])
    return preds

# --------------------------- Data Iteration ---------------------------
def read_json(path: str):
    print(f"  Reading: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} tasks")
    return data

def iter_challenges(ch_path: str, sol_path: Optional[str]):
    ch = read_json(ch_path)
    sol = read_json(sol_path) if (sol_path and os.path.exists(sol_path)) else None
    for k,ex in ch.items():
        if sol:
            yield k, {'train': ex['train'], 'test': [{'input': t['input'], 'output': s} for t,s in zip(ex['test'], sol[k]) ]}
        else:
            yield k, {'train': ex['train'], 'test': ex['test']}

# ------------------------------ Main ------------------------------
# Auto-detect if running on Kaggle or locally
if os.path.exists("/kaggle/input/arc-prize-2025"):
    BASE = "/kaggle/input/arc-prize-2025"
    OUT  = "/kaggle/working"
else:
    # Local path
    BASE = "arc-prize-2025"
    OUT  = "outputs"

# Choose mode: "train", "evaluation", or "test"
MODE = "evaluation"  # change to "evaluation" to sanity check against eval set

print(f"\nData directory: {BASE}")
print(f"Output directory: {OUT}")

# Create output directory if needed
os.makedirs(OUT, exist_ok=True)

print(f"\n{'='*80}")
print(f"MODE: {MODE.upper()}")
print(f"{'='*80}")

if MODE == "train":
    ch = os.path.join(BASE,'arc-agi_training_challenges.json')
    sl = os.path.join(BASE,'arc-agi_training_solutions.json')
elif MODE == "evaluation":
    ch = os.path.join(BASE,'arc-agi_evaluation_challenges.json')
    sl = os.path.join(BASE,'arc-agi_evaluation_solutions.json')
else:
    ch = os.path.join(BASE,'arc-agi_test_challenges.json')
    sl = None

start_time = time.time()

if MODE in ("train","evaluation"):
    print("\nRunning evaluation...")
    correct=0; total=0; task_count=0
    for tid,task in iter_challenges(ch, sl):
        task_count += 1
        if VERBOSE: 
            print(f"\n[Task {task_count}] {tid}: {len(task['train'])} train → {len(task['test'])} test")
        else:
            if task_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = task_count / elapsed if elapsed > 0 else 0
                print(f"  Progress: {task_count} tasks | {rate:.1f} tasks/sec | {elapsed:.1f}s elapsed")
        
        preds=solve_task(task)
        targets=[x['output'] for x in task.get('test',[]) if 'output' in x]
        if targets:
            c=sum(1 for p,t in zip(preds,targets) if p==t)
            correct+=c; total+=len(targets)
            if VERBOSE and c > 0:
                print(f"    SOLVED {c}/{len(targets)} test cases")
    
    elapsed = time.time() - start_time
    acc = (correct/total*100) if total else 0.0
    print(f"\n{'='*80}")
    print(f"RESULTS ({MODE.upper()})")
    print(f"{'='*80}")
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"Tasks processed: {task_count}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Rate: {task_count/elapsed:.2f} tasks/sec")
    print(f"{'='*80}")

else:
    # Generate submission.json for test set
    print("\nGenerating submission.json...")
    submission={}
    task_count = 0
    
    for tid,task in iter_challenges(ch, sl):
        task_count += 1
        if VERBOSE:
            print(f"\n[Task {task_count}] Solving {tid}...")
        else:
            if task_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = task_count / elapsed if elapsed > 0 else 0
                print(f"  Progress: {task_count} tasks | {rate:.1f} tasks/sec | {elapsed:.1f}s elapsed")
        
        preds=solve_task(task)
        # two attempts per test output (duplicate best for simplicity)
        submission[tid]=[{'attempt_1': p, 'attempt_2': p} for p in preds]
    
    out_path = os.path.join(OUT, 'submission.json')
    print(f"\nWriting submission to: {out_path}")
    with open(out_path, 'w') as f:
        json.dump(submission, f)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"SUBMISSION COMPLETE")
    print(f"{'='*80}")
    print(f"File: {out_path}")
    print(f"Tasks processed: {task_count}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Rate: {task_count/elapsed:.2f} tasks/sec")
    print(f"{'='*80}")
    print("\nReady to submit to Kaggle.")

