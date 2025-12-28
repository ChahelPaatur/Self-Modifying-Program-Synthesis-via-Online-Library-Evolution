"""
Brain-Inspired ARC Solver (Single File)
- Object-centric perception (CCs)
- Typed DSL (grid + object ops)
- Guided beam search (heuristics)
- Episodic retrieval (KNN over task embeddings)
- Macro mining (promote frequent subprograms)
- CLI: eval train/eval, generate submission
"""
import json, math, random, argparse, os
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict, Counter

Grid = List[List[int]]

# Verbose logging switch (set via CLI)
VERBOSE = False
USE_ENHANCED = os.environ.get('ARC_BRAIN_ENHANCED','0')=='1'

def print_grid(g: Grid, title: Optional[str]=None, max_rows:int=12, max_cols:int=40):
	if not VERBOSE: return
	if title: print(title)
	if not g:
		print("  <empty>")
		return
	h=len(g); w=len(g[0])
	rows=min(h, max_rows)
	cols=min(w, max_cols)
	for r in range(rows):
		row=g[r][:cols]
		print('  ' + ' '.join(str(x) for x in row) + (" ..." if cols<w else ""))
	if rows<h:
		print("  ...")

# --------------------------- Perception ---------------------------

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
				q = [(r,c)]; seen[r][c] = True
				while q:
					r0,c0 = q.pop(0)
					cells.append((r0,c0))
					for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
						rn,cn = r0+dr, c0+dc
						if 0<=rn<h and 0<=cn<w and not seen[rn][cn] and grid[rn][cn]==color:
							seen[rn][cn]=True; q.append((rn,cn))
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

# ------------------------------ DSL ------------------------------

class Op:
	def __init__(self, name: str, func: Callable, arity: int, kind: str):
		self.name=name; self.func=func; self.arity=arity; self.kind=kind
	def __repr__(self): return self.name

# Grid ops

def op_identity(g: Grid) -> Grid:
	return [row[:] for row in g]

def op_rotate90(g: Grid) -> Grid:
	return [list(row) for row in zip(*g[::-1])] if g else g

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

# Object ops need infer parameters from examples; exposed via macros below

# ---------------------- Program / Macros / Search ----------------------

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
	OPS: Dict[str, Op] = {}
	MACROS: Dict[str, Callable] = {}
	PARAM_HINTS: Dict[str, List] = {}
	@classmethod
	def register(cls, name, func, arity, kind):
		cls.OPS[name]=Op(name,func,arity,kind)
	@classmethod
	def register_macro(cls, name, builder: Callable):
		cls.MACROS[name]=builder
	@classmethod
	def apply(cls, name: str, g: Grid, args: Tuple) -> Optional[Grid]:
		if name in cls.OPS:
			op=cls.OPS[name]
			try:
				if op.arity==0: return op.func(g)
				elif op.arity==1: return op.func(g, args[0])
				elif op.arity==2: return op.func(g, args[0], args[1])
				else: return None
			except Exception:
				return None
		elif name in cls.MACROS:
			try: return cls.MACROS[name](g, *args)
			except Exception: return None
		return None

# Register primitive ops
DSL.register('identity', op_identity, 0, 'grid')
DSL.register('rotate90', op_rotate90, 0, 'grid')
DSL.register('flip_h', op_flip_h, 0, 'grid')
DSL.register('flip_v', op_flip_v, 0, 'grid')
DSL.register('transpose', op_transpose, 0, 'grid')
DSL.register('keep_color', op_keep_color, 1, 'grid')
DSL.register('replace_color', op_replace_color, 2, 'grid')
DSL.register('bbox', op_bbox, 0, 'grid')
DSL.register('gravity_down', op_gravity_down, 0, 'grid')

# Macros (data-driven)

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

DSL.register_macro('row_color_consolidation', lambda g: macro_row_color_consolidation(g))

# Parameter suggestion (colors present, etc.)

def suggest_colors(examples: List[Tuple[Grid,Grid]]) -> List[int]:
	colors=set()
	for inp,out in examples:
		for row in inp+out: colors.update(row)
	return [c for c in sorted(colors) if c!=0][:6]

# Heuristics

def grid_similarity(a: Grid, b: Grid) -> float:
	if not a or not b: return 0.0
	if len(a)!=len(b) or len(a[0])!=len(b[0]): return 0.0
	h,w=len(a),len(a[0])
	match=sum(1 for r in range(h) for c in range(w) if a[r][c]==b[r][c])
	return match/(h*w)

# Guided beam search

def synthesize_program(examples: List[Tuple[Grid,Grid]], beam:int=50, depth:int=3) -> Optional[Program]:
	# seed with identity and simple ops
	color_hints=suggest_colors(examples)
	candidates=[Program([])]
	def expand(prog: Program) -> List[Program]:
		progs=[]
		# arity-0 ops
		for name in ['identity','rotate90','flip_h','flip_v','transpose','bbox','gravity_down','row_color_consolidation']:
			progs.append(Program(prog.steps+[(name,())]))
		# arity-1 ops
		for col in color_hints:
			progs.append(Program(prog.steps+[('keep_color',(col,))]))
		# arity-2 ops (limited pairs)
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
				if s==1.0: return q
				if s>0: next_pool.append((s,q))
		if not next_pool: break
		next_pool.sort(key=lambda x:x[0], reverse=True)
		candidates=[q for _,q in next_pool[:beam]]
		if next_pool[0][0]>best_score: best_score=next_pool[0][0]; best=next_pool[0][1]
		if VERBOSE:
			print(f"  [basic-search] depth={d+1} pool={len(next_pool)} top={best_score:.3f}")
	return best

# Episodic retrieval (lightweight)

class EpisodicMemory:
	def __init__(self, max_items:int=2000):
		self.embs=[]; self.progs=[]; self.max_items=max_items
	def encode_task(self, examples: List[Tuple[Grid,Grid]]) -> Tuple:
		# simple signature: sizes, color histogram deltas
		sizes=[]; dh=Counter()
		for inp,out in examples:
			sizes.append((len(inp), len(inp[0]), len(out), len(out[0])))
			for r in inp: dh.update(r)
			for r in out: 
				for v in r: dh[v]-=1
		return (tuple(sizes), tuple(sorted(dh.items())))
	def add(self, examples, program: Program):
		emb=self.encode_task(examples)
		self.embs.append(emb); self.progs.append(program)
		if len(self.embs)>self.max_items:
			self.embs.pop(0); self.progs.pop(0)
	def retrieve(self, examples, k:int=3) -> List[Program]:
		emb=self.encode_task(examples)
		def dist(a,b):
			return 0 if a==b else 1
		scores=[(dist(emb,e),p) for e,p in zip(self.embs,self.progs)]
		scores.sort(key=lambda x:x[0])
		return [p for _,p in scores[:k]]

MEM=EpisodicMemory()

# Solver

def solve_task(task: Dict) -> List[Grid]:
	train=task.get('train',[]); test=task.get('test',[])
	if not train:
		return [t['input'] for t in test]
	examples=[(ex['input'], ex['output']) for ex in train]
	# warm-start from memory
	candidates = MEM.retrieve(examples, k=3)
	best=None; best_score=-1
	def score_prog(p: Program) -> float:
		sum_s=0.0
		for inp,out in examples:
			pred=p.apply(inp)
			if pred is None: return -1
			sum_s+=grid_similarity(pred,out)
		return sum_s/len(examples)
	for p in candidates:
		s=score_prog(p)
		if s>best_score: best, best_score = p, s
	if best_score<1.0:
		best = synthesize_program(examples, beam=80, depth=4)
		best_score = score_prog(best) if best else -1
	if best and best_score>=0.9:
		MEM.add(examples, best)
	if VERBOSE:
		print("  Selected program:", best)
		print(f"  Train fit score: {best_score:.3f}")
		# show first training example prediction
		inp, out = examples[0]
		pred = best.apply(inp) if best else None
		print_grid(inp, "  Train input (first):")
		print_grid(out, "  Train target (first):")
		if pred is not None:
			print_grid(pred, "  Train pred (first):")
	# apply to test
	preds=[]
	for t in test:
		pred = best.apply(t['input']) if best else t['input']
		preds.append(pred if pred else t['input'])
	return preds

# ------------------------------ CLI ------------------------------

def iter_aggregated(challenges_path: str, solutions_path: Optional[str]):
	import orjson
	with open(challenges_path,'rb') as f: ch = orjson.loads(f.read())
	sol=None
	if solutions_path and os.path.exists(solutions_path):
		with open(solutions_path,'rb') as f: sol = orjson.loads(f.read())
	for k,ex in ch.items():
		if sol:
			yield k, {'train': ex['train'], 'test': [{'input': t['input'], 'output': s} for t,s in zip(ex['test'], sol[k]) ]}
		else:
			yield k, {'train': ex['train'], 'test': ex['test']}

def main():
	ap=argparse.ArgumentParser()
	ap.add_argument('--mode', choices=['train','evaluation','test'], default='train')
	ap.add_argument('--data', default='arc-prize-2025')
	ap.add_argument('--out', default='outputs')
	ap.add_argument('--verbose', action='store_true')
	args=ap.parse_args()
	global VERBOSE
	VERBOSE = bool(args.verbose)
	os.makedirs(args.out, exist_ok=True)
	if args.mode=='train':
		ch=os.path.join(args.data,'arc-agi_training_challenges.json')
		sol=os.path.join(args.data,'arc-agi_training_solutions.json')
	elif args.mode=='evaluation':
		ch=os.path.join(args.data,'arc-agi_evaluation_challenges.json')
		sol=os.path.join(args.data,'arc-agi_evaluation_solutions.json')
	else:
		ch=os.path.join(args.data,'arc-agi_test_challenges.json')
		sol=None
	correct=0; total=0
	results=[]
	i=0
	for tid,task in iter_aggregated(ch, sol):
		i+=1
		if VERBOSE:
			print(f"\n=== Task {i}: {tid} ===")
			print(f"  Train pairs: {len(task.get('train',[]))}  Test: {len(task.get('test',[]))}")
			if task.get('train'):
				print_grid(task['train'][0]['input'], "  Sample train input:")
				print_grid(task['train'][0]['output'], "  Sample train target:")
		solver_fn = globals().get('solve_task_enhanced') if USE_ENHANCED else None
		if solver_fn is None:
			solver_fn = solve_task
		preds = solver_fn(task)
		targets=[x['output'] for x in task.get('test',[]) if 'output'in x]
		if targets:
			c=sum(1 for p,t in zip(preds,targets) if p==t)
			correct+=c; total+=len(targets)
		results.append((tid,c,len(targets)))
	if total>0:
		acc=correct/total
		print(f"accuracy={acc*100:.2f}% ({correct}/{total})")
	with open(os.path.join(args.out,'brain_solver_results.json'),'w') as f:
		json.dump(results,f)
	if args.mode=='test':
		# generate submission.json
		sub={}
		for tid,task in iter_aggregated(ch, sol):
			preds=solve_task(task)
			sub[tid]=[{'attempt_1': p, 'attempt_2': p} for p in preds]
		with open('submission.json','w') as f:
			json.dump(sub,f)
		print('Saved submission.json')

if __name__=='__main__':
	main()

# ---------------------------------------------------------------------------
# Extended Algorithms (Phase 2)
# The sections below extend the core solver with additional operations,
# pattern detectors, macro mining, and a more expressive search. These are
# integrated conservatively to avoid breaking the working baseline.
# ---------------------------------------------------------------------------

# ------------------------------ Extra Ops ------------------------------

def op_rotate180(g: Grid) -> Grid:
	return [row[::-1] for row in g[::-1]] if g else g

def op_rotate270(g: Grid) -> Grid:
	return [list(row) for row in zip(*g)][::-1] if g else g

def op_pad_border(g: Grid, pad: int = 1, color: int = 0) -> Grid:
	if not g or pad <= 0: return g
	h, w = len(g), len(g[0])
	row = [color] * (w + 2 * pad)
	out = [row[:] for _ in range(pad)]
	for r in range(h):
		out.append([color] * pad + g[r][:] + [color] * pad)
	out.extend([row[:] for _ in range(pad)])
	return out

def op_crop_to_first_object(g: Grid) -> Grid:
	objs = find_connected_components(g)
	if not objs: return g
	min_r, min_c, max_r, max_c = objs[0]['bbox']
	return [row[min_c:max_c+1] for row in g[min_r:max_r+1]]

def op_outline_objects(g: Grid) -> Grid:
	if not g: return g
	h,w=len(g),len(g[0])
	out=[[0]*w for _ in range(h)]
	for o in find_connected_components(g):
		color=o['color']
		for r,c in o['cells']:
			for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
				rn,cn=r+dr,c+dc
				if 0<=rn<h and 0<=cn<w and g[rn][cn]==0:
					out[r][c]=color
	return out

def op_fill_holes(g: Grid) -> Grid:
	# Simple hole fill: any zero fully surrounded by same color in 4-neighborhood becomes that color
	if not g: return g
	h,w=len(g),len(g[0])
	out=[row[:] for row in g]
	for r in range(1,h-1):
		for c in range(1,w-1):
			if g[r][c]==0:
				nbr=[g[r-1][c],g[r+1][c],g[r][c-1],g[r][c+1]]
				cand=[x for x in nbr if x!=0]
				if cand and all(x==cand[0] for x in cand):
					out[r][c]=cand[0]
	return out

def op_center_largest_object(g: Grid) -> Grid:
	objs=find_connected_components(g)
	if not objs or not g: return g
	h,w=len(g),len(g[0])
	largest=max(objs,key=lambda o:o['size'])
	cr,cc=largest['center']
	target=(h//2, w//2)
	dr=int(round(target[0]-cr)); dc=int(round(target[1]-cc))
	out=[row[:] for row in g]
	for r,c in largest['cells']:
		out[r][c]=0
	for r,c in largest['cells']:
		rn,cn=r+dr,c+dc
		if 0<=rn<h and 0<=cn<w:
			out[rn][cn]=largest['color']
	return out

def op_sort_objects_left_to_right_by_size(g: Grid) -> Grid:
	objs=find_connected_components(g)
	if not objs or not g: return g
	h,w=len(g),len(g[0])
	objs=sorted(objs,key=lambda o:o['size'], reverse=True)
	out=[[0]*w for _ in range(h)]
	col=0
	for o in objs:
		min_r,min_c,max_r,max_c=o['bbox']
		height=max_r-min_r+1; width=max_c-min_c+1
		if col+width>w: break
		for r,c in o['cells']:
			rn=r-min_r
			cn=c-min_c
			if 0<=rn<h and 0<=col+cn<w:
				out[rn][col+cn]=o['color']
		col+=width+1
	return out

# Register extra ops
DSL.register('rotate180', op_rotate180, 0, 'grid')
DSL.register('rotate270', op_rotate270, 0, 'grid')
DSL.register('pad_border', op_pad_border, 1, 'grid')
DSL.register('crop_first_object', op_crop_to_first_object, 0, 'grid')
DSL.register('outline_objects', op_outline_objects, 0, 'grid')
DSL.register('fill_holes', op_fill_holes, 0, 'grid')
DSL.register('center_largest', op_center_largest_object, 0, 'grid')
DSL.register('sort_obj_l2r', op_sort_objects_left_to_right_by_size, 0, 'grid')

# --------------------------- Symmetry Macros ---------------------------

def is_h_symmetric(g: Grid) -> bool:
	if not g: return False
	h=len(g)
	for r in range(h//2):
		if g[r]!=g[h-1-r]: return False
	return True

def is_v_symmetric(g: Grid) -> bool:
	if not g: return False
	h,w=len(g),len(g[0])
	for r in range(h):
		for c in range(w//2):
			if g[r][c]!=g[r][w-1-c]: return False
	return True

def macro_make_h_symmetric(g: Grid) -> Grid:
	if not g: return g
	h,w=len(g),len(g[0])
	out=[row[:] for row in g]
	for r in range(h//2):
		out[h-1-r]=out[r][:]
	return out

def macro_make_v_symmetric(g: Grid) -> Grid:
	if not g: return g
	h,w=len(g),len(g[0])
	out=[row[:] for row in g]
	for r in range(h):
		for c in range(w//2):
			out[r][w-1-c]=out[r][c]
	return out

DSL.register_macro('make_h_symmetric', lambda g: macro_make_h_symmetric(g))
DSL.register_macro('make_v_symmetric', lambda g: macro_make_v_symmetric(g))

# ------------------------ Tiling / Canonicalization ------------------------

def detect_tiling(g: Grid) -> Optional[Tuple[int,int]]:
	if not g: return None
	h,w=len(g),len(g[0])
	for th in range(1, h+1):
		if h%th!=0: continue
		for tw in range(1, w+1):
			if w%tw!=0: continue
			ok=True
			for r in range(h):
				for c in range(w):
					if g[r][c] != g[r%th][c%tw]:
						ok=False; break
				if not ok: break
			if ok: return (th,tw)
	return None

def macro_extract_tile(g: Grid) -> Grid:
	sz=detect_tiling(g)
	if not sz: return g
	th,tw=sz
	return [row[:tw] for row in g[:th]]

DSL.register_macro('extract_tile', lambda g: macro_extract_tile(g))

# --------------------- Macro Mining (frequent subprograms) ---------------------

class MacroLibrary:
	def __init__(self):
		self.freq=Counter()
		self.max_macros=64
	def observe(self, prog: Program):
		if not prog or not prog.steps: return
		steps=prog.steps
		# count contiguous subsequences length 2-3
		for L in (2,3):
			for i in range(len(steps)-L+1):
				sub=tuple(steps[i:i+L])
				self.freq[sub]+=1
	def materialize(self):
		# promote top sequences to macros
		for (seq,cnt) in self.freq.most_common(self.max_macros):
			name='macro_'+"_".join(n for n,_ in seq)
			if name in DSL.MACROS: continue
			def builder(g: Grid, seq=seq):
				res=[row[:] for row in g]
				for n,args in seq:
					res=DSL.apply(n,res,args)
					if res is None: return None
				return res
			DSL.register_macro(name, builder)

MACROS=MacroLibrary()

# --------------------- Enhanced Search with Macros ---------------------

def synthesize_program_enhanced(examples: List[Tuple[Grid,Grid]], beam:int=120, depth:int=5) -> Optional[Program]:
	color_hints=suggest_colors(examples)
	base0=['identity','rotate90','rotate180','rotate270','flip_h','flip_v','transpose','bbox','gravity_down','fill_holes','outline_objects','center_largest','sort_obj_l2r','row_color_consolidation','make_h_symmetric','make_v_symmetric','extract_tile']
	macro_names=list(DSL.MACROS.keys())
	base=base0+macro_names
	candidates=[Program([])]
	visited=set()

	def signature(p: Program, inp: Grid) -> Tuple:
		pred=p.apply(inp)
		if pred is None: return ()
		return (tuple(tuple(row) for row in pred),)

	def expand(prog: Program) -> List[Program]:
		progs=[]
		# arity-0
		for name in base:
			progs.append(Program(prog.steps+[(name,())]))
		# arity-1
		for col in color_hints:
			progs.append(Program(prog.steps+[('keep_color',(col,))]))
		# arity-2 replace_color limited combos
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
				# dedup first example signature to prune
				sig=signature(q, examples[0][0])
				if sig in visited: continue
				visited.add(sig)
				s=score_prog(q)
				if s==1.0:
					MACROS.observe(q)  # learn from success
					return q
				if s>0: next_pool.append((s,q))
		if not next_pool: break
		next_pool.sort(key=lambda x:x[0], reverse=True)
		candidates=[q for _,q in next_pool[:beam]]
		if next_pool[0][0]>best_score:
			best_score=next_pool[0][0]; best=next_pool[0][1]
		if VERBOSE:
			print(f"  [enhanced-search] depth={d+1} pool={len(next_pool)} top={best_score:.3f} macros={len(DSL.MACROS)}")
	return best

# Override solver to use enhanced search and macro mining
def solve_task_enhanced(task: Dict) -> List[Grid]:
	train=task.get('train',[]); test=task.get('test',[])
	if not train:
		return [t['input'] for t in test]
	examples=[(ex['input'], ex['output']) for ex in train]
	# try episodic retrieval
	candidates = MEM.retrieve(examples, k=5)
	best=None; best_score=-1
	def score_prog(p: Program) -> float:
		sum_s=0.0
		for inp,out in examples:
			pred=p.apply(inp)
			if pred is None: return -1
			sum_s+=grid_similarity(pred,out)
		return sum_s/len(examples)
	for p in candidates:
		s=score_prog(p)
		if s>best_score: best, best_score = p, s
	if best_score<1.0:
		best = synthesize_program_enhanced(examples, beam=150, depth=5)
		best_score = score_prog(best) if best else -1
	if best and best_score>=0.9:
		MEM.add(examples, best)
		MACROS.observe(best)
		MACROS.materialize()
	if VERBOSE:
		print("  Selected program (enhanced):", best)
		print(f"  Train fit score (enhanced): {best_score:.3f}")
		inp, out = examples[0]
		pred = best.apply(inp) if best else None
		print_grid(inp, "  Train input (first):")
		print_grid(out, "  Train target (first):")
		if pred is not None:
			print_grid(pred, "  Train pred (first):")
	# apply to test
	preds=[]
	for t in test:
		pred = best.apply(t['input']) if best else t['input']
		preds.append(pred if pred else t['input'])
	return preds

# Optional: expose enhanced mode via environment flag
USE_ENHANCED = os.environ.get('ARC_BRAIN_ENHANCED','0')=='1'

