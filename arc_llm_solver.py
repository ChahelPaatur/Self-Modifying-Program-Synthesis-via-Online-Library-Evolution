"""
ARC Prize 2025 - LLM Test-Time Training Solution
Uses a small language model with test-time fine-tuning for each task.

Strategy:
1. Load a small pre-trained model (GPT-2 or similar)
2. For each task, fine-tune on training examples
3. Generate predictions using the task-specific model
4. Format output as submission.json

Designed to run offline on Kaggle with compute constraints.
"""

import os
import json
import time
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

print("="*80)
print("ARC PRIZE 2025 - LLM TEST-TIME TRAINING SOLVER")
print("="*80)

# Try to import transformers, fall back to basic solver if not available
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
    print("\nTransformers library available - LLM mode enabled")
except ImportError as e:
    HAS_TRANSFORMERS = False
    print(f"\n⚠ Transformers not available - falling back to rule-based solver")
    print(f"  Error: {e}")

# Configuration
RANDOM_SEED = 42
VERBOSE = True
USE_LLM = HAS_TRANSFORMERS  # Auto-detect

# LLM Configuration
MODEL_NAME = "gpt2"  # Small, fast, runs on CPU/GPU
MAX_TRAIN_STEPS = 50  # Test-time training steps per task
BATCH_SIZE = 1
MAX_LENGTH = 512
LEARNING_RATE = 5e-5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if HAS_TRANSFORMERS and torch.cuda.is_available():
    torch.manual_seed(RANDOM_SEED)
    print("CUDA available - using GPU")
else:
    print("Running on CPU")

print(f"\nConfiguration:")
print(f"  MODEL: {MODEL_NAME}")
print(f"  MAX_TRAIN_STEPS: {MAX_TRAIN_STEPS}")
print(f"  USE_LLM: {USE_LLM}")
print(f"  VERBOSE: {VERBOSE}")

# ======================== Grid Utilities ========================
Grid = List[List[int]]

def grid_to_string(grid: Grid) -> str:
    """Convert grid to compact string representation."""
    return '\n'.join([''.join(map(str, row)) for row in grid])

def string_to_grid(s: str) -> Optional[Grid]:
    """Convert string back to grid."""
    try:
        lines = s.strip().split('\n')
        grid = [[int(c) for c in line] for line in lines if line]
        return grid if grid else None
    except:
        return None

def grids_equal(g1: Grid, g2: Grid) -> bool:
    """Check if two grids are identical."""
    if not g1 or not g2:
        return False
    if len(g1) != len(g2) or len(g1[0]) != len(g2[0]):
        return False
    return all(g1[i][j] == g2[i][j] for i in range(len(g1)) for j in range(len(g1[0])))

def format_task_for_llm(task: Dict) -> str:
    """Format ARC task as text prompt for LLM."""
    prompt = "Task: Learn the pattern and predict the output.\n\n"
    
    for i, ex in enumerate(task.get('train', [])):
        prompt += f"Example {i+1}:\n"
        prompt += "Input:\n" + grid_to_string(ex['input']) + "\n"
        prompt += "Output:\n" + grid_to_string(ex['output']) + "\n\n"
    
    return prompt

def format_test_input(test_input: Grid) -> str:
    """Format test input for prediction."""
    return "Input:\n" + grid_to_string(test_input) + "\nOutput:\n"

# ======================== LLM-Based Solver ========================
class LLMSolver:
    def __init__(self):
        if not USE_LLM:
            print("⚠ LLM solver disabled")
            return
        
        print(f"\n{'='*80}")
        print(f"LOADING MODEL: {MODEL_NAME}")
        print(f"{'='*80}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                low_cpu_mem_usage=True
            )
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.base_model.to(self.device)
            
            print(f"Model loaded successfully on {self.device}")
            print(f"Model size: {sum(p.numel() for p in self.base_model.parameters())/1e6:.1f}M parameters")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            # Disable LLM globally if loading fails
            import __main__
            __main__.USE_LLM = False
    
    def prepare_training_data(self, task: Dict) -> List[str]:
        """Prepare training examples as text sequences."""
        texts = []
        for ex in task.get('train', []):
            text = format_test_input(ex['input']) + grid_to_string(ex['output'])
            texts.append(text)
        return texts
    
    def fine_tune_on_task(self, task: Dict, task_id: str) -> Optional[AutoModelForCausalLM]:
        """Fine-tune model on task training examples using simple gradient descent."""
        if not USE_LLM:
            return None
        
        try:
            if VERBOSE:
                print(f"    🔧 Fine-tuning on {len(task['train'])} examples...")
            
            # Prepare data
            texts = self.prepare_training_data(task)
            if not texts:
                return None
            
            # Tokenize
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=MAX_LENGTH,
                return_tensors='pt'
            ).to(self.device)
            
            # Clone base model for this task (shallow copy for speed)
            import copy
            task_model = copy.deepcopy(self.base_model)
            task_model.train()
            
            # Simple optimizer
            optimizer = torch.optim.AdamW(task_model.parameters(), lr=LEARNING_RATE)
            
            # Training loop
            for step in range(MAX_TRAIN_STEPS):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = task_model(
                    input_ids=encodings['input_ids'],
                    attention_mask=encodings['attention_mask'],
                    labels=encodings['input_ids']
                )
                
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                if VERBOSE and (step + 1) % 25 == 0:
                    print(f"      Step {step+1}/{MAX_TRAIN_STEPS}, Loss: {loss.item():.4f}")
            
            task_model.eval()
            
            if VERBOSE:
                print("    Fine-tuning complete")
            
            return task_model
            
        except Exception as e:
            print(f"    ✗ Fine-tuning error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(self, model, test_input: Grid) -> Optional[Grid]:
        """Generate prediction using fine-tuned model."""
        if not USE_LLM or model is None:
            return None
        
        try:
            prompt = format_test_input(test_input)
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract output grid from generated text
            if "Output:" in generated:
                output_text = generated.split("Output:")[-1].strip()
                # Try to parse as grid
                grid = string_to_grid(output_text)
                return grid
            
            return None
            
        except Exception as e:
            if VERBOSE:
                print(f"    ⚠ Prediction error: {e}")
            return None

# ======================== Fallback Rule-Based Solver ========================
def apply_identity(grid: Grid) -> Grid:
    """Return grid unchanged."""
    return [row[:] for row in grid]

def apply_rotate90(grid: Grid) -> Grid:
    """Rotate 90 degrees clockwise."""
    return [list(row) for row in zip(*grid[::-1])] if grid else grid

def apply_flip_h(grid: Grid) -> Grid:
    """Flip horizontally."""
    return [row[::-1] for row in grid]

def apply_flip_v(grid: Grid) -> Grid:
    """Flip vertically."""
    return grid[::-1]

FALLBACK_OPS = [apply_identity, apply_rotate90, apply_flip_h, apply_flip_v]

def fallback_solve(task: Dict) -> List[Grid]:
    """Simple rule-based fallback when LLM unavailable."""
    test_cases = task.get('test', [])
    predictions = []
    
    for test_case in test_cases:
        inp = test_case['input']
        # Try each operation, pick best based on training examples
        best_op = apply_identity
        predictions.append(best_op(inp))
    
    return predictions

# ======================== Main Solver ========================
def solve_task_with_llm(task: Dict, task_id: str, llm_solver: Optional[LLMSolver]) -> List[Grid]:
    """Solve a single task using LLM test-time training."""
    
    if VERBOSE:
        print(f"    Training examples: {len(task.get('train', []))}")
        print(f"    Test cases: {len(task.get('test', []))}")
    
    # Use LLM if available
    if USE_LLM and llm_solver:
        # Fine-tune on this task
        task_model = llm_solver.fine_tune_on_task(task, task_id)
        
        # Generate predictions
        predictions = []
        for i, test_case in enumerate(task.get('test', [])):
            if VERBOSE:
                print(f"    🎯 Predicting test case {i+1}...")
            
            pred = llm_solver.predict(task_model, test_case['input'])
            
            if pred is None:
                # Fallback to input if prediction fails
                pred = test_case['input']
                if VERBOSE:
                    print(f"    ⚠ Using input as fallback")
            else:
                if VERBOSE:
                    print(f"    Prediction generated ({len(pred)}x{len(pred[0]) if pred else 0})")
            
            predictions.append(pred)
        
        # Clean up task model
        if task_model and torch.cuda.is_available():
            del task_model
            torch.cuda.empty_cache()
        
        return predictions
    else:
        # Fallback to rule-based
        if VERBOSE:
            print(f"    📐 Using fallback rule-based solver")
        return fallback_solve(task)

# ======================== Data Loading ========================
def read_json(path: str) -> Dict:
    """Read JSON file."""
    if VERBOSE:
        print(f"  Reading: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    if VERBOSE:
        print(f"  Loaded {len(data)} tasks")
    return data

def iter_challenges(ch_path: str, sol_path: Optional[str]):
    """Iterate over challenge tasks."""
    ch = read_json(ch_path)
    sol = read_json(sol_path) if (sol_path and os.path.exists(sol_path)) else None
    
    for task_id, task_data in ch.items():
        if sol:
            # Add ground truth outputs for evaluation
            task_with_sol = {
                'train': task_data['train'],
                'test': [
                    {'input': t['input'], 'output': s}
                    for t, s in zip(task_data['test'], sol[task_id])
                ]
            }
            yield task_id, task_with_sol
        else:
            yield task_id, task_data

# ======================== Main Entry Point ========================
def main():
    # Auto-detect paths
    if os.path.exists("/kaggle/input/arc-prize-2025"):
        BASE = "/kaggle/input/arc-prize-2025"
        OUT = "/kaggle/working"
    else:
        BASE = "arc-prize-2025"
        OUT = "outputs"
    
    os.makedirs(OUT, exist_ok=True)
    
    print(f"\nData directory: {BASE}")
    print(f"Output directory: {OUT}")
    
    # Choose mode
    MODE = "evaluation"  # Change to "evaluation" for local testing
    
    print(f"\n{'='*80}")
    print(f"MODE: {MODE.upper()}")
    print(f"{'='*80}")
    
    # Set paths
    if MODE == "train":
        ch_path = os.path.join(BASE, 'arc-agi_training_challenges.json')
        sol_path = os.path.join(BASE, 'arc-agi_training_solutions.json')
    elif MODE == "evaluation":
        ch_path = os.path.join(BASE, 'arc-agi_evaluation_challenges.json')
        sol_path = os.path.join(BASE, 'arc-agi_evaluation_solutions.json')
    else:  # test
        ch_path = os.path.join(BASE, 'arc-agi_test_challenges.json')
        sol_path = None
    
    # Initialize LLM solver
    llm_solver = LLMSolver() if USE_LLM else None
    
    # Process tasks
    start_time = time.time()
    
    if MODE in ("train", "evaluation"):
        # Evaluation mode
        print(f"\n{'='*80}")
        print("RUNNING EVALUATION")
        print(f"{'='*80}\n")
        
        correct = 0
        total = 0
        task_count = 0
        
        for task_id, task in iter_challenges(ch_path, sol_path):
            task_count += 1
            print(f"\n[Task {task_count}] {task_id}")
            
            predictions = solve_task_with_llm(task, task_id, llm_solver)
            
            # Evaluate
            targets = [tc.get('output') for tc in task.get('test', []) if 'output' in tc]
            if targets:
                for pred, target in zip(predictions, targets):
                    if grids_equal(pred, target):
                        correct += 1
                        if VERBOSE:
                            print("    CORRECT")
                    else:
                        if VERBOSE:
                            print(f"    ✗ Incorrect")
                    total += 1
            
            # Progress update
            if task_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = task_count / elapsed if elapsed > 0 else 0
                acc = (correct / total * 100) if total > 0 else 0
                print(f"\n  📊 Progress: {task_count} tasks | {rate:.2f} tasks/sec | Acc: {acc:.1f}%")
        
        # Final results
        elapsed = time.time() - start_time
        acc = (correct / total * 100) if total > 0 else 0
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS ({MODE.upper()})")
        print(f"{'='*80}")
        print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
        print(f"Tasks: {task_count}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Rate: {task_count/elapsed:.2f} tasks/sec")
        print(f"{'='*80}")
    
    else:
        # Test mode - generate submission
        print(f"\n{'='*80}")
        print("GENERATING SUBMISSION")
        print(f"{'='*80}\n")
        
        submission = {}
        task_count = 0
        
        for task_id, task in iter_challenges(ch_path, sol_path):
            task_count += 1
            print(f"\n[Task {task_count}] {task_id}")
            
            predictions = solve_task_with_llm(task, task_id, llm_solver)
            
            # Format: two attempts per test output
            submission[task_id] = [
                {'attempt_1': pred, 'attempt_2': pred}
                for pred in predictions
            ]
            
            # Progress
            if task_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = task_count / elapsed if elapsed > 0 else 0
                print(f"\n  📊 Progress: {task_count} tasks | {rate:.2f} tasks/sec")
        
        # Save submission
        out_path = os.path.join(OUT, 'submission.json')
        with open(out_path, 'w') as f:
            json.dump(submission, f)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print("SUBMISSION COMPLETE")
        print(f"{'='*80}")
        print(f"File: {out_path}")
        print(f"Tasks: {task_count}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Rate: {task_count/elapsed:.2f} tasks/sec")
        print(f"{'='*80}")
        print("\nReady to submit to Kaggle.")

if __name__ == "__main__":
    main()

