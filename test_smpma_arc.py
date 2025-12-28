#!/usr/bin/env python3
"""
Test SMPMA on real ARC-AGI tasks
"""

import json
from smpma_agi import SMPMA

def load_arc_tasks(path, limit=30):
    """Load ARC tasks from JSON"""
    with open(path) as f:
        data = json.load(f)
    
    tasks = {}
    for task_id, task_data in list(data.items())[:limit]:
        examples = []
        for ex in task_data['train']:
            examples.append((ex['input'], ex['output']))
        tasks[task_id] = examples
    
    return tasks

def main():
    print("="*80)
    print("TESTING SMPMA ON REAL ARC-AGI TASKS")
    print("="*80)
    
    # Load tasks
    train_path = "arc-prize-2025/arc-agi_training_challenges.json"
    print(f"\nLoading tasks from {train_path}...")
    
    try:
        tasks = load_arc_tasks(train_path, limit=30)
        print(f"Loaded {len(tasks)} tasks")
    except FileNotFoundError:
        print(f"File not found: {train_path}")
        print("Make sure arc-prize-2025 data is in the current directory")
        return
    
    # Initialize SMPMA
    smpma = SMPMA()
    
    # Evaluate
    print(f"\n{'='*80}")
    print("EVALUATION")
    print(f"{'='*80}")
    
    results = smpma.evaluate(tasks)
    
    # Print results
    print(f"\n{'='*80}")
    print("FINAL RESULTS ON ARC-AGI")
    print(f"{'='*80}")
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Solved (>95% accuracy): {results['solved']} ({results['solved']/results['total_tasks']*100:.1f}%)")
    print(f"Average score: {results['avg_score']:.3f}")
    print(f"Learned abstractions: {results['abstraction_count']}")
    
    # System summary
    smpma.summary()
    
    print(f"\n{'='*80}")
    print("ARC-AGI evaluation complete")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

