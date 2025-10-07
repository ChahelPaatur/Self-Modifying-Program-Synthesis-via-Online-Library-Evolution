"""
Generate Kaggle submission file for ARC Prize 2025
"""
import orjson
import json
from models.hybrid_super.solver import HybridSuperSolver

# You can switch to any solver here
SOLVER = HybridSuperSolver()

def create_submission(test_challenges_path: str, output_path: str):
    """Create submission.json for Kaggle"""
    
    # Load test challenges
    with open(test_challenges_path, 'rb') as f:
        test_challenges = orjson.loads(f.read())
    
    submission = {}
    
    print(f"Processing {len(test_challenges)} test tasks...")
    
    for i, (task_id, task_data) in enumerate(test_challenges.items()):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_challenges)} tasks...")
        
        # Create task format
        task = {
            'train': task_data.get('train', []),
            'test': task_data.get('test', [])
        }
        
        try:
            # Get predictions
            predictions = SOLVER.predict(task)
            
            # Format: each test case gets 2 attempts
            task_predictions = []
            for pred in predictions:
                # Attempt 1: our prediction
                # Attempt 2: copy of same prediction (can be different if you have multiple strategies)
                task_predictions.append({
                    'attempt_1': pred,
                    'attempt_2': pred
                })
            
            submission[task_id] = task_predictions
            
        except Exception as e:
            print(f"Error on task {task_id}: {e}")
            # Fallback: return input as prediction
            task_predictions = []
            for test_item in task['test']:
                inp = test_item['input']
                task_predictions.append({
                    'attempt_1': inp,
                    'attempt_2': inp
                })
            submission[task_id] = task_predictions
    
    # Write submission file
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    
    print(f"\n✅ Submission file created: {output_path}")
    print(f"   Total tasks: {len(submission)}")
    print(f"\nNext steps:")
    print("1. Upload this code to a Kaggle notebook")
    print("2. Make sure the notebook has 'Internet OFF' and 'GPU ON' (optional)")
    print("3. Run the notebook")
    print("4. Click 'Submit to Competition'")


if __name__ == '__main__':
    import sys
    
    test_path = 'arc-prize-2025/arc-agi_test_challenges.json'
    output_path = 'submission.json'
    
    create_submission(test_path, output_path)
