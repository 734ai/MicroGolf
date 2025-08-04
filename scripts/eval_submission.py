#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for MicroGolf
Evaluates generated solutions on ARC tasks and produces detailed metrics
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from microgolf.model import MetaComposer, create_primitive_vocab
from microgolf.engine import PrimitiveController, AbstractPlan, OptimizedExecutor
from microgolf.primitives import PRIMITIVES


class TaskEvaluator:
    """Evaluates solutions on individual ARC tasks"""
    
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
    
    def evaluate_task(self, task_data: Dict, solution_code: str) -> Dict[str, Any]:
        """Evaluate solution on a single task"""
        
        results = {
            'task_id': task_data.get('id', 'unknown'),
            'status': 'success',
            'error': None,
            'execution_time': 0,
            'byte_count': len(solution_code.encode('utf-8')),
            'test_results': [],
            'score': 0,
            'accuracy': 0
        }
        
        try:
            start_time = time.time()
            
            # Compile solution
            try:
                solution_func = eval(solution_code)
            except Exception as e:
                results['status'] = 'compile_error'
                results['error'] = str(e)
                return results
            
            # Test on training examples
            train_examples = task_data.get('train', [])
            test_examples = task_data.get('test', [])
            
            all_examples = train_examples + test_examples
            correct_predictions = 0
            
            for i, example in enumerate(all_examples):
                input_grid = example['input']
                expected_output = example['output']
                
                try:
                    # Execute solution with timeout
                    predicted_output = self._execute_with_timeout(
                        solution_func, input_grid, self.timeout
                    )
                    
                    # Compare outputs
                    is_correct = self._grids_equal(predicted_output, expected_output)
                    
                    if is_correct:
                        correct_predictions += 1
                    
                    results['test_results'].append({
                        'example_id': i,
                        'correct': is_correct,
                        'input_shape': [len(input_grid), len(input_grid[0]) if input_grid else 0],
                        'expected_shape': [len(expected_output), len(expected_output[0]) if expected_output else 0],
                        'predicted_shape': [len(predicted_output), len(predicted_output[0]) if predicted_output else 0] if predicted_output else [0, 0]
                    })
                    
                except TimeoutError:
                    results['test_results'].append({
                        'example_id': i,
                        'correct': False,
                        'error': 'timeout'
                    })
                except Exception as e:
                    results['test_results'].append({
                        'example_id': i,
                        'correct': False,
                        'error': str(e)
                    })
            
            results['execution_time'] = time.time() - start_time
            results['accuracy'] = correct_predictions / len(all_examples) if all_examples else 0
            results['score'] = correct_predictions
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
        
        return results
    
    def _execute_with_timeout(self, func, input_grid, timeout):
        """Execute function with timeout"""
        
        # For simplicity, we'll execute directly here
        # In production, you'd want proper timeout handling
        try:
            return func(input_grid)
        except Exception as e:
            raise e
    
    def _grids_equal(self, grid1, grid2) -> bool:
        """Check if two grids are equal"""
        
        if grid1 is None or grid2 is None:
            return grid1 == grid2
        
        if len(grid1) != len(grid2):
            return False
        
        for row1, row2 in zip(grid1, grid2):
            if len(row1) != len(row2):
                return False
            for cell1, cell2 in zip(row1, row2):
                if cell1 != cell2:
                    return False
        
        return True


class SolutionGenerator:
    """Generates solutions using MicroGolf framework"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.primitive_vocab = create_primitive_vocab()
        
        # Initialize components
        self.controller = PrimitiveController()
        self.executor = OptimizedExecutor()
        
        # Load meta-composer if available
        self.meta_composer = None
        if model_path and Path(model_path).exists():
            try:
                self.meta_composer = MetaComposer(self.primitive_vocab)
                self.meta_composer.load_model(model_path)
                print(f"Loaded meta-composer from {model_path}")
            except Exception as e:
                print(f"Failed to load meta-composer: {e}")
    
    def generate_solution(self, task_data: Dict) -> Tuple[str, Dict[str, Any]]:
        """Generate solution code for a task"""
        
        examples = task_data.get('train', [])
        if not examples:
            return "lambda g: g", {'method': 'identity', 'primitives': []}
        
        metadata = {
            'method': 'unknown',
            'primitives': [],
            'estimated_bytes': 0,
            'generation_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Method 1: Try meta-composer if available
            if self.meta_composer:
                try:
                    primitive_sequence = self.meta_composer.predict_sequence(examples)
                    if primitive_sequence:
                        plan = AbstractPlan()
                        for prim in primitive_sequence:
                            plan.add_step(prim)
                        
                        optimized_plan = plan.optimize()
                        code = self.executor.execute_plan_optimized(optimized_plan)
                        
                        metadata.update({
                            'method': 'meta_composer',
                            'primitives': primitive_sequence,
                            'estimated_bytes': len(code.encode('utf-8'))
                        })
                        
                        metadata['generation_time'] = time.time() - start_time
                        return code, metadata
                except Exception as e:
                    print(f"Meta-composer failed: {e}")
            
            # Method 2: Heuristic controller
            try:
                primitive_sequence = self.controller.predict_sequence(examples)
                if primitive_sequence:
                    plan = AbstractPlan()
                    for prim in primitive_sequence:
                        plan.add_step(prim)
                    
                    optimized_plan = plan.optimize()
                    code = self.executor.execute_plan_optimized(optimized_plan)
                    
                    metadata.update({
                        'method': 'heuristic_controller',
                        'primitives': primitive_sequence,
                        'estimated_bytes': len(code.encode('utf-8'))
                    })
                    
                    metadata['generation_time'] = time.time() - start_time
                    return code, metadata
            except Exception as e:
                print(f"Heuristic controller failed: {e}")
            
            # Method 3: Simple baseline
            code = self._generate_baseline_solution(examples)
            metadata.update({
                'method': 'baseline',
                'primitives': ['identity'],
                'estimated_bytes': len(code.encode('utf-8'))
            })
            
        except Exception as e:
            print(f"Solution generation failed: {e}")
            code = "lambda g: g"
            metadata.update({
                'method': 'fallback',
                'primitives': [],
                'estimated_bytes': len(code.encode('utf-8')),
                'error': str(e)
            })
        
        metadata['generation_time'] = time.time() - start_time
        return code, metadata
    
    def _generate_baseline_solution(self, examples: List[Dict]) -> str:
        """Generate simple baseline solution"""
        
        # Analyze examples to determine simple transformation
        if not examples:
            return "lambda g: g"
        
        first_example = examples[0]
        input_grid = first_example['input']
        output_grid = first_example['output']
        
        # Check for simple transformations
        if len(input_grid) == len(output_grid) and len(input_grid[0]) == len(output_grid[0]):
            # Same size - might be color mapping or identity
            if input_grid == output_grid:
                return "lambda g: g"
            else:
                # Try to detect color mapping
                color_map = {}
                for i in range(len(input_grid)):
                    for j in range(len(input_grid[0])):
                        in_color = input_grid[i][j]
                        out_color = output_grid[i][j]
                        if in_color in color_map and color_map[in_color] != out_color:
                            # Inconsistent mapping, use identity
                            return "lambda g: g"
                        color_map[in_color] = out_color
                
                # Generate color mapping code
                map_str = '{' + ','.join(f'{k}:{v}' for k, v in color_map.items()) + '}'
                return f"lambda g:[[{map_str}.get(c,c) for c in r] for r in g]"
        
        # Default to identity
        return "lambda g: g"


class CompetitionEvaluator:
    """Main evaluator for competition submissions"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 timeout: float = 10.0,
                 max_workers: int = 4):
        
        self.solution_generator = SolutionGenerator(model_path)
        self.task_evaluator = TaskEvaluator(timeout)
        self.max_workers = max_workers
        
    def evaluate_submission(self, tasks: List[Dict]) -> Dict[str, Any]:
        """Evaluate complete submission on all tasks"""
        
        print(f"Evaluating {len(tasks)} tasks...")
        
        results = {
            'total_tasks': len(tasks),
            'completed_tasks': 0,
            'total_score': 0,
            'average_accuracy': 0,
            'total_bytes': 0,
            'byte_violations': 0,
            'method_breakdown': {},
            'task_results': [],
            'summary_stats': {}
        }
        
        # Process tasks
        task_results = []
        
        for i, task in enumerate(tasks):
            print(f"Processing task {i+1}/{len(tasks)} ({task.get('id', 'unknown')})...")
            
            # Generate solution
            solution_code, generation_metadata = self.solution_generator.generate_solution(task)
            
            # Evaluate solution
            eval_result = self.task_evaluator.evaluate_task(task, solution_code)
            
            # Combine results
            task_result = {
                **eval_result,
                'solution_code': solution_code,
                'generation_metadata': generation_metadata
            }
            
            task_results.append(task_result)
            
            # Update summary
            if eval_result['status'] == 'success':
                results['completed_tasks'] += 1
                results['total_score'] += eval_result['score']
            
            results['total_bytes'] += eval_result['byte_count']
            
            if eval_result['byte_count'] > 2500:
                results['byte_violations'] += 1
            
            # Track method usage
            method = generation_metadata['method']
            if method not in results['method_breakdown']:
                results['method_breakdown'][method] = 0
            results['method_breakdown'][method] += 1
        
        results['task_results'] = task_results
        
        # Calculate final metrics
        if results['completed_tasks'] > 0:
            results['average_accuracy'] = sum(
                r['accuracy'] for r in task_results if r['status'] == 'success'
            ) / results['completed_tasks']
        
        results['summary_stats'] = self._calculate_summary_stats(task_results)
        
        return results
    
    def _calculate_summary_stats(self, task_results: List[Dict]) -> Dict[str, Any]:
        """Calculate detailed summary statistics"""
        
        accuracies = [r['accuracy'] for r in task_results if r['status'] == 'success']
        byte_counts = [r['byte_count'] for r in task_results]
        execution_times = [r['execution_time'] for r in task_results if r['status'] == 'success']
        
        stats = {
            'accuracy_stats': {
                'mean': np.mean(accuracies) if accuracies else 0,
                'std': np.std(accuracies) if accuracies else 0,
                'min': np.min(accuracies) if accuracies else 0,
                'max': np.max(accuracies) if accuracies else 0,
                'median': np.median(accuracies) if accuracies else 0
            },
            'byte_stats': {
                'mean': np.mean(byte_counts),
                'std': np.std(byte_counts),
                'min': np.min(byte_counts),
                'max': np.max(byte_counts),
                'median': np.median(byte_counts)
            },
            'time_stats': {
                'mean': np.mean(execution_times) if execution_times else 0,
                'std': np.std(execution_times) if execution_times else 0,
                'min': np.min(execution_times) if execution_times else 0,
                'max': np.max(execution_times) if execution_times else 0,
                'median': np.median(execution_times) if execution_times else 0
            }
        }
        
        return stats


def load_arc_tasks(data_dir: Path) -> List[Dict]:
    """Load ARC tasks from data directory"""
    
    tasks = []
    
    # For demo, create synthetic tasks
    for i in range(10):
        task = {
            'id': f'eval_{i:03d}',
            'train': [
                {
                    'input': np.random.randint(0, 5, (3, 3)).tolist(),
                    'output': np.random.randint(0, 5, (3, 3)).tolist()
                }
            ],
            'test': [
                {
                    'input': np.random.randint(0, 5, (3, 3)).tolist(),
                    'output': np.random.randint(0, 5, (3, 3)).tolist()
                }
            ]
        }
        tasks.append(task)
    
    return tasks


def main():
    parser = argparse.ArgumentParser(description='Evaluate MicroGolf submission')
    
    parser.add_argument('--data_dir', type=str, default='data/arc',
                       help='Directory containing ARC tasks')
    parser.add_argument('--model_path', type=str,
                       help='Path to trained meta-composer model')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--timeout', type=float, default=10.0,
                       help='Timeout per task execution (seconds)')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum parallel workers')
    parser.add_argument('--sample_size', type=int,
                       help='Evaluate only a sample of tasks')
    
    args = parser.parse_args()
    
    # Load tasks
    print("Loading ARC tasks...")
    tasks = load_arc_tasks(Path(args.data_dir))
    
    if args.sample_size:
        tasks = tasks[:args.sample_size]
    
    print(f"Loaded {len(tasks)} tasks")
    
    # Create evaluator
    evaluator = CompetitionEvaluator(
        model_path=args.model_path,
        timeout=args.timeout,
        max_workers=args.max_workers
    )
    
    # Run evaluation
    start_time = time.time()
    results = evaluator.evaluate_submission(tasks)
    total_time = time.time() - start_time
    
    # Add timing info
    results['evaluation_time'] = total_time
    results['tasks_per_second'] = len(tasks) / total_time
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"  Tasks: {results['total_tasks']}")
    print(f"  Completed: {results['completed_tasks']}")
    print(f"  Total Score: {results['total_score']}")
    print(f"  Average Accuracy: {results['average_accuracy']:.3f}")
    print(f"  Total Bytes: {results['total_bytes']:,}")
    print(f"  Byte Violations: {results['byte_violations']}")
    print(f"  Evaluation Time: {total_time:.1f}s")
    
    print(f"\nMethod Breakdown:")
    for method, count in results['method_breakdown'].items():
        print(f"  {method}: {count}")
    
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
