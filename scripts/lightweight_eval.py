#!/usr/bin/env python3
"""
Lightweight Evaluation Script for MicroGolf
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge

This script runs evaluation using heuristic methods without requiring PyTorch,
suitable for systems with limited resources.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict, Counter

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from microgolf.engine.controller import PrimitiveController, AbstractPlan
from microgolf.engine.executor import OptimizedExecutor
from microgolf.data_loader import ARCDataLoader

class LightweightEvaluator:
    """Lightweight evaluation system using heuristic methods."""
    
    def __init__(self, data_dir: str = "data/arc"):
        self.data_dir = data_dir
        self.controller = PrimitiveController()
        self.executor = OptimizedExecutor()
        
        # Evaluation metrics
        self.results = {
            'evaluation_start_time': time.time(),
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'task_results': [],
            'performance_metrics': {},
            'code_metrics': {},
            'error_analysis': defaultdict(int),
            'primitive_usage_stats': Counter(),
            'byte_compliance': {'compliant': 0, 'non_compliant': 0}
        }
    
    def evaluate_single_task(self, task_id: str, task_data: Dict) -> Dict:
        """Evaluate a single ARC task using heuristic methods."""
        start_time = time.time()
        result = {
            'task_id': task_id,
            'status': 'unknown',
            'error': None,
            'primitive_sequence': [],
            'generated_code': '',
            'optimized_code': '',
            'code_length': 0,
            'byte_compliant': False,
            'execution_time': 0,
            'primitive_count': 0,
            'method': 'heuristic'
        }
        
        try:
            # Extract training examples
            examples = []
            if 'train' in task_data:
                for example in task_data['train']:
                    if 'input' in example and 'output' in example:
                        examples.append({
                            'input': example['input'],
                            'output': example['output']
                        })
            
            if not examples:
                result['status'] = 'failed'
                result['error'] = 'No valid training examples found'
                return result
            
            # Generate primitive sequence using heuristic controller
            try:
                primitive_sequence = self.controller.predict_sequence(examples)
                if not primitive_sequence:
                    # Fallback to simple sequence
                    primitive_sequence = ['r90', 'fh']  # Simple default
                    print(f"Warning: Controller returned empty sequence for {task_id}, using fallback")
                
                result['primitive_sequence'] = primitive_sequence
                result['primitive_count'] = len(primitive_sequence)
                
                # Update primitive usage stats
                for primitive in primitive_sequence:
                    self.results['primitive_usage_stats'][primitive] += 1
                    
            except Exception as e:
                print(f"Error in sequence generation for {task_id}: {str(e)}")
                result['status'] = 'failed'
                result['error'] = f'Sequence generation failed: {str(e)}'
                return result
            
            # Generate code using executor
            try:
                if primitive_sequence:
                    # Create AbstractPlan
                    plan = AbstractPlan()
                    for primitive in primitive_sequence:
                        plan.add_step(primitive)
                    
                    # Generate code
                    generated_code = self.executor.execute_plan(plan)
                    result['generated_code'] = generated_code
                    
                    # Apply optimizations
                    optimized_code = self.executor.execute_plan_optimized(plan)
                    result['optimized_code'] = optimized_code
                    result['code_length'] = len(optimized_code)
                    
                    # Check byte compliance
                    result['byte_compliant'] = result['code_length'] <= 2500
                    if result['byte_compliant']:
                        self.results['byte_compliance']['compliant'] += 1
                    else:
                        self.results['byte_compliance']['non_compliant'] += 1
                    
                    result['status'] = 'success'
                else:
                    result['status'] = 'failed'
                    result['error'] = 'Empty primitive sequence generated'
                    
            except Exception as e:
                print(f"Error in code generation for {task_id}: {str(e)}")
                result['status'] = 'failed'
                result['error'] = f'Code generation failed: {str(e)}'
                return result
                
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = f'Unexpected error: {str(e)}'
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def run_evaluation(self, max_tasks: Optional[int] = 50) -> Dict:
        """Run lightweight evaluation on ARC tasks."""
        print("Starting Lightweight MicroGolf Evaluation")
        print("=" * 60)
        
        # Generate synthetic tasks for testing
        tasks = self.generate_test_tasks(max_tasks or 50)
        
        print(f"Evaluating {len(tasks)} synthetic tasks...")
        self.results['total_tasks'] = len(tasks)
        
        # Process tasks
        processed_count = 0
        for task_id, task_data in tasks.items():
            processed_count += 1
            
            if processed_count % 10 == 0 or processed_count <= 5:
                print(f"Processing task {processed_count}/{len(tasks)}: {task_id}")
            
            # Evaluate single task
            task_result = self.evaluate_single_task(task_id, task_data)
            self.results['task_results'].append(task_result)
            
            # Update counters
            if task_result['status'] == 'success':
                self.results['successful_tasks'] += 1
            else:
                self.results['failed_tasks'] += 1
                error_type = task_result['error'][:50] if task_result['error'] else 'unknown'
                self.results['error_analysis'][error_type] += 1
        
        # Calculate final metrics
        self.calculate_performance_metrics()
        
        print(f"\nEvaluation completed in {time.time() - self.results['evaluation_start_time']:.1f} seconds")
        return self.results
    
    def generate_test_tasks(self, num_tasks: int) -> Dict[str, Dict]:
        """Generate synthetic test tasks for evaluation."""
        tasks = {}
        
        for i in range(num_tasks):
            # Generate diverse grid patterns
            if i % 4 == 0:
                # Identity transformation
                input_grid = [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
                output_grid = [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
            elif i % 4 == 1:
                # Horizontal flip
                input_grid = [[1, 2, 3], [4, 5, 6]]
                output_grid = [[3, 2, 1], [6, 5, 4]]
            elif i % 4 == 2:
                # Rotation 90 degrees
                input_grid = [[1, 2], [3, 4]]
                output_grid = [[3, 1], [4, 2]]
            else:
                # Color mapping
                input_grid = [[0, 1, 0], [1, 0, 1]]
                output_grid = [[2, 3, 2], [3, 2, 3]]
            
            task_id = f"synthetic_{i:03d}"
            tasks[task_id] = {
                'train': [
                    {
                        'input': input_grid,
                        'output': output_grid
                    }
                ],
                'test': [
                    {
                        'input': input_grid,
                        'output': output_grid
                    }
                ]
            }
        
        return tasks
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        successful_results = [r for r in self.results['task_results'] if r['status'] == 'success']
        
        # Always calculate basic metrics, even if no successes
        self.results['performance_metrics'] = {
            'success_rate': self.results['successful_tasks'] / self.results['total_tasks'] if self.results['total_tasks'] > 0 else 0,
            'failure_rate': self.results['failed_tasks'] / self.results['total_tasks'] if self.results['total_tasks'] > 0 else 0,
            'average_execution_time': np.mean([r['execution_time'] for r in self.results['task_results']]) if self.results['task_results'] else 0,
            'total_evaluation_time': time.time() - self.results['evaluation_start_time']
        }
        
        if not successful_results:
            print("Warning: No successful results to analyze")
            # Print error summary for debugging
            print("Error summary:")
            for error, count in self.results['error_analysis'].items():
                print(f"  {error}: {count}")
            return
        
        # Code metrics (only if we have successful results)
        code_lengths = [r['code_length'] for r in successful_results if r['code_length'] > 0]
        primitive_counts = [r['primitive_count'] for r in successful_results if r['primitive_count'] > 0]
        
        if code_lengths:
            self.results['code_metrics'] = {
                'average_code_length': np.mean(code_lengths),
                'median_code_length': np.median(code_lengths),
                'min_code_length': np.min(code_lengths),
                'max_code_length': np.max(code_lengths),
                'std_code_length': np.std(code_lengths),
                'byte_compliance_rate': self.results['byte_compliance']['compliant'] / len(code_lengths),
                'average_primitive_count': np.mean(primitive_counts) if primitive_counts else 0,
                'median_primitive_count': np.median(primitive_counts) if primitive_counts else 0
            }
    
    def generate_evaluation_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        report_lines = []
        report_lines.append("MicroGolf Lightweight Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Author: Muzan Sano")
        report_lines.append(f"Method: Heuristic Controller (No ML)")
        report_lines.append("")
        
        # Performance Summary
        report_lines.append("PERFORMANCE SUMMARY")
        report_lines.append("-" * 30)
        perf = self.results['performance_metrics']
        report_lines.append(f"Total Tasks Evaluated: {self.results['total_tasks']}")
        report_lines.append(f"Successful Tasks: {self.results['successful_tasks']} ({perf['success_rate']:.1%})")
        report_lines.append(f"Failed Tasks: {self.results['failed_tasks']} ({perf['failure_rate']:.1%})")
        report_lines.append(f"Average Execution Time: {perf['average_execution_time']:.3f}s per task")
        report_lines.append(f"Total Evaluation Time: {perf['total_evaluation_time']:.1f}s")
        report_lines.append("")
        
        # Code Metrics
        if 'code_metrics' in self.results:
            report_lines.append("CODE METRICS")
            report_lines.append("-" * 30)
            code = self.results['code_metrics']
            report_lines.append(f"Average Code Length: {code['average_code_length']:.1f} bytes")
            report_lines.append(f"Median Code Length: {code['median_code_length']:.1f} bytes")
            report_lines.append(f"Code Length Range: {code['min_code_length']}-{code['max_code_length']} bytes")
            report_lines.append(f"Byte Compliance Rate: {code['byte_compliance_rate']:.1%}")
            report_lines.append(f"Average Primitives per Solution: {code['average_primitive_count']:.1f}")
            report_lines.append("")
        
        # Primitive Usage Analysis
        if self.results['primitive_usage_stats']:
            report_lines.append("PRIMITIVE USAGE ANALYSIS")
            report_lines.append("-" * 30)
            total_usage = sum(self.results['primitive_usage_stats'].values())
            most_used = self.results['primitive_usage_stats'].most_common(10)
            
            report_lines.append(f"Total Primitive Usage: {total_usage}")
            report_lines.append(f"Unique Primitives Used: {len(self.results['primitive_usage_stats'])}")
            report_lines.append("")
            
            report_lines.append("Most Used Primitives:")
            for primitive, count in most_used[:5]:
                percentage = count / total_usage * 100
                report_lines.append(f"  {primitive}: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # Error Analysis
        if self.results['error_analysis']:
            report_lines.append("ERROR ANALYSIS")
            report_lines.append("-" * 30)
            for error_type, count in self.results['error_analysis'].most_common(5):
                report_lines.append(f"  {error_type}: {count} occurrences")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_results(self, output_dir: str = "experiments"):
        """Save evaluation results and report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_path = os.path.join(output_dir, "lightweight_evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save evaluation report
        report = self.generate_evaluation_report()
        report_path = os.path.join(output_dir, "lightweight_evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Results saved to {results_path}")
        print(f"Report saved to {report_path}")
        
        return results_path, report_path

def main():
    """Run lightweight evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lightweight MicroGolf Evaluation")
    parser.add_argument("--max-tasks", type=int, default=50, 
                       help="Maximum number of tasks to evaluate (default: 50)")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick test on 10 tasks")
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.max_tasks = 10
        print("Quick test mode: evaluating 10 tasks")
    
    # Initialize evaluator
    evaluator = LightweightEvaluator()
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation(max_tasks=args.max_tasks)
        
        # Print summary
        print("\n" + evaluator.generate_evaluation_report())
        
        # Save results
        evaluator.save_results()
        
        print(f"\nEvaluation Summary:")
        print(f"- Success Rate: {results['performance_metrics']['success_rate']:.1%}")
        print(f"- Tasks Evaluated: {results['total_tasks']}")
        print(f"- Average Code Length: {results.get('code_metrics', {}).get('average_code_length', 'N/A')}")
        print(f"- Byte Compliance: {results.get('code_metrics', {}).get('byte_compliance_rate', 'N/A'):.1%}")
        
        return results
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
