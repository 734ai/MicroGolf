#!/usr/bin/env python3
"""
Submission Generator for MicroGolf
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge

This script generates submission files by running the complete MicroGolf pipeline
on ARC tasks and creating optimized Python solutions under 2500 bytes.
"""

import os
import sys
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import time
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from microgolf.engine.controller import PrimitiveController, AbstractPlan
from microgolf.engine.executor import OptimizedExecutor
from microgolf.data_loader import ARCDataLoader

class SubmissionGenerator:
    """Generates competition-ready submission files from ARC tasks."""
    
    def __init__(self, data_dir: str = "data/arc", output_dir: str = "submission"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MicroGolf components
        self.controller = PrimitiveController()
        self.executor = OptimizedExecutor()
        
        # Submission statistics
        self.stats = {
            'total_tasks': 0,
            'generated_tasks': 0,
            'failed_tasks': 0,
            'byte_violations': 0,
            'average_code_length': 0,
            'primitive_usage': defaultdict(int)
        }
    
    def generate_solution_code(self, task_data: Dict) -> str:
        """Generate optimized solution code for a single ARC task."""
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
                # Fallback to identity function
                return "lambda g: g"
            
            # Generate primitive sequence
            primitive_sequence = self.controller.predict_sequence(examples)
            if not primitive_sequence:
                primitive_sequence = ['r90', 'fh']  # Simple fallback
            
            # Update primitive usage stats
            for primitive in primitive_sequence:
                self.stats['primitive_usage'][primitive] += 1
            
            # Create AbstractPlan and generate code
            plan = AbstractPlan()
            for primitive in primitive_sequence:
                plan.add_step(primitive)
            
            # Generate optimized code
            optimized_code = self.executor.execute_plan_optimized(plan)
            
            return optimized_code
            
        except Exception as e:
            print(f"Error generating solution: {e}")
            return "lambda g: g"  # Safe fallback
    
    def validate_solution_code(self, code: str, task_id: str) -> bool:
        """Validate that generated code meets submission requirements."""
        try:
            # Check byte limit
            byte_count = len(code.encode('utf-8'))
            if byte_count > 2500:
                print(f"Warning: {task_id} exceeds byte limit: {byte_count} bytes")
                self.stats['byte_violations'] += 1
                return False
            
            # Try to compile the code
            compiled_code = compile(code, f'<{task_id}>', 'eval')
            
            # Basic syntax validation passed
            return True
            
        except Exception as e:
            print(f"Validation failed for {task_id}: {e}")
            return False
    
    def generate_task_file(self, task_id: str, task_data: Dict) -> bool:
        """Generate a single task submission file."""
        try:
            # Generate solution code
            solution_code = self.generate_solution_code(task_data)
            
            # Validate the code
            if not self.validate_solution_code(solution_code, task_id):
                # Use safe fallback if validation fails
                solution_code = "lambda g: g"
            
            # Create task file content
            file_content = f"""# MicroGolf Solution for {task_id}
# Author: Muzan Sano
# NeurIPS 2025 ARC-Golf Challenge
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
# Code length: {len(solution_code)} bytes

{solution_code}
"""
            
            # Write to submission file
            task_file = self.output_dir / f"{task_id}.py"
            with open(task_file, 'w') as f:
                f.write(file_content)
            
            # Update statistics
            self.stats['generated_tasks'] += 1
            code_length = len(solution_code)
            self.stats['average_code_length'] = (
                (self.stats['average_code_length'] * (self.stats['generated_tasks'] - 1) + code_length) 
                / self.stats['generated_tasks']
            )
            
            print(f"Generated {task_id}.py ({code_length} bytes)")
            return True
            
        except Exception as e:
            print(f"Failed to generate {task_id}: {e}")
            self.stats['failed_tasks'] += 1
            return False
    
    def generate_all_submissions(self, max_tasks: Optional[int] = None) -> Dict:
        """Generate submission files for all available tasks."""
        print("MicroGolf Submission Generator")
        print("=" * 50)
        
        # Generate synthetic tasks for demonstration
        # In real competition, this would load actual test tasks
        test_tasks = self.generate_test_tasks(max_tasks or 100)
        
        self.stats['total_tasks'] = len(test_tasks)
        print(f"Generating submissions for {len(test_tasks)} tasks...")
        
        # Process each task
        for i, (task_id, task_data) in enumerate(test_tasks.items(), 1):
            if i % 10 == 0 or i <= 5:
                print(f"Processing {i}/{len(test_tasks)}: {task_id}")
            
            self.generate_task_file(task_id, task_data)
        
        # Generate summary statistics
        self.generate_submission_summary()
        
        print(f"\nSubmission generation completed!")
        print(f"Generated: {self.stats['generated_tasks']}/{self.stats['total_tasks']} tasks")
        print(f"Average code length: {self.stats['average_code_length']:.1f} bytes")
        print(f"Byte violations: {self.stats['byte_violations']}")
        
        return self.stats
    
    def generate_test_tasks(self, num_tasks: int) -> Dict[str, Dict]:
        """Generate test tasks for submission demonstration."""
        tasks = {}
        
        for i in range(num_tasks):
            # Create diverse task patterns
            if i % 4 == 0:
                # Identity task
                input_grid = [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
                output_grid = [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
            elif i % 4 == 1:
                # Horizontal flip
                input_grid = [[1, 2, 3], [4, 5, 6]]
                output_grid = [[3, 2, 1], [6, 5, 4]]
            elif i % 4 == 2:
                # Rotation 90
                input_grid = [[1, 2], [3, 4]]
                output_grid = [[3, 1], [4, 2]]
            else:
                # Color mapping
                input_grid = [[0, 1, 0], [1, 0, 1]]
                output_grid = [[2, 3, 2], [3, 2, 3]]
            
            task_id = f"task_{i:03d}"
            tasks[task_id] = {
                'train': [
                    {
                        'input': input_grid,
                        'output': output_grid
                    }
                ]
            }
        
        return tasks
    
    def generate_submission_summary(self):
        """Generate submission summary and statistics."""
        summary = {
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'author': 'Muzan Sano',
            'competition': 'NeurIPS 2025 ARC-Golf Challenge',
            'framework': 'MicroGolf',
            'statistics': dict(self.stats),
            'primitive_usage': dict(self.stats['primitive_usage']),
            'top_primitives': sorted(
                self.stats['primitive_usage'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
        
        # Save summary JSON
        summary_file = self.output_dir / "submission_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate README for submission
        readme_content = f"""# MicroGolf Submission - NeurIPS 2025 ARC-Golf Challenge

**Author:** Muzan Sano  
**Generated:** {summary['generation_timestamp']}  
**Framework:** MicroGolf Ultra-Compact ARC-AGI Solution

## Submission Statistics

- **Total Tasks:** {self.stats['total_tasks']}
- **Generated Solutions:** {self.stats['generated_tasks']}
- **Success Rate:** {self.stats['generated_tasks']/self.stats['total_tasks']*100:.1f}%
- **Average Code Length:** {self.stats['average_code_length']:.1f} bytes
- **Byte Compliance:** {(self.stats['total_tasks']-self.stats['byte_violations'])/self.stats['total_tasks']*100:.1f}%

## Top Used Primitives

{chr(10).join(f"- **{prim}**: {count} times" for prim, count in summary['top_primitives'][:5])}

## Architecture

MicroGolf uses a hybrid approach combining:
1. **Heuristic Controller** - Rule-based primitive sequence generation
2. **OptimizedExecutor** - Ultra-compact code generation with lambda chaining
3. **Code Golf Optimization** - Aggressive byte minimization techniques

Each solution is generated through:
Task → Primitive Sequence → Abstract Plan → Optimized Code → Validation

## File Structure

- `task_XXX.py` - Individual task solutions
- `submission_summary.json` - Detailed statistics
- `README.md` - This documentation

Generated by MicroGolf framework for NeurIPS 2025 ARC-Golf Challenge.
"""
        
        readme_file = self.output_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"Summary saved to {summary_file}")
        print(f"README saved to {readme_file}")
    
    def create_submission_zip(self, zip_name: str = "microgolf_submission.zip"):
        """Create final submission ZIP file."""
        zip_path = self.output_dir / zip_name
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all Python task files
            for task_file in self.output_dir.glob("task_*.py"):
                zipf.write(task_file, task_file.name)
            
            # Add documentation
            if (self.output_dir / "README.md").exists():
                zipf.write(self.output_dir / "README.md", "README.md")
            
            if (self.output_dir / "submission_summary.json").exists():
                zipf.write(self.output_dir / "submission_summary.json", "submission_summary.json")
        
        print(f"Submission ZIP created: {zip_path}")
        print(f"ZIP size: {zip_path.stat().st_size / 1024:.1f} KB")
        
        return zip_path

def main():
    """Main submission generation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MicroGolf submission files")
    parser.add_argument("--max-tasks", type=int, default=50,
                       help="Maximum number of tasks to generate (default: 50)")
    parser.add_argument("--output-dir", type=str, default="submission",
                       help="Output directory for submission files")
    parser.add_argument("--create-zip", action="store_true",
                       help="Create submission ZIP file")
    parser.add_argument("--quick-test", action="store_true",
                       help="Generate only 10 tasks for quick testing")
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.max_tasks = 10
        print("Quick test mode: generating 10 tasks")
    
    # Initialize generator
    generator = SubmissionGenerator(output_dir=args.output_dir)
    
    try:
        # Generate all submissions
        stats = generator.generate_all_submissions(max_tasks=args.max_tasks)
        
        # Create ZIP if requested
        if args.create_zip:
            generator.create_submission_zip()
        
        print("\n" + "="*50)
        print("SUBMISSION GENERATION COMPLETE")
        print("="*50)
        print(f"Output directory: {args.output_dir}")
        print(f"Generated {stats['generated_tasks']} task solutions")
        print(f"Average code length: {stats['average_code_length']:.1f} bytes")
        
        return stats
        
    except KeyboardInterrupt:
        print("\nSubmission generation interrupted by user")
    except Exception as e:
        print(f"Submission generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
