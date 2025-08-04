"""
ARC Dataset Loader for MicroGolf
Loads and processes the Abstraction and Reasoning Corpus dataset
"""
"""
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge
License: Apache 2.0
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
import random


class ARCTask:
    """Represents a single ARC task with training examples and test case"""
    
    def __init__(self, task_id: str, data: Dict):
        self.task_id = task_id
        self.train_examples = data['train']
        self.test_examples = data['test']
    
    def get_training_pairs(self) -> List[Tuple[List[List[int]], List[List[int]]]]:
        """Get list of (input, output) pairs for training"""
        return [(example['input'], example['output']) for example in self.train_examples]
    
    def get_test_input(self) -> List[List[int]]:
        """Get the test input grid"""
        return self.test_examples[0]['input']
    
    def get_test_output(self) -> List[List[int]]:
        """Get the test output grid (for validation)"""
        return self.test_examples[0]['output']
    
    def analyze_patterns(self) -> Dict:
        """Analyze patterns in the task for feature extraction"""
        patterns = {
            'grid_sizes': [],
            'colors_used': set(),
            'transformations': [],
            'complexity': 0
        }
        
        for example in self.train_examples:
            input_grid = example['input']
            output_grid = example['output']
            
            # Grid dimensions
            patterns['grid_sizes'].append((len(input_grid), len(input_grid[0])))
            
            # Colors used
            for row in input_grid + output_grid:
                patterns['colors_used'].update(row)
            
            # Basic transformation analysis
            if input_grid != output_grid:
                patterns['transformations'].append(self._analyze_transformation(input_grid, output_grid))
        
        patterns['colors_used'] = list(patterns['colors_used'])
        patterns['complexity'] = len(patterns['colors_used']) * len(patterns['transformations'])
        
        return patterns
    
    def _analyze_transformation(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict:
        """Analyze the transformation between input and output"""
        transform = {
            'type': 'unknown',
            'preserves_size': len(input_grid) == len(output_grid) and len(input_grid[0]) == len(output_grid[0]),
            'preserves_colors': set(self._flatten(input_grid)) == set(self._flatten(output_grid)),
            'similarity': self._grid_similarity(input_grid, output_grid)
        }
        
        # Check for common transformations
        if self._is_rotation(input_grid, output_grid):
            transform['type'] = 'rotation'
        elif self._is_reflection(input_grid, output_grid):
            transform['type'] = 'reflection'
        elif self._is_color_mapping(input_grid, output_grid):
            transform['type'] = 'color_mapping'
        elif self._is_shape_addition(input_grid, output_grid):
            transform['type'] = 'shape_addition'
        
        return transform
    
    def _flatten(self, grid: List[List[int]]) -> List[int]:
        """Flatten a 2D grid to 1D list"""
        return [cell for row in grid for cell in row]
    
    def _grid_similarity(self, grid1: List[List[int]], grid2: List[List[int]]) -> float:
        """Calculate similarity between two grids (0-1)"""
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return 0.0
        
        total_cells = len(grid1) * len(grid1[0])
        matching_cells = sum(
            1 for i in range(len(grid1)) 
            for j in range(len(grid1[0])) 
            if grid1[i][j] == grid2[i][j]
        )
        
        return matching_cells / total_cells
    
    def _is_rotation(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """Check if output is a rotation of input"""
        # Simple check for 90-degree rotation
        if len(input_grid) != len(output_grid[0]) or len(input_grid[0]) != len(output_grid):
            return False
        
        # Check 90-degree clockwise rotation
        rotated = [[input_grid[len(input_grid)-1-j][i] for j in range(len(input_grid))] 
                  for i in range(len(input_grid[0]))]
        
        return rotated == output_grid
    
    def _is_reflection(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """Check if output is a reflection of input"""
        # Horizontal flip
        h_flip = input_grid[::-1]
        if h_flip == output_grid:
            return True
        
        # Vertical flip
        v_flip = [row[::-1] for row in input_grid]
        if v_flip == output_grid:
            return True
        
        return False
    
    def _is_color_mapping(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """Check if output is a color mapping of input"""
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return False
        
        # Check if it's just a color substitution
        color_map = {}
        for i in range(len(input_grid)):
            for j in range(len(input_grid[0])):
                in_color = input_grid[i][j]
                out_color = output_grid[i][j]
                
                if in_color in color_map:
                    if color_map[in_color] != out_color:
                        return False
                else:
                    color_map[in_color] = out_color
        
        return len(color_map) > 1  # More than just identity mapping
    
    def _is_shape_addition(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """Check if output adds shapes to input"""
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return False
        
        # Count non-zero cells
        input_non_zero = sum(1 for row in input_grid for cell in row if cell != 0)
        output_non_zero = sum(1 for row in output_grid for cell in row if cell != 0)
        
        return output_non_zero > input_non_zero


class ARCDataLoader:
    """Loads and manages ARC dataset for MicroGolf training"""
    
    def __init__(self, data_dir: str = "data/arc"):
        self.data_dir = Path(data_dir)
        self.training_dir = self.data_dir / "training" / "training"
        self.evaluation_dir = self.data_dir / "evaluation" / "evaluation"
        
        self.training_tasks = {}
        self.evaluation_tasks = {}
        
        self._load_tasks()
    
    def _load_tasks(self):
        """Load all ARC tasks from disk"""
        print("Loading ARC training tasks...")
        self._load_task_set(self.training_dir, self.training_tasks)
        
        print("Loading ARC evaluation tasks...")
        self._load_task_set(self.evaluation_dir, self.evaluation_tasks)
        
        print(f"Loaded {len(self.training_tasks)} training tasks and {len(self.evaluation_tasks)} evaluation tasks")
    
    def _load_task_set(self, task_dir: Path, task_dict: Dict[str, ARCTask]):
        """Load tasks from a directory"""
        if not task_dir.exists():
            print(f"Warning: {task_dir} does not exist")
            return
        
        for json_file in task_dir.glob("*.json"):
            task_id = json_file.stem
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                task_dict[task_id] = ARCTask(task_id, data)
            except Exception as e:
                print(f"Error loading task {task_id}: {e}")
    
    def get_training_tasks(self) -> Dict[str, ARCTask]:
        """Get all training tasks"""
        return self.training_tasks
    
    def get_evaluation_tasks(self) -> Dict[str, ARCTask]:
        """Get all evaluation tasks"""
        return self.evaluation_tasks
    
    def get_task(self, task_id: str) -> Optional[ARCTask]:
        """Get a specific task by ID"""
        return self.training_tasks.get(task_id) or self.evaluation_tasks.get(task_id)
    
    def get_random_tasks(self, n: int, from_training: bool = True) -> List[ARCTask]:
        """Get n random tasks"""
        task_set = self.training_tasks if from_training else self.evaluation_tasks
        task_ids = list(task_set.keys())
        selected_ids = random.sample(task_ids, min(n, len(task_ids)))
        return [task_set[task_id] for task_id in selected_ids]
    
    def get_tasks_by_complexity(self, max_complexity: int = 10, from_training: bool = True) -> List[ARCTask]:
        """Get tasks filtered by complexity"""
        task_set = self.training_tasks if from_training else self.evaluation_tasks
        filtered_tasks = []
        
        for task in task_set.values():
            patterns = task.analyze_patterns()
            if patterns['complexity'] <= max_complexity:
                filtered_tasks.append(task)
        
        return filtered_tasks
    
    def create_training_batch(self, batch_size: int = 32) -> Iterator[List[Tuple[ARCTask, Tuple]]]:
        """Create batches for training the meta-learning model"""
        tasks = list(self.training_tasks.values())
        random.shuffle(tasks)
        
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i+batch_size]
            batch_data = []
            
            for task in batch_tasks:
                # Get training examples for this task
                training_pairs = task.get_training_pairs()
                # Add task and its training examples to batch
                batch_data.append((task, training_pairs))
            
            yield batch_data
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the dataset"""
        stats = {
            'num_training_tasks': len(self.training_tasks),
            'num_evaluation_tasks': len(self.evaluation_tasks),
            'grid_sizes': {},
            'colors_distribution': {},
            'transformation_types': {},
            'complexity_distribution': {}
        }
        
        # Analyze all tasks
        all_tasks = list(self.training_tasks.values()) + list(self.evaluation_tasks.values())
        
        for task in all_tasks:
            patterns = task.analyze_patterns()
            
            # Grid sizes
            for size in patterns['grid_sizes']:
                size_key = f"{size[0]}x{size[1]}"
                stats['grid_sizes'][size_key] = stats['grid_sizes'].get(size_key, 0) + 1
            
            # Colors
            num_colors = len(patterns['colors_used'])
            stats['colors_distribution'][num_colors] = stats['colors_distribution'].get(num_colors, 0) + 1
            
            # Transformations
            for transform in patterns['transformations']:
                t_type = transform['type']
                stats['transformation_types'][t_type] = stats['transformation_types'].get(t_type, 0) + 1
            
            # Complexity
            complexity = patterns['complexity']
            complexity_bucket = f"{complexity//5*5}-{complexity//5*5+4}"
            stats['complexity_distribution'][complexity_bucket] = stats['complexity_distribution'].get(complexity_bucket, 0) + 1
        
        return stats


if __name__ == "__main__":
    # Demo usage
    loader = ARCDataLoader()
    
    # Get dataset statistics
    stats = loader.get_dataset_statistics()
    print("\n=== ARC Dataset Statistics ===")
    print(f"Training tasks: {stats['num_training_tasks']}")
    print(f"Evaluation tasks: {stats['num_evaluation_tasks']}")
    
    print(f"\nMost common grid sizes:")
    for size, count in sorted(stats['grid_sizes'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {size}: {count} tasks")
    
    print(f"\nColor distribution:")
    for num_colors, count in sorted(stats['colors_distribution'].items()):
        print(f"  {num_colors} colors: {count} tasks")
    
    # Analyze a sample task
    print("\n=== Sample Task Analysis ===")
    sample_tasks = loader.get_random_tasks(3)
    for task in sample_tasks:
        patterns = task.analyze_patterns()
        print(f"Task {task.task_id}:")
        print(f"  Grid sizes: {patterns['grid_sizes']}")
        print(f"  Colors used: {patterns['colors_used']}")
        print(f"  Complexity: {patterns['complexity']}")
        print(f"  Transformations: {[t['type'] for t in patterns['transformations']]}")
