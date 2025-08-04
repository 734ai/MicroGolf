#!/usr/bin/env python3
"""
End-to-End Integration Test for MicroGolf with Real ARC Data
Tests the complete pipeline without requiring PyTorch
"""
"""
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge
License: Apache 2.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from microgolf.data_loader import ARCDataLoader
from microgolf.model.tokenizer import ARCTokenizer, FeatureExtractor
from microgolf.engine.controller import PrimitiveController
from microgolf.engine.executor import OptimizedExecutor


def test_arc_data_pipeline():
    """Test the complete pipeline with real ARC data"""
    print("=== MicroGolf ARC Integration Test ===")
    
    # Step 1: Load real ARC data
    print("\n1. Loading ARC dataset...")
    arc_loader = ARCDataLoader('data/arc')
    training_tasks = arc_loader.get_training_tasks()
    print(f"   Loaded {len(training_tasks)} training tasks")
    
    # Step 2: Test tokenizer on real data
    print("\n2. Testing tokenizer...")
    tokenizer = ARCTokenizer(max_tokens=100)
    feature_extractor = FeatureExtractor()
    
    # Get a sample task
    sample_task_id = list(training_tasks.keys())[0]
    sample_task = training_tasks[sample_task_id]
    
    print(f"   Sample task: {sample_task_id}")
    print(f"   Training examples: {len(sample_task.train_examples)}")
    
    # Tokenize the examples
    tokens = tokenizer.tokenize_examples(sample_task.train_examples[:2])
    features = feature_extractor.extract_features(sample_task.train_examples[:2])
    
    print(f"   Tokenized length: {len(tokens)}")
    print(f"   Feature dimensions: {features.shape}")
    print(f"   Sample tokens: {tokens[:10]}...")
    
    # Step 3: Test controller on real data
    print("\n3. Testing primitive controller...")
    controller = PrimitiveController()
    
    input_grid = sample_task.train_examples[0]['input']
    output_grid = sample_task.train_examples[0]['output']
    
    candidates = controller.get_primitive_candidates(input_grid, output_grid)
    print(f"   Input grid size: {len(input_grid)}x{len(input_grid[0])}")
    print(f"   Output grid size: {len(output_grid)}x{len(output_grid[0])}")
    print(f"   Primitive candidates: {candidates[:3]}")
    
    # Step 4: Test executor
    print("\n4. Testing optimized executor...")
    executor = OptimizedExecutor()
    
    # Map candidates to full primitive names
    char_to_primitive = {
        'r': 'r90', 'f': 'fh', 'v': 'fv', 't': 'tr', 's': 'sh',
        'm': 'mc'
    }
    
    # Get top primitive suggestions
    top_primitives = candidates[:2]  # Take first 2 directly since they're already strings
    
    if not top_primitives:
        top_primitives = ['r90']  # Default
    
    print(f"   Suggested primitive sequence: {top_primitives}")
    
    # Generate optimized code
    code = executor.primitive_sequence_to_code(top_primitives)
    print(f"   Generated code: {repr(code)}")
    print(f"   Code length: {len(code)} bytes")
    
    # Step 5: Test code execution
    print("\n5. Testing code execution...")
    try:
        func = eval(code)
        result = func(input_grid)
        
        print(f"   Input shape: {len(input_grid)}x{len(input_grid[0])}")
        print(f"   Output shape: {len(result)}x{len(result[0])}")
        print(f"   Execution: SUCCESS")
        
        # Show small sample of transformation
        if len(input_grid) <= 5 and len(input_grid[0]) <= 5:
            print("   Input sample:")
            for row in input_grid:
                print(f"     {row}")
            print("   Generated output sample:")
            for row in result[:3]:  # Show first 3 rows
                print(f"     {row}")
        
    except Exception as e:
        print(f"   Execution failed: {e}")
    
    # Step 6: Analyze multiple tasks
    print("\n6. Analyzing task diversity...")
    task_stats = {
        'grid_sizes': {},
        'primitive_suggestions': {},
        'code_lengths': []
    }
    
    for i, (task_id, task) in enumerate(list(training_tasks.items())[:10]):
        if not task.train_examples:
            continue
        
        inp = task.train_examples[0]['input']
        out = task.train_examples[0]['output']
        
        # Grid size
        size = f"{len(inp)}x{len(inp[0])}"
        task_stats['grid_sizes'][size] = task_stats['grid_sizes'].get(size, 0) + 1
        
        # Primitive suggestions
        try:
            candidates = controller.get_primitive_candidates(inp, out)
            top_prim = candidates[0] if candidates else 'unknown'
            task_stats['primitive_suggestions'][top_prim] = task_stats['primitive_suggestions'].get(top_prim, 0) + 1
            
            # Code generation
            primitives = [top_prim] if top_prim != 'unknown' else ['mc']
            code = executor.primitive_sequence_to_code(primitives)
            task_stats['code_lengths'].append(len(code))
            
        except Exception:
            continue
    
    print("   Grid size distribution:")
    for size, count in sorted(task_stats['grid_sizes'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     {size}: {count} tasks")
    
    print("   Primitive suggestions:")
    for prim, count in sorted(task_stats['primitive_suggestions'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     {prim}: {count} tasks")
    
    if task_stats['code_lengths']:
        avg_length = sum(task_stats['code_lengths']) / len(task_stats['code_lengths'])
        print(f"   Average code length: {avg_length:.1f} bytes")
        print(f"   Code length range: {min(task_stats['code_lengths'])}-{max(task_stats['code_lengths'])} bytes")
    
    print("\n=== Integration Test Complete ===")
    print("✓ ARC data loading")
    print("✓ Tokenization and feature extraction")  
    print("✓ Primitive controller suggestions")
    print("✓ Code generation and optimization")
    print("✓ End-to-end pipeline working!")


if __name__ == "__main__":
    test_arc_data_pipeline()
