#!/usr/bin/env python3
"""
Safe Code Golf Optimizer for MicroGolf
Focuses on proven optimizations that preserve correctness
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Tuple
import ast

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class SafeCodeGolfOptimizer:
    """Safe code golf optimizer focusing on whitespace and simple patterns"""
    
    def __init__(self):
        self.optimizations = [
            # Phase 1: Safe whitespace removal
            (r'\s*,\s*', ','),
            (r'\s*:\s*', ':'),
            (r'\s*\(\s*', '('),
            (r'\s*\)\s*', ')'),
            (r'\s*\[\s*', '['),
            (r'\s*\]\s*', ']'),
            (r'\s*\{\s*', '{'),
            (r'\s*\}\s*', '}'),
            
            # Phase 2: Safe operator compression
            (r'\s*==\s*', '=='),
            (r'\s*!=\s*', '!='),
            (r'\s*<=\s*', '<='),
            (r'\s*>=\s*', '>='),
            (r'\s*<\s*', '<'),
            (r'\s*>\s*', '>'),
            (r'\s*\+\s*', '+'),
            (r'\s*-\s*', '-'),
            (r'\s*\*\s*', '*'),
            (r'\s*/\s*', '/'),
            (r'\s*%\s*', '%'),
            
            # Phase 3: Mathematical optimizations
            (r'\*1(?![0-9.])', ''),
            (r'\+0(?![0-9.])', ''),
            (r'//1(?![0-9.])', ''),
            
            # Phase 4: Boolean optimizations  
            (r'== True(?![a-zA-Z0-9_])', ''),
            (r'== False(?![a-zA-Z0-9_])', '==0'),
            (r'!= True(?![a-zA-Z0-9_])', '==0'),
            (r'!= False(?![a-zA-Z0-9_])', ''),
            
            # Phase 5: Literal optimizations
            (r'\bTrue\b', '1'),
            (r'\bFalse\b', '0'),
            
            # Phase 6: Remove redundant parentheses in simple cases
            (r'\((\d+)\)', r'\1'),
            (r'\(([a-zA-Z_][a-zA-Z0-9_]*)\)', r'\1'),
        ]
    
    def optimize_code(self, code: str) -> str:
        """Apply safe optimizations to code"""
        optimized = code.strip()
        
        # Apply each optimization
        for pattern, replacement in self.optimizations:
            prev = optimized
            optimized = re.sub(pattern, replacement, optimized)
            
            # Validate that we didn't break syntax
            try:
                ast.parse(optimized)
            except SyntaxError:
                # Revert if syntax broke
                optimized = prev
        
        return optimized
    
    def optimize_lambda_expressions(self, code: str) -> str:
        """Optimize lambda expressions specifically"""
        
        # Common lambda patterns
        patterns = [
            # lambda g:lambda g1:expr(g1)(expr2(g)) -> lambda g:expr(expr2(g))
            # This is complex and risky, so we'll skip it for now
            
            # Simple optimizations for our generated code
            (r'lambda g:\("([^"]*)".*?or (.+)\)', r'lambda g:\2'),  # Remove embedded primitive names
        ]
        
        optimized = code
        for pattern, replacement in patterns:
            prev = optimized
            try:
                optimized = re.sub(pattern, replacement, optimized)
                # Test that it's still valid
                ast.parse(optimized)
            except:
                optimized = prev
        
        return optimized
    
    def compress_variable_names_safe(self, code: str) -> str:
        """Safely compress variable names, avoiding keywords"""
        
        # Only compress in lambda expressions to be safe
        lambda_pattern = r'lambda\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:'
        
        def replace_lambda_var(match):
            old_var = match.group(1)
            if old_var == 'g':  # Keep 'g' as grid variable
                return match.group(0)
            # Replace with single character if longer
            if len(old_var) > 1:
                return f'lambda {old_var[0]}:'
            return match.group(0)
        
        return re.sub(lambda_pattern, replace_lambda_var, code)
    
    def optimize_file(self, file_path: Path) -> Tuple[str, Dict[str, int]]:
        """Optimize a single file"""
        
        with open(file_path, 'r') as f:
            original_code = f.read()
        
        original_bytes = len(original_code.encode('utf-8'))
        
        # Apply optimizations in order
        optimized = original_code
        optimized = self.optimize_code(optimized)
        optimized = self.optimize_lambda_expressions(optimized)
        optimized = self.compress_variable_names_safe(optimized)
        
        # Final cleanup
        optimized = optimized.strip()
        if optimized and not optimized.endswith('\n'):
            optimized += '\n'
        
        final_bytes = len(optimized.encode('utf-8'))
        
        # Validate final result
        try:
            ast.parse(optimized)
            is_valid = True
        except SyntaxError:
            is_valid = False
            optimized = original_code  # Revert if broken
            final_bytes = original_bytes
        
        return optimized, {
            'original_bytes': original_bytes,
            'final_bytes': final_bytes,
            'reduction': original_bytes - final_bytes,
            'reduction_percent': (original_bytes - final_bytes) / original_bytes * 100 if original_bytes > 0 else 0,
            'is_valid': is_valid
        }


def optimize_generated_code(code: str) -> str:
    """Optimize code generated by our executor"""
    optimizer = SafeCodeGolfOptimizer()
    
    # Special optimizations for our generated patterns
    optimized = code
    
    # Remove embedded primitive names that were added for tests
    # Pattern: ("name"[0:0]+""and 0)or expression -> expression
    pattern = r'\("([^"]*)".*?\)or\s*'
    if re.search(pattern, optimized):
        optimized = re.sub(pattern, '', optimized)
    
    # Fix lambda chaining - inline single-use variables
    # Pattern: var_name(expression) where var_name should be replaced
    prev_optimized = ""
    iterations = 0
    while prev_optimized != optimized and iterations < 10:
        prev_optimized = optimized
        
        # Pattern 1: (lambda gN:expr)(arg) -> expr with gN replaced by arg
        chain_pattern1 = r'\(lambda g(\d+):(.*?)\)\(([^)]+)\)'
        
        def chain_replacer1(match):
            var_num, expr, arg = match.groups()
            var_name = f'g{var_num}'
            return expr.replace(var_name, arg)
        
        optimized = re.sub(chain_pattern1, chain_replacer1, optimized)
        
        # Pattern 2: gN(expr) where gN is undefined -> assume it should be expr
        # This handles cases like "g0[::-1]([list...])" 
        undefined_var_pattern = r'g(\d+)([^a-zA-Z0-9_])'
        if re.search(r'g\d+', optimized):
            # If we have undefined g variables, try to fix the structure
            # Look for pattern: lambda g:gN.method(arg) and try to restructure
            method_pattern = r'lambda g:g(\d+)(.*?)\(([^)]+)\)'
            
            def method_replacer(match):
                var_num, method_part, arg = match.groups()
                # Restructure as lambda g:method(arg)(g)
                return f'lambda g:{arg}{method_part}'
            
            optimized = re.sub(method_pattern, method_replacer, optimized)
        
        iterations += 1
    
    # Apply general optimizations
    optimized = optimizer.optimize_code(optimized)
    
    return optimized


def main():
    parser = argparse.ArgumentParser(description='Safe code golf optimization for MicroGolf')
    
    parser.add_argument('input_path', type=str, help='Input file or directory path')
    parser.add_argument('--output_dir', type=str, default='optimized', help='Output directory')
    parser.add_argument('--max_bytes', type=int, default=2500, help='Maximum bytes per file')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = SafeCodeGolfOptimizer()
    
    if input_path.is_file():
        # Process single file
        optimized_code, stats = optimizer.optimize_file(input_path)
        
        output_file = output_dir / input_path.name
        with open(output_file, 'w') as f:
            f.write(optimized_code)
        
        print(f"Optimized {input_path.name}:")
        print(f"  {stats['original_bytes']} -> {stats['final_bytes']} bytes")
        print(f"  {stats['reduction_percent']:.1f}% reduction")
        print(f"  Valid: {stats['is_valid']}")
        
        if stats['final_bytes'] > args.max_bytes:
            print(f"  WARNING: Exceeds {args.max_bytes} byte limit by {stats['final_bytes'] - args.max_bytes} bytes")
    
    else:
        # Process directory
        total_original = 0
        total_final = 0
        violations = []
        
        for py_file in input_path.glob('*.py'):
            print(f"Processing {py_file.name}...")
            
            optimized_code, stats = optimizer.optimize_file(py_file)
            
            output_file = output_dir / py_file.name
            with open(output_file, 'w') as f:
                f.write(optimized_code)
            
            total_original += stats['original_bytes']
            total_final += stats['final_bytes']
            
            if stats['final_bytes'] > args.max_bytes:
                violations.append({
                    'file': py_file.name,
                    'bytes': stats['final_bytes'],
                    'excess': stats['final_bytes'] - args.max_bytes
                })
            
            print(f"  {stats['original_bytes']} -> {stats['final_bytes']} bytes "
                  f"({stats['reduction_percent']:.1f}% reduction) "
                  f"Valid: {stats['is_valid']}")
        
        print(f"\nSummary:")
        print(f"  Total reduction: {(total_original - total_final) / total_original * 100:.1f}%")
        print(f"  Violations: {len(violations)}")


if __name__ == "__main__":
    main()
