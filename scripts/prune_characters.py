#!/usr/bin/env python3
"""
Advanced Code Golf Pruning Script for MicroGolf
Applies aggressive AST-based and regex optimizations to minimize byte count
"""
"""
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge
License: Apache 2.0
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import astor  # For AST to code conversion

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class ASTOptimizer(ast.NodeTransformer):
    """AST-based code optimizer for aggressive byte reduction"""
    
    def __init__(self):
        self.optimizations_applied = []
    
    def visit_For(self, node):
        """Convert simple for loops to list comprehensions"""
        self.generic_visit(node)
        
        # Check if this is a simple accumulation loop
        if (len(node.body) == 1 and 
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Call) and
            hasattr(node.body[0].value.func, 'attr') and
            node.body[0].value.func.attr == 'append'):
            
            # Convert to list comprehension
            self.optimizations_applied.append("for_to_comprehension")
            # This is a simplified example - real implementation would be more complex
        
        return node
    
    def visit_If(self, node):
        """Optimize conditional expressions"""
        self.generic_visit(node)
        
        # Convert if-else to ternary operator when possible
        if (len(node.body) == 1 and 
            len(node.orelse) == 1 and
            isinstance(node.body[0], ast.Return) and
            isinstance(node.orelse[0], ast.Return)):
            
            # Create ternary expression
            ternary = ast.IfExp(
                test=node.test,
                body=node.body[0].value,
                orelse=node.orelse[0].value
            )
            
            self.optimizations_applied.append("if_to_ternary")
            return ast.Return(value=ternary)
        
        return node
    
    def visit_FunctionDef(self, node):
        """Optimize function definitions"""
        self.generic_visit(node)
        
        # Convert single-expression functions to lambdas (when possible)
        if (len(node.body) == 1 and 
            isinstance(node.body[0], ast.Return)):
            
            self.optimizations_applied.append("func_to_lambda")
            # Note: This would need special handling in the calling code
        
        return node
    
    def visit_BinOp(self, node):
        """Optimize binary operations"""
        self.generic_visit(node)
        
        # Optimize mathematical operations
        if isinstance(node.op, ast.Mult):
            # Check for multiplication by 1
            if (isinstance(node.right, ast.Constant) and node.right.value == 1):
                self.optimizations_applied.append("mult_by_one")
                return node.left
            if (isinstance(node.left, ast.Constant) and node.left.value == 1):
                self.optimizations_applied.append("mult_by_one")
                return node.right
        
        if isinstance(node.op, ast.Add):
            # Check for addition of 0
            if (isinstance(node.right, ast.Constant) and node.right.value == 0):
                self.optimizations_applied.append("add_zero")
                return node.left
            if (isinstance(node.left, ast.Constant) and node.left.value == 0):
                self.optimizations_applied.append("add_zero")
                return node.right
        
        return node


class RegexOptimizer:
    """Regex-based code optimizations for final polishing"""
    
    def __init__(self):
        self.optimizations = [
            # Remove unnecessary whitespace
            (r'\s*,\s*', ','),
            (r'\s*:\s*', ':'),
            (r'\s*\(\s*', '('),
            (r'\s*\)\s*', ')'),
            (r'\s*\[\s*', '['),
            (r'\s*\]\s*', ']'),
            (r'\s*\{\s*', '{'),
            (r'\s*\}\s*', '}'),
            
            # Compress operators
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
            
            # Compress keywords
            (r'\s+for\s+', 'for '),
            (r'\s+in\s+', 'in '),
            (r'\s+if\s+', 'if '),
            (r'\s+else\s+', 'else '),
            (r'\s+and\s+', 'and '),
            (r'\s+or\s+', 'or '),
            (r'\s+not\s+', 'not '),
            
            # Remove redundant operations
            (r'\*1(?![0-9])', ''),
            (r'\+0(?![0-9])', ''),
            (r'//1(?![0-9])', ''),
            
            # Compress common patterns
            (r'range\(len\(([^)]+)\)\)', r'range(len(\1))'),
            (r'list\(range\((\d+)\)\)', r'[*range(\1)]'),
            
            # Boolean optimizations
            (r'== True', ''),
            (r'== False', '==0'),
            (r'!= True', '==0'),
            (r'!= False', ''),
            
            # String optimizations
            (r"'([^']*)'", r'"\1"'),  # Prefer double quotes if shorter
        ]
    
    def optimize(self, code: str) -> str:
        """Apply all regex optimizations"""
        optimized = code
        
        for pattern, replacement in self.optimizations:
            optimized = re.sub(pattern, replacement, optimized)
        
        return optimized


class UltraCompressor:
    """Ultra-aggressive compression techniques for competition"""
    
    def __init__(self):
        self.variable_map = {}
        self.next_var_id = 0
    
    def compress_variable_names(self, code: str) -> str:
        """Replace variable names with single characters"""
        
        # Find all variable names
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        variables = set(re.findall(var_pattern, code))
        
        # Remove Python keywords and builtins
        keywords = {
            'def', 'for', 'in', 'if', 'else', 'and', 'or', 'not', 'is', 'lambda',
            'return', 'yield', 'import', 'from', 'as', 'try', 'except', 'finally',
            'with', 'while', 'break', 'continue', 'pass', 'class', 'global', 'nonlocal',
            'len', 'sum', 'max', 'min', 'abs', 'int', 'float', 'str', 'list', 'dict',
            'set', 'tuple', 'range', 'enumerate', 'zip', 'map', 'filter', 'any', 'all'
        }
        
        variables = variables - keywords
        
        # Create single-character mappings
        chars = 'abcdefghijklmnopqrstuvwxyz'
        compressed_code = code
        
        for i, var in enumerate(variables):
            if i < len(chars) and len(var) > 1:
                new_name = chars[i]
                # Use word boundaries to avoid partial replacements
                compressed_code = re.sub(r'\b' + re.escape(var) + r'\b', new_name, compressed_code)
        
        return compressed_code
    
    def inline_single_use_variables(self, code: str) -> str:
        """Inline variables that are used only once"""
        
        # This is a simplified version - real implementation would use AST
        lines = code.split(';') if ';' in code else code.split('\n')
        
        # Find variable assignments
        assignments = {}
        usage_count = {}
        
        for line in lines:
            if '=' in line and not ('==' in line or '!=' in line or '>=' in line or '<=' in line):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    value = parts[1].strip()
                    assignments[var_name] = value
                    usage_count[var_name] = 0
        
        # Count usage
        for line in lines:
            for var_name in assignments:
                if var_name in line and not line.strip().startswith(var_name + '='):
                    usage_count[var_name] += line.count(var_name)
        
        # Inline single-use variables
        inlined_code = code
        for var_name, count in usage_count.items():
            if count == 1 and var_name in assignments:
                # Replace the variable with its value
                value = assignments[var_name]
                inlined_code = inlined_code.replace(var_name, f'({value})')
                # Remove the assignment
                assignment_pattern = rf'{re.escape(var_name)}\s*=\s*{re.escape(value)}\s*[;\n]?'
                inlined_code = re.sub(assignment_pattern, '', inlined_code)
        
        return inlined_code
    
    def compress_literals(self, code: str) -> str:
        """Compress literal values where possible"""
        
        # Replace True/False with 1/0 where appropriate
        code = re.sub(r'\bTrue\b', '1', code)
        code = re.sub(r'\bFalse\b', '0', code)
        
        # Compress common numeric literals
        code = re.sub(r'\b0\.0\b', '0', code)
        code = re.sub(r'\b1\.0\b', '1', code)
        
        # Compress empty collections
        code = re.sub(r'\[\s*\]', '[]', code)
        code = re.sub(r'\{\s*\}', '{}', code)
        code = re.sub(r'\(\s*\)', '()', code)
        
        return code


class CodeGolfPruner:
    """Main class for aggressive code optimization"""
    
    def __init__(self):
        self.ast_optimizer = ASTOptimizer()
        self.regex_optimizer = RegexOptimizer()
        self.ultra_compressor = UltraCompressor()
        
        self.original_bytes = 0
        self.final_bytes = 0
        self.optimizations_log = []
    
    def prune_file(self, file_path: Path) -> Tuple[str, Dict[str, int]]:
        """Prune a single Python file"""
        
        with open(file_path, 'r') as f:
            original_code = f.read()
        
        self.original_bytes = len(original_code.encode('utf-8'))
        
        try:
            # Phase 1: AST-based optimizations
            tree = ast.parse(original_code)
            optimized_tree = self.ast_optimizer.visit(tree)
            
            # Convert back to code
            ast_optimized = astor.to_source(optimized_tree)
            
            # Phase 2: Regex optimizations
            regex_optimized = self.regex_optimizer.optimize(ast_optimized)
            
            # Phase 3: Ultra-aggressive compression
            variable_compressed = self.ultra_compressor.compress_variable_names(regex_optimized)
            inlined = self.ultra_compressor.inline_single_use_variables(variable_compressed)
            final_code = self.ultra_compressor.compress_literals(inlined)
            
            # Final cleanup
            final_code = self._final_cleanup(final_code)
            
            self.final_bytes = len(final_code.encode('utf-8'))
            
            return final_code, {
                'original_bytes': self.original_bytes,
                'final_bytes': self.final_bytes,
                'reduction': self.original_bytes - self.final_bytes,
                'reduction_percent': (self.original_bytes - self.final_bytes) / self.original_bytes * 100
            }
            
        except Exception as e:
            print(f"Error optimizing {file_path}: {e}")
            return original_code, {
                'original_bytes': self.original_bytes,
                'final_bytes': self.original_bytes,
                'reduction': 0,
                'reduction_percent': 0,
                'error': str(e)
            }
    
    def _final_cleanup(self, code: str) -> str:
        """Final cleanup pass"""
        
        # Remove unnecessary newlines
        code = re.sub(r'\n\s*\n', '\n', code)
        
        # Remove trailing whitespace
        code = re.sub(r'\s+$', '', code, flags=re.MULTILINE)
        
        # Remove leading/trailing newlines
        code = code.strip()
        
        # Ensure single newline at end if needed
        if not code.endswith('\n') and code:
            code += '\n'
        
        return code
    
    def validate_code(self, original_code: str, optimized_code: str) -> bool:
        """Validate that optimized code is syntactically correct"""
        
        try:
            # Check syntax
            ast.parse(optimized_code)
            
            # Basic functionality test (if possible)
            # This would need more sophisticated testing in practice
            
            return True
        except SyntaxError:
            return False
        except Exception:
            return False


def process_directory(input_dir: Path, output_dir: Path, max_bytes: int = 2500) -> Dict[str, any]:
    """Process all Python files in directory"""
    
    pruner = CodeGolfPruner()
    results = {}
    total_original = 0
    total_final = 0
    violations = []
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all Python files
    for py_file in input_dir.glob('*.py'):
        print(f"Processing {py_file.name}...")
        
        optimized_code, stats = pruner.prune_file(py_file)
        
        # Validate
        with open(py_file, 'r') as f:
            original_code = f.read()
        
        is_valid = pruner.validate_code(original_code, optimized_code)
        
        if is_valid:
            # Save optimized version
            output_file = output_dir / py_file.name
            with open(output_file, 'w') as f:
                f.write(optimized_code)
        else:
            print(f"Warning: Optimization of {py_file.name} failed validation")
            optimized_code = original_code
            output_file = output_dir / py_file.name
            with open(output_file, 'w') as f:
                f.write(original_code)
        
        # Check byte limit
        if stats['final_bytes'] > max_bytes:
            violations.append({
                'file': py_file.name,
                'bytes': stats['final_bytes'],
                'limit': max_bytes,
                'excess': stats['final_bytes'] - max_bytes
            })
        
        results[py_file.name] = stats
        total_original += stats['original_bytes']
        total_final += stats['final_bytes']
        
        print(f"  {stats['original_bytes']} -> {stats['final_bytes']} bytes "
              f"({stats['reduction_percent']:.1f}% reduction)")
    
    # Summary
    summary = {
        'total_files': len(results),
        'total_original_bytes': total_original,
        'total_final_bytes': total_final,
        'total_reduction': total_original - total_final,
        'total_reduction_percent': (total_original - total_final) / total_original * 100 if total_original > 0 else 0,
        'violations': violations,
        'files': results
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Prune Python files for code golf')
    
    parser.add_argument('input_path', type=str,
                       help='Input file or directory path')
    parser.add_argument('--output_dir', type=str, default='submission',
                       help='Output directory for optimized files')
    parser.add_argument('--max_bytes', type=int, default=2500,
                       help='Maximum bytes per file (default: 2500)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate optimized code')
    parser.add_argument('--report', type=str,
                       help='Path to save optimization report (JSON)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    
    if input_path.is_file():
        # Process single file
        pruner = CodeGolfPruner()
        optimized_code, stats = pruner.prune_file(input_path)
        
        # Save result
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / input_path.name
        
        with open(output_file, 'w') as f:
            f.write(optimized_code)
        
        print(f"Optimized {input_path.name}:")
        print(f"  {stats['original_bytes']} -> {stats['final_bytes']} bytes")
        print(f"  {stats['reduction_percent']:.1f}% reduction")
        
        if stats['final_bytes'] > args.max_bytes:
            print(f"  WARNING: Exceeds {args.max_bytes} byte limit by {stats['final_bytes'] - args.max_bytes} bytes")
    
    else:
        # Process directory
        summary = process_directory(input_path, output_dir, args.max_bytes)
        
        print(f"\nOptimization Summary:")
        print(f"  Files processed: {summary['total_files']}")
        print(f"  Total size: {summary['total_original_bytes']} -> {summary['total_final_bytes']} bytes")
        print(f"  Total reduction: {summary['total_reduction_percent']:.1f}%")
        
        if summary['violations']:
            print(f"\nByte limit violations ({len(summary['violations'])}):")
            for violation in summary['violations']:
                print(f"  {violation['file']}: {violation['bytes']} bytes (+{violation['excess']})")
        
        # Save report
        if args.report:
            import json
            with open(args.report, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nReport saved to {args.report}")


if __name__ == "__main__":
    main()
