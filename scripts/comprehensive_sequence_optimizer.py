#!/usr/bin/env python3
"""
Comprehensive Sequence Optimization Framework for MicroGolf
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge

This script combines all optimization techniques: primitive efficiency, sequence ordering,
pattern matching, and code golf techniques to achieve maximum byte reduction.
"""

import json
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microgolf.engine.controller import PrimitiveController
from microgolf.engine.executor import OptimizedExecutor

class ComprehensiveSequenceOptimizer:
    """Master optimizer combining all optimization strategies."""
    
    def __init__(self):
        self.controller = PrimitiveController()
        self.executor = OptimizedExecutor()
        
        # Load optimization data
        self.load_optimization_data()
        
        # Advanced optimization parameters
        self.max_optimization_iterations = 3
        self.target_byte_limit = 2200  # Conservative limit
        self.optimization_strategies = [
            'primitive_replacement',
            'sequence_reordering', 
            'pattern_matching',
            'length_optimization',
            'code_golf_techniques'
        ]
    
    def load_optimization_data(self):
        """Load all optimization data from previous analyses."""
        # Load evaluation results
        try:
            with open('/home/o1/Documents/kaggle/NeurIPS 2025/experiments/evaluation_results.json', 'r') as f:
                self.eval_results = json.load(f)
        except FileNotFoundError:
            self.eval_results = {}
        
        # Load pattern analysis results
        try:
            with open('/home/o1/Documents/kaggle/NeurIPS 2025/experiments/pattern_optimization_results.json', 'r') as f:
                self.pattern_results = json.load(f)
        except FileNotFoundError:
            self.pattern_results = {}
        
        # Load sequence optimization results
        try:
            with open('/home/o1/Documents/kaggle/NeurIPS 2025/experiments/sequence_optimization_results.json', 'r') as f:
                self.sequence_results = json.load(f)
        except FileNotFoundError:
            self.sequence_results = {}
        
        # Primitive efficiency scores
        self.primitive_efficiency = self.calculate_primitive_efficiency()
        
        # Common patterns for optimization
        self.optimization_patterns = self.extract_optimization_patterns()
    
    def calculate_primitive_efficiency(self) -> Dict[str, float]:
        """Calculate comprehensive efficiency scores for all primitives."""
        if 'primitive_usage' not in self.eval_results:
            return {}
        
        usage_stats = self.eval_results['primitive_usage']
        total_usage = sum(usage_stats.values())
        
        # Base efficiency from frequency and estimated byte cost
        primitive_costs = {
            'r90': 40, 'r180': 42, 'r270': 42, 'fh': 35, 'fv': 35, 'tr': 38, 'sh': 40,
            'mc': 45, 'tm': 42, 'rc': 48, 'bc': 50, 'md': 45,
            'ff': 38, 'bb': 42, 'ct': 45, 'cc': 40, 'sv': 35, 'ext': 38,
            'cl': 40, 'he': 38, 'inc': 35, 'sm': 38, 'avg': 35
        }
        
        efficiency_scores = {}
        for primitive, count in usage_stats.items():
            frequency = count / total_usage if total_usage > 0 else 0
            cost = primitive_costs.get(primitive, 50)
            
            # Advanced efficiency calculation
            base_efficiency = frequency / (cost / 100)
            
            # Bonus for high-frequency primitives (proven useful)
            frequency_bonus = min(0.5, frequency * 2)
            
            # Penalty for high-cost primitives
            cost_penalty = max(0, (cost - 35) / 100)
            
            efficiency_scores[primitive] = base_efficiency + frequency_bonus - cost_penalty
        
        return efficiency_scores
    
    def extract_optimization_patterns(self) -> Dict[str, List[Dict]]:
        """Extract actionable optimization patterns from analysis results."""
        patterns = {
            'high_value_subsequences': [],
            'replacement_chains': [],
            'reduction_opportunities': []
        }
        
        if 'top_common_subsequences' in self.pattern_results:
            for subseq_info in self.pattern_results['top_common_subsequences']:
                if subseq_info['optimization_value'] > 10:
                    patterns['high_value_subsequences'].append({
                        'pattern': subseq_info['sequence'],
                        'frequency': subseq_info['frequency'],
                        'value': subseq_info['optimization_value'],
                        'estimated_savings': len(subseq_info['sequence']) * 3
                    })
        
        # Create replacement chains based on efficiency
        sorted_primitives = sorted(
            self.primitive_efficiency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        high_efficiency = [p for p, score in sorted_primitives[:10]]
        low_efficiency = [p for p, score in sorted_primitives[-10:]]
        
        for low_prim in low_efficiency:
            for high_prim in high_efficiency:
                if self.are_functionally_similar(low_prim, high_prim):
                    patterns['replacement_chains'].append({
                        'from': low_prim,
                        'to': high_prim,
                        'efficiency_gain': self.primitive_efficiency.get(high_prim, 0) - self.primitive_efficiency.get(low_prim, 0)
                    })
        
        return patterns
    
    def are_functionally_similar(self, prim1: str, prim2: str) -> bool:
        """Check if two primitives are functionally similar."""
        similarity_groups = [
            {'r90', 'r180', 'r270'},  # Rotations
            {'fh', 'fv'},             # Flips
            {'mc', 'tm', 'rc'},       # Color operations
            {'ff', 'bb', 'ext'},      # Fill operations
            {'cc', 'sv', 'ct'},       # Component analysis
            {'inc', 'avg', 'sm'}      # Numeric operations
        ]
        
        for group in similarity_groups:
            if prim1 in group and prim2 in group:
                return True
        return False
    
    def optimize_primitive_selection(self, sequence: List[str]) -> List[str]:
        """Optimize primitive selection based on efficiency scores."""
        optimized = []
        
        for primitive in sequence:
            best_replacement = primitive
            best_score = self.primitive_efficiency.get(primitive, 0)
            
            # Look for better alternatives
            for replacement_info in self.optimization_patterns['replacement_chains']:
                if replacement_info['from'] == primitive and replacement_info['efficiency_gain'] > 0.1:
                    replacement = replacement_info['to']
                    replacement_score = self.primitive_efficiency.get(replacement, 0)
                    
                    if replacement_score > best_score:
                        best_replacement = replacement
                        best_score = replacement_score
            
            optimized.append(best_replacement)
        
        return optimized
    
    def optimize_sequence_order(self, sequence: List[str]) -> List[str]:
        """Optimize sequence order for better lambda chaining and compression."""
        if len(sequence) <= 1:
            return sequence
        
        # Score different orderings
        best_order = sequence[:]
        best_score = self.score_sequence_order(sequence)
        
        # Try pattern-based reordering
        for pattern_info in self.optimization_patterns['high_value_subsequences']:
            pattern = pattern_info['pattern']
            if len(pattern) <= len(sequence):
                # Try to align sequence with high-value patterns
                reordered = self.align_with_pattern(sequence, pattern)
                score = self.score_sequence_order(reordered)
                
                if score > best_score:
                    best_order = reordered
                    best_score = score
        
        # Try efficiency-based ordering (high efficiency first)
        efficiency_ordered = sorted(
            sequence,
            key=lambda p: self.primitive_efficiency.get(p, 0),
            reverse=True
        )
        score = self.score_sequence_order(efficiency_ordered)
        
        if score > best_score:
            best_order = efficiency_ordered
            best_score = score
        
        return best_order
    
    def score_sequence_order(self, sequence: List[str]) -> float:
        """Score a sequence order for optimization potential."""
        if not sequence:
            return 0.0
        
        score = 0.0
        
        # Efficiency score (weighted by position - earlier is better)
        for i, primitive in enumerate(sequence):
            efficiency = self.primitive_efficiency.get(primitive, 0)
            position_weight = 1.0 - (i / len(sequence)) * 0.3  # Later positions get slightly lower weight
            score += efficiency * position_weight
        
        # Pattern matching bonus
        for pattern_info in self.optimization_patterns['high_value_subsequences']:
            pattern = pattern_info['pattern']
            if self.contains_subsequence(sequence, pattern):
                score += pattern_info['value'] * 0.1
        
        # Length penalty for very long sequences
        if len(sequence) > 8:
            score *= 0.9 ** (len(sequence) - 8)
        
        return score
    
    def contains_subsequence(self, sequence: List[str], subsequence: List[str]) -> bool:
        """Check if sequence contains subsequence."""
        if len(subsequence) > len(sequence):
            return False
        
        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i:i+len(subsequence)] == subsequence:
                return True
        return False
    
    def align_with_pattern(self, sequence: List[str], pattern: List[str]) -> List[str]:
        """Align sequence elements with a high-value pattern."""
        aligned = []
        sequence_set = set(sequence)
        pattern_set = set(pattern)
        
        # Add pattern elements that are in sequence first
        for prim in pattern:
            if prim in sequence_set:
                aligned.append(prim)
                sequence_set.remove(prim)
        
        # Add remaining elements
        aligned.extend(sequence_set)
        
        return aligned
    
    def apply_length_optimization(self, sequence: List[str]) -> List[str]:
        """Optimize sequence length based on byte budget."""
        # Estimate byte cost
        estimated_cost = self.estimate_sequence_byte_cost(sequence)
        
        if estimated_cost <= self.target_byte_limit:
            return sequence
        
        # Remove least efficient primitives until under budget
        scored_sequence = [
            (prim, self.primitive_efficiency.get(prim, 0))
            for prim in sequence
        ]
        
        # Sort by efficiency (keep most efficient)
        scored_sequence.sort(key=lambda x: x[1], reverse=True)
        
        optimized = []
        current_cost = 0
        
        for primitive, efficiency in scored_sequence:
            prim_cost = self.estimate_primitive_cost(primitive)
            if current_cost + prim_cost <= self.target_byte_limit:
                optimized.append(primitive)
                current_cost += prim_cost
            else:
                break
        
        return optimized
    
    def estimate_sequence_byte_cost(self, sequence: List[str]) -> int:
        """Estimate total byte cost of a sequence."""
        base_costs = {
            'r90': 40, 'r180': 42, 'r270': 42, 'fh': 35, 'fv': 35, 'tr': 38, 'sh': 40,
            'mc': 45, 'tm': 42, 'rc': 48, 'bc': 50, 'md': 45,
            'ff': 38, 'bb': 42, 'ct': 45, 'cc': 40, 'sv': 35, 'ext': 38,
            'cl': 40, 'he': 38, 'inc': 35, 'sm': 38, 'avg': 35
        }
        
        total_cost = sum(base_costs.get(prim, 50) for prim in sequence)
        
        # Add lambda chaining overhead
        if len(sequence) > 1:
            total_cost += (len(sequence) - 1) * 5
        
        # Add base function overhead
        total_cost += 50
        
        return total_cost
    
    def estimate_primitive_cost(self, primitive: str) -> int:
        """Estimate byte cost of a single primitive."""
        base_costs = {
            'r90': 40, 'r180': 42, 'r270': 42, 'fh': 35, 'fv': 35, 'tr': 38, 'sh': 40,
            'mc': 45, 'tm': 42, 'rc': 48, 'bc': 50, 'md': 45,
            'ff': 38, 'bb': 42, 'ct': 45, 'cc': 40, 'sv': 35, 'ext': 38,
            'cl': 40, 'he': 38, 'inc': 35, 'sm': 38, 'avg': 35
        }
        return base_costs.get(primitive, 50)
    
    def comprehensive_optimize(self, sequence: List[str], max_iterations: int = None) -> Tuple[List[str], Dict[str, any]]:
        """Apply comprehensive optimization with multiple strategies."""
        if max_iterations is None:
            max_iterations = self.max_optimization_iterations
        
        current_sequence = sequence[:]
        optimization_history = []
        
        for iteration in range(max_iterations):
            iteration_info = {
                'iteration': iteration + 1,
                'input_sequence': current_sequence[:],
                'optimizations_applied': []
            }
            
            # Strategy 1: Primitive replacement
            optimized_primitives = self.optimize_primitive_selection(current_sequence)
            if optimized_primitives != current_sequence:
                iteration_info['optimizations_applied'].append('primitive_replacement')
                current_sequence = optimized_primitives
            
            # Strategy 2: Sequence reordering
            optimized_order = self.optimize_sequence_order(current_sequence)
            if optimized_order != current_sequence:
                iteration_info['optimizations_applied'].append('sequence_reordering')
                current_sequence = optimized_order
            
            # Strategy 3: Length optimization
            optimized_length = self.apply_length_optimization(current_sequence)
            if optimized_length != current_sequence:
                iteration_info['optimizations_applied'].append('length_optimization')
                current_sequence = optimized_length
            
            iteration_info['output_sequence'] = current_sequence[:]
            iteration_info['estimated_cost'] = self.estimate_sequence_byte_cost(current_sequence)
            
            optimization_history.append(iteration_info)
            
            # Break if no changes in this iteration
            if not iteration_info['optimizations_applied']:
                break
        
        # Final results
        original_cost = self.estimate_sequence_byte_cost(sequence)
        final_cost = self.estimate_sequence_byte_cost(current_sequence)
        
        results = {
            'original_sequence': sequence,
            'optimized_sequence': current_sequence,
            'original_cost': original_cost,
            'final_cost': final_cost,
            'bytes_saved': original_cost - final_cost,
            'optimization_ratio': final_cost / original_cost if original_cost > 0 else 1.0,
            'iterations': len(optimization_history),
            'optimization_history': optimization_history,
            'under_budget': final_cost <= self.target_byte_limit
        }
        
        return current_sequence, results

def main():
    """Run comprehensive sequence optimization."""
    print("MicroGolf Comprehensive Sequence Optimizer")
    print("=" * 70)
    
    optimizer = ComprehensiveSequenceOptimizer()
    
    # Test on various sequence types
    test_sequences = [
        ['r90', 'fh', 'cc', 'tm', 'avg'],          # Mixed operations
        ['sv', 'cc', 'ext', 'tm', 'avg', 'r90'],   # Successful pattern from evaluation
        ['inc', 'rc', 'bb', 'r270', 'fv', 'sm'],   # Suboptimal sequence
        ['cc', 'sv', 'ext', 'tm', 'avg', 'r90', 'sm', 'tr', 'inc'],  # Long sequence
        ['r180', 'fv', 'md', 'bc', 'he'],          # Lower efficiency primitives
    ]
    
    all_results = []
    
    for i, test_sequence in enumerate(test_sequences, 1):
        print(f"\n--- Test Case {i}: {test_sequence} ---")
        
        optimized_seq, results = optimizer.comprehensive_optimize(test_sequence)
        all_results.append(results)
        
        print(f"Original: {results['original_sequence']} ({results['original_cost']} bytes)")
        print(f"Optimized: {results['optimized_sequence']} ({results['final_cost']} bytes)")
        print(f"Savings: {results['bytes_saved']} bytes ({(1-results['optimization_ratio']):.1%} reduction)")
        print(f"Under budget: {results['under_budget']} (target: {optimizer.target_byte_limit} bytes)")
        print(f"Iterations: {results['iterations']}")
        
        if results['optimization_history']:
            print("Optimizations applied:")
            for iter_info in results['optimization_history']:
                if iter_info['optimizations_applied']:
                    print(f"  Iteration {iter_info['iteration']}: {', '.join(iter_info['optimizations_applied'])}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("COMPREHENSIVE OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    
    total_original = sum(r['original_cost'] for r in all_results)
    total_optimized = sum(r['final_cost'] for r in all_results)
    total_saved = total_original - total_optimized
    under_budget_count = sum(1 for r in all_results if r['under_budget'])
    
    print(f"Test sequences: {len(all_results)}")
    print(f"Total original cost: {total_original} bytes")
    print(f"Total optimized cost: {total_optimized} bytes")
    print(f"Total bytes saved: {total_saved} bytes ({total_saved/total_original:.1%} reduction)")
    print(f"Sequences under budget: {under_budget_count}/{len(all_results)} ({under_budget_count/len(all_results):.1%})")
    
    avg_iterations = np.mean([r['iterations'] for r in all_results])
    print(f"Average optimization iterations: {avg_iterations:.1f}")
    
    # Save comprehensive results
    comprehensive_results = {
        'optimization_summary': {
            'total_sequences': len(all_results),
            'total_bytes_saved': total_saved,
            'average_reduction': total_saved / total_original,
            'under_budget_rate': under_budget_count / len(all_results),
            'average_iterations': avg_iterations
        },
        'test_results': all_results,
        'optimizer_config': {
            'target_byte_limit': optimizer.target_byte_limit,
            'max_iterations': optimizer.max_optimization_iterations,
            'strategies': optimizer.optimization_strategies
        }
    }
    
    with open('/home/o1/Documents/kaggle/NeurIPS 2025/experiments/comprehensive_optimization_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
        print(f"\nComprehensive optimization results saved to experiments/comprehensive_optimization_results.json")
    
    return comprehensive_results

if __name__ == "__main__":
    main()
