#!/usr/bin/env python3
"""
Pattern-Based Sequence Optimization for MicroGolf
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge

This script identifies common patterns in successful primitive sequences and
creates optimized templates for better code generation.
"""

import json
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SequencePatternAnalyzer:
    """Analyzes patterns in successful primitive sequences to create optimized templates."""
    
    def __init__(self):
        self.load_evaluation_results()
        self.pattern_templates = {}
        self.common_subsequences = {}
        
    def load_evaluation_results(self):
        """Load evaluation results for pattern analysis."""
        try:
            with open('/home/o1/Documents/kaggle/NeurIPS 2025/experiments/evaluation_results.json', 'r') as f:
                self.eval_results = json.load(f)
                print(f"Loaded {len(self.eval_results['successful_sequences'])} successful sequences for pattern analysis")
        except FileNotFoundError:
            print("ERROR: No evaluation results found")
            self.eval_results = {'successful_sequences': []}
    
    def find_common_subsequences(self, min_length: int = 2, min_frequency: int = 3) -> Dict[Tuple[str, ...], int]:
        """Find commonly occurring subsequences in successful solutions."""
        subsequence_counts = defaultdict(int)
        
        for sequence in self.eval_results['successful_sequences']:
            # Generate all subsequences of specified minimum length
            for length in range(min_length, min(len(sequence) + 1, 6)):  # Cap at length 5
                for start in range(len(sequence) - length + 1):
                    subseq = tuple(sequence[start:start + length])
                    subsequence_counts[subseq] += 1
        
        # Filter by minimum frequency
        common_subsequences = {
            subseq: count for subseq, count in subsequence_counts.items()
            if count >= min_frequency
        }
        
        return common_subsequences
    
    def analyze_sequence_patterns(self) -> Dict[str, List[Tuple[str, ...]]]:
        """Analyze sequences by functional patterns."""
        patterns = {
            'rotation_chains': [],
            'color_transformations': [],
            'shape_analysis': [],
            'geometric_operations': [],
            'mixed_operations': []
        }
        
        # Define primitive categories
        rotation_ops = {'r90', 'r180', 'r270', 'fh', 'fv', 'tr'}
        color_ops = {'mc', 'tm', 'rc', 'bc', 'md', 'avg', 'inc'}
        shape_ops = {'ff', 'bb', 'ct', 'cc', 'sv', 'ext'}
        numeric_ops = {'cl', 'he', 'sm'}
        
        for sequence in self.eval_results['successful_sequences']:
            seq_tuple = tuple(sequence)
            
            # Categorize sequences based on dominant operation types
            rotation_count = sum(1 for op in sequence if op in rotation_ops)
            color_count = sum(1 for op in sequence if op in color_ops)
            shape_count = sum(1 for op in sequence if op in shape_ops)
            numeric_count = sum(1 for op in sequence if op in numeric_ops)
            
            total_ops = len(sequence)
            if total_ops == 0:
                continue
                
            # Classify based on dominant pattern (>50% of operations)
            if rotation_count / total_ops > 0.5:
                patterns['rotation_chains'].append(seq_tuple)
            elif color_count / total_ops > 0.5:
                patterns['color_transformations'].append(seq_tuple)
            elif shape_count / total_ops > 0.5:
                patterns['shape_analysis'].append(seq_tuple)
            elif numeric_count / total_ops > 0.5:
                patterns['geometric_operations'].append(seq_tuple)
            else:
                patterns['mixed_operations'].append(seq_tuple)
        
        return patterns
    
    def create_pattern_templates(self) -> Dict[str, Dict]:
        """Create optimized templates for common patterns."""
        common_subseqs = self.find_common_subsequences()
        sequence_patterns = self.analyze_sequence_patterns()
        
        templates = {}
        
        # Template 1: Most common subsequences
        templates['frequent_subsequences'] = {
            'patterns': [],
            'optimization_potential': 0
        }
        
        for subseq, frequency in sorted(common_subseqs.items(), key=lambda x: x[1], reverse=True)[:10]:
            templates['frequent_subsequences']['patterns'].append({
                'sequence': list(subseq),
                'frequency': frequency,
                'length': len(subseq),
                'estimated_savings': len(subseq) * 2  # Assuming 2 bytes saved per primitive in pattern
            })
        
        # Template 2: Category-specific optimizations
        for category, sequences in sequence_patterns.items():
            if sequences:
                # Find most common sequence in each category
                seq_counts = Counter(sequences)
                most_common = seq_counts.most_common(3)
                
                templates[category] = {
                    'common_patterns': [
                        {
                            'sequence': list(seq),
                            'frequency': count,
                            'optimization_priority': count * len(seq)
                        }
                        for seq, count in most_common
                    ]
                }
        
        return templates
    
    def suggest_sequence_optimizations(self, input_sequence: List[str]) -> Dict[str, any]:
        """Suggest optimizations for a given sequence based on discovered patterns."""
        suggestions = {
            'original_sequence': input_sequence,
            'pattern_matches': [],
            'recommended_optimizations': [],
            'estimated_savings': 0
        }
        
        # Check for pattern matches
        common_subseqs = self.find_common_subsequences()
        input_tuple = tuple(input_sequence)
        
        # Find matching subsequences
        for length in range(2, len(input_sequence) + 1):
            for start in range(len(input_sequence) - length + 1):
                subseq = tuple(input_sequence[start:start + length])
                if subseq in common_subseqs:
                    suggestions['pattern_matches'].append({
                        'subsequence': list(subseq),
                        'position': start,
                        'frequency': common_subseqs[subseq],
                        'length': length
                    })
        
        # Generate optimization recommendations
        if suggestions['pattern_matches']:
            # Sort by frequency and length
            best_matches = sorted(
                suggestions['pattern_matches'],
                key=lambda x: x['frequency'] * x['length'],
                reverse=True
            )
            
            for match in best_matches[:3]:  # Top 3 recommendations
                suggestions['recommended_optimizations'].append({
                    'type': 'pattern_optimization',
                    'target_subsequence': match['subsequence'],
                    'reason': f"Common pattern (appears {match['frequency']} times in successful solutions)",
                    'estimated_bytes_saved': match['length'] * 2
                })
                suggestions['estimated_savings'] += match['length'] * 2
        
        # Check for category-specific optimizations
        templates = self.create_pattern_templates()
        for category, template_data in templates.items():
            if category == 'frequent_subsequences':
                continue
                
            if 'common_patterns' in template_data:
                for pattern in template_data['common_patterns']:
                    pattern_seq = pattern['sequence']
                    # Check if input sequence matches or contains this pattern
                    if self.sequence_similarity(input_sequence, pattern_seq) > 0.6:
                        suggestions['recommended_optimizations'].append({
                            'type': 'category_optimization',
                            'category': category,
                            'suggested_pattern': pattern_seq,
                            'reason': f"High similarity to successful {category} pattern",
                            'estimated_bytes_saved': max(0, len(input_sequence) - len(pattern_seq)) * 3
                        })
        
        return suggestions
    
    def sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two sequences using Jaccard similarity."""
        set1, set2 = set(seq1), set(seq2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def generate_optimization_report(self) -> Dict[str, any]:
        """Generate comprehensive optimization report based on pattern analysis."""
        common_subseqs = self.find_common_subsequences()
        sequence_patterns = self.analyze_sequence_patterns()
        templates = self.create_pattern_templates()
        
        report = {
            'analysis_summary': {
                'total_sequences_analyzed': len(self.eval_results['successful_sequences']),
                'common_subsequences_found': len(common_subseqs),
                'pattern_categories': len(sequence_patterns),
                'optimization_templates': len(templates)
            },
            'top_common_subsequences': [
                {
                    'sequence': list(subseq),
                    'frequency': frequency,
                    'optimization_value': frequency * len(subseq)
                }
                for subseq, frequency in sorted(common_subseqs.items(), key=lambda x: x[1] * len(x[0]), reverse=True)[:10]
            ],
            'pattern_distribution': {
                category: len(sequences) for category, sequences in sequence_patterns.items()
            },
            'optimization_recommendations': {
                'high_priority': [],
                'medium_priority': [],
                'low_priority': []
            }
        }
        
        # Prioritize optimizations
        for subseq_info in report['top_common_subsequences']:
            priority_score = subseq_info['optimization_value']
            recommendation = {
                'type': 'subsequence_template',
                'pattern': subseq_info['sequence'],
                'frequency': subseq_info['frequency'],
                'estimated_impact': f"{priority_score} optimization units"
            }
            
            if priority_score > 20:
                report['optimization_recommendations']['high_priority'].append(recommendation)
            elif priority_score > 10:
                report['optimization_recommendations']['medium_priority'].append(recommendation)
            else:
                report['optimization_recommendations']['low_priority'].append(recommendation)
        
        return report

def main():
    """Run pattern-based sequence optimization analysis."""
    print("MicroGolf Pattern-Based Sequence Optimizer")
    print("=" * 60)
    
    analyzer = SequencePatternAnalyzer()
    
    # Generate optimization report
    report = analyzer.generate_optimization_report()
    
    print(f"\nAnalysis Summary:")
    print(f"- Sequences analyzed: {report['analysis_summary']['total_sequences_analyzed']}")
    print(f"- Common subsequences found: {report['analysis_summary']['common_subsequences_found']}")
    print(f"- Pattern categories: {report['analysis_summary']['pattern_categories']}")
    
    print(f"\nTop Common Subsequences:")
    for i, subseq_info in enumerate(report['top_common_subsequences'][:5], 1):
        print(f"{i}. {subseq_info['sequence']} (frequency: {subseq_info['frequency']}, value: {subseq_info['optimization_value']})")
    
    print(f"\nPattern Distribution:")
    for category, count in report['pattern_distribution'].items():
        print(f"- {category}: {count} sequences")
    
    print(f"\nOptimization Recommendations:")
    for priority, recommendations in report['optimization_recommendations'].items():
        if recommendations:
            print(f"\n{priority.upper()} PRIORITY ({len(recommendations)} items):")
            for rec in recommendations[:3]:  # Show top 3 per priority
                print(f"  • {rec['pattern']} - {rec['estimated_impact']}")
    
    # Test optimization suggestions on sample sequences
    print(f"\n{'='*60}")
    print("TESTING OPTIMIZATION SUGGESTIONS")
    print(f"{'='*60}")
    
    test_sequences = [
        ['sv', 'cc', 'ext', 'tm'],
        ['r90', 'fh', 'avg'],
        ['inc', 'tm', 'rc', 'avg'],
        ['ext', 'sv', 'cc', 'tm', 'avg']
    ]
    
    for i, test_seq in enumerate(test_sequences, 1):
        print(f"\nTest Sequence {i}: {test_seq}")
        suggestions = analyzer.suggest_sequence_optimizations(test_seq)
        
        if suggestions['pattern_matches']:
            print(f"  Pattern matches found: {len(suggestions['pattern_matches'])}")
            for match in suggestions['pattern_matches'][:2]:
                print(f"    • {match['subsequence']} (frequency: {match['frequency']})")
        
        if suggestions['recommended_optimizations']:
            print(f"  Recommendations: {len(suggestions['recommended_optimizations'])}")
            for rec in suggestions['recommended_optimizations'][:2]:
                print(f"    • {rec['type']}: {rec['reason']}")
        
        print(f"  Estimated savings: {suggestions['estimated_savings']} bytes")
    
    # Save results
    with open('/home/o1/Documents/kaggle/NeurIPS 2025/experiments/pattern_optimization_results.json', 'w') as f:
        json.dump(report, f, indent=2)
        print(f"\nPattern analysis results saved to experiments/pattern_optimization_results.json")
    
    return report

if __name__ == "__main__":
    main()
