"""
Stub Meta-Composer module when PyTorch is not available
"""
"""
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge
License: Apache 2.0
"""

import numpy as np
from typing import List, Dict, Any


class MetaComposer:
    """Stub meta-composer when PyTorch not available"""
    
    def __init__(self, primitive_vocab: Dict[str, int], model_config=None):
        self.primitive_vocab = primitive_vocab
        self.reverse_vocab = {v: k for k, v in primitive_vocab.items()}
        
    def predict_sequence(self, examples: List[Dict]) -> List[str]:
        """Fallback prediction using simple heuristics"""
        if not examples:
            return ['mc']  # Default to color mapping
        
        example = examples[0]
        inp, out = example.get('input', []), example.get('output', [])
        
        # Simple heuristic based on grid properties
        if not inp or not out:
            return ['mc']
        
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        
        sequence = []
        
        # Check for rotation
        if ih == ow and iw == oh:
            sequence.append('r90')
        
        # Check for flipping
        if ih == oh and iw == ow:
            # Same size - might be flip or color change
            if inp != out:
                sequence.append('fh')  # Assume horizontal flip
        
        # Always add color mapping as fallback
        if not sequence:
            sequence.append('mc')
        
        return sequence[:3]  # Limit to 3 operations


class MicroTransformer:
    """Stub transformer when PyTorch not available"""
    
    def __init__(self, *args, **kwargs):
        raise ImportError("MicroTransformer requires PyTorch. Please install torch>=2.0.1")


class ARCDataset:
    """Stub dataset when PyTorch not available"""
    
    def __init__(self, *args, **kwargs):
        raise ImportError("ARCDataset requires PyTorch. Please install torch>=2.0.1")


def create_primitive_vocab() -> Dict[str, int]:
    """Create vocabulary mapping for primitives"""
    primitives = [
        'r90', 'fh', 'fv', 'tr', 'sh',  # geometry
        'mc', 'tm', 'rc', 'bc', 'md',   # color ops
        'ff', 'bb', 'ct', 'cc',         # shape ops
        'inc', 'cl', 'he', 'sm', 'avg', # numeric
    ]
    
    return {prim: i for i, prim in enumerate(primitives)}
