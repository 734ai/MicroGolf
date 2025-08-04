"""Ultra-compact primitives for ARC-AGI tasks (<= 20 bytes each)"""
"""
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge
License: Apache 2.0
"""

from .geometry import r90, fh, fv, tr, sh
from .color_ops import mc, tm, rc, bc, md  
from .shape_ops import ff, bb, ct, cc
from .numeric import inc, cl, he, sm, avg

# All primitive functions with their byte counts
PRIMITIVES = {
    # Geometry - 89 bytes total
    'r90': (r90, 19),  # rotate 90 degrees
    'fh': (fh, 18),    # flip horizontal  
    'fv': (fv, 20),    # flip vertical
    'tr': (tr, 17),    # transpose
    'sh': (sh, 15),    # shift
    
    # Color operations - 90 bytes total
    'mc': (mc, 20),    # map colors
    'tm': (tm, 18),    # threshold mask
    'rc': (rc, 17),    # replace color
    'bc': (bc, 19),    # blend colors
    'md': (md, 16),    # max difference
    
    # Shape operations - 74 bytes total
    'ff': (ff, 20),    # flood fill
    'bb': (bb, 18),    # bounding box
    'ct': (ct, 17),    # centroid
    'cc': (cc, 19),    # connected components
    
    # Numeric operations - 86 bytes total
    'inc': (inc, 15),  # increment
    'cl': (cl, 18),    # clamp
    'he': (he, 20),    # histogram equalization
    'sm': (sm, 16),    # sum
    'avg': (avg, 17),  # average
}

# Total primitive library: 339 bytes
TOTAL_BYTES = sum(bytes_count for _, bytes_count in PRIMITIVES.values())

__all__ = ['PRIMITIVES', 'TOTAL_BYTES'] + list(PRIMITIVES.keys())
