"""
MicroGolf: Ultra-compact ARC-AGI solution framework
NeurIPS 2025 Google Code Golf Championship

A state-of-the-art framework for generating ultra-compact (<2500 bytes)
Python solutions using modular primitives, DSL, and meta-learning.
"""

__version__ = "1.0.0"
__author__ = "MicroGolf Team"
__license__ = "CC BY 4.0"

from . import primitives
from . import engine
from . import model

__all__ = ["primitives", "engine", "model"]
