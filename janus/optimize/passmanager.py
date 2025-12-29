# This code is part of Janus Quantum Compiler.
"""
PassManager module for optimize passes.

This module re-exports PassManager classes from janus infrastructure.
"""

# Import PassManager from janus infrastructure (hybrid dependency strategy)
from janus.compat.passmanager import PassManager

__all__ = [
    "PassManager",
]
