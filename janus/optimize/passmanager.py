# This code is part of Janus Quantum Compiler.
"""
PassManager module for optimize passes.

This module re-exports PassManager classes from qiskit infrastructure.
"""

# Import PassManager from qiskit infrastructure (hybrid dependency strategy)
from janus.compat.passmanager import PassManager

__all__ = [
    "PassManager",
]
