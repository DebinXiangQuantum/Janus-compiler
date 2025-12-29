# This code is part of Janus Quantum Compiler.
"""
Janus Transpiler Passes Module

This module provides optimization and analysis passes for quantum circuits.
"""

# Optimization Passes
from .optimization import (
    # Technology 1: Clifford+Rz Optimization
    OptimizeCliffordT,
    TChinMerger,
    OptimizeCliffords,
    CliffordMerger,
    CollectCliffords,
    CliffordCollector,
    LitinskiTransformation,
    CliffordRzTransform,
    # Technology 2: Gate Fusion Optimization
    ConsolidateBlocks,
    Optimize1qGates,
    Optimize1qGatesDecomposition,
    Collect1qRuns,
    Collect2qBlocks,
    CollectMultiQBlocks,
    CollectAndCollapse,
    Split2QUnitaries,
    # Technology 3: Commutativity Optimization
    CommutativeCancellation,
    InverseCancellation,
    CommutativeInverseCancellation,
    CommutationAnalysis,
    Optimize1qGatesSimpleCommutation,
    # Technology 4: Template Matching
    TemplateOptimization,
    TemplateMatching,
    TemplateSubstitution,
    BackwardMatch,
    ForwardMatch,
    MaximalMatches,
)

# Analysis Passes
from .analysis import (
    # Technology 9: Metrics Analysis
    ResourceEstimation,
    Depth,
    Width,
    Size,
    CountOps,
    CountOpsLongestPath,
    NumTensorFactors,
    DAGLongestPath,
)
