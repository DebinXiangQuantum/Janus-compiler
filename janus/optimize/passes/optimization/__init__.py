# This code is part of Janus Quantum Compiler.
"""Module containing transpiler optimization passes."""

# Technology 1: Clifford+Rz Optimization
from .optimize_clifford_t import OptimizeCliffordT, TChinMerger
from .optimize_cliffords import OptimizeCliffords, CliffordMerger
from .collect_cliffords import CollectCliffords, CliffordCollector
from .litinski_transformation import LitinskiTransformation, CliffordRzTransform

# Technology 2: Gate Fusion Optimization
from .consolidate_blocks import ConsolidateBlocks, BlockConsolidator
from .optimize_1q_gates import Optimize1qGates, SingleQubitGateOptimizer
from .optimize_1q_decomposition import Optimize1qGatesDecomposition, SingleQubitGateDecomposer
from .collect_1q_runs import Collect1qRuns, SingleQubitRunCollector
from .collect_2q_blocks import Collect2qBlocks, TwoQubitBlockCollector
from .collect_multiqubit_blocks import CollectMultiQBlocks, MultiQubitBlockCollector
from .collect_and_collapse import CollectAndCollapse, BlockCollectCollapser
from .split_2q_unitaries import Split2QUnitaries, TwoQubitUnitarySplitter

# Technology 3: Commutativity Optimization
from .commutative_cancellation import CommutativeCancellation, CommutativeGateCanceller
from .inverse_cancellation import InverseCancellation, InverseGateCanceller
from .commutative_inverse_cancellation import CommutativeInverseCancellation, CommutativeInverseGateCanceller
from .commutation_analysis import CommutationAnalysis, GateCommutationAnalyzer
from .optimize_1q_commutation import Optimize1qGatesSimpleCommutation, SingleQubitCommutationOptimizer

# Technology 4: Template Matching
from .template_optimization import TemplateOptimization, CircuitTemplateOptimizer
from .template_matching import (
    TemplateMatching,
    TemplatePatternMatcher,
    TemplateSubstitution,
    TemplateCircuitSubstitutor,
    BackwardMatch,
    ForwardMatch,
    MaximalMatches,
)
