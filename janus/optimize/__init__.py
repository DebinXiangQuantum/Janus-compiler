# This code is part of Janus Quantum Compiler.
"""
Janus Optimize Module - Quantum Circuit Optimization and Synthesis

This module provides a comprehensive suite of quantum circuit optimization
techniques and synthesis algorithms. It implements 10 optimization technologies:

**Optimization Passes:**
- Technology 1: Clifford+Rz instruction set optimization
- Technology 2: Gate fusion optimization
- Technology 3: Commutativity-based optimization
- Technology 4: Template pattern matching

**Synthesis Algorithms:**
- Technology 5: KAK decomposition for two-qubit gates
- Technology 6: Clifford circuit synthesis
- Technology 7: CNOT circuit optimization

**Analysis and Benchmarking:**
- Technology 9: Circuit metrics analysis
- Technology 8: Benchmarking (to be implemented)
- Technology 10: Auto-selection (to be implemented)

**Usage:**
    from janus.optimize import TChinMerger, CliffordMerger
    from janus.optimize.synthesis import synthesize_clifford_circuit
    from janus.optimize.passes.analysis import Depth, CountOps

**Architecture:**
This module provides quantum circuit optimization functionality:
- Core functionality is implemented in the optimize namespace
- Infrastructure components (QuantumCircuit, DAGCircuit, etc.) are imported from janus
- Performance-critical functions use optimized implementations
"""

# === Optimization Passes (Technologies 1-4) ===

# Technology 1: Clifford+Rz Optimization
from .passes.optimization import (
    OptimizeCliffordT,
    TChinMerger,
    OptimizeCliffords,
    CliffordMerger,
    CollectCliffords,
    CliffordCollector,
    LitinskiTransformation,
    CliffordRzTransform,
)

# Technology 2: Gate Fusion Optimization
from .passes.optimization import (
    ConsolidateBlocks,
    BlockConsolidator,
    Optimize1qGates,
    SingleQubitGateOptimizer,
    Optimize1qGatesDecomposition,
    SingleQubitGateDecomposer,
    Collect1qRuns,
    SingleQubitRunCollector,
    Collect2qBlocks,
    TwoQubitBlockCollector,
    CollectMultiQBlocks,
    MultiQubitBlockCollector,
    CollectAndCollapse,
    BlockCollectCollapser,
    Split2QUnitaries,
    TwoQubitUnitarySplitter,
)

# Technology 3: Commutativity Optimization
from .passes.optimization import (
    CommutativeCancellation,
    CommutativeGateCanceller,
    InverseCancellation,
    InverseGateCanceller,
    CommutativeInverseCancellation,
    CommutativeInverseGateCanceller,
    CommutationAnalysis,
    GateCommutationAnalyzer,
    Optimize1qGatesSimpleCommutation,
    SingleQubitCommutationOptimizer,
)

# Technology 4: Template Matching
from .passes.optimization import (
    TemplateOptimization,
    CircuitTemplateOptimizer,
    TemplateMatching,
    TemplatePatternMatcher,
    TemplateSubstitution,
    TemplateCircuitSubstitutor,
)

# === Synthesis Algorithms (Technologies 5-7) ===

# Technology 5: KAK Decomposition
from .synthesis.two_qubit import (
    TwoQubitWeylDecomposition,
    KAKDecomposition,
    TwoQubitBasisDecomposer,
    KAKBasisDecomposer,
    two_qubit_cnot_decompose,
    TwoQubitControlledUDecomposer,
    ControlledUKAKDecomposer,
)

# Technology 6: Clifford Synthesis
from .synthesis.clifford import (
    synthesize_clifford_circuit,
    synthesize_clifford_aaronson_gottesman,
    synthesize_clifford_bravyi_maslov,
    synthesize_clifford_greedy,
    synthesize_clifford_layered,
    synthesize_clifford_depth_lnn,
)

# Technology 7: CNOT Optimization
from .synthesis.linear import (
    synthesize_cnot_count_pmh,
    synthesize_cnot_depth_lnn_kms,
)

from .synthesis.linear_phase import (
    synthesize_cnot_phase_aam,
    synthesize_cx_cz_depth_lnn_my,
    synthesize_cz_depth_lnn_mr,
)

# === Analysis Passes (Technology 9) ===

from .passes.analysis import (
    ResourceEstimation,
    CircuitResourceAnalyzer,
    Depth,
    CircuitDepthAnalyzer,
    Width,
    CircuitWidthAnalyzer,
    Size,
    CircuitSizeAnalyzer,
    CountOps,
    GateCountAnalyzer,
    CountOpsLongestPath,
    LongestPathGateCounter,
    NumTensorFactors,
    TensorFactorCounter,
    DAGLongestPath,
    DAGLongestPathAnalyzer,
)

# === Technology 10: Smart Optimizer ===

from .smart_optimizer import (
    SmartOptimizer,
    smart_compile,
    analyze_and_optimize,
)

# === Unified API (统一接口) ===

from .unified_api import (
    optimize_clifford_rz,      # 技术1: Clifford+RZ优化
    optimize_gate_fusion,      # 技术2: 门融合优化
    optimize_commutativity,    # 技术3: 交换性优化
    optimize_template,         # 技术4: 模板匹配优化
    optimize_kak,              # 技术5: KAK分解优化
    optimize_clifford_synth,   # 技术6: Clifford合成优化
    optimize_cnot_synth,       # 技术7: CNOT合成优化
    run_benchmark,             # 技术8: 基准测试
    analyze_circuit,           # 技术9: 电路分析
    smart_optimize,            # 技术10: 智能优化
)

# === Base Classes ===

from .basepasses import TransformationPass, AnalysisPass

# === Version Info ===

__version__ = "1.0.0"
__author__ = "Janus Quantum Compiler Team"

# === All Exports ===

__all__ = [
    # Technology 1 (Old and New names)
    "OptimizeCliffordT",
    "TChinMerger",
    "OptimizeCliffords",
    "CliffordMerger",
    "CollectCliffords",
    "CliffordCollector",
    "LitinskiTransformation",
    "CliffordRzTransform",
    # Technology 2 (Old and New names)
    "ConsolidateBlocks",
    "BlockConsolidator",
    "Optimize1qGates",
    "SingleQubitGateOptimizer",
    "Optimize1qGatesDecomposition",
    "SingleQubitGateDecomposer",
    "Collect1qRuns",
    "SingleQubitRunCollector",
    "Collect2qBlocks",
    "TwoQubitBlockCollector",
    "CollectMultiQBlocks",
    "MultiQubitBlockCollector",
    "CollectAndCollapse",
    "BlockCollectCollapser",
    "Split2QUnitaries",
    "TwoQubitUnitarySplitter",
    # Technology 3 (Old and New names)
    "CommutativeCancellation",
    "CommutativeGateCanceller",
    "InverseCancellation",
    "InverseGateCanceller",
    "CommutativeInverseCancellation",
    "CommutativeInverseGateCanceller",
    "CommutationAnalysis",
    "GateCommutationAnalyzer",
    "Optimize1qGatesSimpleCommutation",
    "SingleQubitCommutationOptimizer",
    # Technology 4 (Old and New names)
    "TemplateOptimization",
    "CircuitTemplateOptimizer",
    "TemplateMatching",
    "TemplatePatternMatcher",
    "TemplateSubstitution",
    "TemplateCircuitSubstitutor",
    # Technology 5 (Old and New names)
    "TwoQubitWeylDecomposition",
    "KAKDecomposition",
    "TwoQubitBasisDecomposer",
    "KAKBasisDecomposer",
    "two_qubit_cnot_decompose",
    "TwoQubitControlledUDecomposer",
    "ControlledUKAKDecomposer",
    # Technology 6
    "synthesize_clifford_circuit",
    "synthesize_clifford_aaronson_gottesman",
    "synthesize_clifford_bravyi_maslov",
    "synthesize_clifford_greedy",
    "synthesize_clifford_layered",
    "synthesize_clifford_depth_lnn",
    # Technology 7
    "synthesize_cnot_count_pmh",
    "synthesize_cnot_depth_lnn_kms",
    "synthesize_cnot_phase_aam",
    "synthesize_cx_cz_depth_lnn_my",
    "synthesize_cz_depth_lnn_mr",
    # Technology 9 (Old and New names)
    "ResourceEstimation",
    "CircuitResourceAnalyzer",
    "Depth",
    "CircuitDepthAnalyzer",
    "Width",
    "CircuitWidthAnalyzer",
    "Size",
    "CircuitSizeAnalyzer",
    "CountOps",
    "GateCountAnalyzer",
    "CountOpsLongestPath",
    "LongestPathGateCounter",
    "NumTensorFactors",
    "TensorFactorCounter",
    "DAGLongestPath",
    "DAGLongestPathAnalyzer",
    # Technology 10 (Smart Optimizer)
    "SmartOptimizer",
    "smart_compile",
    "analyze_and_optimize",
    # Unified API (统一接口)
    "optimize_clifford_rz",
    "optimize_gate_fusion",
    "optimize_commutativity",
    "optimize_template",
    "optimize_kak",
    "optimize_clifford_synth",
    "optimize_cnot_synth",
    "run_benchmark",
    "analyze_circuit",
    "smart_optimize",
    # Base classes
    "TransformationPass",
    "AnalysisPass",
]
