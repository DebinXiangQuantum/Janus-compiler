# This code is part of Janus Quantum Compiler.
"""Module containing circuit analysis passes."""

# Technology 9: Metrics Analysis
from .resource_estimation import ResourceEstimation, CircuitResourceAnalyzer
from .depth import Depth, CircuitDepthAnalyzer
from .width import Width, CircuitWidthAnalyzer
from .size import Size, CircuitSizeAnalyzer
from .count_ops import CountOps, GateCountAnalyzer
from .count_ops_longest_path import CountOpsLongestPath, LongestPathGateCounter
from .num_tensor_factors import NumTensorFactors, TensorFactorCounter
from .num_qubits import NumQubits
from .dag_longest_path import DAGLongestPath, DAGLongestPathAnalyzer
