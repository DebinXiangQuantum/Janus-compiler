"""Automatically require analysis passes for resource estimation."""

from janus.optimize.basepasses import AnalysisPass
from janus.optimize.passes.analysis.depth import Depth
from janus.optimize.passes.analysis.width import Width
from janus.optimize.passes.analysis.size import Size
from janus.optimize.passes.analysis.count_ops import CountOps
from janus.optimize.passes.analysis.num_tensor_factors import NumTensorFactors
from janus.optimize.passes.analysis.num_qubits import NumQubits


class CircuitResourceAnalyzer(AnalysisPass):
    """Automatically require analysis passes for resource estimation.

    An analysis pass for automatically running:
    * Depth()
    * Width()
    * Size()
    * CountOps()
    * NumTensorFactors()
    * NumQubits()
    """

    def __init__(self):
        super().__init__()
        # Store instances of required passes
        self._depth_pass = Depth()
        self._width_pass = Width()
        self._size_pass = Size()
        self._count_ops_pass = CountOps()
        self._num_tensor_factors_pass = NumTensorFactors()
        self._num_qubits_pass = NumQubits()

        self.requires += [
            self._depth_pass,
            self._width_pass,
            self._size_pass,
            self._count_ops_pass,
            self._num_tensor_factors_pass,
            self._num_qubits_pass
        ]

    def estimate_resources(self, dag):
        """Estimate the resources needed for the circuit in the given DAG."""
        # Run all required passes manually to ensure results are available
        self._depth_pass.run(dag)
        self._width_pass.run(dag)
        self._size_pass.run(dag)
        self._count_ops_pass.run(dag)
        self._num_tensor_factors_pass.run(dag)
        self._num_qubits_pass.run(dag)

        # Copy results to our property_set
        self.property_set['depth'] = self._depth_pass.property_set.get('depth')
        self.property_set['width'] = self._width_pass.property_set.get('width')
        self.property_set['size'] = self._size_pass.property_set.get('size')
        self.property_set['count_ops'] = self._count_ops_pass.property_set.get('count_ops')
        self.property_set['num_tensor_factors'] = self._num_tensor_factors_pass.property_set.get('num_tensor_factors')
        self.property_set['num_qubits'] = self._num_qubits_pass.property_set.get('num_qubits')

    def run(self, dag):
        """Alias for estimate_resources() to maintain backward compatibility."""
        self.estimate_resources(dag)


# Backward compatibility alias
ResourceEstimation = CircuitResourceAnalyzer
