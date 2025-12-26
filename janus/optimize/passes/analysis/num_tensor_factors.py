"""Calculate the number of tensor factors of a DAG circuit."""

from janus.optimize.basepasses import AnalysisPass


class TensorFactorCounter(AnalysisPass):
    """Calculate the number of tensor factors of a DAG circuit.

    The result is saved in ``property_set['num_tensor_factors']`` as an integer.
    """

    def count_tensor_factors(self, dag):
        """Count the tensor factors in the given DAG circuit."""
        self.property_set["num_tensor_factors"] = dag.num_tensor_factors()

    def run(self, dag):
        """Alias for count_tensor_factors() to maintain backward compatibility."""
        return self.count_tensor_factors(dag)


# Backward compatibility alias
NumTensorFactors = TensorFactorCounter
