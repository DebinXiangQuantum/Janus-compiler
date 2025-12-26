"""Calculate the size of a DAG circuit."""

from janus.optimize.basepasses import AnalysisPass


class CircuitSizeAnalyzer(AnalysisPass):
    """Calculate the size of a DAG circuit.

    The result is saved in ``property_set['size']`` as an integer.
    """

    def __init__(self, *, recurse=False):
        """
        Args:
            recurse: whether to allow recursion into control flow.  If this is ``False`` (default),
                the pass will throw an error when control flow is present, to avoid returning a
                number with little meaning.
        """
        super().__init__()
        self.recurse = recurse

    def analyze_size(self, dag):
        """Analyze the size of the circuit in the given DAG."""
        self.property_set["size"] = dag.size(recurse=self.recurse)

    def run(self, dag):
        """Alias for analyze_size() to maintain backward compatibility."""
        return self.analyze_size(dag)


# Backward compatibility alias
Size = CircuitSizeAnalyzer
