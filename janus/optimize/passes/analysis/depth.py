"""Calculate the depth of a DAG circuit."""

from janus.optimize.basepasses import AnalysisPass


class CircuitDepthAnalyzer(AnalysisPass):
    """Calculate the depth of a DAG circuit.

    The result is saved in ``property_set['depth']`` as an integer that
    represents the longest path in the DAG.
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

    def analyze_depth(self, dag):
        """Analyze the depth of the circuit in the given DAG."""
        self.property_set["depth"] = dag.depth(recurse=self.recurse)

    def run(self, dag):
        """Alias for analyze_depth() to maintain backward compatibility."""
        return self.analyze_depth(dag)


# Backward compatibility alias
Depth = CircuitDepthAnalyzer
