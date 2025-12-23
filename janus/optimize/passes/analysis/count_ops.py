"""Count the operations in a DAG circuit."""

from janus.optimize.basepasses import AnalysisPass


class GateCountAnalyzer(AnalysisPass):
    """Count the operations in a DAG circuit.

    The result is saved in ``property_set['count_ops']`` as a dict that
    maps each operation to the number of times it appears in the DAG.
    """

    def __init__(self, *, recurse=True):
        """
        Args:
            recurse: whether to allow recursion into control flow. Default is ``True``.
        """
        super().__init__()
        self.recurse = recurse

    def count_gates(self, dag):
        """Count the gates in the given DAG circuit."""
        self.property_set["count_ops"] = dag.count_ops(recurse=self.recurse)

    def run(self, dag):
        """Alias for count_gates() to maintain backward compatibility."""
        return self.count_gates(dag)


# Backward compatibility alias
CountOps = GateCountAnalyzer
