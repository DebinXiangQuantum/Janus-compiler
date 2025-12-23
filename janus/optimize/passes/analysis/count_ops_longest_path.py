"""Count the operations on the longest path in a DAGCircuit."""

from janus.optimize.basepasses import AnalysisPass


class LongestPathGateCounter(AnalysisPass):
    """Count the operations on the longest path in a :class:`.DAGCircuit`.

    The result is saved in ``property_set['count_ops_longest_path']`` as an integer.
    """

    def count_longest_path_gates(self, dag):
        """Count the gates on the longest path in the given DAG."""
        self.property_set["count_ops_longest_path"] = dag.count_ops_longest_path()

    def run(self, dag):
        """Alias for count_longest_path_gates() to maintain backward compatibility."""
        return self.count_longest_path_gates(dag)


# Backward compatibility alias
CountOpsLongestPath = LongestPathGateCounter
