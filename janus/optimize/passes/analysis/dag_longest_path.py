"""Return the longest path in a :class:`.DAGCircuit` as a list of DAGNodes."""

from janus.optimize.basepasses import AnalysisPass


class DAGLongestPathAnalyzer(AnalysisPass):
    """Return the longest path in a :class:`.DAGCircuit` as a list of
    :class:`.DAGOpNode`\\ s, :class:`.DAGInNode`\\ s, and :class:`.DAGOutNode`\\ s."""

    def find_longest_path(self, dag):
        """Find the longest path in the given DAG circuit."""
        self.property_set["dag_longest_path"] = dag.longest_path()

    def run(self, dag):
        """Alias for find_longest_path() to maintain backward compatibility."""
        return self.find_longest_path(dag)


# Backward compatibility alias
DAGLongestPath = DAGLongestPathAnalyzer
