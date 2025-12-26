"""Collect sequences of uninterrupted gates acting on 1 qubit."""

from janus.optimize.basepasses import AnalysisPass


class SingleQubitRunCollector(AnalysisPass):
    """Collect one-qubit subcircuits."""

    def collect_runs(self, dag):
        """Run the SingleQubitRunCollector pass on `dag`.

        The blocks contain "op" nodes in topological order such that all gates
        in a block act on the same qubits and are adjacent in the circuit.

        After the execution, ``property_set['run_list']`` is set to a list of
        tuples of "op" node.
        """
        self.property_set["run_list"] = dag.collect_1q_runs()
        return dag

    def run(self, dag):
        """Alias for collect_runs() to maintain backward compatibility."""
        return self.collect_runs(dag)


# Backward compatibility alias
Collect1qRuns = SingleQubitRunCollector
