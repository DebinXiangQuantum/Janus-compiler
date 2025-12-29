"""Collect sequences of uninterrupted gates acting on 2 qubits."""

from collections import defaultdict

from janus.optimize.basepasses import AnalysisPass


class TwoQubitBlockCollector(AnalysisPass):
    """Collect two-qubit subcircuits."""

    def collect_blocks(self, dag):
        """Run the TwoQubitBlockCollector pass on `dag`.

        The blocks contain "op" nodes in topological order such that all gates
        in a block act on the same qubits and are adjacent in the circuit.

        After the execution, ``property_set['block_list']`` is set to a list of
        tuples of "op" node.
        """
        self.property_set["commutation_set"] = defaultdict(list)
        self.property_set["block_list"] = dag.collect_2q_runs()

        return dag

    def run(self, dag):
        """Alias for collect_blocks() to maintain backward compatibility."""
        return self.collect_blocks(dag)


# Backward compatibility alias
Collect2qBlocks = TwoQubitBlockCollector
