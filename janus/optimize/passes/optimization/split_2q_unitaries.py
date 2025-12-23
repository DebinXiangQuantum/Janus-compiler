"""Splits each two-qubit gate in the `dag` into two single-qubit gates, if possible without error."""

from janus.optimize.basepasses import TransformationPass
# FIXME: Remove qiskit import import Layout
from janus.circuit import DAGCircuit
# FIXME: from # FIXME: qiskit._accelerate.split_2q_unitaries import split_2q_unitaries


class TwoQubitUnitarySplitter(TransformationPass):
    """Attempt to splits two-qubit unitaries in a :class:`.DAGCircuit` into two single-qubit gates.

    This pass will analyze all :class:`.UnitaryGate` instances and determine whether the
    matrix is actually a product of 2 single qubit gates. In these cases the 2q gate can be
    simplified into two single qubit gates and this pass will perform this optimization and will
    replace the two qubit gate with two single qubit :class:`.UnitaryGate`.

    If some of the gates can be viewed as a swap joined by the product of 2 single qubit gates,
    the pass will recreate the DAG, permuting the swapped qubits similar
    to how it's done in :class:`ElidePermutations`.
    """

    def __init__(self, fidelity: float = 1.0 - 1e-16, split_swap: bool = False):
        """
        Args:
            fidelity: Allowed tolerance for splitting two-qubit unitaries and gate decompositions.
            split_swap: Whether to attempt to split swap gates, resulting in a permutation of the qubits.
        """
        super().__init__()
        self.requested_fidelity = fidelity
        self.split_swap = split_swap

    def split_unitaries(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the TwoQubitUnitarySplitter pass on `dag`."""
        result = split_2q_unitaries(dag, self.requested_fidelity, self.split_swap)
        if result is None:
            return dag

        (new_dag, qubit_mapping) = result
        input_qubit_mapping = {qubit: index for index, qubit in enumerate(dag.qubits)}
        self.property_set["original_layout"] = Layout(input_qubit_mapping)
        if self.property_set["original_qubit_indices"] is None:
            self.property_set["original_qubit_indices"] = input_qubit_mapping

        new_layout = Layout({dag.qubits[out]: idx for idx, out in enumerate(qubit_mapping)})
        if current_layout := self.property_set["virtual_permutation_layout"]:
            self.property_set["virtual_permutation_layout"] = new_layout.compose(
                current_layout, dag.qubits
            )
        else:
            self.property_set["virtual_permutation_layout"] = new_layout
        return new_dag

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Alias for split_unitaries() to maintain backward compatibility."""
        return self.split_unitaries(dag)


# Backward compatibility alias
Split2QUnitaries = TwoQubitUnitarySplitter
