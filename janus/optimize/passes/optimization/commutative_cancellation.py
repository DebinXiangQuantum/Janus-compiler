"""Cancel the redundant (self-adjoint) gates through commutation relations."""
from janus.optimize.basepasses import TransformationPass
from janus.optimize.commutation_library import StandardGateCommutations
from janus.compat.control_flow_utils import trivial_recurse
from janus.compat.commutation_checker import CommutationChecker
from janus.compat import commutation_cancellation

from janus.circuit.library.u1 import U1Gate
from janus.circuit.library.p import PhaseGate
from janus.circuit.library.rz import RZGate

_CUTOFF_PRECISION = 1e-5


class CommutativeGateCanceller(TransformationPass):
    """Cancel the redundant (self-adjoint) gates through commutation relations.

    Pass for cancelling self-inverse gates/rotations. The cancellation utilizes
    the commutation relations in the circuit. Gates considered include::

        H, X, Y, Z, CX, CY, CZ
    """

    def __init__(self, basis_gates=None, target=None):
        """
        CommutativeGateCanceller initializer.

        Args:
            basis_gates (list[str]): Basis gates to consider, e.g.
                ``['u3', 'cx']``. For the effects of this pass, the basis is
                the set intersection between the ``basis_gates`` parameter
                and the gates in the dag.
            target (Target): The :class:`~.Target` representing the target backend, if both
                ``basis_gates`` and ``target`` are specified then this argument will take
                precedence and ``basis_gates`` will be ignored.
        """
        super().__init__()
        if basis_gates:
            self.basis = set(basis_gates)
        else:
            self.basis = set()
        self.target = target
        if target is not None:
            self.basis = set(target.operation_names)

        self._var_z_map = {"rz": RZGate, "p": PhaseGate, "u1": U1Gate}

        self._z_rotations = {"p", "z", "u1", "rz", "t", "s"}
        self._x_rotations = {"x", "rx"}
        self._gates = {"cx", "cy", "cz", "h", "y"}  # Now the gates supported are hard-coded

        # build a commutation checker restricted to the gates we cancel -- the others we
        # do not have to investigate, which allows to save time
        self._commutation_checker = CommutationChecker(
            StandardGateCommutations, gates=self._gates | self._z_rotations | self._x_rotations
        )

    @trivial_recurse
    def cancel_commutative_gates(self, dag):
        """Run the CommutativeGateCanceller pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        commutation_cancellation.cancel_commutations(
            dag, self._commutation_checker, sorted(self.basis)
        )
        return dag

    def run(self, dag):
        """Alias for cancel_commutative_gates() to maintain backward compatibility.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        return self.cancel_commutative_gates(dag)


# Backward compatibility alias
CommutativeCancellation = CommutativeGateCanceller
