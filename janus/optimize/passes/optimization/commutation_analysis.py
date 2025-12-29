"""Analysis pass to find commutation relations between DAG nodes."""

from janus.optimize.commutation_library import SessionCommutationChecker as scc
from janus.optimize.basepasses import AnalysisPass
# Accelerated implementation.commutation_analysis import analyze_commutations

def analyze_commutations(dag, commutation_checker):
    """
    Python实现: 分析DAG中门的交换关系

    Args:
        dag: DAGCircuit对象
        commutation_checker: 交换性检查器

    Returns:
        dict: 交换集合,每个量子比特映射到可交换的门组
    """
    commutation_set = {}

    # 对每个量子比特分析交换关系
    for qubit in range(dag.n_qubits):
        qubit_gates = []
        for node in dag.topological_op_nodes():
            if qubit in node.qubits:
                qubit_gates.append(node)

        # 将门分组为可交换的集合
        groups = []
        for gate in qubit_gates:
            placed = False
            for group in groups:
                # 检查gate是否与group中所有门交换
                # commute方法需要: op1, qargs1, op2, qargs2
                if all(commutation_checker.commute(
                    gate.op, gate.qubits,
                    g.op, g.qubits
                ) for g in group):
                    group.append(gate)
                    placed = True
                    break
            if not placed:
                groups.append([gate])

        commutation_set[qubit] = groups

    return commutation_set


class GateCommutationAnalyzer(AnalysisPass):
    r"""Analysis pass to find commutation relations between DAG nodes.

    This sets ``property_set['commutation_set']`` to a dictionary that describes
    the commutation relations on a given wire: all the gates on a wire
    are grouped into a set of gates that commute.
    """

    def __init__(self, *, _commutation_checker=None):
        super().__init__()
        # allow setting a private commutation checker, this allows better performance if we
        # do not care about commutations of all gates, but just a subset
        if _commutation_checker is None:
            _commutation_checker = scc

        self.comm_checker = _commutation_checker

    def analyze_commutation(self, dag):
        """Run the GateCommutationAnalyzer pass on `dag`.

        Run the pass on the DAG, and write the discovered commutation relations
        into the ``property_set``.
        """
        # Initiate the commutation set
        self.property_set["commutation_set"] = analyze_commutations(dag, self.comm_checker.cc)
        return dag

    def run(self, dag):
        """Alias for analyze_commutation() to maintain backward compatibility.

        Run the pass on the DAG, and write the discovered commutation relations
        into the ``property_set``.
        """
        return self.analyze_commutation(dag)


# Backward compatibility alias
CommutationAnalysis = GateCommutationAnalyzer
