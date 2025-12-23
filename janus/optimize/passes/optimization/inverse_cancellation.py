"""
A generic InverseCancellation pass for any set of gate-inverse pairs.
"""
from __future__ import annotations

from typing import List, Tuple, Union

from janus.circuit import Gate
from janus.circuit import DAGCircuit
from janus.optimize.basepasses import TransformationPass
from janus.compat.exceptions import TranspilerError
from janus.compat.control_flow_utils import trivial_recurse
# STUB: control_flow utils

# FIXME: qiskit._accelerate.inverse_cancellation not available
# from janus.compat.accelerate.inverse_cancellation import (
#     inverse_cancellation,
#     run_inverse_cancellation_standard_gates,
# )
# TODO: Implement Python versions

def inverse_cancellation(*args, **kwargs):
    raise NotImplementedError("inverse_cancellation not yet implemented")

def run_inverse_cancellation_standard_gates(dag: DAGCircuit):
    """
    Python实现: 消除标准门的逆门对

    增强版本:
    - 支持自逆门: H, X, Y, Z, CX, CY, CZ, SWAP, CH, CCX, CCZ等
    - 支持逆门对: (T, Tdg), (S, Sdg), (SX, SXdg), (CS, CSdg)
    - 支持带参数的旋转门逆操作
    """
    # 自逆门列表 (门名)
    self_inverse_gates = {
        'h', 'x', 'y', 'z',
        'cx', 'cy', 'cz',
        'swap', 'ch', 'ccx', 'ccz',
        'ecr', 'rccx', 'cswap', 'c3x'
    }

    # 逆门对映射 (门名 -> 其逆门名)
    inverse_pairs = {
        't': 'tdg', 'tdg': 't',
        's': 'sdg', 'sdg': 's',
        'sx': 'sxdg', 'sxdg': 'sx',
        'cs': 'csdg', 'csdg': 'cs',
    }

    # 遍历所有量子比特,检测并消除逆门对
    nodes_to_remove = []

    # 对于每个量子比特的每个位置，检查门序列
    for qubit_idx in range(dag.n_qubits):
        # 收集该量子比特上的所有门节点（按拓扑顺序）
        all_nodes = list(dag.topological_op_nodes())
        qubit_nodes = []

        for node in all_nodes:
            # 检查节点是否涉及当前量子比特
            # node.qargs可能是Qubit对象列表或整数列表
            node_qubits = []
            for q in node.qargs:
                if hasattr(q, 'index'):
                    node_qubits.append(q.index)
                else:
                    node_qubits.append(int(q))

            if qubit_idx in node_qubits:
                qubit_nodes.append(node)

        # 在该量子比特的门序列中查找可以消除的逆门对
        i = 0
        while i < len(qubit_nodes) - 1:
            current = qubit_nodes[i]
            j = i + 1

            # 向前搜索可以与current消除的门
            while j < len(qubit_nodes):
                next_node = qubit_nodes[j]

                # 获取两个节点的量子比特索引
                current_qubits = set()
                for q in current.qargs:
                    current_qubits.add(q.index if hasattr(q, 'index') else int(q))

                next_qubits = set()
                for q in next_node.qargs:
                    next_qubits.add(q.index if hasattr(q, 'index') else int(q))

                # 检查量子比特是否完全匹配
                if current_qubits != next_qubits:
                    # 如果是单量子门，可以跨过不相关的门继续搜索
                    if len(current.qargs) == 1 and len(next_node.qargs) == 1:
                        if current_qubits != next_qubits:
                            j += 1
                            continue
                    # 多量子门必须严格匹配
                    break

                current_name = current.name.lower()
                next_name = next_node.name.lower()

                should_cancel = False

                # 检查是否是自逆门对 (如 H-H, X-X)
                if current_name == next_name and current_name in self_inverse_gates:
                    # 还要检查量子比特顺序是否一致
                    if current.qargs == next_node.qargs:
                        should_cancel = True

                # 检查是否是互逆门对 (如 T-Tdg)
                elif current_name in inverse_pairs:
                    if inverse_pairs[current_name] == next_name and current.qargs == next_node.qargs:
                        should_cancel = True

                # 检查参数化旋转门 (如 Rx(θ)-Rx(-θ))
                elif current_name in ['rx', 'ry', 'rz'] and current_name == next_name:
                    if current.qargs == next_node.qargs:
                        # 检查参数是否相反
                        if hasattr(current.op, 'params') and hasattr(next_node.op, 'params'):
                            if len(current.op.params) > 0 and len(next_node.op.params) > 0:
                                import math
                                if abs(current.op.params[0] + next_node.op.params[0]) < 1e-10:
                                    should_cancel = True

                if should_cancel:
                    nodes_to_remove.append(current)
                    nodes_to_remove.append(next_node)
                    # 移除j节点，继续处理
                    break

                j += 1

            i += 1

    # 移除标记的节点
    removed = set()
    for node in nodes_to_remove:
        if node.node_id not in removed and node.node_id in dag._nodes:
            dag.remove_op_node(node)
            removed.add(node.node_id)

    return dag



class InverseGateCanceller(TransformationPass):
    """Cancel specific Gates which are inverses of each other when they occur back-to-
    back."""

    def __init__(
        self,
        gates_to_cancel: List[Union[Gate, Tuple[Gate, Gate]]] | None = None,
        run_default: bool = False,
    ):
        """Initialize InverseGateCanceller pass.

        Args:
            gates_to_cancel: List describing the gates to cancel. Each element of the
                list is either a single gate or a pair of gates. If a single gate, then
                it should be self-inverse. If a pair of gates, then the gates in the
                pair should be inverses of each other. If ``None`` a default list of
                self-inverse gates and a default list of inverse gate pairs will be used.
                The current default list of self-inverse gates is:

                  * :class:`.CXGate`
                  * :class:`.ECRGate`
                  * :class:`.CYGate`
                  * :class:`.CZGate`
                  * :class:`.XGate`
                  * :class:`.YGate`
                  * :class:`.ZGate`
                  * :class:`.HGate`
                  * :class:`.SwapGate`
                  * :class:`.CHGate`
                  * :class:`.CCXGate`
                  * :class:`.CCZGate`
                  * :class:`.RCCXGate`
                  * :class:`.CSwapGate`
                  * :class:`.C3XGate`

                and the default list of inverse gate pairs is:

                  * :class:`.TGate` and :class:`.TdgGate`
                  * :class:`.SGate` and :class:`.SdgGate`
                  * :class:`.SXGate` and :class:`.SXdgGate`
                  * :class:`.CSGate` and :class:`.CSdgGate`

            run_default: If set to true and ``gates_to_cancel`` is set to a list then in
                addition to the gates listed in ``gates_to_cancel`` the default list of gate
                inverses (the same as when ``gates_to_cancel`` is set to ``None``) will be
                run. The order of evaluation is significant in how sequences of gates are
                cancelled and the default gates will be evaluated after the provided gates
                in ``gates_to_cancel``. If ``gates_to_cancel`` is ``None`` this option has
                no impact.

        Raises:
            TranspilerError: Input is not a self-inverse gate or a pair of inverse gates.
        """
        self.self_inverse_gates = []
        self.inverse_gate_pairs = []
        self.self_inverse_gate_names = set()
        self.inverse_gate_pairs_names = set()
        self._also_default = run_default
        if gates_to_cancel is None:
            self._use_standard_gates = True
        else:
            self._use_standard_gates = False
            for gates in gates_to_cancel:
                if isinstance(gates, Gate):
                    if gates != gates.inverse():
                        raise TranspilerError(f"Gate {gates.name} is not self-inverse")
                elif isinstance(gates, tuple):
                    if len(gates) != 2:
                        raise TranspilerError(
                            f"Too many or too few inputs: {gates}. Only two are allowed."
                        )
                    if gates[0] != gates[1].inverse():
                        raise TranspilerError(
                            f"Gate {gates[0].name} and {gates[1].name} are not inverse."
                        )
                else:
                    raise TranspilerError(
                        f"InverseGateCanceller pass does not take input type {type(gates)}. Input must be"
                        " a Gate."
                    )

            for gates in gates_to_cancel:
                if isinstance(gates, Gate):
                    self.self_inverse_gates.append(gates)
                    self.self_inverse_gate_names.add(gates.name)
                else:
                    self.inverse_gate_pairs.append(gates)
                    self.inverse_gate_pairs_names.update(x.name for x in gates)
        super().__init__()

    @trivial_recurse
    def cancel_inverse_pairs(self, dag: DAGCircuit):
        """Run the InverseGateCanceller pass on `dag`.

        Args:
            dag: the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        if self._use_standard_gates:
            run_inverse_cancellation_standard_gates(dag)
        else:
            inverse_cancellation(
                dag,
                self.inverse_gate_pairs,
                self.self_inverse_gates,
                self.inverse_gate_pairs_names,
                self.self_inverse_gate_names,
                self._also_default,
            )
        return dag

    def run(self, dag: DAGCircuit):
        """Alias for cancel_inverse_pairs() to maintain backward compatibility.

        Args:
            dag: the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        return self.cancel_inverse_pairs(dag)


# Backward compatibility alias
InverseCancellation = InverseGateCanceller
