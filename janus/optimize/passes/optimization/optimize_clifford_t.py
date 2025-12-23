"""Combine consecutive T/Tdg gates in a Clifford+T circuit."""

from janus.circuit import DAGCircuit
from janus.optimize.basepasses import TransformationPass
from janus.circuit.library import SGate, SdgGate


class TChinMerger(TransformationPass):
    """An optimization pass for Clifford+T circuits.

    Currently all the pass does is merging pairs of consecutive T-gates into
    S-gates, and pair of consecutive Tdg-gates into Sdg-gates.
    """

    def merge_t_gates(self, dag: DAGCircuit):
        """
        Run the TChinMerger pass on `dag`.

        Enhanced version: Merges T gates more intelligently by creating a new DAG
        - T + T = S
        - T + T + T + T = Z
        - Tdg + Tdg = Sdg
        - Tdg + Tdg + Tdg + Tdg = Z

        Args:
            dag: The directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        from janus.circuit.library import ZGate, TGate, TdgGate

        new_dag = dag.copy_empty_like()

        # 按量子比特分组处理
        for qubit_idx in range(dag.n_qubits):
            # 收集该量子比特上的所有门（按拓扑顺序）
            qubit_gates = []
            for node in dag.topological_op_nodes():
                node_qubits = []
                for q in node.qargs:
                    if hasattr(q, 'index'):
                        node_qubits.append(q.index)
                    else:
                        node_qubits.append(int(q))

                # 只处理单量子门作用在当前量子比特
                if len(node_qubits) == 1 and node_qubits[0] == qubit_idx:
                    qubit_gates.append(node)

            # 在这个量子比特上合并T/Tdg门
            i = 0
            while i < len(qubit_gates):
                gate = qubit_gates[i]
                if gate.name not in ['t', 'tdg']:
                    # 非T/Tdg门直接复制
                    new_dag.apply_operation_back(gate.op, gate.qargs, gate.cargs)
                    i += 1
                    continue

                # T/Tdg门 - 统计连续的相同类型
                gate_type = gate.name
                qargs = gate.qargs
                cargs = gate.cargs
                count = 1
                j = i + 1

                while j < len(qubit_gates) and qubit_gates[j].name == gate_type:
                    count += 1
                    j += 1

                # 根据数量进行优化
                if count >= 4:
                    # 4个T/Tdg = Z, 剩余的继续处理
                    num_z = count // 4
                    remainder = count % 4

                    # 添加Z门
                    for _ in range(num_z):
                        new_dag.apply_operation_back(ZGate(), qargs, cargs)

                    # 处理剩余
                    if remainder >= 2:
                        gate = SGate() if gate_type == 't' else SdgGate()
                        new_dag.apply_operation_back(gate, qargs, cargs)
                        remainder -= 2

                    # 添加剩余的单个T/Tdg
                    for _ in range(remainder):
                        gate_obj = TGate() if gate_type == 't' else TdgGate()
                        new_dag.apply_operation_back(gate_obj, qargs, cargs)

                elif count >= 2:
                    # 2个T = S, 2个Tdg = Sdg
                    gate_obj = SGate() if gate_type == 't' else SdgGate()

                    # 添加S/Sdg门
                    for _ in range(count // 2):
                        new_dag.apply_operation_back(gate_obj, qargs, cargs)

                    # 处理奇数剩余
                    if count % 2 == 1:
                        gate_single = TGate() if gate_type == 't' else TdgGate()
                        new_dag.apply_operation_back(gate_single, qargs, cargs)
                else:
                    # 单个T/Tdg直接复制
                    new_dag.apply_operation_back(gate.op, qargs, cargs)

                i = j

        # 复制多量子门
        for node in dag.topological_op_nodes():
            if len(node.qargs) > 1:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return new_dag

    # Keep original method name for backward compatibility
    def run(self, dag: DAGCircuit):
        """Alias for merge_t_gates() to maintain backward compatibility."""
        return self.merge_t_gates(dag)


# Keep original class name for backward compatibility
OptimizeCliffordT = TChinMerger
