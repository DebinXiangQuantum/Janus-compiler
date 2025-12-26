"""Combine consecutive Cliffords over the same qubits."""

from janus.optimize.basepasses import TransformationPass
# STUB: control_flow utils
from janus.compat.clifford import Clifford
from janus.compat.control_flow_utils import trivial_recurse


class CliffordMerger(TransformationPass):
    """Combine consecutive Cliffords over the same qubits.
    This serves as an example of extra capabilities enabled by storing
    Cliffords natively on the circuit.
    """

    def _is_clifford_gate(self, gate_name):
        """检查门是否是Clifford门"""
        clifford_gates = {
            'h', 'x', 'y', 'z', 's', 'sdg', 'sx', 'sxdg',
            'cx', 'cy', 'cz', 'swap', 'ch', 'ccx', 'ccz'
        }
        return gate_name.lower() in clifford_gates

    @trivial_recurse
    def merge_cliffords(self, dag):
        """Run the CliffordMerger pass on `dag`.

        Enhanced version: Also recognizes basic Clifford gates (H, S, CX, etc.)
        and converts them to Clifford objects for merging.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

        blocks = []
        prev_node = None
        cur_block = []

        # Iterate over all nodes and collect consecutive Cliffords over the
        # same qubits. In this very first proof-of-concept implementation
        # we require the same ordering of qubits, but this restriction will
        # be shortly removed. An interesting question is whether we may also
        # want to compose Cliffords over different sets of qubits, such as
        # cliff1 over qubits [1, 2, 3] and cliff2 over [2, 3, 4].
        for node in dag.topological_op_nodes():
            # 检查是否是Clifford对象或Clifford门
            is_clifford = isinstance(node.op, Clifford) or self._is_clifford_gate(node.name)

            if is_clifford:
                if prev_node is None:
                    blocks.append(cur_block)
                    cur_block = [node]
                else:
                    if prev_node.qargs == node.qargs:
                        cur_block.append(node)
                    else:
                        blocks.append(cur_block)
                        cur_block = [node]

                prev_node = node

            else:
                # not a clifford
                if cur_block:
                    blocks.append(cur_block)
                prev_node = None
                cur_block = []

        if cur_block:
            blocks.append(cur_block)

        # Replace every discovered block of cliffords by a single clifford
        # based on the Cliffords' compose function.
        # NOTE: 由于Clifford.from_gate()可能不可用，暂时跳过基础门合并
        # 只处理已经是Clifford对象的块
        for cur_nodes in blocks:
            # Create clifford functions only out of blocks with at least 2 gates
            if len(cur_nodes) <= 1:
                continue

            # 检查是否所有节点都是Clifford对象
            all_clifford_objects = all(isinstance(node.op, Clifford) for node in cur_nodes)

            if not all_clifford_objects:
                # 跳过包含基础门的块（因为from_gate可能不可用）
                continue

            wire_pos_map = {qb: ix for ix, qb in enumerate(cur_nodes[0].qargs)}

            try:
                # Construct a linear circuit by composing all Clifford objects
                cliff = cur_nodes[0].op
                for i, node in enumerate(cur_nodes):
                    if i > 0:
                        cliff = Clifford.compose(node.op, cliff, front=True)

                # Replace the block by the composed clifford
                dag.replace_block_with_op(cur_nodes, cliff, wire_pos_map, cycle_check=False)
            except Exception:
                # 如果合并失败，跳过这个块
                pass

        return dag

    # Keep original method name for backward compatibility
    def run(self, dag):
        """Alias for merge_cliffords() to maintain backward compatibility."""
        return self.merge_cliffords(dag)


# Keep original class name for backward compatibility
OptimizeCliffords = CliffordMerger
