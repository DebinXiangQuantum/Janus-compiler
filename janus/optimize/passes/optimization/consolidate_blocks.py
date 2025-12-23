"""Replace each block of consecutive gates by a single Unitary node."""
from __future__ import annotations

from janus.compat.synthesis.two_qubit import TwoQubitBasisDecomposer, TwoQubitControlledUDecomposer
from janus.circuit.library import (
    CXGate,
    CZGate,
    iSwapGate,
    ECRGate,
    RXXGate,
    RYYGate,
    RZZGate,
    RZXGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CPhaseGate,
)

from janus.optimize.basepasses import TransformationPass
from janus.optimize.passmanager import PassManager
# FIXME: from # FIXME: qiskit._accelerate.consolidate_blocks import consolidate_blocks
# Python stub implementation
def consolidate_blocks(dag, decomposer, basis_gate_name, force_consolidate=False,
                      target=None, basis_gates=None, blocks=None, runs=None, qubit_map=None):
    """
    Python stub: 合并门块为Unitary操作

    这是一个占位实现，真实实现需要Rust加速
    目前简单返回原DAG
    """
    # TODO: 实现实际的块合并逻辑
    # 这需要：
    # 1. 将块中的门合成为单个unitary矩阵
    # 2. 使用decomposer将unitary分解为基础门
    # 3. 替换DAG中的块
    return dag

from .collect_1q_runs import Collect1qRuns
from .collect_2q_blocks import Collect2qBlocks

KAK_GATE_NAMES = {
    "cx": CXGate(),
    "cz": CZGate(),
    "iswap": iSwapGate(),
    "ecr": ECRGate(),
}

KAK_GATE_PARAM_NAMES = {
    "rxx": RXXGate,
    "rzz": RZZGate,
    "ryy": RYYGate,
    "rzx": RZXGate,
    "cphase": CPhaseGate,
    "crx": CRXGate,
    "cry": CRYGate,
    "crz": CRZGate,
}


class BlockConsolidator(TransformationPass):
    """Replace each block of consecutive gates by a single Unitary node.

    Pass to consolidate sequences of uninterrupted gates acting on
    the same qubits into a Unitary node, to be resynthesized later,
    to a potentially more optimal subcircuit.

    This pass reads the :class:`.PropertySet` key ``ConsolidateBlocks_qubit_map`` which it uses to
    communicate with recursive worker instances of itself for control-flow operations.  The key
    should never be observable in a user-facing :class:`.PassManager` pipeline (it is only set in
    internal :class:`.PassManager` instances), but the pass may return incorrect results or error if
    another pass sets this key.

    Notes:
        This pass assumes that the 'blocks_list' property that it reads is
        given such that blocks are in topological order. The blocks are
        collected by a previous pass, such as `Collect2qBlocks`.
    """

    _QUBIT_MAP_KEY = "ConsolidateBlocks_qubit_map"

    def __init__(
        self,
        kak_basis_gate=None,
        force_consolidate=False,
        basis_gates=None,
        approximation_degree=1.0,
        target=None,
    ):
        """BlockConsolidator initializer.

        If ``kak_basis_gate`` is not ``None`` it will be used as the basis gate for KAK decomposition.
        Otherwise, if ``basis_gates`` is not ``None`` a basis gate will be chosen from this list.
        Otherwise, the basis gate will be :class:`.CXGate`.

        Args:
            kak_basis_gate (Gate): Basis gate for KAK decomposition.
            force_consolidate (bool): Force block consolidation.
            basis_gates (List(str)): Basis gates from which to choose a KAK gate.
            approximation_degree (float): a float between :math:`[0.0, 1.0]`. Lower approximates more.
            target (Target): The target object for the compilation target backend.
        """
        super().__init__()
        self.basis_gates = None
        self.basis_gate_name = None
        # Bypass target if it doesn't contain any basis gates (i.e. it's a _FakeTarget), as this
        # not part of the official target model.
        self.target = target if target is not None and len(target.operation_names) > 0 else None
        if basis_gates is not None:
            self.basis_gates = set(basis_gates)
        self.force_consolidate = force_consolidate
        if kak_basis_gate is not None:
            self.decomposer = TwoQubitBasisDecomposer(kak_basis_gate)
            self.basis_gate_name = kak_basis_gate.name
        elif basis_gates is not None:
            kak_gates = KAK_GATE_NAMES.keys() & (basis_gates or [])
            kak_param_gates = KAK_GATE_PARAM_NAMES.keys() & (basis_gates or [])
            if kak_param_gates:
                self.decomposer = TwoQubitControlledUDecomposer(
                    KAK_GATE_PARAM_NAMES[list(kak_param_gates)[0]]
                )
                self.basis_gate_name = list(kak_param_gates)[0]
            elif kak_gates:
                self.decomposer = TwoQubitBasisDecomposer(
                    KAK_GATE_NAMES[list(kak_gates)[0]], basis_fidelity=approximation_degree or 1.0
                )
                self.basis_gate_name = list(kak_gates)[0]
            else:
                self.decomposer = None
        else:
            self.decomposer = TwoQubitBasisDecomposer(CXGate())
            self.basis_gate_name = "cx"

    def consolidate_blocks(self, dag):
        """Run the BlockConsolidator pass on `dag`.

        Iterate over each block and replace it with an equivalent Unitary
        on the same wires.
        """
        if self.decomposer is None:
            return dag

        # 如果property_set中没有block_list，自动收集
        if "block_list" not in self.property_set:
            # 自动收集2量子比特块
            from janus.optimize.passes.optimization.collect_2q_blocks import TwoQubitBlockCollector
            collector = TwoQubitBlockCollector()
            # 共享property_set
            collector.property_set = self.property_set
            collector.run(dag)

        blocks = self.property_set.get("block_list", None)
        if blocks is not None:
            blocks = [[node.node_id for node in block] for block in blocks]
        runs = self.property_set.get("run_list", None)
        if runs is not None:
            runs = [[node.node_id for node in run] for run in runs]

        qubit_map = self.property_set.get(self._QUBIT_MAP_KEY, None)
        if qubit_map is None:
            qubit_map = list(range(dag.num_qubits()))
        consolidate_blocks(
            dag,
            self.decomposer._inner_decomposer,
            self.basis_gate_name,
            self.force_consolidate,
            target=self.target,
            basis_gates=self.basis_gates,
            blocks=blocks,
            runs=runs,
            qubit_map=qubit_map,
        )
        dag = self._handle_control_flow_ops(dag, qubit_map)

        # Clear collected blocks and runs as they are no longer valid after consolidation
        if "run_list" in self.property_set:
            del self.property_set["run_list"]
        if "block_list" in self.property_set:
            del self.property_set["block_list"]

        return dag

    def _handle_control_flow_ops(self, dag, qubit_map):
        """
        This is similar to transpiler/passes/utils/control_flow.py except that the
        collect blocks is redone for the control flow blocks.
        """

        pass_manager = PassManager()
        if "run_list" in self.property_set:
            pass_manager.append(Collect1qRuns())
            pass_manager.append(Collect2qBlocks())
        pass_manager.append(self)

        for node in dag.control_flow_op_nodes():
            inner_qubit_map = [qubit_map[dag.find_bit(q).index] for q in node.qargs]
            new_op = node.op.replace_blocks(
                pass_manager.run(block, property_set={self._QUBIT_MAP_KEY: inner_qubit_map})
                for block in node.op.blocks
            )
            dag.substitute_node(node, new_op)
        return dag

    def run(self, dag):
        """Alias for consolidate_blocks() to maintain backward compatibility."""
        return self.consolidate_blocks(dag)


# Backward compatibility alias
ConsolidateBlocks = BlockConsolidator
