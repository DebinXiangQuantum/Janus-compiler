from __future__ import annotations
import numpy as np
from janus.circuit.circuit import Circuit
from janus.circuit.dag import DAGCircuit, circuit_to_dag
from janus.circuit.library.standard_gates import HGate, CXGate, TGate, TdgGate, ZGate
from .exceptions import UnsupportedMethodError


def decompose_multi_control_toffoli(
    num_ctrl_qubits: int,
    num_ancilla_qubits: int = None,
    ancilla_type: str = "clean",
    use_dag: bool = False,
    **kwargs
) -> Circuit:
    """
    分解多控制Toffoli门为基本门
    
    Args:
        num_ctrl_qubits: 控制量子比特数量
        num_ancilla_qubits: 辅助量子比特数量
        ancilla_type: 辅助量子比特类型
        use_dag: 是否返回DAGCircuit
        
    Returns:
        分解后的电路
    """
    # 创建电路
    n_qubits = num_ctrl_qubits + 1
    qc = Circuit(n_qubits=n_qubits)
    
    # 简单情况处理
    if num_ctrl_qubits == 0:
        qc.x(0)
    elif num_ctrl_qubits == 1:
        qc.cx(0, 1)
    elif num_ctrl_qubits == 2:
        # 3-qubit Toffoli gate
        qc.h(2)
        qc.cx(1, 2)
        qc.tdg(2)
        qc.cx(0, 2)
        qc.t(2)
        qc.cx(1, 2)
        qc.tdg(2)
        qc.cx(0, 2)
        qc.t(1)
        qc.t(2)
        qc.h(2)
        qc.cx(0, 1)
        qc.t(0)
        qc.tdg(1)
        qc.cx(0, 1)
    else:
        # 多控制情况的递归实现
        qubits = list(range(n_qubits))
        controls = qubits[:num_ctrl_qubits]
        target = qubits[-1]
        
        # 递归构建多控制Toffoli门
        _recursive_mct(qc, controls, target)
    
    if use_dag:
        return circuit_to_dag(qc)
    return qc


def _recursive_mct(qc: Circuit, controls: list[int], target: int):
    """
    递归构建多控制Toffoli门
    
    Args:
        qc: 电路对象
        controls: 控制量子比特列表
        target: 目标量子比特
    """
    n = len(controls)
    
    if n == 2:
        # 3-qubit Toffoli gate
        qc.h(target)
        qc.cx(controls[1], target)
        qc.tdg(target)
        qc.cx(controls[0], target)
        qc.t(target)
        qc.cx(controls[1], target)
        qc.tdg(target)
        qc.cx(controls[0], target)
        qc.t(controls[1])
        qc.t(target)
        qc.h(target)
        qc.cx(controls[0], controls[1])
        qc.t(controls[0])
        qc.tdg(controls[1])
        qc.cx(controls[0], controls[1])
    else:
        # 创建一个新的目标比特（使用倒数第二个控制比特）
        new_target = controls[-1]
        new_controls = controls[:-1]
        
        # 递归构建(n-1)-控制的Toffoli门
        _recursive_mct(qc, new_controls, new_target)
        
        # 应用当前层的3-qubit Toffoli门
        qc.h(target)
        qc.cx(new_target, target)
        qc.tdg(target)
        qc.cx(new_controls[-1], target)
        qc.t(target)
        qc.cx(new_target, target)
        qc.tdg(target)
        qc.cx(new_controls[-1], target)
        qc.t(new_target)
        qc.t(target)
        qc.h(target)
        qc.cx(new_controls[-1], new_target)
        qc.t(new_controls[-1])
        qc.tdg(new_target)
        qc.cx(new_controls[-1], new_target)
        
        # 再次递归构建(n-1)-控制的Toffoli门
        _recursive_mct(qc, new_controls, new_target)