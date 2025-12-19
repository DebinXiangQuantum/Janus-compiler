from __future__ import annotations
from typing import Optional
import numpy as np

from ..circuit.circuit import Circuit
from ..circuit.dag import DAGCircuit, circuit_to_dag
from ..circuit.gate import Gate
from ..circuit.library.standard_gates import XGate, YGate, ZGate, RXGate, RYGate, RZGate, CXGate, HGate, CZGate, CRZGate, SGate
from .exceptions import UnsupportedMethodError, CircuitError
from .decompose_multi_control_toffoli import decompose_multi_control_toffoli

def decompose_controlled_gate(
    gate: Gate,
    num_ctrl_qubits: int = 1,
    num_ancilla_qubits: int = 0,
    ancilla_type: str = "clean",
    use_dag: bool = False
) -> Circuit:
    """
    分解受控量子门
    
    Args:
        gate: 要分解的门
        num_ctrl_qubits: 控制量子比特数量
        num_ancilla_qubits: 辅助量子比特数量
        ancilla_type: 辅助量子比特类型（"clean" 或 "dirty"）
        use_dag: 是否返回DAGCircuit
        
    Returns:
        分解后的电路
    """

    # 如果是 X 门，使用多控制 Toffoli 分解
    if gate.name == "x":
        result = decompose_multi_control_toffoli(num_ctrl_qubits)
    else:
        # 其他门的分解
        result = _decompose_general_controlled_gate(gate, num_ctrl_qubits)
    
    if use_dag:
        return circuit_to_dag(result)
    return result

def _decompose_general_controlled_gate(gate: Gate, num_ctrl_qubits: int) -> Circuit:
    """
    分解一般的受控门
    
    Args:
        gate: 要分解的门
        num_ctrl_qubits: 控制量子比特数量
        
    Returns:
        分解后的电路
    """
    num_qubits = num_ctrl_qubits + 1
    circuit = Circuit(n_qubits=num_qubits)
    
    # 将受控门分解为单比特旋转和 CNOT 门的组合
    # 这里只处理旋转门的情况
    if gate.name in ["rx", "ry", "rz"]:
        # 对于旋转门，我们可以使用 Euler 分解和 CNOT 门来构建受控版本
        target = num_qubits - 1
        
        if gate.name == "rx":
            # RX 旋转的受控版本
            # 分解为 H, CRZ, H 的组合
            circuit.h(target)
            circuit = _controlled_rz_decomposition(circuit, gate.params[0], list(range(num_ctrl_qubits)), target)
            circuit.h(target)
        elif gate.name == "ry":
            # RY 旋转的受控版本
            # 分解为 Sdg, H, CRZ, H, S 的组合
            circuit.sdg(target)
            circuit.h(target)
            circuit = _controlled_rz_decomposition(circuit, gate.params[0], list(range(num_ctrl_qubits)), target)
            circuit.h(target)
            circuit.s(target)
        elif gate.name == "rz":
            # RZ 旋转的受控版本
            circuit = _controlled_rz_decomposition(circuit, gate.params[0], list(range(num_ctrl_qubits)), target)
    else:
        # 对于其他门，使用通用方法
        target = num_qubits - 1
        
        if gate.name == "y":
            # 多控制 Y 门分解：Y = iXZ，所以受控 Y = i受控 X 受控 Z
            # 这里简化处理，先转换为 X 门分解
            # 在目标比特上应用 H 和 S 门，将 Y 转换为 X
            circuit.h(target)
            circuit.s(target)
            # 使用多控制 Toffoli 分解 X 门
            temp_circuit = decompose_multi_control_toffoli(num_ctrl_qubits)
            # 将temp_circuit中的所有指令添加到circuit中
            for inst in temp_circuit.instructions:
                circuit.append(inst.operation, inst.qubits)
            # 转换回 Y 门的效果
            circuit.sdg(target)
            circuit.h(target)
        elif gate.name == "z":
            # 多控制 Z 门分解：使用 H 门将 Z 转换为 X，然后使用 Toffoli 门
            # 在目标比特上应用 H 门
            circuit.h(target)
            # 使用多控制 Toffoli 分解 X 门
            temp_circuit = decompose_multi_control_toffoli(num_ctrl_qubits)
            # 将temp_circuit中的所有指令添加到circuit中
            for inst in temp_circuit.instructions:
                circuit.append(inst.operation, inst.qubits)
            # 转换回 Z 门的效果
            circuit.h(target)
    
    return circuit

def _controlled_rz_decomposition(circuit: Circuit, theta: float, controls: list, target: int) -> Circuit:
    """
    分解多控制 RZ 门
    
    Args:
        circuit: 要添加分解门的电路
        theta: 旋转角度
        controls: 控制量子比特列表
        target: 目标量子比特
        
    Returns:
        更新后的电路
    """
    n = len(controls)
    
    if n == 1:
        # 单控制 RZ 门
        circuit.crz(theta, controls[0], target)
    else:
        # 多控制 RZ 门分解为多个单控制和双控制门
        # 使用递归方法
        mid = n // 2
        new_theta = theta / (2 ** (n - mid))
        
        # 分解为第一部分控制
        _controlled_rz_decomposition(circuit, new_theta, controls[:mid], target)
        
        # 分解为第二部分控制
        _controlled_rz_decomposition(circuit, new_theta, controls[mid:], target)
    
    return circuit