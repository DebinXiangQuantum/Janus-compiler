"""
Janus 电路转换器

提供电路数组格式转换功能
"""
from typing import Optional

from .circuit import Circuit
from .gate import Gate
from .library.standard_gates import (
    HGate, XGate, YGate, ZGate, SGate, TGate,
    RXGate, RYGate, RZGate, UGate,
    CXGate, CZGate, CRZGate, SwapGate
)


# 门名称到类的映射
GATE_MAP = {
    'h': HGate,
    'x': XGate,
    'y': YGate,
    'z': ZGate,
    's': SGate,
    't': TGate,
    'rx': RXGate,
    'ry': RYGate,
    'rz': RZGate,
    'u': UGate,
    'u3': UGate,  # u3 等价于 u
    'cx': CXGate,
    'cz': CZGate,
    'crz': CRZGate,
    'swap': SwapGate,
}


def _create_gate(name: str, params: list) -> Optional[Gate]:
    """根据名称和参数创建门"""
    name = name.lower()
    
    if name == 'u3':
        name = 'u'
    
    if name not in GATE_MAP:
        print(f"Warning: Unknown gate '{name}', skipping")
        return None
    
    gate_class = GATE_MAP[name]
    
    # 根据门类型创建实例
    if name in ('h', 'x', 'y', 'z', 's', 't', 'cx', 'cz', 'swap'):
        return gate_class()
    elif name in ('rx', 'ry', 'rz', 'crz'):
        return gate_class(params[0])
    elif name == 'u':
        return gate_class(params[0], params[1], params[2])
    
    return None


def to_instruction_list(circuit: Circuit) -> list:
    """
    将 Janus Circuit 转换为指令数组 (元组格式)
    
    Args:
        circuit: Janus 电路
    
    Returns:
        [(name, qubits, params), ...] 格式的列表
    """
    return [(inst.name, inst.qubits, inst.params) for inst in circuit.instructions]


def from_instruction_list(instructions: list, n_qubits: int = None) -> Circuit:
    """
    从指令数组创建 Janus Circuit
    
    Args:
        instructions: [(name, qubits, params), ...] 或 [{'name':..., 'qubits':..., 'params':...}, ...]
        n_qubits: 量子比特数，如果不指定则自动推断
    
    Returns:
        Janus Circuit
    """
    # 自动推断量子比特数
    if n_qubits is None:
        max_qubit = 0
        for inst in instructions:
            if isinstance(inst, dict):
                qubits = inst['qubits']
            else:
                qubits = inst[1]
            if qubits:
                max_qubit = max(max_qubit, max(qubits))
        n_qubits = max_qubit + 1
    
    circuit = Circuit(n_qubits)
    
    for inst in instructions:
        # 支持两种格式
        if isinstance(inst, dict):
            name = inst['name']
            qubits = inst['qubits']
            params = inst.get('params', [])
        else:
            name, qubits, params = inst
        
        gate = _create_gate(name, params)
        if gate is not None:
            circuit.append(gate, qubits)
    
    return circuit
