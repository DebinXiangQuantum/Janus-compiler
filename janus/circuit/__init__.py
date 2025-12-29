"""
Janus 量子电路模块

提供量子电路的构建、操作和表示
"""
import json
import os
from pathlib import Path

from .operation import Operation
from .gate import Gate
from .instruction import Instruction
from .layer import Layer
from .circuit import Circuit
# Alias for compatibility
QuantumCircuit = Circuit

from .qubit import Qubit, QuantumRegister
from .clbit import Clbit, ClassicalRegister
from .parameter import Parameter, ParameterExpression
from .dag import DAGCircuit, DAGNode, DAGOpNode, DAGInNode, DAGOutNode, circuit_to_dag, dag_to_circuit

# Compatibility stubs
class CircuitInstruction:
    """Stub for CircuitInstruction - represents a gate application in a circuit."""
    def __init__(self, operation, qubits=None, clbits=None):
        self.operation = operation
        self.qubits = qubits or []
        self.clbits = clbits or []

# Bit is an alias for base bit class
class Bit:
    """Stub for Bit class."""
    def __init__(self, index=None):
        self.index = index

# 标准门
from .library import (
    HGate,
    XGate,
    YGate,
    ZGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    RXGate,
    RYGate,
    RZGate,
    UGate,
    CXGate,
    CZGate,
    CRZGate,
    SwapGate,
    Barrier,
    Measure,
    Reset,
    Delay,
)

__all__ = [
    # 核心类
    'Operation',
    'Gate',
    'Instruction',
    'Layer',
    'Circuit',
    'QuantumCircuit',  # Alias for Circuit
    'Qubit',
    'QuantumRegister',
    'Clbit',
    'ClassicalRegister',
    'CircuitInstruction',  # Added
    'Bit',  # Added
    # 参数化
    'Parameter',
    'ParameterExpression',
    # DAG
    'DAGCircuit',
    'DAGNode',
    'DAGOpNode',
    'DAGInNode',
    'DAGOutNode',
    'circuit_to_dag',
    'dag_to_circuit',
    # 标准门
    'HGate',
    'XGate',
    'YGate',
    'ZGate',
    'SGate',
    'SdgGate',
    'TGate',
    'TdgGate',
    'RXGate',
    'RYGate',
    'RZGate',
    'UGate',
    'CXGate',
    'CZGate',
    'CRZGate',
    'SwapGate',
    'Barrier',
    'Measure',
    'Reset',
    'Delay',
    # 电路文件工具
    'load_circuit',
    'save_circuit',
    'list_circuits',
]


# 电路文件加载工具
def _get_circuits_dir():
    """获取默认电路目录"""
    return Path(__file__).parent.parent.parent / 'circuits'


def list_circuits(directory=None):
    """
    列出目录中所有可用的电路文件
    
    Args:
        directory: 电路目录路径，默认为 janus 包的 circuits 目录
        
    Returns:
        list: 电路名称列表（不含 .json 后缀）
    """
    if directory is None:
        directory = _get_circuits_dir()
    else:
        directory = Path(directory)
    
    if not directory.exists():
        return []
    
    circuits = []
    for f in directory.glob('*.json'):
        circuits.append(f.stem)
    return sorted(circuits)


def load_circuit(name=None, filepath=None):
    """
    从 JSON 文件加载电路
    
    Args:
        name: 电路名称（不含 .json 后缀），从默认目录加载
        filepath: 完整文件路径，优先于 name
        
    Returns:
        Circuit: 加载的电路对象
        
    Examples:
        >>> qc = load_circuit(name='bell')
        >>> qc = load_circuit(filepath='./my_circuit.json')
    """
    if filepath is not None:
        path = Path(filepath)
    elif name is not None:
        path = _get_circuits_dir() / f'{name}.json'
    else:
        raise ValueError("必须提供 name 或 filepath 参数")
    
    if not path.exists():
        raise FileNotFoundError(f"电路文件不存在: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 支持两种格式：
    # 1. {'n_qubits': N, 'layers': [...]}
    # 2. 直接是 layers 列表
    if isinstance(data, dict):
        layers = data.get('layers', [])
        n_qubits = data.get('n_qubits')
    else:
        layers = data
        n_qubits = None
    
    return Circuit.from_layers(layers, n_qubits=n_qubits)


def save_circuit(circuit, filepath):
    """
    将电路保存为 JSON 文件
    
    Args:
        circuit: Circuit 对象
        filepath: 保存路径
    """
    data = {
        'n_qubits': circuit.n_qubits,
        'layers': circuit.to_layers()
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
