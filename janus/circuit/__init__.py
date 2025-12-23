"""
Janus 量子电路模块

提供量子电路的构建、操作和表示
"""
from .operation import Operation
from .gate import Gate
from .instruction import Instruction
from .layer import Layer
from .circuit import Circuit
# Alias for Qiskit compatibility
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
]
