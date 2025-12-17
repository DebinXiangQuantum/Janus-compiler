"""
Janus 量子电路模块

提供量子电路的构建、操作和表示
"""
from .operation import Operation
from .gate import Gate
from .instruction import Instruction
from .layer import Layer
from .circuit import Circuit
from .qubit import Qubit, QuantumRegister
from .clbit import Clbit, ClassicalRegister
from .parameter import Parameter, ParameterExpression
from .dag import DAGCircuit, DAGNode, circuit_to_dag, dag_to_circuit

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
)

__all__ = [
    # 核心类
    'Operation',
    'Gate',
    'Instruction',
    'Layer',
    'Circuit',
    'Qubit',
    'QuantumRegister',
    'Clbit',
    'ClassicalRegister',
    # 参数化
    'Parameter',
    'ParameterExpression',
    # DAG
    'DAGCircuit',
    'DAGNode',
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
]
