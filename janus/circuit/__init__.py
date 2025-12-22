"""
Janus 量子电路模块

提供量子电路的构建、操作和表示
"""
from .operation import Operation
from .gate import Gate, ControlledGate
from .instruction import Instruction
from .layer import Layer
from .circuit import Circuit, SeperatableCircuit
from .qubit import Qubit, QuantumRegister
from .clbit import Clbit, ClassicalRegister
from .parameter import Parameter, ParameterExpression
from .dag import (
    DAGCircuit, DAGNode, DAGOpNode, DAGInNode, DAGOutNode,
    circuit_to_dag, dag_to_circuit,
    DAGDependency, circuit_to_dag_dependency, dag_dependency_to_circuit,
    BlockCollector, BlockSplitter, BlockCollapser, split_block_into_layers
)

# 标准门
from .library import (
    # 单比特 Pauli 门
    IGate,
    XGate,
    YGate,
    ZGate,
    # Hadamard 和 Clifford 门
    HGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    SXGate,
    SXdgGate,
    # 单比特旋转门
    RXGate,
    RYGate,
    RZGate,
    PhaseGate,
    U1Gate,
    U2Gate,
    U3Gate,
    UGate,
    RGate,
    # 两比特旋转门
    RXXGate,
    RYYGate,
    RZZGate,
    RZXGate,
    # 两比特门
    CXGate,
    CYGate,
    CZGate,
    CHGate,
    CSGate,
    CSdgGate,
    CSXGate,
    DCXGate,
    ECRGate,
    SwapGate,
    iSwapGate,
    # 受控旋转门
    CRXGate,
    CRYGate,
    CRZGate,
    CPhaseGate,
    CU1Gate,
    CU3Gate,
    CUGate,
    # 三比特及多比特门
    CCXGate,
    CCZGate,
    CSwapGate,
    RCCXGate,
    RC3XGate,
    C3XGate,
    C4XGate,
    # 特殊两比特门
    XXMinusYYGate,
    XXPlusYYGate,
    # 特殊操作
    Barrier,
    Measure,
    Reset,
    Delay,
    GlobalPhaseGate,
    # 多控制门
    C3SXGate,
    MCXGate,
    MCXGrayCode,
    MCXRecursive,
    MCXVChain,
    MCPhaseGate,
    MCU1Gate,
    # 多控制旋转门
    MCRXGate,
    MCRYGate,
    MCRZGate,
)

__all__ = [
    # 核心类
    'Operation',
    'Gate',
    'ControlledGate',
    'Instruction',
    'Layer',
    'Circuit',
    'SeperatableCircuit',
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
    'DAGOpNode',
    'DAGInNode',
    'DAGOutNode',
    'circuit_to_dag',
    'dag_to_circuit',
    # DAGDependency (交换性分析)
    'DAGDependency',
    'circuit_to_dag_dependency',
    'dag_dependency_to_circuit',
    # 块操作
    'BlockCollector',
    'BlockSplitter',
    'BlockCollapser',
    'split_block_into_layers',
    # 单比特 Pauli 门
    'IGate',
    'XGate',
    'YGate',
    'ZGate',
    # Hadamard 和 Clifford 门
    'HGate',
    'SGate',
    'SdgGate',
    'TGate',
    'TdgGate',
    'SXGate',
    'SXdgGate',
    # 单比特旋转门
    'RXGate',
    'RYGate',
    'RZGate',
    'PhaseGate',
    'U1Gate',
    'U2Gate',
    'U3Gate',
    'UGate',
    'RGate',
    # 两比特旋转门
    'RXXGate',
    'RYYGate',
    'RZZGate',
    'RZXGate',
    # 两比特门
    'CXGate',
    'CYGate',
    'CZGate',
    'CHGate',
    'CSGate',
    'CSdgGate',
    'CSXGate',
    'DCXGate',
    'ECRGate',
    'SwapGate',
    'iSwapGate',
    # 受控旋转门
    'CRXGate',
    'CRYGate',
    'CRZGate',
    'CPhaseGate',
    'CU1Gate',
    'CU3Gate',
    'CUGate',
    # 三比特及多比特门
    'CCXGate',
    'CCZGate',
    'CSwapGate',
    'RCCXGate',
    'RC3XGate',
    'C3XGate',
    'C4XGate',
    # 特殊两比特门
    'XXMinusYYGate',
    'XXPlusYYGate',
    # 特殊操作
    'Barrier',
    'Measure',
    'Reset',
    'Delay',
    'GlobalPhaseGate',
    # 多控制门
    'C3SXGate',
    'MCXGate',
    'MCXGrayCode',
    'MCXRecursive',
    'MCXVChain',
    'MCPhaseGate',
    'MCU1Gate',
    # 多控制旋转门
    'MCRXGate',
    'MCRYGate',
    'MCRZGate',
]
