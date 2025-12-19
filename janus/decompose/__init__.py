"""
Janus 分解模块

该模块提供了各种量子门和电路的分解功能，包括：
- 单量子比特门分解
- 两量子比特门分解
- 多控制 Toffoli 门分解
- 电路到指定指令集的转换
- KAK 分解
"""

# 导入统一的错误类型
from .exceptions import (
    DecomposeError,
    UnsupportedMethodError,
    GateNotSupportedError,
    ParameterError,
    CircuitError
)

# 导入分解函数
from .decompose_one_qubit import decompose_one_qubit, OneQubitEulerDecomposer
from .decompose_two_qubit_gate import decompose_two_qubit_gate
from .decompose_multi_control_toffoli import decompose_multi_control_toffoli
from .decompose_controlled_gate import decompose_controlled_gate
from .decompose_kak import decompose_kak
from .convert_circuit_to_instruction_set import convert_circuit_to_instruction_set

# 导出所有公开API
__all__ = [
    # 错误类型
    "DecomposeError",
    "UnsupportedMethodError",
    "GateNotSupportedError",
    "ParameterError",
    "CircuitError",
    
    # 分解函数
    "decompose_one_qubit",
    "OneQubitEulerDecomposer",
    "decompose_two_qubit_gate",
    "decompose_multi_control_toffoli",
    "decompose_controlled_gate",
    "decompose_kak",
    "convert_circuit_to_instruction_set"
]
