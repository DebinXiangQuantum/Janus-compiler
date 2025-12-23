"""
Janus 标准量子门库

"""
from typing import Optional, Type
from .standard_gates import (
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

from .qft import QFT

__all__ = [
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
    # QFT
    'QFT',
    # 工具函数
    'get_gate_class',
]


# 门名称到类的映射
_GATE_MAP = {
    # 单比特 Pauli 门
    'id': IGate, 'i': IGate,
    'x': XGate,
    'y': YGate,
    'z': ZGate,
    # Hadamard 和 Clifford 门
    'h': HGate,
    's': SGate,
    'sdg': SdgGate,
    't': TGate,
    'tdg': TdgGate,
    'sx': SXGate,
    'sxdg': SXdgGate,
    # 单比特旋转门
    'rx': RXGate,
    'ry': RYGate,
    'rz': RZGate,
    'p': PhaseGate, 'phase': PhaseGate,
    'u1': U1Gate,
    'u2': U2Gate,
    'u3': U3Gate,
    'u': UGate,
    'r': RGate,
    # 两比特旋转门
    'rxx': RXXGate,
    'ryy': RYYGate,
    'rzz': RZZGate,
    'rzx': RZXGate,
    # 两比特门
    'cx': CXGate, 'cnot': CXGate,
    'cy': CYGate,
    'cz': CZGate,
    'ch': CHGate,
    'cs': CSGate,
    'csdg': CSdgGate,
    'csx': CSXGate,
    'dcx': DCXGate,
    'ecr': ECRGate,
    'swap': SwapGate,
    'iswap': iSwapGate,
    # 受控旋转门
    'crx': CRXGate,
    'cry': CRYGate,
    'crz': CRZGate,
    'cp': CPhaseGate, 'cphase': CPhaseGate,
    'cu1': CU1Gate,
    'cu3': CU3Gate,
    'cu': CUGate,
    # 三比特及多比特门
    'ccx': CCXGate, 'toffoli': CCXGate,
    'ccz': CCZGate,
    'cswap': CSwapGate, 'fredkin': CSwapGate,
    'rccx': RCCXGate,
    'rc3x': RC3XGate,
    'c3x': C3XGate,
    'c4x': C4XGate,
    # 特殊两比特门
    'xx_minus_yy': XXMinusYYGate,
    'xx_plus_yy': XXPlusYYGate,
    # 特殊操作
    'barrier': Barrier,
    'measure': Measure,
    'reset': Reset,
    'delay': Delay,
    'global_phase': GlobalPhaseGate,
    # 多控制门
    'c3sx': C3SXGate,
    'mcx': MCXGate,
    'mcx_gray': MCXGrayCode,
    'mcx_recursive': MCXRecursive,
    'mcx_vchain': MCXVChain,
    'mcp': MCPhaseGate, 'mcphase': MCPhaseGate,
    'mcu1': MCU1Gate,
    # 多控制旋转门
    'mcrx': MCRXGate,
    'mcry': MCRYGate,
    'mcrz': MCRZGate,
}


def get_gate_class(name: str) -> Optional[Type]:
    """
    根据门名称获取对应的门类
    
    Args:
        name: 门名称（如 'rx', 'cx', 'h' 等）
    
    Returns:
        对应的门类，如果不存在则返回 None
    
    Example:
        gate_cls = get_gate_class('rx')
        gate = gate_cls(3.14)  # 创建 RX(3.14) 门
    """
    return _GATE_MAP.get(name.lower())
