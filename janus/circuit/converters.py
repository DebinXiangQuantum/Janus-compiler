"""
Janus 电路转换器

提供电路数组格式转换功能，支持所有标准门
"""
from heapq import merge
from typing import Optional

from .circuit import Circuit
from .gate import Gate
from .library.standard_gates import (
    # 单比特 Pauli 门
    IGate, XGate, YGate, ZGate,
    # Clifford 门
    HGate, SGate, SdgGate, TGate, TdgGate, SXGate, SXdgGate,
    # 单比特旋转门
    RXGate, RYGate, RZGate, PhaseGate, U1Gate, U2Gate, U3Gate, UGate, RGate,
    # 两比特旋转门
    RXXGate, RYYGate, RZZGate, RZXGate,
    # 两比特门
    CXGate, CYGate, CZGate, CHGate, CSGate, CSdgGate, CSXGate,
    DCXGate, ECRGate, SwapGate, iSwapGate,
    # 受控旋转门
    CRXGate, CRYGate, CRZGate, CPhaseGate, CU1Gate, CU3Gate, CUGate,
    # 三比特及多比特门
    CCXGate, CCZGate, CSwapGate, RCCXGate, RC3XGate, C3XGate, C4XGate, C3SXGate,
    # 多控制门
    MCXGate, MCXGrayCode, MCXRecursive, MCXVChain,
    MCPhaseGate, MCU1Gate, MCRXGate, MCRYGate, MCRZGate,
    # 特殊门
    XXMinusYYGate, XXPlusYYGate, GlobalPhaseGate,
    # 特殊操作
    Barrier, Measure, Reset, Delay,
)


# 无参数门映射
NO_PARAM_GATES = {
    'id': IGate,
    'x': XGate,
    'y': YGate,
    'z': ZGate,
    'h': HGate,
    's': SGate,
    'sdg': SdgGate,
    't': TGate,
    'tdg': TdgGate,
    'sx': SXGate,
    'sxdg': SXdgGate,
    'cx': CXGate,
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
    'ccx': CCXGate,
    'ccz': CCZGate,
    'cswap': CSwapGate,
    'rccx': RCCXGate,
    'rc3x': RC3XGate,
    'c3x': C3XGate,
    'c4x': C4XGate,
    'c3sx': C3SXGate,
    'measure': Measure,
    'reset': Reset,
}

# 单参数门映射 (theta)
SINGLE_PARAM_GATES = {
    'rx': RXGate,
    'ry': RYGate,
    'rz': RZGate,
    'p': PhaseGate,
    'u1': U1Gate,
    'crx': CRXGate,
    'cry': CRYGate,
    'crz': CRZGate,
    'cp': CPhaseGate,
    'cu1': CU1Gate,
    'rxx': RXXGate,
    'ryy': RYYGate,
    'rzz': RZZGate,
    'rzx': RZXGate,
    'global_phase': GlobalPhaseGate,
}

# 双参数门映射
TWO_PARAM_GATES = {
    'u2': U2Gate,
    'r': RGate,
    'xx_minus_yy': XXMinusYYGate,
    'xx_plus_yy': XXPlusYYGate,
}

# 三参数门映射
THREE_PARAM_GATES = {
    'u': UGate,
    'u3': U3Gate,
    'cu3': CU3Gate,
}

# 四参数门映射
FOUR_PARAM_GATES = {
    'cu': CUGate,
}

GATE_MAP = {
    **NO_PARAM_GATES,
    **SINGLE_PARAM_GATES,
    **TWO_PARAM_GATES,
    **THREE_PARAM_GATES,
    **FOUR_PARAM_GATES,
}

def _create_gate(name: str, params: list, qubits: list = None) -> Optional[Gate]:
    """根据名称和参数创建门
    
    Args:
        name: 门名称
        params: 参数列表
        qubits: 量子比特列表（用于推断多控制门的控制比特数）
    
    Returns:
        Gate 实例，如果门不支持则返回 None
    """
    name = name.lower()
    
    # 无参数门
    if name in NO_PARAM_GATES:
        gate_class = NO_PARAM_GATES[name]
        # Barrier 需要特殊处理
        if name == 'barrier' and qubits:
            return Barrier(len(qubits))
        return gate_class()
    
    # 单参数门
    if name in SINGLE_PARAM_GATES:
        if not params:
            print(f"Warning: Gate '{name}' requires 1 parameter, got 0")
            return None
        return SINGLE_PARAM_GATES[name](params[0])
    
    # 双参数门
    if name in TWO_PARAM_GATES:
        if len(params) < 2:
            print(f"Warning: Gate '{name}' requires 2 parameters, got {len(params)}")
            return None
        return TWO_PARAM_GATES[name](params[0], params[1])
    
    # 三参数门
    if name in THREE_PARAM_GATES:
        if len(params) < 3:
            print(f"Warning: Gate '{name}' requires 3 parameters, got {len(params)}")
            return None
        return THREE_PARAM_GATES[name](params[0], params[1], params[2])
    
    # 四参数门
    if name in FOUR_PARAM_GATES:
        if len(params) < 4:
            print(f"Warning: Gate '{name}' requires 4 parameters, got {len(params)}")
            return None
        return FOUR_PARAM_GATES[name](params[0], params[1], params[2], params[3])
    
    # 多控制门（需要根据 qubits 推断控制比特数）
    if name in ('mcx', 'mcx_gray', 'mcx_recursive', 'mcx_vchain'):
        if qubits is None or len(qubits) < 2:
            print(f"Warning: Gate '{name}' requires at least 2 qubits")
            return None
        num_ctrl = len(qubits) - 1
        if name == 'mcx':
            return MCXGate(num_ctrl)
        elif name == 'mcx_gray':
            return MCXGrayCode(num_ctrl)
        elif name == 'mcx_recursive':
            return MCXRecursive(num_ctrl)
        elif name == 'mcx_vchain':
            return MCXVChain(num_ctrl)
    
    if name in ('mcp', 'mcphase'):
        if qubits is None or len(qubits) < 2:
            print(f"Warning: Gate '{name}' requires at least 2 qubits")
            return None
        if not params:
            print(f"Warning: Gate '{name}' requires 1 parameter")
            return None
        num_ctrl = len(qubits) - 1
        return MCPhaseGate(params[0], num_ctrl)
    
    if name == 'mcu1':
        if qubits is None or len(qubits) < 2:
            print(f"Warning: Gate '{name}' requires at least 2 qubits")
            return None
        if not params:
            print(f"Warning: Gate '{name}' requires 1 parameter")
            return None
        num_ctrl = len(qubits) - 1
        return MCU1Gate(params[0], num_ctrl)
    
    if name == 'mcrx':
        if qubits is None or len(qubits) < 2:
            print(f"Warning: Gate '{name}' requires at least 2 qubits")
            return None
        if not params:
            print(f"Warning: Gate '{name}' requires 1 parameter")
            return None
        num_ctrl = len(qubits) - 1
        return MCRXGate(params[0], num_ctrl)
    
    if name == 'mcry':
        if qubits is None or len(qubits) < 2:
            print(f"Warning: Gate '{name}' requires at least 2 qubits")
            return None
        if not params:
            print(f"Warning: Gate '{name}' requires 1 parameter")
            return None
        num_ctrl = len(qubits) - 1
        return MCRYGate(params[0], num_ctrl)
    
    if name == 'mcrz':
        if qubits is None or len(qubits) < 2:
            print(f"Warning: Gate '{name}' requires at least 2 qubits")
            return None
        if not params:
            print(f"Warning: Gate '{name}' requires 1 parameter")
            return None
        num_ctrl = len(qubits) - 1
        return MCRZGate(params[0], num_ctrl)
    
    # Barrier 特殊处理
    if name == 'barrier':
        num_qubits = len(qubits) if qubits else 1
        return Barrier(num_qubits)
    
    # Delay 特殊处理
    if name == 'delay':
        if not params:
            print(f"Warning: Gate 'delay' requires duration parameter")
            return None
        return Delay(params[0])
    
    print(f"Warning: Unknown gate '{name}', skipping")
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


def from_instruction_list(instructions: list, n_qubits: int = None, n_clbits: int = None) -> Circuit:
    """
    从指令数组创建 Janus Circuit
    
    Args:
        instructions: [(name, qubits, params), ...] 或 [{'name':..., 'qubits':..., 'params':..., 'clbits':...}, ...]
        n_qubits: 量子比特数，如果不指定则自动推断
        n_clbits: 经典比特数，如果不指定则自动推断（根据测量门）
    
    Returns:
        Janus Circuit
    """
    # 自动推断量子比特数和经典比特数
    max_qubit = 0
    max_clbit = -1
    
    for inst in instructions:
        if isinstance(inst, dict):
            qubits = inst['qubits']
            clbits = inst.get('clbits', [])
        else:
            qubits = inst[1]
            clbits = inst[3] if len(inst) > 3 else []
        
        if qubits:
            max_qubit = max(max_qubit, max(qubits))
        if clbits:
            max_clbit = max(max_clbit, max(clbits))
    
    if n_qubits is None:
        n_qubits = max_qubit + 1
    
    if n_clbits is None:
        n_clbits = max_clbit + 1 if max_clbit >= 0 else 0
    
    circuit = Circuit(n_qubits, n_clbits=n_clbits)
    
    for inst in instructions:
        # 支持两种格式
        if isinstance(inst, dict):
            name = inst['name']
            qubits = inst['qubits']
            params = inst.get('params', [])
            clbits = inst.get('clbits', [])
        else:
            name = inst[0]
            qubits = inst[1]
            params = inst[2] if len(inst) > 2 else []
            clbits = inst[3] if len(inst) > 3 else []
        
        gate = _create_gate(name, params, qubits)
        if gate is not None:
            circuit.append(gate, qubits, clbits if clbits else None)
    
    return circuit
