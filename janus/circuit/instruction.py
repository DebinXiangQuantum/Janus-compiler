"""
Janus 电路指令

表示电路中的一条指令，包含操作和作用的量子比特
"""
from typing import List, Tuple, Optional, Union
from .gate import Gate


class Instruction:
    """
    电路指令类
    
    将一个操作（Gate）与具体的量子比特绑定
    
    Attributes:
        operation: 量子操作（门）
        qubits: 作用的量子比特索引
        clbits: 作用的经典比特索引（用于测量等）
    """
    
    def __init__(
        self,
        operation: Gate,
        qubits: List[int],
        clbits: Optional[List[int]] = None
    ):
        self._operation = operation
        self._qubits = qubits
        self._clbits = clbits if clbits is not None else []
        
        # 同步设置 operation 的 qubits
        operation.qubits = qubits
    
    @property
    def operation(self) -> Gate:
        return self._operation
    
    @property
    def qubits(self) -> List[int]:
        return self._qubits
    
    @property
    def clbits(self) -> List[int]:
        return self._clbits
    
    @property
    def name(self) -> str:
        return self._operation.name
    
    @property
    def params(self) -> List[float]:
        return self._operation.params
    
    def copy(self) -> 'Instruction':
        """创建指令的副本"""
        return Instruction(
            self._operation.copy(),
            self._qubits.copy(),
            self._clbits.copy()
        )
    
    def __repr__(self) -> str:
        return f"Instruction({self._operation}, qubits={self._qubits})"
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'name': self._operation.name,
            'qubits': self._qubits,
            'params': self._operation.params
        }
