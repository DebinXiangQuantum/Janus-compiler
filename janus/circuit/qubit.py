"""
Janus 量子比特定义
"""
from typing import Optional


class Qubit:
    """
    量子比特类
    
    Attributes:
        index: 量子比特在电路中的索引
        register: 所属的量子寄存器（可选）
    """
    
    def __init__(self, index: int, register: Optional['QuantumRegister'] = None):
        self._index = index
        self._register = register
    
    @property
    def index(self) -> int:
        return self._index
    
    @property
    def register(self) -> Optional['QuantumRegister']:
        return self._register
    
    def __repr__(self) -> str:
        if self._register:
            return f"{self._register.name}[{self._index}]"
        return f"Qubit({self._index})"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Qubit):
            return self._index == other._index
        return False
    
    def __hash__(self) -> int:
        return hash(self._index)


class QuantumRegister:
    """
    量子寄存器 - 一组量子比特的集合
    
    Attributes:
        size: 寄存器中量子比特的数量
        name: 寄存器名称
    """
    
    def __init__(self, size: int, name: str = "q"):
        self._size = size
        self._name = name
        self._qubits = [Qubit(i, self) for i in range(size)]
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def name(self) -> str:
        return self._name
    
    def __len__(self) -> int:
        return self._size
    
    def __getitem__(self, index: int) -> Qubit:
        if index < 0 or index >= self._size:
            raise IndexError(f"Qubit index {index} out of range [0, {self._size})")
        return self._qubits[index]
    
    def __iter__(self):
        return iter(self._qubits)
    
    def __repr__(self) -> str:
        return f"QuantumRegister({self._size}, '{self._name}')"
