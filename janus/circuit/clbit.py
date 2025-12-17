"""
Janus 经典比特定义
"""
from typing import Optional, List


class Clbit:
    """
    经典比特类
    
    Attributes:
        index: 经典比特在电路中的索引
        register: 所属的经典寄存器（可选）
    """
    
    def __init__(self, index: int, register: Optional['ClassicalRegister'] = None):
        self._index = index
        self._register = register
    
    @property
    def index(self) -> int:
        return self._index
    
    @property
    def register(self) -> Optional['ClassicalRegister']:
        return self._register
    
    def __repr__(self) -> str:
        if self._register:
            return f"{self._register.name}[{self._index}]"
        return f"Clbit({self._index})"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Clbit):
            return self._index == other._index
        return False
    
    def __hash__(self) -> int:
        return hash(('clbit', self._index))


class ClassicalRegister:
    """
    经典寄存器 - 一组经典比特的集合
    
    Attributes:
        size: 寄存器中经典比特的数量
        name: 寄存器名称
    """
    
    def __init__(self, size: int, name: str = "c"):
        self._size = size
        self._name = name
        self._clbits = [Clbit(i, self) for i in range(size)]
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def name(self) -> str:
        return self._name
    
    def __len__(self) -> int:
        return self._size
    
    def __getitem__(self, index: int) -> Clbit:
        if index < 0 or index >= self._size:
            raise IndexError(f"Clbit index {index} out of range [0, {self._size})")
        return self._clbits[index]
    
    def __iter__(self):
        return iter(self._clbits)
    
    def __repr__(self) -> str:
        return f"ClassicalRegister({self._size}, '{self._name}')"
