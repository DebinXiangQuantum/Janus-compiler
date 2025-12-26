"""
Janus 电路层

表示可以并行执行的一组量子门
"""
from typing import List, Iterator, Optional
from .instruction import Instruction
from .gate import Gate


class Layer:
    """
    电路层类
    
    表示量子电路中的一层，包含可以并行执行的量子门集合。
    同一层中的门作用在不同的量子比特上，因此可以同时执行。
    
    Attributes:
        index: 层在电路中的索引
    """
    
    def __init__(self, instructions: Optional[List[Instruction]] = None, index: int = 0):
        """
        初始化电路层
        
        Args:
            instructions: 该层包含的指令列表
            index: 层的索引
        """
        self._instructions: List[Instruction] = instructions if instructions is not None else []
        self._index = index
    
    @property
    def index(self) -> int:
        return self._index
    
    @index.setter
    def index(self, value: int):
        self._index = value
    
    @property
    def instructions(self) -> List[Instruction]:
        return self._instructions
    
    def append(self, instruction: Instruction):
        """添加一条指令到层中"""
        self._instructions.append(instruction)
    
    def remove(self, instruction: Instruction):
        """从层中移除一条指令"""
        self._instructions.remove(instruction)
    
    def get_qubits(self) -> List[int]:
        """获取该层中所有被操作的量子比特"""
        qubits = []
        for inst in self._instructions:
            qubits.extend(inst.qubits)
        return list(set(qubits))
    
    def can_add(self, qubits: List[int]) -> bool:
        """检查是否可以添加作用于指定量子比特的门（无冲突）"""
        layer_qubits = set(self.get_qubits())
        return not any(q in layer_qubits for q in qubits)
    
    def is_empty(self) -> bool:
        """检查层是否为空"""
        return len(self._instructions) == 0
    
    def copy(self) -> 'Layer':
        """创建层的副本"""
        return Layer(
            [inst.copy() for inst in self._instructions],
            self._index
        )
    
    def __len__(self) -> int:
        return len(self._instructions)
    
    def __iter__(self) -> Iterator[Instruction]:
        return iter(self._instructions)
    
    def __getitem__(self, index: int) -> Instruction:
        return self._instructions[index]
    
    def __repr__(self) -> str:
        gates_str = ", ".join(str(inst.operation) for inst in self._instructions)
        return f"Layer[{self._index}]({gates_str})"
    
    def to_list(self) -> List[dict]:
        """转换为字典列表格式（兼容旧格式）"""
        return [inst.to_dict() for inst in self._instructions]
