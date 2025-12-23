"""
Janus 量子门基类

定义量子门的基本结构和接口
"""
from typing import List, Optional, Union
import numpy as np
from .operation import Operation


class Gate(Operation):
    """
    量子门基类
    
    量子门是酉操作，可以用酉矩阵表示
    
    Attributes:
        name: 门的名称
        qubits: 门作用的量子比特索引列表
        params: 门的参数列表（如旋转角度）
        label: 可选的标签
    """
    
    def __init__(
        self,
        name: str,
        num_qubits: int,
        params: Optional[List[float]] = None,
        label: Optional[str] = None
    ):
        self._name = name
        self._num_qubits = num_qubits
        self._params = params if params is not None else []
        self._label = label
        self._qubits: List[int] = []  # 实际作用的量子比特，在添加到电路时设置
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def num_qubits(self) -> int:
        return self._num_qubits
    
    @property
    def params(self) -> List[float]:
        return self._params
    
    @params.setter
    def params(self, value: List[float]):
        self._params = value
    
    @property
    def qubits(self) -> List[int]:
        return self._qubits
    
    @qubits.setter
    def qubits(self, value: List[int]):
        if len(value) != self._num_qubits:
            raise ValueError(f"Gate {self._name} requires {self._num_qubits} qubits, got {len(value)}")
        self._qubits = value
    
    @property
    def label(self) -> Optional[str]:
        return self._label
    
    @label.setter
    def label(self, value: str):
        self._label = value
    
    def to_matrix(self) -> np.ndarray:
        """
        返回门的酉矩阵表示
        
        子类应该重写此方法
        """
        raise NotImplementedError(f"to_matrix not implemented for {self._name}")
    
    def inverse(self) -> 'Gate':
        """
        返回门的逆操作
        
        子类应该重写此方法
        """
        raise NotImplementedError(f"inverse not implemented for {self._name}")
    
    def copy(self) -> 'Gate':
        """创建门的副本"""
        new_gate = Gate(self._name, self._num_qubits, self._params.copy(), self._label)
        new_gate._qubits = self._qubits.copy()
        return new_gate
    
    def __repr__(self) -> str:
        if self._params:
            return f"{self._name}({', '.join(map(str, self._params))})"
        return self._name
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Gate):
            return False
        return (self._name == other._name and 
                self._num_qubits == other._num_qubits and
                self._params == other._params)
    
    def to_dict(self) -> dict:
        """转换为字典格式（兼容旧格式）"""
        return {
            'name': self._name,
            'qubits': self._qubits,
            'params': self._params
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Gate':
        """从字典创建门"""
        gate = cls(data['name'], len(data['qubits']), data.get('params', []))
        gate.qubits = data['qubits']
        return gate

    def control(self, num_ctrl_qubits: int = 1, label: Optional[str] = None, 
                ctrl_state: Optional[Union[str, int]] = None) -> 'ControlledGate':
        """
        返回该门的受控版本
        
        Args:
            num_ctrl_qubits: 控制比特数量，默认为 1
            label: 可选的标签
            ctrl_state: 控制状态（暂未实现，保留接口）
        
        Returns:
            ControlledGate: 受控门
        
        Example:
            # 创建受控 U3 门
            u3 = U3Gate(np.pi/4, np.pi/4, np.pi/4)
            cu3 = u3.control(1)  # 单控制
            ccu3 = u3.control(2)  # 双控制
        """
        return ControlledGate(
            base_gate=self,
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state
        )


class ControlledGate(Gate):
    """
    受控门类
    
    将任意门转换为受控版本
    
    Attributes:
        base_gate: 基础门
        num_ctrl_qubits: 控制比特数量
        ctrl_state: 控制状态
    """
    
    def __init__(
        self,
        base_gate: Gate,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None
    ):
        self._base_gate = base_gate
        self._num_ctrl_qubits = num_ctrl_qubits
        self._ctrl_state = ctrl_state if ctrl_state is not None else (1 << num_ctrl_qubits) - 1
        
        # 构建名称
        if num_ctrl_qubits == 1:
            name = f"c{base_gate.name}"
        else:
            name = f"mc{base_gate.name}"
        
        # 总量子比特数 = 控制比特 + 基础门的量子比特
        total_qubits = num_ctrl_qubits + base_gate.num_qubits
        
        super().__init__(
            name=name,
            num_qubits=total_qubits,
            params=base_gate.params.copy(),
            label=label
        )
    
    @property
    def base_gate(self) -> Gate:
        """获取基础门"""
        return self._base_gate
    
    @property
    def ctrl_qubits(self) -> int:
        """获取控制比特数量"""
        return self._num_ctrl_qubits
    
    @property
    def ctrl_state(self) -> int:
        """获取控制状态"""
        return self._ctrl_state
    
    def to_matrix(self) -> np.ndarray:
        """
        返回受控门的酉矩阵
        
        构建方式：在控制比特为 |1...1⟩ 时应用基础门
        """
        base_matrix = self._base_gate.to_matrix()
        base_dim = base_matrix.shape[0]
        total_dim = 2 ** self._num_qubits
        
        # 创建单位矩阵
        matrix = np.eye(total_dim, dtype=complex)
        
        # 控制状态对应的索引范围
        ctrl_mask = self._ctrl_state << self._base_gate.num_qubits
        
        # 在控制状态为全 1 时，替换对应的子矩阵
        for i in range(base_dim):
            for j in range(base_dim):
                row = ctrl_mask | i
                col = ctrl_mask | j
                matrix[row, col] = base_matrix[i, j]
        
        return matrix
    
    def inverse(self) -> 'ControlledGate':
        """返回受控门的逆"""
        return ControlledGate(
            base_gate=self._base_gate.inverse(),
            num_ctrl_qubits=self._num_ctrl_qubits,
            label=self._label,
            ctrl_state=self._ctrl_state
        )
    
    def copy(self) -> 'ControlledGate':
        """创建受控门的副本"""
        new_gate = ControlledGate(
            base_gate=self._base_gate.copy(),
            num_ctrl_qubits=self._num_ctrl_qubits,
            label=self._label,
            ctrl_state=self._ctrl_state
        )
        new_gate._qubits = self._qubits.copy()
        return new_gate
    
    def control(self, num_ctrl_qubits: int = 1, label: Optional[str] = None,
                ctrl_state: Optional[Union[str, int]] = None) -> 'ControlledGate':
        """
        在受控门上再添加控制
        
        Args:
            num_ctrl_qubits: 额外的控制比特数量
        
        Returns:
            新的受控门，控制比特数量累加
        """
        return ControlledGate(
            base_gate=self._base_gate,
            num_ctrl_qubits=self._num_ctrl_qubits + num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state
        )
