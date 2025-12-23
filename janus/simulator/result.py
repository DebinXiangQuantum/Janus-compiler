"""
Janus 模拟器结果类

封装模拟结果，包括测量计数、状态向量、概率分布等
"""
from typing import Dict, List, Optional, Union
import numpy as np
from collections import Counter


class Counts(Counter):
    """
    测量计数类
    
    继承自 Counter，提供额外的量子计算相关方法
    
    Example:
        counts = Counts({'00': 512, '11': 488})
        print(counts.most_frequent())  # '00'
        print(counts.int_outcomes())   # {0: 512, 3: 488}
    """
    
    def __init__(self, data: Optional[Dict[str, int]] = None):
        super().__init__(data or {})
    
    def most_frequent(self) -> str:
        """返回出现次数最多的测量结果"""
        if not self:
            return ""
        return self.most_common(1)[0][0]
    
    def int_outcomes(self) -> Dict[int, int]:
        """将二进制字符串结果转换为整数"""
        return {int(k, 2): v for k, v in self.items()}
    
    def hex_outcomes(self) -> Dict[str, int]:
        """将二进制字符串结果转换为十六进制"""
        return {hex(int(k, 2)): v for k, v in self.items()}
    
    def total_shots(self) -> int:
        """返回总测量次数"""
        return sum(self.values())
    
    def probabilities(self) -> Dict[str, float]:
        """返回概率分布"""
        total = self.total_shots()
        if total == 0:
            return {}
        return {k: v / total for k, v in self.items()}
    
    def marginal(self, indices: List[int]) -> 'Counts':
        """
        边缘化到指定的比特位置
        
        Args:
            indices: 要保留的比特索引列表（从右到左，0 是最低位）
        
        Returns:
            Counts: 边缘化后的计数
        """
        new_counts = Counter()
        for bitstring, count in self.items():
            # 提取指定位置的比特
            new_bits = ''.join(bitstring[-(i+1)] for i in sorted(indices, reverse=True))
            new_counts[new_bits] += count
        return Counts(dict(new_counts))


class SimulatorResult:
    """
    模拟器结果类
    
    封装单次模拟运行的所有结果
    
    Attributes:
        counts: 测量计数
        statevector: 最终状态向量（可选）
        probabilities: 概率分布
        shots: 测量次数
        metadata: 额外元数据
    """
    
    def __init__(
        self,
        counts: Optional[Dict[str, int]] = None,
        statevector: Optional[np.ndarray] = None,
        shots: int = 0,
        metadata: Optional[Dict] = None
    ):
        self._counts = Counts(counts) if counts else Counts()
        self._statevector = statevector
        self._shots = shots
        self._metadata = metadata or {}
        self._probabilities: Optional[np.ndarray] = None
    
    @property
    def counts(self) -> Counts:
        """获取测量计数"""
        return self._counts
    
    @property
    def statevector(self) -> Optional[np.ndarray]:
        """获取最终状态向量"""
        return self._statevector
    
    @property
    def shots(self) -> int:
        """获取测量次数"""
        return self._shots
    
    @property
    def metadata(self) -> Dict:
        """获取元数据"""
        return self._metadata
    
    @property
    def probabilities(self) -> np.ndarray:
        """获取概率分布"""
        if self._probabilities is None and self._statevector is not None:
            self._probabilities = np.abs(self._statevector) ** 2
        return self._probabilities
    
    def get_counts(self, threshold: float = 0.0) -> Counts:
        """
        获取测量计数，可选过滤低于阈值的结果
        
        Args:
            threshold: 概率阈值，低于此值的结果将被过滤
        """
        if threshold <= 0:
            return self._counts
        
        total = self._counts.total_shots()
        if total == 0:
            return self._counts
        
        filtered = {k: v for k, v in self._counts.items() 
                   if v / total >= threshold}
        return Counts(filtered)
    
    def get_statevector(self) -> Optional[np.ndarray]:
        """获取状态向量的副本"""
        if self._statevector is None:
            return None
        return self._statevector.copy()
    
    def get_probabilities(self, decimals: Optional[int] = None) -> Dict[str, float]:
        """
        获取概率分布字典
        
        Args:
            decimals: 小数位数，None 表示不舍入
        """
        if self._statevector is None:
            return self._counts.probabilities()
        
        probs = np.abs(self._statevector) ** 2
        n_qubits = int(np.log2(len(probs)))
        
        result = {}
        for i, p in enumerate(probs):
            if p > 1e-15:  # 过滤极小值
                bitstring = format(i, f'0{n_qubits}b')
                if decimals is not None:
                    p = round(p, decimals)
                result[bitstring] = p
        
        return result
    
    def expectation_value(self, observable: np.ndarray) -> complex:
        """
        计算可观测量的期望值
        
        Args:
            observable: 可观测量矩阵（厄米矩阵）
        
        Returns:
            期望值 <ψ|O|ψ>
        """
        if self._statevector is None:
            raise ValueError("No statevector available for expectation value calculation")
        
        return np.vdot(self._statevector, observable @ self._statevector)
    
    def __repr__(self) -> str:
        return f"SimulatorResult(shots={self._shots}, counts={dict(self._counts)})"
