"""
Janus 量子操作基础接口

定义所有量子操作的抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class Operation(ABC):
    """
    量子操作的抽象基类
    
    所有量子门、指令都应该继承此类
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """操作名称"""
        pass
    
    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """操作作用的量子比特数"""
        pass
    
    @property
    def num_clbits(self) -> int:
        """操作作用的经典比特数，默认为0"""
        return 0
