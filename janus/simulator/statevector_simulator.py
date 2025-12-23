"""
Janus 状态向量模拟器

基于状态向量的量子电路模拟器
"""
from __future__ import annotations
from typing import List, Optional, Union, Dict, Any
import numpy as np

from .statevector import Statevector
from .result import SimulatorResult, Counts
from .exceptions import SimulatorError, InvalidCircuitError, ParameterBindingError


class StatevectorSimulator:
    """
    状态向量模拟器
    
    使用完整状态向量进行精确模拟，适用于小规模量子电路（< 25 qubits）
    
    Features:
        - 精确模拟（无采样误差）
        - 支持参数化电路
        - 支持初始状态设置
        - 支持部分测量
        - 支持期望值计算
    
    Example:
        from janus.circuit import Circuit
        from janus.simulator import StatevectorSimulator
        
        # 创建 Bell 态电路
        qc = Circuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        # 模拟
        sim = StatevectorSimulator()
        result = sim.run(qc, shots=1000)
        
        print(result.counts)  # {'00': ~500, '11': ~500}
    """
    
    def __init__(
        self,
        seed: Optional[Union[int, np.random.Generator]] = None,
        precision: str = 'double'
    ):
        """
        初始化模拟器
        
        Args:
            seed: 随机数种子
            precision: 精度 ('single' 或 'double')
        """
        self._seed = seed
        self._precision = precision
        self._dtype = np.complex128 if precision == 'double' else np.complex64
        
        if seed is not None:
            if isinstance(seed, np.random.Generator):
                self._rng = seed
            else:
                self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()
    
    @property
    def seed(self) -> Optional[Union[int, np.random.Generator]]:
        """获取随机数种子"""
        return self._seed
    
    def run(
        self,
        circuit,
        shots: int = 1024,
        initial_state: Optional[Union[Statevector, np.ndarray, str]] = None,
        parameter_binds: Optional[Dict] = None,
        measure_qubits: Optional[List[int]] = None,
        return_statevector: bool = True
    ) -> SimulatorResult:
        """
        运行模拟
        
        Args:
            circuit: 要模拟的电路
            shots: 测量次数
            initial_state: 初始状态（默认 |0...0⟩）
            parameter_binds: 参数绑定字典
            measure_qubits: 要测量的量子比特（默认全部）
            return_statevector: 是否返回状态向量
        
        Returns:
            SimulatorResult: 模拟结果
        """
        # 绑定参数
        if parameter_binds:
            circuit = self._bind_parameters(circuit, parameter_binds)
        
        # 验证电路
        self._validate_circuit(circuit)
        
        # 准备初始状态
        statevector = self._prepare_initial_state(circuit.n_qubits, initial_state)
        statevector.seed(self._rng)
        
        # 演化电路
        statevector = statevector.evolve_circuit(circuit)
        
        # 确定测量的量子比特
        if measure_qubits is None:
            if circuit.measured_qubits is not None:
                measure_qubits = circuit.measured_qubits
            else:
                measure_qubits = list(range(circuit.n_qubits))
        
        # 采样测量
        counts = statevector.sample_counts(shots, measure_qubits)
        
        # 构建结果
        result = SimulatorResult(
            counts=dict(counts),
            statevector=statevector.data if return_statevector else None,
            shots=shots,
            metadata={
                'num_qubits': circuit.n_qubits,
                'circuit_depth': circuit.depth,
                'measured_qubits': measure_qubits,
                'simulator': 'statevector'
            }
        )
        
        return result
    
    def run_batch(
        self,
        circuits: List,
        shots: int = 1024,
        **kwargs
    ) -> List[SimulatorResult]:
        """
        批量运行多个电路
        
        Args:
            circuits: 电路列表
            shots: 每个电路的测量次数
            **kwargs: 传递给 run() 的其他参数
        
        Returns:
            List[SimulatorResult]: 结果列表
        """
        return [self.run(circuit, shots=shots, **kwargs) for circuit in circuits]
    
    def statevector(
        self,
        circuit,
        initial_state: Optional[Union[Statevector, np.ndarray, str]] = None,
        parameter_binds: Optional[Dict] = None
    ) -> Statevector:
        """
        获取电路的最终状态向量（不进行测量）
        
        Args:
            circuit: 要模拟的电路
            initial_state: 初始状态
            parameter_binds: 参数绑定
        
        Returns:
            Statevector: 最终状态向量
        """
        if parameter_binds:
            circuit = self._bind_parameters(circuit, parameter_binds)
        
        self._validate_circuit(circuit)
        
        sv = self._prepare_initial_state(circuit.n_qubits, initial_state)
        return sv.evolve_circuit(circuit)
    
    def expectation_value(
        self,
        circuit,
        observable: np.ndarray,
        qargs: Optional[List[int]] = None,
        initial_state: Optional[Union[Statevector, np.ndarray, str]] = None,
        parameter_binds: Optional[Dict] = None
    ) -> complex:
        """
        计算可观测量的期望值
        
        Args:
            circuit: 量子电路
            observable: 可观测量矩阵
            qargs: 可观测量作用的量子比特
            initial_state: 初始状态
            parameter_binds: 参数绑定
        
        Returns:
            complex: 期望值
        """
        sv = self.statevector(circuit, initial_state, parameter_binds)
        return sv.expectation_value(observable, qargs)
    
    def probabilities(
        self,
        circuit,
        qargs: Optional[List[int]] = None,
        initial_state: Optional[Union[Statevector, np.ndarray, str]] = None,
        parameter_binds: Optional[Dict] = None
    ) -> np.ndarray:
        """
        计算测量概率分布
        
        Args:
            circuit: 量子电路
            qargs: 要测量的量子比特
            initial_state: 初始状态
            parameter_binds: 参数绑定
        
        Returns:
            np.ndarray: 概率分布
        """
        sv = self.statevector(circuit, initial_state, parameter_binds)
        return sv.probabilities(qargs)
    
    def _prepare_initial_state(
        self,
        num_qubits: int,
        initial_state: Optional[Union[Statevector, np.ndarray, str]]
    ) -> Statevector:
        """准备初始状态"""
        if initial_state is None:
            return Statevector.from_int(0, num_qubits)
        elif isinstance(initial_state, Statevector):
            if initial_state.num_qubits != num_qubits:
                raise SimulatorError(
                    f"Initial state has {initial_state.num_qubits} qubits, "
                    f"but circuit has {num_qubits} qubits"
                )
            return initial_state.copy()
        elif isinstance(initial_state, str):
            sv = Statevector.from_label(initial_state)
            if sv.num_qubits != num_qubits:
                raise SimulatorError(
                    f"Initial state label has {sv.num_qubits} qubits, "
                    f"but circuit has {num_qubits} qubits"
                )
            return sv
        elif isinstance(initial_state, np.ndarray):
            return Statevector(initial_state, num_qubits)
        else:
            raise SimulatorError(f"Invalid initial state type: {type(initial_state)}")
    
    def _validate_circuit(self, circuit):
        """验证电路"""
        from ..circuit.parameter import Parameter, ParameterExpression
        
        # 检查是否有未绑定的参数
        for inst in circuit.instructions:
            for param in inst.operation.params:
                if isinstance(param, Parameter):
                    raise ParameterBindingError(
                        f"Circuit contains unbound parameter '{param.name}' in gate {inst.name}. "
                        "Please provide parameter_binds."
                    )
                elif isinstance(param, ParameterExpression) and not param.is_real():
                    raise ParameterBindingError(
                        f"Circuit contains unbound parameter expression in gate {inst.name}. "
                        "Please provide parameter_binds."
                    )
    
    def _bind_parameters(self, circuit, parameter_binds: Dict):
        """绑定参数到电路"""
        from ..circuit.parameter import Parameter, ParameterExpression
        
        # 将字符串键转换为 Parameter 对象
        converted_binds = {}
        for key, value in parameter_binds.items():
            if isinstance(key, Parameter):
                converted_binds[key] = value
            elif isinstance(key, str):
                # 查找电路中对应名称的参数
                for param in circuit.parameters:
                    if param.name == key:
                        converted_binds[param] = value
                        break
                else:
                    # 如果找不到，保留字符串键（可能电路有自己的处理方式）
                    converted_binds[key] = value
            else:
                converted_binds[key] = value
        
        # 使用电路的 bind_parameters 方法
        if hasattr(circuit, 'bind_parameters'):
            return circuit.bind_parameters(converted_binds)
        
        # 手动绑定作为后备
        import copy
        bound_circuit = copy.deepcopy(circuit)
        
        param_map = {p.name if isinstance(p, Parameter) else str(p): v 
                     for p, v in converted_binds.items()}
        
        for inst in bound_circuit._instructions:
            new_params = []
            for param in inst.operation._params:
                if isinstance(param, Parameter):
                    if param.name in param_map:
                        new_params.append(param_map[param.name])
                    else:
                        raise ParameterBindingError(
                            f"No binding provided for parameter {param.name}"
                        )
                elif isinstance(param, ParameterExpression):
                    expr_param_map = {}
                    for p in param.parameters:
                        if p.name in param_map:
                            expr_param_map[p] = param_map[p.name]
                    bound_value = param.bind(expr_param_map)
                    if isinstance(bound_value, (int, float)):
                        new_params.append(float(bound_value))
                    elif isinstance(bound_value, ParameterExpression) and bound_value.is_real():
                        new_params.append(float(bound_value))
                    else:
                        new_params.append(bound_value)
                else:
                    new_params.append(param)
            inst.operation._params = new_params
        
        return bound_circuit
