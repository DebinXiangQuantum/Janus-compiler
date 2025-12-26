"""
Janus 噪声模拟器

基于密度矩阵的带噪声量子电路模拟器
"""
from __future__ import annotations
from typing import List, Optional, Union, Dict
import numpy as np

from .density_matrix import DensityMatrix
from .statevector import Statevector
from .result import SimulatorResult, Counts
from .noise import NoiseModel, NoiseChannel
from .exceptions import SimulatorError, ParameterBindingError


class NoisySimulator:
    """
    噪声模拟器
    
    使用密度矩阵进行带噪声的量子电路模拟
    
    Example:
        from janus.simulator import NoisySimulator
        from janus.simulator.noise import NoiseModel, depolarizing_channel
        
        # 创建噪声模型
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            depolarizing_channel(0.01), ['cx']
        )
        
        # 模拟
        sim = NoisySimulator(noise_model)
        result = sim.run(circuit, shots=1000)
    """
    
    def __init__(
        self,
        noise_model: Optional[NoiseModel] = None,
        seed: Optional[Union[int, np.random.Generator]] = None
    ):
        """
        初始化噪声模拟器
        
        Args:
            noise_model: 噪声模型
            seed: 随机数种子
        """
        self._noise_model = noise_model
        self._seed = seed
        
        if seed is not None:
            if isinstance(seed, np.random.Generator):
                self._rng = seed
            else:
                self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()
    
    @property
    def noise_model(self) -> Optional[NoiseModel]:
        """获取噪声模型"""
        return self._noise_model
    
    @noise_model.setter
    def noise_model(self, value: NoiseModel):
        """设置噪声模型"""
        self._noise_model = value
    
    def run(
        self,
        circuit,
        shots: int = 1024,
        initial_state: Optional[Union[DensityMatrix, Statevector, np.ndarray]] = None,
        parameter_binds: Optional[Dict] = None,
        measure_qubits: Optional[List[int]] = None
    ) -> SimulatorResult:
        """
        运行噪声模拟
        
        Args:
            circuit: 要模拟的电路
            shots: 测量次数
            initial_state: 初始状态
            parameter_binds: 参数绑定
            measure_qubits: 要测量的量子比特
        
        Returns:
            SimulatorResult: 模拟结果
        """
        # 绑定参数
        if parameter_binds:
            circuit = self._bind_parameters(circuit, parameter_binds)
        
        # 准备初始状态
        dm = self._prepare_initial_state(circuit.n_qubits, initial_state)
        dm.seed(self._rng)
        
        # 演化电路（带噪声）
        dm = self._evolve_with_noise(dm, circuit)
        
        # 确定测量的量子比特
        if measure_qubits is None:
            if circuit.measured_qubits is not None:
                measure_qubits = circuit.measured_qubits
            else:
                measure_qubits = list(range(circuit.n_qubits))
        
        # 采样测量（可能带读出错误）
        counts = self._sample_with_readout_error(dm, shots, measure_qubits)
        
        # 构建结果
        result = SimulatorResult(
            counts=dict(counts),
            statevector=None,  # 噪声模拟不返回状态向量
            shots=shots,
            metadata={
                'num_qubits': circuit.n_qubits,
                'circuit_depth': circuit.depth,
                'measured_qubits': measure_qubits,
                'simulator': 'noisy',
                'purity': dm.purity()
            }
        )
        
        return result
    
    def density_matrix(
        self,
        circuit,
        initial_state: Optional[Union[DensityMatrix, Statevector, np.ndarray]] = None,
        parameter_binds: Optional[Dict] = None
    ) -> DensityMatrix:
        """
        获取电路的最终密度矩阵
        
        Args:
            circuit: 要模拟的电路
            initial_state: 初始状态
            parameter_binds: 参数绑定
        
        Returns:
            DensityMatrix: 最终密度矩阵
        """
        if parameter_binds:
            circuit = self._bind_parameters(circuit, parameter_binds)
        
        dm = self._prepare_initial_state(circuit.n_qubits, initial_state)
        return self._evolve_with_noise(dm, circuit)
    
    def _prepare_initial_state(
        self,
        num_qubits: int,
        initial_state: Optional[Union[DensityMatrix, Statevector, np.ndarray]]
    ) -> DensityMatrix:
        """准备初始状态"""
        if initial_state is None:
            sv = Statevector.from_int(0, num_qubits)
            return DensityMatrix.from_statevector(sv)
        elif isinstance(initial_state, DensityMatrix):
            if initial_state.num_qubits != num_qubits:
                raise SimulatorError(
                    f"Initial state has {initial_state.num_qubits} qubits, "
                    f"but circuit has {num_qubits} qubits"
                )
            return initial_state.copy()
        elif isinstance(initial_state, Statevector):
            if initial_state.num_qubits != num_qubits:
                raise SimulatorError(
                    f"Initial state has {initial_state.num_qubits} qubits, "
                    f"but circuit has {num_qubits} qubits"
                )
            return DensityMatrix.from_statevector(initial_state)
        elif isinstance(initial_state, np.ndarray):
            if initial_state.ndim == 1:
                return DensityMatrix.from_statevector(Statevector(initial_state))
            else:
                return DensityMatrix(initial_state)
        else:
            raise SimulatorError(f"Invalid initial state type: {type(initial_state)}")
    
    def _evolve_with_noise(self, dm: DensityMatrix, circuit) -> DensityMatrix:
        """带噪声演化"""
        from janus.circuit.parameter import is_parameterized
        
        for inst in circuit.instructions:
            if is_parameterized(inst.operation):
                raise ParameterBindingError(
                    f"Circuit contains unbound parameter in gate {inst.name}"
                )
            
            # 应用门
            matrix = inst.operation.to_matrix()
            dm.evolve(matrix, inst.qubits)
            
            # 应用噪声
            if self._noise_model is not None:
                error = self._noise_model.get_gate_error(inst.name, inst.qubits)
                if error is not None:
                    # 对每个量子比特应用单比特噪声
                    if error.num_qubits == 1:
                        for q in inst.qubits:
                            dm.apply_channel(error.kraus_ops, [q])
                    else:
                        # 多比特噪声直接应用到所有相关量子比特
                        dm.apply_channel(error.kraus_ops, inst.qubits)
        
        return dm
    
    def _sample_with_readout_error(
        self,
        dm: DensityMatrix,
        shots: int,
        measure_qubits: List[int]
    ) -> Counts:
        """带读出错误的采样"""
        # 先进行理想采样
        counts = dm.sample_counts(shots, measure_qubits)
        
        # 如果没有噪声模型或没有读出错误，直接返回
        if self._noise_model is None:
            return counts
        
        # 应用读出错误
        new_counts = {}
        for bitstring, count in counts.items():
            # 对每个采样结果应用读出错误
            for _ in range(count):
                noisy_bits = list(bitstring)
                for i, q in enumerate(measure_qubits):
                    error = self._noise_model.get_readout_error(q)
                    if error is not None:
                        p0_given_1, p1_given_0 = error
                        bit_idx = len(bitstring) - 1 - i
                        if noisy_bits[bit_idx] == '0':
                            if self._rng.random() < p1_given_0:
                                noisy_bits[bit_idx] = '1'
                        else:
                            if self._rng.random() < p0_given_1:
                                noisy_bits[bit_idx] = '0'
                
                noisy_bitstring = ''.join(noisy_bits)
                new_counts[noisy_bitstring] = new_counts.get(noisy_bitstring, 0) + 1
        
        return Counts(new_counts)
    
    def _bind_parameters(self, circuit, parameter_binds: Dict):
        """绑定参数"""
        from janus.circuit.parameter import Parameter
        
        if hasattr(circuit, 'bind_parameters'):
            return circuit.bind_parameters(parameter_binds)
        
        import copy
        bound_circuit = copy.deepcopy(circuit)
        
        for inst in bound_circuit.instructions:
            new_params = []
            for param in inst.operation.params:
                if isinstance(param, Parameter):
                    if param in parameter_binds:
                        new_params.append(parameter_binds[param])
                    elif param.name in parameter_binds:
                        new_params.append(parameter_binds[param.name])
                    else:
                        raise ParameterBindingError(
                            f"No binding provided for parameter {param.name}"
                        )
                else:
                    new_params.append(param)
            inst.operation.params = new_params
        
        return bound_circuit
