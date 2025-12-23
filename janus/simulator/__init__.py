"""
Janus 量子电路模拟器

提供状态向量模拟、密度矩阵模拟、噪声模拟等功能

Example:
    from janus.circuit import Circuit
    from janus.simulator import StatevectorSimulator, Statevector
    
    # 创建电路
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    # 状态向量模拟
    sim = StatevectorSimulator()
    result = sim.run(qc, shots=1000)
    print(result.counts)  # {'00': ~500, '11': ~500}
    
    # 直接获取状态向量
    sv = Statevector.from_circuit(qc)
    print(sv.probabilities())
    
    # 噪声模拟
    from janus.simulator import NoisySimulator
    from janus.simulator.noise import NoiseModel, depolarizing_channel
    
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_channel(0.01), ['cx'])
    
    noisy_sim = NoisySimulator(noise_model)
    noisy_result = noisy_sim.run(qc, shots=1000)
"""

from .statevector import Statevector
from .density_matrix import DensityMatrix
from .statevector_simulator import StatevectorSimulator
from .noisy_simulator import NoisySimulator
from .result import SimulatorResult, Counts
from .exceptions import SimulatorError, InvalidStateError, InvalidCircuitError, ParameterBindingError
from .noise import (
    NoiseChannel,
    NoiseModel,
    depolarizing_channel,
    amplitude_damping_channel,
    phase_damping_channel,
    bit_flip_channel,
    phase_flip_channel,
    bit_phase_flip_channel,
    thermal_relaxation_channel,
    readout_error_channel,
)

__all__ = [
    # 核心类
    'Statevector',
    'DensityMatrix',
    'StatevectorSimulator',
    'NoisySimulator',
    'SimulatorResult',
    'Counts',
    
    # 噪声
    'NoiseChannel',
    'NoiseModel',
    'depolarizing_channel',
    'amplitude_damping_channel',
    'phase_damping_channel',
    'bit_flip_channel',
    'phase_flip_channel',
    'bit_phase_flip_channel',
    'thermal_relaxation_channel',
    'readout_error_channel',
    
    # 异常
    'SimulatorError',
    'InvalidStateError',
    'InvalidCircuitError',
    'ParameterBindingError',
]
