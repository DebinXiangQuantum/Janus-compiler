# Janus 量子电路框架

[![PyPI version](https://badge.fury.io/py/janus-quantum.svg)](https://badge.fury.io/py/janus-quantum)
[![Documentation Status](https://readthedocs.org/projects/janus-quantum/badge/?version=latest)](https://janus-quantum.readthedocs.io/zh-cn/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Janus** 是一个轻量级、高性能的量子电路构建、模拟和编译框架。

## ✨ 特性

- 🚀 **轻量高效** - 纯 Python 实现，依赖少，启动快
- 🔧 **60+ 量子门** - 完整的标准门库，支持参数化电路
- 📊 **多种模拟器** - 状态向量、密度矩阵、噪声模拟
- ⚡ **电路优化** - 多级优化 Pass，自动门合并与消除
- 🎨 **可视化** - 文本和图像电路绘制
- 📁 **JSON 格式** - 简洁的电路存储和交换格式

## 📦 安装

```bash
pip install janus-quantum
```

完整安装（包含可视化）：

```bash
pip install janus-quantum[full]
```

## 🚀 快速开始

### 创建 Bell 态

```python
from janus.circuit import Circuit
from janus.simulator import StatevectorSimulator

# 创建电路
qc = Circuit(2)
qc.h(0)
qc.cx(0, 1)

# 模拟运行
sim = StatevectorSimulator()
result = sim.run(qc, shots=1000)
print(result.counts)  # {'00': ~500, '11': ~500}

# 查看电路
print(qc.draw())
```

输出：
```
q0: ─H─●─
       │
q1: ───X─
```

### 参数化电路

```python
from janus.circuit import Circuit, Parameter
import numpy as np

theta = Parameter('theta')

qc = Circuit(1)
qc.ry(theta, 0)

# 绑定参数
bound_qc = qc.bind_parameters({theta: np.pi/2})
```

### 噪声模拟

```python
from janus.simulator import NoisySimulator, NoiseModel, depolarizing_channel

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(
    depolarizing_channel(0.01),
    ['h', 'x', 'cx']
)

noisy_sim = NoisySimulator(noise_model)
result = noisy_sim.run(qc, shots=1000)
```

### 电路优化

```python
from janus.compiler import compile_circuit

optimized = compile_circuit(qc, optimization_level=2)
```

### 电路文件操作

```python
from janus.circuit import load_circuit, save_circuit, list_circuits

# 列出预置电路
print(list_circuits())  # ['bell', 'test']

# 加载预置电路
qc = load_circuit(name='bell')

# 从文件加载
qc = load_circuit(filepath='./my_circuit.json')

# 保存电路
save_circuit(qc, 'my_circuit.json')
```

## 📚 文档

完整文档请访问：[https://janus-quantum.readthedocs.io/zh-cn/latest/](https://janus-quantum.readthedocs.io/zh-cn/latest/)

## 🔧 支持的量子门

### 单比特门
`H`, `X`, `Y`, `Z`, `S`, `T`, `RX`, `RY`, `RZ`, `U`, `P`, ...

### 两比特门
`CX`, `CY`, `CZ`, `CH`, `SWAP`, `iSWAP`, `CRX`, `CRY`, `CRZ`, `RXX`, `RYY`, `RZZ`, ...

### 多比特门
`CCX` (Toffoli), `CCZ`, `CSWAP` (Fredkin), `MCX`, `MCP`, ...

## 📁 项目结构

```
janus/
├── circuit/      # 电路构建和表示
├── simulator/    # 量子模拟器
├── compiler/     # 电路编译器
├── decompose/    # 门分解
├── optimize/     # 高级优化
└── encode/       # 量子态编码
```

## 🤝 贡献

欢迎贡献代码！请查看 [贡献指南](https://janus-quantum.readthedocs.io/zh-cn/latest/contributing.html)。

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。
