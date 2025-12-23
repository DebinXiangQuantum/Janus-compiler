# Janus 量子电路框架

轻量级量子电路构建、表示和编译框架。

## 安装

```bash
pip install numpy matplotlib
```

## 快速开始

```python
from janus.circuit import Circuit
import numpy as np

# 创建电路并添加门
qc = Circuit(2)
qc.h(0)
qc.cx(0, 1)
qc.rx(np.pi/2, 0)

# 查看电路
print(qc.draw())
```

## 电路创建

### 方法 1：从层列表创建

```python
circuit = Circuit.from_layers([
    [{'name': 'h', 'qubits': [0], 'params': []}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
    [{'name': 'rx', 'qubits': [0], 'params': [1.57]}]
], n_qubits=2)
```

### 方法 2：逐个添加门

```python
circuit = Circuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.rx(np.pi/4, 2)
```

### 方法 3：指定门所在层

使用 `layer_index` 参数指定门添加到哪一层：

```python
circuit = Circuit(3)
circuit.h(0, layer_index=0)
circuit.x(1, layer_index=0)      # 与 h 门在同一层
circuit.cx(0, 1, layer_index=1)
circuit.rx(np.pi/4, 2, layer_index=0)
```

### 可分离电路

组合多个独立子电路：

```python
from janus.circuit import Circuit, SeperatableCircuit

c1 = Circuit(2)
c1.rx(np.pi/4, 0)

c2 = Circuit(3)
c2.h(2)

sep_circuit = SeperatableCircuit([c1, c2], n_qubits=4)
```

## 电路属性

```python
circuit.n_qubits            # 量子比特数
circuit.depth               # 电路深度（层数）
circuit.n_gates             # 门总数
circuit.num_two_qubit_gate  # 两比特门数量
circuit.duration            # 估算执行时间
circuit.gates               # 门列表（字典格式）
circuit.layers              # 分层表示
circuit.operated_qubits     # 实际被操作的量子比特
circuit.measured_qubits     # 需要测量的量子比特（可读写）
```

## 电路操作

### 门移动

```python
# 获取门可移动的层范围
available = circuit.get_available_space(gate_index=0)
print(available)  # range(0, 2)

# 移动门到新层
new_circuit = circuit.move_gate(gate_index=0, new_layer=1)

# 清理空层
circuit.clean_empty_layers()
```

### 复制与组合

```python
# 复制
qc_copy = qc.copy()

# 组合电路
qc1.compose(qc2)

# 电路逆
qc_inv = qc.inverse()
```

### 导出格式

```python
qc.to_dict_list()   # [{'name': 'h', 'qubits': [0], 'params': []}, ...]
qc.to_tuple_list()  # [('h', [0], []), ...]
qc.to_layers()      # 分层字典格式
```

## 电路可视化

### 文本绘图

```python
print(qc.draw())
print(qc.draw(fold=3))        # 每行最多 3 层
print(qc.draw(line_length=80)) # 指定行宽
print(qc.draw(fold=-1))       # 禁用折叠
```

### 图像导出

```python
qc.draw(output='png', filename='circuit.png')
qc.draw(output='png', filename='circuit.png', figsize=(12, 6), dpi=200)

fig = qc.draw(output='mpl')
fig.savefig('circuit.pdf')
```

## 支持的量子门 (60+)

### 单比特门

| 门 | 方法 | 说明 |
|---|------|------|
| I | `qc.id(q)` | 恒等门 |
| X | `qc.x(q)` | Pauli-X |
| Y | `qc.y(q)` | Pauli-Y |
| Z | `qc.z(q)` | Pauli-Z |
| H | `qc.h(q)` | Hadamard |
| S | `qc.s(q)` | √Z |
| S† | `qc.sdg(q)` | S 共轭转置 |
| T | `qc.t(q)` | √S |
| T† | `qc.tdg(q)` | T 共轭转置 |
| √X | `qc.sx(q)` | √X |

### 单比特旋转门

| 门 | 方法 | 参数 |
|---|------|------|
| RX | `qc.rx(θ, q)` | θ: 旋转角度 |
| RY | `qc.ry(θ, q)` | θ: 旋转角度 |
| RZ | `qc.rz(θ, q)` | θ: 旋转角度 |
| P | `qc.p(λ, q)` | λ: 相位 |
| U | `qc.u(θ, φ, λ, q)` | 通用单比特门 |
| U1 | `qc.u1(λ, q)` | 相位门 |
| U2 | `qc.u2(φ, λ, q)` | 两参数门 |
| U3 | `qc.u3(θ, φ, λ, q)` | 三参数门 |

### 两比特门

| 门 | 方法 | 说明 |
|---|------|------|
| CX | `qc.cx(c, t)` | CNOT |
| CY | `qc.cy(c, t)` | 受控 Y |
| CZ | `qc.cz(c, t)` | 受控 Z |
| CH | `qc.ch(c, t)` | 受控 H |
| SWAP | `qc.swap(q1, q2)` | 交换门 |
| iSWAP | `qc.iswap(q1, q2)` | iSWAP |

### 受控旋转门

| 门 | 方法 | 说明 |
|---|------|------|
| CRX | `qc.crx(θ, c, t)` | 受控 RX |
| CRY | `qc.cry(θ, c, t)` | 受控 RY |
| CRZ | `qc.crz(θ, c, t)` | 受控 RZ |
| CP | `qc.cp(θ, c, t)` | 受控 Phase |
| CU | `qc.cu(θ, φ, λ, γ, c, t)` | 受控 U |

### 两比特旋转门

| 门 | 方法 | 说明 |
|---|------|------|
| RXX | `qc.rxx(θ, q1, q2)` | XX 旋转 |
| RYY | `qc.ryy(θ, q1, q2)` | YY 旋转 |
| RZZ | `qc.rzz(θ, q1, q2)` | ZZ 旋转 |
| RZX | `qc.rzx(θ, q1, q2)` | ZX 旋转 |

### 三比特及多比特门

| 门 | 方法 | 说明 |
|---|------|------|
| CCX | `qc.ccx(c1, c2, t)` | Toffoli |
| CCZ | `qc.ccz(c1, c2, t)` | 双控制 Z |
| CSWAP | `qc.cswap(c, t1, t2)` | Fredkin |
| C3X | `qc.c3x(c1, c2, c3, t)` | 三控制 X |
| C4X | `qc.c4x(c1, c2, c3, c4, t)` | 四控制 X |

### 多控制门

```python
qc.mcx([0, 1], 2)              # 多控制 X
qc.mcp(np.pi/4, [0, 1], 2)     # 多控制 Phase
qc.mcrx(np.pi/4, [0, 1], 2)    # 多控制 RX
qc.mcry(np.pi/3, [0, 1, 2], 3) # 多控制 RY
qc.mcrz(np.pi/2, [0], 1)       # 多控制 RZ
```

### 链式调用创建受控门

```python
from janus.circuit.library import U3Gate, RXGate, HGate

qc.gate(RXGate(np.pi/4), 2).control(0)           # 单控制 RX
qc.gate(HGate(), 2).control([0, 1])              # 双控制 H
qc.gate(U3Gate(np.pi/4, 0, 0), 3).control([0, 1, 2])  # 三控制 U3
```

### 特殊操作

| 操作 | 方法 | 说明 |
|------|------|------|
| Barrier | `qc.barrier()` | 屏障 |
| Measure | `qc.measure(q, c)` | 测量 |
| Reset | `qc.reset(q)` | 重置 |
| Delay | `qc.delay(duration, q)` | 延迟 |

## 参数化电路

```python
from janus.circuit import Circuit, Parameter

theta = Parameter('theta')
phi = Parameter('phi')

qc = Circuit(2)
qc.rx(theta, 0)
qc.ry(phi, 1)

# 检查参数
print(qc.parameters)          # {Parameter(theta), Parameter(phi)}
print(qc.is_parameterized())  # True

# 绑定参数
bound_qc = qc.bind_parameters({theta: np.pi/2, phi: np.pi/4})
```

## DAG 表示

```python
from janus.circuit.dag import circuit_to_dag, dag_to_circuit

# 电路转 DAG
dag = circuit_to_dag(qc)

print(dag.depth())      # 深度
print(dag.count_ops())  # 门统计

# 遍历节点
for node in dag.op_nodes():
    print(node.name, node.qubits)

# DAG 转回电路
qc2 = dag_to_circuit(dag)
```

### DAGDependency

```python
from janus.circuit.dag import circuit_to_dag_dependency

dag_dep = circuit_to_dag_dependency(qc)
print(dag_dep.size())
print(dag_dep.depth())
```

### 块操作

```python
from janus.circuit.dag import BlockCollector, split_block_into_layers

collector = BlockCollector(dag)
blocks = collector.collect_all_matching_blocks(
    filter_fn=lambda n: len(n.qubits) == 1,
    min_block_size=2
)
```

## 编译器

```python
from janus.compiler import compile_circuit

qc = Circuit(2)
qc.h(0)
qc.h(0)  # 冗余
qc.rz(np.pi/4, 0)
qc.rz(np.pi/4, 0)  # 会合并

optimized = compile_circuit(qc, optimization_level=2)
```

### 优化级别

| 级别 | 内容 |
|-----|------|
| 0 | 无优化 |
| 1 | 移除恒等门、消除逆门对 |
| 2 | 级别1 + 合并连续旋转门 |

### 自定义 Pass

```python
from janus.compiler.passes import CancelInversesPass, MergeRotationsPass

optimized = compile_circuit(qc, passes=[
    CancelInversesPass(),
    MergeRotationsPass(),
])
```

## 编码器

### Schmidt 编码

```python
from janus.encode.schmidt_encode import schmidt_encode

data = [1/np.sqrt(2), 1/np.sqrt(2), 0, 0]
circuit = schmidt_encode(q_size=4, data=data, cutoff=1e-4)
```

## 模块结构

```
project/
├── circuits/           # 电路 JSON 文件存储目录
├── janus/
│   ├── circuit/
│   │   ├── circuit.py      # Circuit、SeperatableCircuit
│   │   ├── gate.py         # 门基类
│   │   ├── instruction.py  # 指令类
│   │   ├── layer.py        # 层表示
│   │   ├── dag.py          # DAG 表示
│   │   ├── parameter.py    # 参数化支持
│   │   ├── io.py           # 文件读写
│   │   ├── cli.py          # 命令行工具
│   │   └── library/        # 标准门库 (60+)
│   ├── compiler/
│   │   ├── compiler.py     # 编译主函数
│   │   └── passes.py       # 优化 Pass
│   └── encode/
│       └── schmidt_encode.py
```

## 电路文件读写

### JSON 文件格式

电路以分层格式存储：

```json
[
  [{"name": "h", "qubits": [0], "params": []}],
  [{"name": "cx", "qubits": [0, 1], "params": []}],
  [{"name": "rx", "qubits": [0], "params": [1.57]}]
]
```

### 从文件加载电路

```python
from janus.circuit import load_circuit, list_circuits

# 列出所有已保存的电路
print(list_circuits())  # ['bell.json', 'ghz.json']

# 从默认目录加载
qc = load_circuit(name='bell')

# 从指定路径加载
qc = load_circuit(filepath='./my_circuit.json')
```

### 命令行工具

```bash
# 查看电路信息
python -m janus.circuit.cli info circuit.json
python -m janus.circuit.cli info circuit.json -v  # 详细信息

# 绘制电路
python -m janus.circuit.cli draw circuit.json
python -m janus.circuit.cli draw circuit.json -o output.png  # 保存图片

# 测试电路功能
python -m janus.circuit.cli test circuit.json
```

## 模拟器

Janus 提供完整的量子电路模拟器，支持状态向量模拟、密度矩阵模拟和噪声模拟。

### 快速开始

```python
from janus.circuit import Circuit
from janus.simulator import StatevectorSimulator

# 创建电路
qc = Circuit.from_layers([
    [{'name': 'h', 'qubits': [0], 'params': []}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}]
], n_qubits=2)

# 模拟
sim = StatevectorSimulator()
result = sim.run(qc, shots=1000)
print(result.counts)  # {'00': 503, '11': 497}
```

### 状态向量模拟

```python
from janus.circuit import Circuit
from janus.simulator import StatevectorSimulator, Statevector

# 从层列表创建电路
qc = Circuit.from_layers([
    [{'name': 'h', 'qubits': [0], 'params': []}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
    [{'name': 'rx', 'qubits': [0], 'params': [1.57]}]
], n_qubits=2)

sim = StatevectorSimulator(seed=42)

# 获取状态向量
sv = sim.statevector(qc)
print(sv)                    # 状态向量表示
print(sv.probabilities())    # 概率分布

# 采样测量
result = sim.run(qc, shots=1000)
print(result.counts)         # 测量结果统计
print(result.counts.most_frequent())  # 最频繁结果
```

### 期望值计算

```python
import numpy as np
from janus.circuit import Circuit
from janus.simulator import StatevectorSimulator

qc = Circuit.from_layers([
    [{'name': 'h', 'qubits': [0], 'params': []}]
], n_qubits=1)

sim = StatevectorSimulator()
sv = sim.statevector(qc)

# Pauli 算符
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

print(sv.expectation_value(Z))  # ⟨Z⟩ ≈ 0
print(sv.expectation_value(X))  # ⟨X⟩ = 1
```

### 参数化电路

```python
from janus.circuit import Circuit, Parameter
from janus.simulator import StatevectorSimulator
import numpy as np

theta = Parameter('theta')

qc = Circuit(1)
qc.ry(theta, 0)

sim = StatevectorSimulator()

# 绑定参数运行
result = sim.run(qc, shots=100, parameter_binds={'theta': np.pi})
print(result.counts)  # {'1': 100}

# 参数扫描
for t in [0, np.pi/2, np.pi]:
    sv = sim.statevector(qc, parameter_binds={'theta': t})
    print(f"θ={t:.2f}: P(1)={sv.probabilities()[1]:.3f}")
```

### 初始状态

```python
from janus.circuit import Circuit
from janus.simulator import StatevectorSimulator, Statevector

qc = Circuit.from_layers([
    [{'name': 'x', 'qubits': [0], 'params': []}]
], n_qubits=2)

sim = StatevectorSimulator()

# 字符串初始状态
result = sim.run(qc, shots=100, initial_state='01')

# Statevector 初始状态
sv_init = Statevector.from_label('+0')
result = sim.run(qc, shots=100, initial_state=sv_init)
```

### 部分测量

```python
from janus.circuit import Circuit
from janus.simulator import StatevectorSimulator

qc = Circuit.from_layers([
    [{'name': 'h', 'qubits': [0], 'params': []}, 
     {'name': 'h', 'qubits': [1], 'params': []},
     {'name': 'h', 'qubits': [2], 'params': []}]
], n_qubits=3)

sim = StatevectorSimulator()

# 只测量部分量子比特
result = sim.run(qc, shots=1000, measure_qubits=[0, 2])
print(result.counts)  # 2-bit 结果
```

### 密度矩阵

```python
from janus.circuit import Circuit
from janus.simulator import DensityMatrix, Statevector

qc = Circuit.from_layers([
    [{'name': 'h', 'qubits': [0], 'params': []}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}]
], n_qubits=2)

# 从电路创建密度矩阵
sv = Statevector.from_circuit(qc)
dm = DensityMatrix.from_statevector(sv)

print(dm.purity())              # 纯度
print(dm.is_pure())             # 是否纯态
print(dm.von_neumann_entropy()) # 冯诺依曼熵

# 部分迹
dm_reduced = dm.partial_trace([0])  # 保留 qubit 0
print(dm_reduced.purity())
```

### 噪声模拟

```python
from janus.circuit import Circuit
from janus.simulator import (
    NoisySimulator, 
    NoiseModel,
    depolarizing_channel,
    amplitude_damping_channel,
    phase_damping_channel
)

# 创建噪声模型
noise_model = NoiseModel()

# 添加去极化噪声到所有单比特门
noise_model.add_all_qubit_quantum_error(
    depolarizing_channel(0.01), 
    ['h', 'x', 'rx', 'ry', 'rz']
)

# 添加去极化噪声到两比特门
noise_model.add_all_qubit_quantum_error(
    depolarizing_channel(0.02), 
    ['cx', 'cz']
)

# 创建电路
qc = Circuit.from_layers([
    [{'name': 'h', 'qubits': [0], 'params': []}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}]
], n_qubits=2)

# 噪声模拟
noisy_sim = NoisySimulator(noise_model, seed=42)
result = noisy_sim.run(qc, shots=1000)
print(result.counts)  # 包含噪声导致的错误

# 获取密度矩阵
dm = noisy_sim.density_matrix(qc)
print(f"纯度: {dm.purity():.4f}")  # < 1 表示混合态
```

### 噪声信道

```python
from janus.simulator import (
    depolarizing_channel,      # 去极化
    amplitude_damping_channel, # 振幅阻尼 (T1)
    phase_damping_channel,     # 相位阻尼 (T2)
    bit_flip_channel,          # 比特翻转
    phase_flip_channel,        # 相位翻转
    reset_channel,             # 重置
)

# 创建噪声信道
dep = depolarizing_channel(p=0.01)      # 1% 去极化
amp = amplitude_damping_channel(gamma=0.05)  # 振幅阻尼
phase = phase_damping_channel(gamma=0.03)    # 相位阻尼
```

### Statevector 类方法

```python
from janus.simulator import Statevector
import numpy as np

# 创建状态
sv0 = Statevector.from_int(0, num_qubits=3)  # |000⟩
sv1 = Statevector.from_label('+-0')          # |+⟩⊗|-⟩⊗|0⟩
sv2 = Statevector(np.array([1, 0, 0, 1]) / np.sqrt(2))  # 自定义

# 状态操作
sv.probabilities()           # 概率分布
sv.probabilities([0])        # 边缘概率
sv.sample_counts(1000)       # 采样
sv.expectation_value(op)     # 期望值

# 状态比较
sv1.inner(sv2)               # 内积
sv1.equiv(sv2)               # 等价（忽略全局相位）

# 张量积
sv_tensor = sv1.tensor(sv2)

# 演化
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
sv.evolve(H, [0])            # 在 qubit 0 上应用 H
```

### 量子算法示例

#### Grover 搜索

```python
from janus.circuit import Circuit
from janus.simulator import StatevectorSimulator

# 2-qubit Grover 搜索 |11⟩
qc = Circuit.from_layers([
    # 初始化
    [{'name': 'h', 'qubits': [0], 'params': []},
     {'name': 'h', 'qubits': [1], 'params': []}],
    # Oracle (标记 |11⟩)
    [{'name': 'cz', 'qubits': [0, 1], 'params': []}],
    # Diffusion
    [{'name': 'h', 'qubits': [0], 'params': []},
     {'name': 'h', 'qubits': [1], 'params': []}],
    [{'name': 'x', 'qubits': [0], 'params': []},
     {'name': 'x', 'qubits': [1], 'params': []}],
    [{'name': 'cz', 'qubits': [0, 1], 'params': []}],
    [{'name': 'x', 'qubits': [0], 'params': []},
     {'name': 'x', 'qubits': [1], 'params': []}],
    [{'name': 'h', 'qubits': [0], 'params': []},
     {'name': 'h', 'qubits': [1], 'params': []}],
], n_qubits=2)

sim = StatevectorSimulator()
result = sim.run(qc, shots=1000)
print(result.counts)  # {'11': ~1000}
```

#### Bell 态

```python
from janus.circuit import Circuit
from janus.simulator import StatevectorSimulator

qc = Circuit.from_layers([
    [{'name': 'h', 'qubits': [0], 'params': []}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}]
], n_qubits=2)

sim = StatevectorSimulator()
sv = sim.statevector(qc)
print(sv)  # 0.707|00⟩ + 0.707|11⟩
```

#### GHZ 态

```python
from janus.circuit import Circuit
from janus.simulator import StatevectorSimulator

qc = Circuit.from_layers([
    [{'name': 'h', 'qubits': [0], 'params': []}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
    [{'name': 'cx', 'qubits': [1, 2], 'params': []}]
], n_qubits=3)

sim = StatevectorSimulator()
result = sim.run(qc, shots=1000)
print(result.counts)  # {'000': ~500, '111': ~500}
```

### 模拟器 API 参考

#### StatevectorSimulator

| 方法 | 说明 |
|------|------|
| `run(circuit, shots, ...)` | 运行电路并采样 |
| `statevector(circuit, ...)` | 获取最终状态向量 |

#### run() 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `circuit` | Circuit | 量子电路 |
| `shots` | int | 采样次数 |
| `initial_state` | str/Statevector | 初始状态 |
| `parameter_binds` | dict | 参数绑定 |
| `measure_qubits` | list | 测量的量子比特 |

#### NoisySimulator

| 方法 | 说明 |
|------|------|
| `run(circuit, shots, ...)` | 噪声模拟并采样 |
| `density_matrix(circuit, ...)` | 获取密度矩阵 |

## 许可证

MIT License
