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
janus/
├── circuit/
│   ├── circuit.py      # Circuit、SeperatableCircuit
│   ├── gate.py         # 门基类
│   ├── instruction.py  # 指令类
│   ├── layer.py        # 层表示
│   ├── dag.py          # DAG 表示
│   ├── parameter.py    # 参数化支持
│   └── library/        # 标准门库 (60+)
├── compiler/
│   ├── compiler.py     # 编译主函数
│   └── passes.py       # 优化 Pass
└── encode/
    └── schmidt_encode.py
```

## 许可证

MIT License
