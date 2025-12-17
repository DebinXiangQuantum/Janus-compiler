# Janus 量子电路框架

轻量级量子电路构建、表示和编译框架。

## 安装

```bash
pip install numpy
```

## 快速开始

### 创建电路

```python
from janus.circuit import Circuit
import numpy as np

# 创建 2 量子比特电路
qc = Circuit(2, name="Bell")

# 添加门
qc.h(0)           # Hadamard 门
qc.cx(0, 1)       # CNOT 门
qc.rx(np.pi/2, 0) # RX 旋转门

print(qc)
print(qc.draw())
```

### 支持的门

| 单比特门 | 两比特门 |
|---------|---------|
| h, x, y, z | cx, cz |
| s, t | swap |
| rx, ry, rz | crz |
| u (通用门) | |

所有门都支持可选的 `params` 参数：
```python
qc.h(0)                      # 默认
qc.h(0, params=[1.0, 2.0])   # 自定义参数
```

### 添加门的方法

#### append()
直接添加 Gate 对象到电路：
```python
from janus.circuit.library import HGate, CXGate, RXGate

qc = Circuit(2)
qc.append(HGate(), [0])                    # 添加 H 门到 qubit 0
qc.append(HGate([1.0, 2.0]), [0])          # 添加带自定义 params 的 H 门
qc.append(CXGate(), [0, 1])                # 添加 CX 门
qc.append(CXGate([0.5]), [0, 1])           # 添加带自定义 params 的 CX 门
qc.append(RXGate(np.pi/2), [0])            # 添加带角度参数的 RX 门
```

#### _add_gate()
内部方法，简化版的 append（不支持经典比特）：
```python
# 等价于 qc.append(HGate(), [0])
qc._add_gate(HGate(), [0])
```

#### 快捷方法 vs append
```python
# 快捷方法 - 更简洁
qc.h(0)
qc.cx(0, 1)
qc.rx(np.pi/2, 0)

# append - 更灵活，可使用自定义 Gate
from janus.circuit.library import HGate
qc.append(HGate(), [0])
```

### 电路属性

```python
qc.n_qubits          # 量子比特数
qc.n_gates           # 门数量
qc.depth             # 电路深度
qc.layers            # 分层表示
qc.num_two_qubit_gates  # 两比特门数量
```

### 导出电路数据

```python
# Janus 格式 (字典列表)
qc.to_dict_list()
# [{'name': 'h', 'qubits': [0], 'params': []}, ...]

# 元组格式(qiskt格式)
qc.to_tuple_list()
# [('h', [0], []), ('cx', [0, 1], []), ...]
```

### 数组转换

```python
from janus.circuit.converters import from_instruction_list, to_instruction_list

# 电路 → 数组
inst_list = to_instruction_list(qc)

# 数组 → 电路 (支持元组和字典格式)
qc2 = from_instruction_list([
    ('h', [0], []),
    ('cx', [0, 1], [])
])
```

## 编译器

### 基础优化

```python
from janus.compiler import compile_circuit

qc = Circuit(2)
qc.h(0)
qc.x(0)
qc.x(0)  # 冗余，会被消除

optimized = compile_circuit(qc, optimization_level=1)
```

### 优化级别

| 级别 | 优化内容 |
|-----|---------|
| 0 | 无优化 |
| 1 | 移除恒等门、消除逆门对 (X-X, H-H) |
| 2 | 级别1 + 合并连续旋转门 |

### 自定义 Pass

```python
from janus.compiler import compile_circuit, CancelInversesPass, MergeRotationsPass

optimized = compile_circuit(qc, passes=[
    CancelInversesPass(),
    MergeRotationsPass(),
])
```

## DAG 表示

```python
from janus.circuit import circuit_to_dag, dag_to_circuit

# 转换为 DAG
dag = circuit_to_dag(qc)

print(dag.depth())      # DAG 深度
print(dag.count_ops())  # 门统计
print(dag.layers())     # 分层

# 转换回电路
qc2 = dag_to_circuit(dag)
```

## 参数化电路

```python
from janus.circuit import Circuit, Parameter

theta = Parameter('θ')
qc = Circuit(1)
qc.rx(theta, 0)

# 绑定参数
bound_qc = qc.assign_parameters({theta: np.pi/2})
```

## 模块结构

```
janus/
├── circuit/
│   ├── circuit.py      # 核心 Circuit 类
│   ├── gate.py         # 门基类
│   ├── dag.py          # DAG 表示
│   ├── converters.py   # 格式转换
│   ├── parameter.py    # 参数化支持
│   └── library/        # 标准门库
└── compiler/
    ├── compiler.py     # 编译主函数
    └── passes.py       # 优化 Pass
```
