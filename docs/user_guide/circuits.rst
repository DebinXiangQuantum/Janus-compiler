电路操作
========

本节详细介绍 Janus 电路的创建和操作方法。

创建电路
--------

**推荐方法：from_layers**

使用 ``from_layers`` 方法创建电路，明确指定每层的门：

.. code-block:: python

   from janus.circuit import Circuit

   # 创建 2 量子比特的电路
   # 每个列表元素代表一层，同一层的门可以并行执行
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],           # 第 0 层
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],       # 第 1 层
       [{'name': 'rx', 'qubits': [0], 'params': [1.57]}],      # 第 2 层
   ], n_qubits=2)

**门的格式**

每个门是一个字典，包含三个字段：

- ``name``: 门的名称（如 'h', 'cx', 'rx'）
- ``qubits``: 作用的量子比特列表
- ``params``: 参数列表（无参数门为空列表）

.. code-block:: python

   # 单比特门示例
   {'name': 'h', 'qubits': [0], 'params': []}           # H 门
   {'name': 'rx', 'qubits': [0], 'params': [1.57]}      # RX(π/2) 门
   {'name': 'u', 'qubits': [0], 'params': [1.57, 0, 0]} # U 门

   # 两比特门示例
   {'name': 'cx', 'qubits': [0, 1], 'params': []}       # CNOT 门
   {'name': 'crx', 'qubits': [0, 1], 'params': [1.57]}  # 受控 RX 门

   # 三比特门示例
   {'name': 'ccx', 'qubits': [0, 1, 2], 'params': []}   # Toffoli 门

**并行门**

同一层的门可以并行执行：

.. code-block:: python

   qc = Circuit.from_layers([
       [  # 这三个门在同一层，可以并行执行
           {'name': 'h', 'qubits': [0], 'params': []},
           {'name': 'h', 'qubits': [1], 'params': []},
           {'name': 'h', 'qubits': [2], 'params': []},
       ],
       [  # 第二层
           {'name': 'cx', 'qubits': [0, 1], 'params': []},
       ],
   ], n_qubits=3)

**其他创建方法**

.. code-block:: python

   # 创建空电路（不推荐，但支持）
   qc = Circuit(3)

   # 指定经典比特数
   qc = Circuit.from_layers([...], n_qubits=3, n_clbits=3)

   # 给电路命名
   qc = Circuit.from_layers([...], n_qubits=2, name='my_circuit')

电路属性
--------

.. code-block:: python

   qc.n_qubits            # 量子比特数
   qc.n_clbits            # 经典比特数
   qc.depth               # 电路深度（层数）
   qc.n_gates             # 门总数
   qc.num_two_qubit_gate  # 两比特门数量
   qc.duration            # 估算执行时间
   qc.gates               # 门列表（字典格式）
   qc.layers              # 分层表示
   qc.operated_qubits     # 实际被操作的量子比特
   qc.measured_qubits     # 需要测量的量子比特
   qc.parameters          # 电路中的参数集合
   qc.name                # 电路名称


添加量子门
----------

虽然推荐使用 ``from_layers`` 创建完整电路，但也可以动态添加门：

**使用 append 方法**

.. code-block:: python

   from janus.circuit import Circuit
   from janus.circuit.library import HGate, CXGate

   qc = Circuit(2)
   qc.append(HGate(), [0])
   qc.append(CXGate(), [0, 1])

**链式创建受控门**

.. code-block:: python

   from janus.circuit.library import RXGate, U3Gate

   qc = Circuit(4)
   # 创建受控 RX 门
   qc.gate(RXGate(np.pi/4), 2).control(0)

   # 创建多控制门
   qc.gate(U3Gate(np.pi/4, 0, 0), 3).control([0, 1, 2])

电路操作
--------

**复制电路**

.. code-block:: python

   qc_copy = qc.copy()

**组合电路**

.. code-block:: python

   qc1 = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
   ], n_qubits=2)

   qc2 = Circuit.from_layers([
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
   ], n_qubits=2)

   # 将 qc2 追加到 qc1
   qc1.compose(qc2)

**逆电路**

.. code-block:: python

   # 创建电路的逆（所有门逆序并取逆）
   qc_inv = qc.inverse()

**门移动**

.. code-block:: python

   # 获取门可移动的层范围
   available = qc.get_available_space(gate_index=0)
   print(available)  # range(0, 2)

   # 移动门到新层
   new_qc = qc.move_gate(gate_index=0, new_layer=1)

   # 清理空层
   qc.clean_empty_layers()


导出格式
--------

.. code-block:: python

   # 导出为字典列表
   qc.to_dict_list()
   # [{'name': 'h', 'qubits': [0], 'params': []}, ...]

   # 导出为元组列表
   qc.to_tuple_list()
   # [('h', [0], []), ...]

   # 导出为分层格式
   qc.to_layers()
   # [[{'name': 'h', ...}], [{'name': 'cx', ...}]]

可分离电路
----------

组合多个独立子电路，适用于可以并行执行的场景：

.. code-block:: python

   from janus.circuit import Circuit, SeperatableCircuit
   import numpy as np

   # 创建两个独立的子电路
   c1 = Circuit.from_layers([
       [{'name': 'rx', 'qubits': [0], 'params': [0.785]}],
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
   ], n_qubits=2)

   c2 = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
       [{'name': 'ry', 'qubits': [1], 'params': [1.047]}],
   ], n_qubits=2)

   # 组合为可分离电路
   sep_circuit = SeperatableCircuit([c1, c2], n_qubits=4)

DAG 表示
--------

有向无环图 (DAG) 表示便于电路分析和优化：

.. code-block:: python

   from janus.circuit import circuit_to_dag, dag_to_circuit

   # 电路转 DAG
   dag = circuit_to_dag(qc)

   # DAG 属性
   print(dag.depth())      # 深度
   print(dag.count_ops())  # 门统计 {'h': 1, 'cx': 1}

   # 遍历节点
   for node in dag.op_nodes():
       print(f"门: {node.name}, 量子比特: {node.qubits}")

   # DAG 转回电路
   qc2 = dag_to_circuit(dag)

**DAGDependency**

用于分析门之间的依赖关系：

.. code-block:: python

   from janus.circuit import circuit_to_dag_dependency

   dag_dep = circuit_to_dag_dependency(qc)
   print(dag_dep.size())   # 节点数
   print(dag_dep.depth())  # 深度

文件读写
--------

Janus 提供便捷的电路文件读写功能。

**JSON 格式**

电路以分层 JSON 格式存储：

.. code-block:: json

   [
     [{"name": "h", "qubits": [0], "params": []}],
     [{"name": "cx", "qubits": [0, 1], "params": []}],
     [{"name": "rx", "qubits": [0], "params": [1.57]}]
   ]

**列出可用电路**

.. code-block:: python

   from janus.circuit import list_circuits

   # 列出预置电路
   circuits = list_circuits()
   print(circuits)  # ['bell', 'test']

   # 列出指定目录的电路
   circuits = list_circuits(directory='./my_circuits')

**加载电路**

.. code-block:: python

   from janus.circuit import load_circuit

   # 从预置目录加载（按名称）
   qc = load_circuit(name='bell')

   # 从指定路径加载
   qc = load_circuit(filepath='./my_circuit.json')

**保存电路**

.. code-block:: python

   from janus.circuit import save_circuit, Circuit

   # 创建电路
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
   ], n_qubits=2)

   # 保存电路（自动包含 n_qubits 信息）
   save_circuit(qc, 'my_circuit.json')

保存的文件格式：

.. code-block:: json

   {
     "n_qubits": 2,
     "layers": [
       [{"name": "h", "qubits": [0], "params": []}],
       [{"name": "cx", "qubits": [0, 1], "params": []}]
     ]
   }

**手动保存（仅层数据）**

.. code-block:: python

   import json

   # 仅保存层数据
   with open('circuit_layers.json', 'w') as f:
       json.dump(qc.to_layers(), f, indent=2)

命令行工具
----------

Janus 提供命令行工具查看和操作电路：

.. code-block:: bash

   # 查看电路信息
   python -m janus.circuit.cli info circuit.json
   python -m janus.circuit.cli info circuit.json -v  # 详细信息

   # 绘制电路
   python -m janus.circuit.cli draw circuit.json
   python -m janus.circuit.cli draw circuit.json -o output.png

   # 测试电路
   python -m janus.circuit.cli test circuit.json
