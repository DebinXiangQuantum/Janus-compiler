快速入门
========

本教程将带你快速了解 Janus 的核心功能，预计阅读时间 10 分钟。

第一步：创建量子电路
--------------------

量子电路是量子计算的基本单元。让我们从创建一个简单的电路开始：

.. code-block:: python

   from janus.circuit import Circuit
   import numpy as np

   # 使用 from_layers 创建电路
   # 每个列表元素代表一层，同一层的门可以并行执行
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],           # 第 0 层：H 门
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],       # 第 1 层：CNOT 门
       [{'name': 'rx', 'qubits': [0], 'params': [0.785]}],     # 第 2 层：RX(π/4) 门
   ], n_qubits=2)

   # 查看电路
   print(qc.draw())

输出::

   q0: ─H─●─Rx(0.79)─
         │
   q1: ───X───────────

.. tip::
   ``from_layers`` 方法使用分层结构创建电路，每层是一个门列表。门的格式为 ``{'name': '门名', 'qubits': [量子比特], 'params': [参数]}``。

第二步：运行模拟
----------------

创建电路后，使用模拟器运行它：

.. code-block:: python

   from janus.simulator import StatevectorSimulator

   # 创建模拟器
   sim = StatevectorSimulator()

   # 运行电路，测量 1000 次
   result = sim.run(qc, shots=1000)

   # 查看测量结果
   print(result.counts)
   # 输出类似: {'00': 503, '11': 497}

   # 获取最终状态向量
   sv = sim.statevector(qc)
   print(sv.probabilities())
   # 输出: [0.5, 0.0, 0.0, 0.5]

.. note::
   ``shots`` 参数指定测量次数。由于量子测量的概率性，每次运行结果可能略有不同。

第三步：参数化电路
------------------

参数化电路允许你创建可调节的量子电路，这在变分量子算法中非常有用：

.. code-block:: python

   from janus.circuit import Circuit, Parameter
   import numpy as np

   # 创建参数
   theta = Parameter('theta')
   phi = Parameter('phi')

   # 创建参数化电路
   qc = Circuit.from_layers([
       [{'name': 'ry', 'qubits': [0], 'params': [theta]}],   # 参数化 RY 门
       [{'name': 'rx', 'qubits': [1], 'params': [phi]}],     # 参数化 RX 门
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],     # CNOT 门
   ], n_qubits=2)

   # 查看电路参数
   print(f"电路参数: {qc.parameters}")
   print(f"是否参数化: {qc.is_parameterized()}")

   # 方法 1：绑定参数创建新电路
   bound_qc = qc.bind_parameters({theta: np.pi/2, phi: np.pi/4})

   # 方法 2：在模拟时绑定参数
   sim = StatevectorSimulator()
   result = sim.run(qc, shots=100, parameter_binds={
       'theta': np.pi/2,
       'phi': np.pi/4
   })

第四步：电路优化
----------------

Janus 提供电路优化功能，可以减少门数量：

.. code-block:: python

   from janus.circuit import Circuit
   from janus.compiler import compile_circuit
   import numpy as np

   # 创建一个有冗余的电路
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
       [{'name': 'h', 'qubits': [0], 'params': []}],              # 两个 H 门相消
       [{'name': 'rz', 'qubits': [0], 'params': [0.785]}],        # π/4
       [{'name': 'rz', 'qubits': [0], 'params': [0.785]}],        # 两个 RZ 门可以合并
   ], n_qubits=2)

   print(f"优化前: {qc.n_gates} 个门")

   # 优化电路
   optimized = compile_circuit(qc, optimization_level=2)
   print(f"优化后: {optimized.n_gates} 个门")

优化级别说明：

- **级别 0**: 无优化
- **级别 1**: 移除恒等门、消除逆门对
- **级别 2**: 级别 1 + 合并连续旋转门

第五步：噪声模拟
----------------

真实的量子计算机存在噪声。Janus 支持噪声模拟：

.. code-block:: python

   from janus.circuit import Circuit
   from janus.simulator import NoisySimulator, NoiseModel, depolarizing_channel

   # 创建电路
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
   ], n_qubits=2)

   # 创建噪声模型
   noise_model = NoiseModel()

   # 为单比特门添加 1% 去极化噪声
   noise_model.add_all_qubit_quantum_error(
       depolarizing_channel(0.01),
       ['h', 'x', 'rx', 'ry', 'rz']
   )

   # 为两比特门添加 2% 去极化噪声
   noise_model.add_all_qubit_quantum_error(
       depolarizing_channel(0.02),
       ['cx', 'cz']
   )

   # 噪声模拟
   noisy_sim = NoisySimulator(noise_model, seed=42)
   result = noisy_sim.run(qc, shots=1000)

   print(result.counts)
   # 输出会包含一些错误结果，如 {'00': 480, '11': 470, '01': 25, '10': 25}

第六步：保存和加载电路
----------------------

Janus 提供便捷的电路文件操作函数：

.. code-block:: python

   from janus.circuit import Circuit, load_circuit, save_circuit, list_circuits

   # 创建电路
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
   ], n_qubits=2)

   # 保存电路
   save_circuit(qc, 'my_bell.json')

   # 列出预置电路
   print(list_circuits())  # ['bell', 'test']

   # 加载预置电路
   bell = load_circuit(name='bell')

   # 加载自定义电路
   qc_loaded = load_circuit(filepath='my_bell.json')

完整示例：Bell 态
-----------------

让我们把所有内容整合到一个完整的例子中：

.. code-block:: python

   """
   创建、模拟和分析 Bell 态
   """
   from janus.circuit import Circuit
   from janus.simulator import StatevectorSimulator, Statevector
   import numpy as np

   # 1. 创建 Bell 态电路
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
   ], n_qubits=2)

   print("=== Bell 态电路 ===")
   print(qc.draw())
   print(f"电路深度: {qc.depth}")
   print(f"门数量: {qc.n_gates}")

   # 2. 模拟
   sim = StatevectorSimulator(seed=42)
   result = sim.run(qc, shots=10000)

   print("\n=== 测量结果 ===")
   print(f"测量统计: {result.counts}")

   # 3. 分析状态向量
   sv = sim.statevector(qc)
   print("\n=== 状态向量分析 ===")
   print(f"状态向量: {sv}")
   print(f"概率分布: {sv.probabilities()}")

   # 4. 验证纠缠
   # Bell 态应该只有 |00⟩ 和 |11⟩，概率各 50%
   probs = sv.probabilities()
   assert abs(probs[0] - 0.5) < 0.01, "P(00) 应该约为 0.5"
   assert abs(probs[3] - 0.5) < 0.01, "P(11) 应该约为 0.5"
   print("\n✓ Bell 态验证通过！")

下一步
------

恭喜你完成了快速入门！接下来可以：

- :doc:`concepts` - 深入了解量子计算核心概念
- :doc:`../user_guide/circuits` - 学习更多电路操作
- :doc:`../user_guide/gates` - 查看完整的量子门库
- :doc:`../user_guide/examples` - 更多实用示例
