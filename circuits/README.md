# 电路 JSON 文件存储目录

此目录用于存放量子电路的 JSON 文件。

## 文件格式

分层格式，每层是一个门列表：

```json
[
  [{"name": "h", "qubits": [0], "params": []}],
  [{"name": "cx", "qubits": [0, 1], "params": []}],
  [{"name": "rx", "qubits": [0], "params": [1.57]}]
]
```

或包含量子比特数的完整格式：

```json
{
  "n_qubits": 2,
  "layers": [
    [{"name": "h", "qubits": [0], "params": []}],
    [{"name": "cx", "qubits": [0, 1], "params": []}]
  ]
}
```

## 使用方法

```python
from janus.circuit import load_circuit, save_circuit, list_circuits

# 列出所有预置电路
print(list_circuits())  # ['bell', 'test']

# 加载预置电路
qc = load_circuit(name='bell')

# 从指定路径加载
qc = load_circuit(filepath='./my_circuit.json')

# 保存电路到文件
save_circuit(qc, './my_circuit.json')
```

## 预置电路

| 文件名 | 描述 |
|--------|------|
| bell.json | Bell 态电路 (2 量子比特) |
| test.json | 测试电路 |

## 命令行工具

```bash
# 查看电路信息
python -m janus.circuit.cli info circuits/bell.json

# 绘制电路
python -m janus.circuit.cli draw circuits/bell.json
```
