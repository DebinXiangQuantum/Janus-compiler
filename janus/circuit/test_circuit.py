"""
Janus Circuit 测试
"""
import numpy as np
from janus.circuit import Circuit, HGate, CXGate


def test_basic_circuit():
    """测试基本电路构建"""
    # 创建 Bell 态电路
    qc = Circuit(2, name="Bell")
    qc.h(0)
    qc.cx(0, 1)
    
    print("=== Bell State Circuit ===")
    print(qc)
    print(f"\nDepth: {qc.depth}")
    print(f"Gates: {qc.n_gates}")
    print(f"Two-qubit gates: {qc.num_two_qubit_gates}")
    print(f"\nLayers: {qc.layers}")
    print(f"\nText drawing:\n{qc.draw()}")


def test_parametric_gates():
    """测试参数化门"""
    qc = Circuit(3, name="Parametric")
    qc.rx(np.pi/2, 0)
    qc.ry(np.pi/4, 1)
    qc.rz(np.pi, 2)
    qc.crz(np.pi/2, 0, 1)
    
    print("\n=== Parametric Circuit ===")
    print(qc)


def test_layer_computation():
    """测试分层计算"""
    qc = Circuit(4)
    # 第一层：并行的单比特门
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    # 第二层：两比特门
    qc.cx(0, 1)
    qc.cx(2, 3)
    # 第三层
    qc.cx(1, 2)
    
    print("\n=== Layer Computation ===")
    print(f"Total gates: {qc.n_gates}")
    print(f"Depth: {qc.depth}")
    for i, layer in enumerate(qc.layers):
        print(f"Layer {i}: {[inst.name for inst in layer]}")


def test_circuit_copy():
    """测试电路复制"""
    qc1 = Circuit(2)
    qc1.h(0)
    qc1.cx(0, 1)
    
    qc2 = qc1.copy()
    qc2.x(0)
    
    print("\n=== Circuit Copy ===")
    print(f"Original: {qc1.n_gates} gates")
    print(f"Copy: {qc2.n_gates} gates")


def test_gate_matrix():
    """测试门矩阵"""
    print("\n=== Gate Matrices ===")
    
    h = HGate()
    print(f"H gate:\n{h.to_matrix()}")
    
    cx = CXGate()
    print(f"\nCX gate:\n{cx.to_matrix()}")


def test_to_dict():
    """测试转换为字典格式"""
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    print("\n=== To Dict Format ===")
    print("Instructions:", qc.to_instructions())
    print("Layers:", qc.to_layers())


def test_array_conversion():
    """测试数组转换"""
    from janus.circuit.converters import to_instruction_list, from_instruction_list
    
    print("\n=== Array Conversion ===")
    
    # 创建电路
    jc = Circuit(2)
    jc.h(0)
    jc.cx(0, 1)
    jc.rx(np.pi/2, 0)
    
    # 转换为元组格式数组
    inst_list = to_instruction_list(jc)
    print("Circuit -> Tuple array:")
    print(inst_list)
    
    # 从元组格式数组重建电路
    jc2 = from_instruction_list(inst_list)
    print("\nTuple array -> Circuit:")
    print(jc2)
    
    # 也支持字典格式
    dict_list = [
        {'name': 'h', 'qubits': [0], 'params': []},
        {'name': 'cx', 'qubits': [0, 1], 'params': []},
        {'name': 'rz', 'qubits': [1], 'params': [np.pi/4]}
    ]
    print("\nDict array -> Circuit:")
    jc3 = from_instruction_list(dict_list)
    print(jc3)


def test_circuit_data_formats():
    """测试电路数据格式"""
    print("\n=== Circuit Data Formats ===")
    
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(np.pi/2, 0)
    
    print("Janus (dict):", qc.to_dict_list())
    print("Tuple:", qc.to_tuple_list())


if __name__ == "__main__":
    test_basic_circuit()
    test_parametric_gates()
    test_layer_computation()
    test_circuit_copy()
    test_gate_matrix()
    test_to_dict()
    test_array_conversion()
    test_circuit_data_formats()
    
    print("\n✓ All tests passed!")
