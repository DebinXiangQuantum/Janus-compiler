import sys
import os

# Add the project root directory to sys.path so Python can find the local qiskit module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from encode import bid_amplitude_encode_qiskit, schmidt_encode_qiskit, efficient_sparse_qiskit

# 准备示例数据：归一化的向量
def normalize_data(data):
    """将数据归一化，使其L2范数为1"""
    norm = np.linalg.norm(data)
    if norm == 0:
        raise ValueError("数据不能全为0")
    return (np.array(data) / norm).tolist()

# 示例1: bid_amplitude_encode_qiskit
def example_bid_amplitude_encode():
    print("\n=== bid_amplitude_encode_qiskit 示例 ===")
    
    # 准备归一化数据
    data = normalize_data([1, 1, 1, 1])  # 4个元素，需要2个量子比特
    print(f"输入数据: {data}")
    print(f"数据范数: {np.linalg.norm(data)}")
    
    # 使用默认split参数
    circuit1 = bid_amplitude_encode_qiskit(data)
    print(f"\n默认split参数的电路深度: {circuit1.depth()}")
    print(f"电路门数: {circuit1.size()}")
    print("电路:")
    print(circuit1)
    
    # 使用自定义split参数
    circuit2 = bid_amplitude_encode_qiskit(data, split=1)
    print(f"\n自定义split=1的电路深度: {circuit2.depth()}")
    print(f"电路门数: {circuit2.size()}")
    print("电路:")
    print(circuit2)

# 示例2: schmidt_encode_qiskit
def example_schmidt_encode():
    print("\n=== schmidt_encode_qiskit 示例 ===")
    
    # 准备归一化数据
    data = normalize_data([1, 0, 0, 1])  # 4个元素，需要2个量子比特
    print(f"输入数据: {data}")
    print(f"数据范数: {np.linalg.norm(data)}")
    
    # 使用不同的截断参数
    q_size = 3  # 可用量子比特数
    
    circuit1 = schmidt_encode_qiskit(q_size, data, cutoff=1e-4)
    print(f"\n截断参数cutoff=1e-4的电路深度: {circuit1.depth()}")
    print(f"电路门数: {circuit1.size()}")
    print("电路:")
    print(circuit1)
    
    circuit2 = schmidt_encode_qiskit(q_size, data, cutoff=1e-2)
    print(f"\n截断参数cutoff=1e-2的电路深度: {circuit2.depth()}")
    print(f"电路门数: {circuit2.size()}")
    print("电路:")
    print(circuit2)

# 示例3: efficient_sparse_qiskit
def example_efficient_sparse_encode():
    print("\n=== efficient_sparse_qiskit 示例 ===")
    
    # 准备归一化的稀疏数据
    data = normalize_data([1, 0, 0, 0, 0, 0, 0, 1])  # 8个元素，其中6个为0
    print(f"输入数据: {data}")
    print(f"数据范数: {np.linalg.norm(data)}")
    
    # 使用列表形式输入
    q_size = 4  # 可用量子比特数
    circuit1 = efficient_sparse_qiskit(q_size, data)
    print(f"\n列表形式输入的电路深度: {circuit1.depth()}")
    print(f"电路门数: {circuit1.size()}")
    print("电路:")
    print(circuit1)
    
    # 使用字典形式输入（更直观地表示稀疏数据）
    sparse_data = {
        '000': 1/np.sqrt(2),
        '111': 1/np.sqrt(2)
    }
    print(f"\n字典形式输入: {sparse_data}")
    circuit2 = efficient_sparse_qiskit(q_size, sparse_data)
    print(f"\n字典形式输入的电路深度: {circuit2.depth()}")
    print(f"电路门数: {circuit2.size()}")
    print("电路:")
    print(circuit2)

# 运行所有示例
if __name__ == "__main__":
    print("量子振幅编码函数示例")
    print("=" * 50)
    
    example_bid_amplitude_encode()
    example_schmidt_encode()
    example_efficient_sparse_encode()
    
    print("\n" + "=" * 50)
    print("所有示例运行完毕！")
