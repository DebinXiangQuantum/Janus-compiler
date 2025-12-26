"""
生成测试用振幅数据文件

使用方法:
    # 生成稠密数据 (4 量子比特 = 16 维)
    python test/encode/generate_data.py -n 4 -o benchmark/dense_data.txt
    
    # 生成稀疏数据 (4 量子比特，只有 5 个非零元素)
    python test/encode/generate_data.py -n 4 -o benchmark/sparse_data.txt --nonzero 5
"""
import argparse
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def generate_random_amplitudes(num_qubits: int, num_nonzero: int = 0, seed: int = None) -> np.ndarray:
    """
    生成随机归一化振幅数据（非负，范围 [0, 1]）
    
    参数:
        num_qubits: 量子比特数，生成 2^num_qubits 个振幅
        num_nonzero: 非零元素数量，0 表示稠密（所有元素非零）
        seed: 随机数种子
    
    返回:
        归一化的振幅数组，所有元素 >= 0，L2 范数 = 1
    """
    if seed is not None:
        np.random.seed(seed)
    
    dim = 2 ** num_qubits
    
    if num_nonzero > 0:
        # 生成稀疏数据
        num_nonzero = min(num_nonzero, dim)  # 不能超过维度
        data = np.zeros(dim)
        indices = np.random.choice(dim, num_nonzero, replace=False)
        # 使用均匀分布生成非负随机数
        data[indices] = np.random.uniform(0.1, 1.0, num_nonzero)
    else:
        # 生成稠密数据，使用均匀分布 [0, 1]
        data = np.random.uniform(0.0, 1.0, dim)
    
    # L2 归一化，使得 sum(|a_i|^2) = 1
    norm = np.linalg.norm(data)
    if norm > 1e-10:
        data = data / norm
    else:
        # 如果全为零，默认设置第一个元素为1
        data[0] = 1.0
    
    return data


def save_amplitudes(data: np.ndarray, filepath: str):
    """
    保存振幅数据到文件
    
    文件格式: 每行一个振幅值
    
    参数:
        data: 振幅数组
        filepath: 输出文件路径
    """
    # 确保目录存在
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for amp in data:
            f.write(f"{amp:.15e}\n")
    
    print(f"Data saved: {filepath}")
    print(f"  Dimension: {len(data)}")
    print(f"  Num qubits: {int(np.log2(len(data)))}")
    print(f"  Nonzero elements: {np.count_nonzero(data)}")
    print(f"  L2 norm: {np.linalg.norm(data):.10f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate quantum state amplitude test data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dense data (4 qubits = 16 dimensions)
  python tests/encode/generate_data.py -n 4 -o benchmark/dense_data.txt
  
  # Generate sparse data (4 qubits, only 5 nonzero elements)
  python tests/encode/generate_data.py -n 4 -o benchmark/sparse_data.txt --nonzero 5
  
  # Generate sparse data with fixed random seed
  python tests/encode/generate_data.py -n 4 -o benchmark/test_data.txt --nonzero 3 --seed 42
        """
    )
    
    parser.add_argument('-n', '--num_qubits', type=int, required=True,
                        help='Number of qubits (generates 2^n amplitudes)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output file path')
    parser.add_argument('--nonzero', type=int, default=0,
                        help='Number of nonzero elements (0 = dense, all nonzero)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible data generation')
    
    args = parser.parse_args()
    
    # 参数验证
    dim = 2 ** args.num_qubits
    
    if args.num_qubits < 1:
        parser.error("Number of qubits must be > 0")
    if args.num_qubits > 20:
        parser.error("Number of qubits cannot exceed 20 (memory limit)")
    if args.nonzero < 0:
        parser.error("Number of nonzero elements must be >= 0")
    if args.nonzero > dim:
        parser.error(f"Number of nonzero elements cannot exceed dimension ({dim})")
    
    # 生成数据
    if args.nonzero > 0:
        print(f"Generating sparse data: {args.num_qubits} qubits, {args.nonzero} nonzero elements...")
    else:
        print(f"Generating dense data: {args.num_qubits} qubits ({dim} dimensions)...")
    
    data = generate_random_amplitudes(args.num_qubits, args.nonzero, args.seed)
    
    # 保存数据
    save_amplitudes(data, args.output)


if __name__ == '__main__':
    main()
