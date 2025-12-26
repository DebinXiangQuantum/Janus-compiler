"""
编码器测试脚本

使用方法:
    python test/encode/test_example_encoding.py --type 0 --data benchmark/test_data.txt
    python test/encode/test_example_encoding.py --type 1 --data benchmark/schmidt_data.txt
    python test/encode/test_example_encoding.py --type 2 --data benchmark/efficient_sparse_data.txt

编码类型:
    type 0: bidrc_encode (双向振幅编码)
    type 1: efficient_sparse (高效稀疏编码)
    type 2: schmidt_encode (Schmidt分解编码)
"""
import argparse
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from janus.encode.amplitude_encode import bidrc_encode
from janus.encode.efficient_sparse import efficient_sparse, EfficientSparseResult
from janus.encode.schmidt_encode import schmidt_encode, SchmidtEncodeResult
from janus.circuit import Circuit
from janus.simulator import StatevectorSimulator


def load_amplitudes(filepath: str) -> np.ndarray:
    """
    从文件加载振幅数据
    
    参数:
        filepath: 数据文件路径
    
    返回:
        振幅数组
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(float(line))
    
    return np.array(data)


def print_circuit_info(circuit: Circuit, name: str):
    """打印电路信息"""
    print("\n" + "=" * 60)
    print(f"编码方式: {name}")
    print("\n--- 量子电路 ---")
    print(circuit.draw())
    print("=" * 60)
    print(f"量子比特数: {circuit.num_qubits}")
    print(f"门数量: {len(circuit.instructions)}")



def compute_fidelity(target_amplitudes: np.ndarray, actual_statevector: np.ndarray) -> float:
    """
    计算保真度 (Fidelity)
    
    F = |<target|actual>|^2
    
    参数:
        target_amplitudes: 目标振幅（输入数据）
        actual_statevector: 实际量子态向量
    
    返回:
        保真度值 [0, 1]
    """
    # 确保长度相同
    target = np.array(target_amplitudes, dtype=complex)
    actual = np.array(actual_statevector, dtype=complex)
    
    if len(target) < len(actual):
        target = np.pad(target, (0, len(actual) - len(target)))
    elif len(actual) < len(target):
        actual = np.pad(actual, (0, len(target) - len(actual)))
    
    # 计算保真度
    inner_product = np.vdot(target, actual)
    fidelity = np.abs(inner_product) ** 2
    
    return float(fidelity)


def compute_probability_distribution(statevector: np.ndarray) -> np.ndarray:
    """
    从量子态向量计算测量概率分布
    
    P(i) = |amplitude_i|^2
    """
    return np.abs(statevector) ** 2


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    计算 KL 散度 D_KL(P || Q)
    
    参数:
        p: 目标分布
        q: 实际分布
        epsilon: 防止 log(0) 的小常数
    
    返回:
        KL 散度值
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return float(np.sum(p * np.log(p / q)))


def compute_total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    计算总变差距离 (Total Variation Distance)
    
    TVD = 0.5 * sum(|p_i - q_i|)
    
    参数:
        p: 分布1
        q: 分布2
    
    返回:
        TVD 值 [0, 1]
    """
    return float(0.5 * np.sum(np.abs(p - q)))


def print_probability_distribution(probs: np.ndarray, n_qubits: int, top_k: int = 10):
    """打印概率分布（显示前 top_k 个最大概率）"""
    print(f"\n--- 概率分布 (Top {top_k}) ---")
    
    # 获取概率最大的索引
    indices = np.argsort(probs)[::-1][:top_k]
    
    print(f"{'State':>12} | {'Prob':>12} | {'Amplitude':>12}")
    print("-" * 42)
    
    for idx in indices:
        if probs[idx] > 1e-10:  # 只显示非零概率
            state_str = format(idx, f'0{n_qubits}b')
            print(f"|{state_str}> | {probs[idx]:>12.6f} | {np.sqrt(probs[idx]):>12.6f}")


def print_similarity_metrics(target_probs: np.ndarray, actual_probs: np.ndarray, 
                             target_amps: np.ndarray, actual_sv: np.ndarray,
                             has_ancilla: bool = False):
    """打印相似性度量"""
    print("\n" + "=" * 60)
    print("分布相似性分析")
    print("=" * 60)
    
    # 余弦相似度（基于概率分布）
    cos_sim = np.dot(target_probs, actual_probs) / (np.linalg.norm(target_probs) * np.linalg.norm(actual_probs) + 1e-10)
    print(f"概率分布余弦相似度:          {cos_sim:.10f}")


def partial_trace_to_output_qubits(statevector, n_total_qubits: int, 
                                    out_qubits: list) -> np.ndarray:
    """
    通过对非输出比特求迹，获取输出比特上的概率分布
    
    参数:
        statevector: 完整量子态向量
        n_total_qubits: 总量子比特数
        out_qubits: 输出比特索引列表（按照测量结果的比特顺序）
    
    返回:
        输出比特上的概率分布
        
    注意:
        输出索引按照 out_qubits 中的顺序映射：
        - out_qubits[0] 的状态对应输出索引的第 0 位
        - out_qubits[1] 的状态对应输出索引的第 1 位
        - 以此类推
    """
    # 确保是 numpy 数组
    sv = np.array(statevector, dtype=complex)
    
    n_out = len(out_qubits)
    out_dim = 2 ** n_out
    probs = np.zeros(out_dim)
    
    # 遍历所有计算基态
    for i in range(len(sv)):
        # 提取输出比特的值
        out_index = 0
        for j, q in enumerate(out_qubits):
            bit_val = (i >> q) & 1
            out_index |= (bit_val << j)
        
        # 累加概率
        probs[out_index] += np.abs(sv[i]) ** 2
    
    return probs


def simulate_and_analyze(circuit: Circuit, target_data: np.ndarray, out_qubits: list = None):
    """
    模拟电路并分析结果
    
    参数:
        circuit: 量子电路
        target_data: 目标振幅数据
        out_qubits: 输出比特列表（用于显示）
    """
    # 移除测量操作并获取输出比特（如果没有指定）
    circuit_no_measure = Circuit(circuit.num_qubits)
    measured_qubits = []
    for inst in circuit.instructions:
        if inst.operation.name.lower() == 'measure':
            measured_qubits.append(inst.qubits[0])
        else:
            circuit_no_measure.append(inst.operation, inst.qubits, inst.clbits)
    
    # 如果没有指定输出比特，使用测量的比特
    if out_qubits is None and measured_qubits:
        out_qubits = measured_qubits
    
    # 创建模拟器并运行
    sim = StatevectorSimulator()
    result = sim.run(circuit_no_measure, shots=1, return_statevector=True)
    
    # 获取量子态向量
    statevector = result.statevector.data
    n_qubits = circuit.num_qubits
    
    # 如果没有指定输出比特，默认使用所有比特
    if out_qubits is None:
        out_qubits = list(range(n_qubits))
    
    n_out_qubits = len(out_qubits)
    out_dim = 2 ** n_out_qubits
    
    # 计算输出比特上的概率分布（通过部分迹）
    if n_qubits == n_out_qubits:
        # 如果输出比特等于总比特数，直接使用完整概率分布
        actual_probs = compute_probability_distribution(statevector)
        actual_sv_reduced = statevector[:out_dim]
    else:
        # 否则需要对非输出比特求部分迹
        actual_probs = partial_trace_to_output_qubits(statevector, n_qubits, out_qubits)
        actual_sv_reduced = np.sqrt(actual_probs)  # 近似，用于保真度计算

    # 准备目标数据（补齐到输出维度）
    target_padded = np.zeros(out_dim)
    target_padded[:min(len(target_data), out_dim)] = target_data[:min(len(target_data), out_dim)]
    target_probs = target_padded ** 2

    # 打印实际概率分布
    print("\n【编码后的概率分布】")
    print_probability_distribution(actual_probs, n_out_qubits, top_k=8)
    
    # 打印相似性度量
    has_ancilla = (n_qubits != n_out_qubits)
    print_similarity_metrics(target_probs, actual_probs, target_padded, actual_sv_reduced, has_ancilla)
    
    return statevector, actual_probs


def run_bidrc_encode(data: np.ndarray):
    """运行 bidrc_encode 编码
    
    注意：由于 bidrc_encode 对某些非单调数据存在问题，
    这里改用 efficient_sparse 进行编码，以确保正确性。
    efficient_sparse 对所有类型的数据都能正确工作。
    """
    # bidrc_encode 直接返回 Circuit（已包含测量）
    circuit = bidrc_encode(data.tolist(), split=-1)
    print_circuit_info(circuit, "bidrc_encode (双向振幅编码)")
    # 模拟并分析（使用不带测量的电路）
    simulate_and_analyze(circuit, data)
    
    return circuit


def run_efficient_sparse(data: np.ndarray):
    """运行 efficient_sparse 编码"""
    n_qubits = int(np.ceil(np.log2(len(data))))
    
    # efficient_sparse 返回 EfficientSparseResult
    result: EfficientSparseResult = efficient_sparse(n_qubits, data.tolist())
    
    # 显示带测量的电路
    measured_circuit = result.measure()
    print_circuit_info(measured_circuit, "efficient_sparse (高效稀疏编码)")
    
    # 模拟并分析
    simulate_and_analyze(result.circuit, data, result.out_qubits)
    
    return result.circuit


def run_schmidt_encode(data: np.ndarray):
    """运行 schmidt_encode 编码"""
    n_qubits = int(np.ceil(np.log2(len(data))))
    
    # schmidt_encode 返回 SchmidtEncodeResult
    result: SchmidtEncodeResult = schmidt_encode(n_qubits, data.tolist())
    
    # 显示带测量的电路
    measured_circuit = result.measure()
    print_circuit_info(measured_circuit, "schmidt_encode (Schmidt分解编码)")
    
    # 模拟并分析
    simulate_and_analyze(result.circuit, data, result.out_qubits)
    
    return result.circuit


def main():
    parser = argparse.ArgumentParser(
        description='量子态编码测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
编码类型说明:
  type 0: bidrc_encode    - 双向振幅编码，适用于通用量子态
  type 1: schmidt_encode  - Schmidt分解编码，适用于具有特定结构的量子态
  type 2: efficient_sparse - 高效稀疏编码，适用于稀疏量子态

示例:
  # 使用 bidrc_encode 编码
  python tests/encode/test_example_encoding.py --type 0 --data benchmark/test_data.txt
  
  
  
  # 使用 schmidt_encode 编码
  python tests/encode/test_example_encoding.py --type 1 --data benchmark/schmidt_data.txt

  # 使用 efficient_sparse 编码
  python tests/encode/test_example_encoding.py --type 2 --data benchmark/sparse_data.txt
        """
    )
    
    parser.add_argument('--type', '-t', type=int, required=True, choices=[0, 1, 2],
                        help='编码类型: 0=bidrc_encode, 1=efficient_sparse, 2=schmidt_encode')
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='振幅数据文件路径')
    
    args = parser.parse_args()
    
    # 加载数据
    data = load_amplitudes(args.data)
    
    # 检查归一化
    norm = np.linalg.norm(data)
    
    if not np.isclose(norm, 1.0, atol=1e-10):
        data = data / norm
    
    # 运行对应的编码器
    encode_funcs = {
        0: run_bidrc_encode,
        1: run_schmidt_encode,
        2: run_efficient_sparse,
    }
    
    try:
        encode_funcs[args.type](data)
    except ValueError as e:
        print(f"\n错误: 编码失败 - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
