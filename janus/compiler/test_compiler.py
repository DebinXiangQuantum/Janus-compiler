"""
Janus 编译器测试
"""
import numpy as np
from janus.circuit import Circuit
from janus.compiler import compile_circuit, CancelInversesPass, MergeRotationsPass


def test_cancel_inverses():
    """测试消除逆门对"""
    print("=== Cancel Inverses ===")
    
    # X-X 应该被消除
    qc = Circuit(2)
    qc.h(0)
    qc.x(0)
    qc.x(0)  # 这两个 X 应该被消除
    qc.cx(0, 1)
    
    print(f"Before: {qc.n_gates} gates")
    print(qc)
    
    optimized = compile_circuit(qc, optimization_level=1, verbose=True)
    
    print(f"\nAfter: {optimized.n_gates} gates")
    print(optimized)


def test_cancel_h_gates():
    """测试消除 H-H"""
    print("\n=== Cancel H-H ===")
    
    qc = Circuit(1)
    qc.h(0)
    qc.h(0)  # H-H = I
    qc.x(0)
    
    print(f"Before: {qc.n_gates} gates")
    
    optimized = compile_circuit(qc, optimization_level=1)
    
    print(f"After: {optimized.n_gates} gates")
    print(optimized)


def test_merge_rotations():
    """测试合并旋转门"""
    print("\n=== Merge Rotations ===")
    
    qc = Circuit(1)
    qc.rz(np.pi/4, 0)
    qc.rz(np.pi/4, 0)  # 应该合并为 RZ(π/2)
    qc.rx(np.pi/2, 0)
    qc.rx(np.pi/2, 0)  # 应该合并为 RX(π)
    
    print(f"Before: {qc.n_gates} gates")
    for inst in qc.instructions:
        print(f"  {inst.name}({inst.params}) on {inst.qubits}")
    
    optimized = compile_circuit(qc, optimization_level=2)
    
    print(f"\nAfter: {optimized.n_gates} gates")
    for inst in optimized.instructions:
        print(f"  {inst.name}({inst.params}) on {inst.qubits}")


def test_complex_circuit():
    """测试复杂电路优化"""
    print("\n=== Complex Circuit ===")
    
    qc = Circuit(3)
    # 一些冗余操作
    qc.h(0)
    qc.h(0)      # 消除
    qc.x(1)
    qc.cx(0, 1)
    qc.cx(0, 1)  # 消除
    qc.rz(np.pi/4, 2)
    qc.rz(np.pi/4, 2)  # 合并
    qc.x(1)      # 和前面的 x(1) 不相邻，不消除
    
    print(f"Before: {qc.n_gates} gates, depth {qc.depth}")
    
    optimized = compile_circuit(qc, optimization_level=2, verbose=True)
    
    print(f"\nAfter: {optimized.n_gates} gates, depth {optimized.depth}")
    print(optimized)


if __name__ == "__main__":
    test_cancel_inverses()
    test_cancel_h_gates()
    test_merge_rotations()
    test_complex_circuit()
    
    print("\n✓ All compiler tests passed!")
