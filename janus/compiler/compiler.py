"""
Janus 编译器主模块
"""
from typing import List, Optional

from janus.circuit import Circuit
from janus.circuit.dag import circuit_to_dag, dag_to_circuit
from .passes import BasePass, RemoveIdentityPass, CancelInversesPass, MergeRotationsPass


def compile_circuit(
    circuit: Circuit,
    passes: Optional[List[BasePass]] = None,
    optimization_level: int = 1,
    verbose: bool = False
) -> Circuit:
    """
    编译/优化量子电路
    
    Args:
        circuit: 输入电路
        passes: 自定义 pass 列表，如果为 None 则使用默认 pass
        optimization_level: 优化级别 (0=无优化, 1=基础优化, 2=更多优化)
        verbose: 是否打印优化过程
    
    Returns:
        优化后的电路
    """
    # 默认 pass
    if passes is None:
        if optimization_level == 0:
            passes = []
        elif optimization_level == 1:
            passes = [
                RemoveIdentityPass(),
                CancelInversesPass(),
            ]
        else:  # level >= 2
            passes = [
                RemoveIdentityPass(),
                CancelInversesPass(),
                MergeRotationsPass(),
            ]
    
    if not passes:
        return circuit.copy()
    
    # 转换为 DAG
    dag = circuit_to_dag(circuit)
    
    if verbose:
        print(f"Initial: {dag}")
    
    # 执行 passes
    for p in passes:
        dag = p.run(dag)
        if verbose:
            print(f"After {p.name}: {dag}")
    
    # 转换回电路
    result = dag_to_circuit(dag)
    
    if verbose:
        print(f"Final: {result.n_gates} gates, depth {result.depth}")
    
    return result
