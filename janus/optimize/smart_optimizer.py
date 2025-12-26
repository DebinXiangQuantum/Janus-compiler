"""
技术10: 智能优化器 - 根据输入量子电路选择最优优化方案

SmartOptimizer 自动分析电路特征并选择最优的优化策略组合。

Usage:
    from optimize import SmartOptimizer, smart_compile, analyze_and_optimize
    
    # 方法1: 快速优化
    qc_opt = smart_compile(qc, verbose=True)
    
    # 方法2: 使用优化器类
    optimizer = SmartOptimizer(verbose=True)
    qc_opt = optimizer.optimize(qc)
    
    # 方法3: 带报告的优化
    report = analyze_and_optimize(qc)
    print(report['strategy'])
    print(report['improvements'])
"""

import time
from typing import Dict, Any, Optional, Union

from circuit import Circuit as QuantumCircuit
from circuit import circuit_to_dag, dag_to_circuit


class SmartOptimizer:
    """
    智能量子电路优化器
    
    自动分析电路特征（门类型、规模、深度），智能选择最优优化策略。
    
    Attributes:
        verbose: 是否打印详细信息
        
    Example:
        >>> optimizer = SmartOptimizer(verbose=True)
        >>> qc_opt = optimizer.optimize(circuit)
    """
    
    def __init__(self, verbose: bool = False):
        """
        初始化智能优化器
        
        Args:
            verbose: 是否打印详细优化信息
        """
        self.verbose = verbose
        
    def _count_ops(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """统计电路中各种门的数量"""
        ops_count = {}
        for instruction in circuit.data:
            gate_name = instruction.operation.name.lower()
            ops_count[gate_name] = ops_count.get(gate_name, 0) + 1
        return ops_count
    
    def _classify_circuit(self, circuit: QuantumCircuit) -> str:
        """
        分类电路类型
        
        Returns:
            电路类型: 't_heavy', 'rotation_heavy', 'cx_heavy', 
                     'clifford_heavy', 'mixed', 'small_full', 'large_fast'
        """
        ops = self._count_ops(circuit)
        total = len(circuit.data)
        
        if total == 0:
            return 'empty'
        
        # 小电路使用全面优化
        if total < 20:
            return 'small_full'
        
        # 大电路使用快速优化
        if total > 100:
            # 计算各类门的比例
            clifford_gates = (ops.get('h', 0) + ops.get('s', 0) + ops.get('sdg', 0) + 
                            ops.get('x', 0) + ops.get('y', 0) + ops.get('z', 0) + 
                            ops.get('cx', 0))
            t_gates = ops.get('t', 0) + ops.get('tdg', 0)
            rotation_gates = (ops.get('rx', 0) + ops.get('ry', 0) + 
                            ops.get('rz', 0) + ops.get('u', 0))
            cx_gates = ops.get('cx', 0) + ops.get('cz', 0)
            
            t_ratio = t_gates / total
            rotation_ratio = rotation_gates / total
            cx_ratio = cx_gates / total
            clifford_ratio = clifford_gates / total
            
            if t_ratio > 0.3:
                return 't_heavy'
            elif rotation_ratio > 0.3:
                return 'rotation_heavy'
            elif cx_ratio > 0.3:
                return 'cx_heavy'
            elif clifford_ratio > 0.6:
                return 'clifford_heavy'
            else:
                return 'mixed'
        
        return 'mixed'
    
    def _get_strategy_name(self, circuit_type: str) -> str:
        """获取策略名称"""
        strategy_names = {
            'empty': '空电路',
            'small_full': '小电路深度优化',
            't_heavy': 'T门优化策略',
            'rotation_heavy': '门融合策略',
            'cx_heavy': 'CNOT优化策略',
            'clifford_heavy': 'Clifford优化策略',
            'mixed': '通用优化策略',
            'large_fast': '大电路快速优化'
        }
        return strategy_names.get(circuit_type, '通用优化策略')
    
    def _get_passes_for_strategy(self, circuit_type: str) -> list:
        """获取策略对应的优化Pass列表"""
        from optimize import (
            TChinMerger, CliffordMerger,
            SingleQubitGateOptimizer, SingleQubitRunCollector,
            CommutativeGateCanceller, InverseGateCanceller
        )
        from compiler import MergeRotationsPass
        
        passes_map = {
            'empty': [],
            'small_full': [
                ('TChinMerger', TChinMerger()),
                ('MergeRotationsPass', MergeRotationsPass()),
                ('CommutativeGateCanceller', CommutativeGateCanceller()),
                ('InverseGateCanceller', InverseGateCanceller()),
            ],
            't_heavy': [
                ('TChinMerger', TChinMerger()),
                ('CliffordMerger', CliffordMerger()),
                ('InverseGateCanceller', InverseGateCanceller()),
            ],
            'rotation_heavy': [
                ('MergeRotationsPass', MergeRotationsPass()),
                ('SingleQubitRunCollector', SingleQubitRunCollector()),
                ('SingleQubitGateOptimizer', SingleQubitGateOptimizer()),
                ('InverseGateCanceller', InverseGateCanceller()),
            ],
            'cx_heavy': [
                ('CommutativeGateCanceller', CommutativeGateCanceller()),
                ('InverseGateCanceller', InverseGateCanceller()),
            ],
            'clifford_heavy': [
                ('CliffordMerger', CliffordMerger()),
                ('InverseGateCanceller', InverseGateCanceller()),
            ],
            'mixed': [
                ('InverseGateCanceller', InverseGateCanceller()),
                ('CommutativeGateCanceller', CommutativeGateCanceller()),
                ('MergeRotationsPass', MergeRotationsPass()),
            ],
            'large_fast': [
                ('InverseGateCanceller', InverseGateCanceller()),
            ],
        }
        return passes_map.get(circuit_type, passes_map['mixed'])
    
    def optimize(self, circuit: QuantumCircuit, 
                 strategy: Optional[str] = None) -> QuantumCircuit:
        """
        优化量子电路
        
        Args:
            circuit: 输入量子电路
            strategy: 可选，强制使用特定策略
                     可选值: 't_heavy', 'rotation_heavy', 'cx_heavy',
                            'clifford_heavy', 'mixed', 'small_full', 'large_fast'
        
        Returns:
            优化后的量子电路
        """
        if strategy is None:
            circuit_type = self._classify_circuit(circuit)
        else:
            circuit_type = strategy
        
        strategy_name = self._get_strategy_name(circuit_type)
        passes = self._get_passes_for_strategy(circuit_type)
        
        if self.verbose:
            print(f"\n=== 电路分析 ===")
            print(f"电路规模: {'small' if len(circuit.data) < 20 else 'medium' if len(circuit.data) < 100 else 'large'} ({len(circuit.data)}门)")
            print(f"主要特征: {circuit_type}")
            print(f"\n=== 选择策略 ===")
            print(f"策略名称: {strategy_name}")
            print(f"优化Pass: {', '.join([p[0] for p in passes])}")
            print(f"\n=== 执行优化 ===")
        
        # 执行优化
        dag = circuit_to_dag(circuit)
        original_size = len(circuit.data)
        
        for pass_name, pass_obj in passes:
            try:
                dag = pass_obj.run(dag)
                if self.verbose:
                    current_circuit = dag_to_circuit(dag)
                    current_size = len(current_circuit.data)
                    if current_size < original_size:
                        print(f"✓ {pass_name}: 门数 {original_size} → {current_size}")
            except Exception as e:
                if self.verbose:
                    print(f"✗ {pass_name}: 跳过 ({str(e)[:50]})")
        
        optimized_circuit = dag_to_circuit(dag)
        
        if self.verbose:
            optimized_size = len(optimized_circuit.data)
            reduction = original_size - optimized_size
            reduction_rate = reduction / original_size * 100 if original_size > 0 else 0
            print(f"\n=== 优化结果 ===")
            print(f"门数: {original_size} → {optimized_size} ({reduction_rate:.1f}% 减少)")
        
        return optimized_circuit


def smart_compile(circuit: QuantumCircuit, 
                  verbose: bool = False,
                  strategy: Optional[str] = None) -> QuantumCircuit:
    """
    智能编译量子电路（便捷函数）
    
    自动分析电路特征并选择最优的优化策略。
    
    Args:
        circuit: 输入量子电路
        verbose: 是否打印详细信息
        strategy: 可选，强制使用特定策略
        
    Returns:
        优化后的量子电路
        
    Example:
        >>> from optimize import smart_compile
        >>> qc_opt = smart_compile(qc, verbose=True)
    """
    optimizer = SmartOptimizer(verbose=verbose)
    return optimizer.optimize(circuit, strategy=strategy)


def analyze_and_optimize(circuit: QuantumCircuit, 
                         verbose: bool = False) -> Dict[str, Any]:
    """
    分析并优化量子电路，返回详细报告
    
    Args:
        circuit: 输入量子电路
        verbose: 是否打印详细信息
        
    Returns:
        包含以下键的字典:
        - 'circuit': 优化后的电路
        - 'strategy': 选择的策略名称
        - 'circuit_features': 电路特征
        - 'original_stats': 原始电路统计
        - 'optimized_stats': 优化后电路统计
        - 'improvements': 改进指标
        - 'optimization_time': 优化耗时
        
    Example:
        >>> from optimize import analyze_and_optimize
        >>> report = analyze_and_optimize(qc)
        >>> print(report['strategy'])
        >>> print(report['improvements'])
    """
    optimizer = SmartOptimizer(verbose=verbose)
    
    # 分析原始电路
    original_ops = optimizer._count_ops(circuit)
    original_size = len(circuit.data)
    original_depth = circuit.depth
    circuit_type = optimizer._classify_circuit(circuit)
    strategy_name = optimizer._get_strategy_name(circuit_type)
    
    # 执行优化
    start_time = time.time()
    optimized_circuit = optimizer.optimize(circuit)
    optimization_time = time.time() - start_time
    
    # 分析优化后电路
    optimized_ops = optimizer._count_ops(optimized_circuit)
    optimized_size = len(optimized_circuit.data)
    optimized_depth = optimized_circuit.depth
    
    # 计算改进
    gate_reduction = (original_size - optimized_size) / original_size * 100 if original_size > 0 else 0
    depth_reduction = (original_depth - optimized_depth) / original_depth * 100 if original_depth > 0 else 0
    
    return {
        'circuit': optimized_circuit,
        'strategy': strategy_name,
        'circuit_features': {
            'type': circuit_type,
            'n_qubits': circuit.n_qubits,
        },
        'original_stats': {
            'size': original_size,
            'depth': original_depth,
            'ops': original_ops,
        },
        'optimized_stats': {
            'size': optimized_size,
            'depth': optimized_depth,
            'ops': optimized_ops,
        },
        'improvements': {
            'gate_reduction': gate_reduction,
            'depth_reduction': depth_reduction,
            'gates_removed': original_size - optimized_size,
        },
        'optimization_time': optimization_time,
    }
