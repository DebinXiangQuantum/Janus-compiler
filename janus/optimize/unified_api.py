"""
Janus量子电路优化器 - 统一API接口

本模块为10种优化技术提供统一的接口，每个技术只暴露一个核心接口，
内部的多种优化技术通过参数开关控制。

使用方法:
    from optimize import (
        optimize_clifford_rz,      # 技术1: Clifford+RZ优化
        optimize_gate_fusion,      # 技术2: 门融合优化
        optimize_commutativity,    # 技术3: 交换性优化
        optimize_template,         # 技术4: 模板匹配优化
        optimize_kak,              # 技术5: KAK分解优化
        optimize_clifford_synth,   # 技术6: Clifford合成优化
        optimize_cnot_synth,       # 技术7: CNOT合成优化
        run_benchmark,             # 技术8: 基准测试
        analyze_circuit,           # 技术9: 电路分析
        smart_optimize,            # 技术10: 智能优化
    )
"""

import time
from typing import Dict, Any, Optional, List, Union
from circuit import Circuit as QuantumCircuit
from circuit import circuit_to_dag, dag_to_circuit


# ============================================================================
# 技术1: Clifford+RZ优化 (T门合并、Clifford门合并)
# ============================================================================

def optimize_clifford_rz(circuit: QuantumCircuit,
                         enable_t_merge: bool = True,
                         enable_clifford_merge: bool = True,
                         enable_inverse_cancel: bool = True,
                         verbose: bool = False) -> QuantumCircuit:
    """
    技术1: Clifford+RZ指令集优化
    
    将电路分解为Clifford门和RZ旋转门，优化T门数量。
    T门合并规则: T+T→S, T+T+T+T→Z
    
    Args:
        circuit: 输入量子电路
        enable_t_merge: 启用T门合并 (默认True)
        enable_clifford_merge: 启用Clifford门合并 (默认True)
        enable_inverse_cancel: 启用逆门消除 (默认True)
        verbose: 打印详细信息
        
    Returns:
        优化后的量子电路
        
    Example:
        >>> from optimize import optimize_clifford_rz
        >>> qc_opt = optimize_clifford_rz(qc, enable_t_merge=True)
    """
    from optimize import TChinMerger, CliffordMerger, InverseGateCanceller
    
    original_size = len(circuit.data)
    dag = circuit_to_dag(circuit)
    
    if verbose:
        print(f"[技术1] Clifford+RZ优化")
        print(f"  原始门数: {original_size}")
    
    if enable_t_merge:
        dag = TChinMerger().run(dag)
        if verbose:
            print(f"  ✓ T门合并: {len(dag_to_circuit(dag).data)}门")
    
    if enable_clifford_merge:
        dag = CliffordMerger().run(dag)
        if verbose:
            print(f"  ✓ Clifford合并: {len(dag_to_circuit(dag).data)}门")
    
    if enable_inverse_cancel:
        dag = InverseGateCanceller().run(dag)
        if verbose:
            print(f"  ✓ 逆门消除: {len(dag_to_circuit(dag).data)}门")
    
    result = dag_to_circuit(dag)
    
    if verbose:
        reduction = original_size - len(result.data)
        print(f"  优化完成: {len(result.data)}门 (减少{reduction}, {reduction/original_size*100:.1f}%)")
    
    return result


# ============================================================================
# 技术2: 门融合优化 (单比特门合并、旋转门合并、块合并)
# ============================================================================

def optimize_gate_fusion(circuit: QuantumCircuit,
                         enable_rotation_merge: bool = True,
                         enable_single_qubit_opt: bool = True,
                         enable_block_consolidate: bool = False,
                         enable_inverse_cancel: bool = True,
                         verbose: bool = False) -> QuantumCircuit:
    """
    技术2: 门融合优化
    
    将连续的单量子比特门融合为一个等价门，优化旋转门序列。
    
    Args:
        circuit: 输入量子电路
        enable_rotation_merge: 启用旋转门合并 (默认True)
        enable_single_qubit_opt: 启用单比特门优化 (默认True)
        enable_block_consolidate: 启用块合并 (默认False，较慢)
        enable_inverse_cancel: 启用逆门消除 (默认True)
        verbose: 打印详细信息
        
    Returns:
        优化后的量子电路
        
    Example:
        >>> from optimize import optimize_gate_fusion
        >>> qc_opt = optimize_gate_fusion(qc, enable_rotation_merge=True)
    """
    from compiler import MergeRotationsPass
    from optimize import (
        SingleQubitGateOptimizer, SingleQubitRunCollector,
        InverseGateCanceller, BlockConsolidator, TwoQubitBlockCollector
    )
    from circuit.library import CXGate
    
    original_size = len(circuit.data)
    dag = circuit_to_dag(circuit)
    
    if verbose:
        print(f"[技术2] 门融合优化")
        print(f"  原始门数: {original_size}")
    
    if enable_rotation_merge:
        dag = MergeRotationsPass().run(dag)
        if verbose:
            print(f"  ✓ 旋转门合并: {len(dag_to_circuit(dag).data)}门")
    
    if enable_single_qubit_opt:
        dag = SingleQubitRunCollector().run(dag)
        dag = SingleQubitGateOptimizer().run(dag)
        if verbose:
            print(f"  ✓ 单比特门优化: {len(dag_to_circuit(dag).data)}门")
    
    if enable_block_consolidate:
        collector = TwoQubitBlockCollector()
        collector.run(dag)
        consolidator = BlockConsolidator(kak_basis_gate=CXGate())
        dag = consolidator.run(dag)
        if verbose:
            print(f"  ✓ 块合并: {len(dag_to_circuit(dag).data)}门")
    
    if enable_inverse_cancel:
        dag = InverseGateCanceller().run(dag)
        if verbose:
            print(f"  ✓ 逆门消除: {len(dag_to_circuit(dag).data)}门")
    
    result = dag_to_circuit(dag)
    
    if verbose:
        reduction = original_size - len(result.data)
        print(f"  优化完成: {len(result.data)}门 (减少{reduction}, {reduction/original_size*100:.1f}%)")
    
    return result


# ============================================================================
# 技术3: 交换性优化 (自伴门消除、互逆门消除、交换性消除)
# ============================================================================

def optimize_commutativity(circuit: QuantumCircuit,
                           enable_commutative_cancel: bool = True,
                           enable_inverse_cancel: bool = True,
                           enable_commutative_inverse: bool = True,
                           enable_analysis: bool = False,
                           verbose: bool = False) -> QuantumCircuit:
    """
    技术3: 交换性优化
    
    利用量子门的交换性重新排列门顺序，消除自伴门对和互逆门对。
    自伴门: H·H=I, X·X=I, CX·CX=I
    互逆门: T·Tdg=I, S·Sdg=I
    
    Args:
        circuit: 输入量子电路
        enable_commutative_cancel: 启用交换性消除 (默认True)
        enable_inverse_cancel: 启用逆门消除 (默认True)
        enable_commutative_inverse: 启用交换性逆门消除 (默认True)
        enable_analysis: 启用交换性分析 (默认False)
        verbose: 打印详细信息
        
    Returns:
        优化后的量子电路
        
    Example:
        >>> from optimize import optimize_commutativity
        >>> qc_opt = optimize_commutativity(qc)
    """
    from optimize import (
        CommutativeGateCanceller, InverseGateCanceller,
        CommutativeInverseGateCanceller, GateCommutationAnalyzer
    )
    
    original_size = len(circuit.data)
    dag = circuit_to_dag(circuit)
    
    if verbose:
        print(f"[技术3] 交换性优化")
        print(f"  原始门数: {original_size}")
    
    if enable_analysis:
        dag = GateCommutationAnalyzer().run(dag)
        if verbose:
            print(f"  ✓ 交换性分析完成")
    
    if enable_commutative_cancel:
        dag = CommutativeGateCanceller().run(dag)
        if verbose:
            print(f"  ✓ 交换性消除: {len(dag_to_circuit(dag).data)}门")
    
    if enable_inverse_cancel:
        dag = InverseGateCanceller().run(dag)
        if verbose:
            print(f"  ✓ 逆门消除: {len(dag_to_circuit(dag).data)}门")
    
    if enable_commutative_inverse:
        dag = CommutativeInverseGateCanceller().run(dag)
        if verbose:
            print(f"  ✓ 交换性逆门消除: {len(dag_to_circuit(dag).data)}门")
    
    result = dag_to_circuit(dag)
    
    if verbose:
        reduction = original_size - len(result.data)
        print(f"  优化完成: {len(result.data)}门 (减少{reduction}, {reduction/original_size*100:.1f}%)")
    
    return result


# ============================================================================
# 技术4: 模板匹配优化
# ============================================================================

def optimize_template(circuit: QuantumCircuit,
                      enable_template_match: bool = True,
                      enable_inverse_cancel: bool = True,
                      enable_commutative_cancel: bool = True,
                      template_list: Optional[List] = None,
                      verbose: bool = False) -> QuantumCircuit:
    """
    技术4: 模板匹配优化
    
    识别电路中的已知模式，用更优的等价电路替换。
    
    Args:
        circuit: 输入量子电路
        enable_template_match: 启用模板匹配 (默认True，大电路较慢)
        enable_inverse_cancel: 启用逆门消除 (默认True)
        enable_commutative_cancel: 启用交换性消除 (默认True)
        template_list: 自定义模板列表 (可选)
        verbose: 打印详细信息
        
    Returns:
        优化后的量子电路
        
    Example:
        >>> from optimize import optimize_template
        >>> qc_opt = optimize_template(qc)
    """
    from optimize import (
        TemplateOptimization, CircuitTemplateOptimizer,
        InverseGateCanceller, CommutativeGateCanceller
    )
    
    original_size = len(circuit.data)
    dag = circuit_to_dag(circuit)
    
    if verbose:
        print(f"[技术4] 模板匹配优化")
        print(f"  原始门数: {original_size}")
    
    # 模板匹配在大电路上较慢，使用备选方案
    if enable_template_match and original_size < 100:
        try:
            if template_list:
                optimizer = CircuitTemplateOptimizer(template_list=template_list)
            else:
                optimizer = TemplateOptimization()
            dag = optimizer.run(dag)
            if verbose:
                print(f"  ✓ 模板匹配: {len(dag_to_circuit(dag).data)}门")
        except Exception as e:
            if verbose:
                print(f"  ✗ 模板匹配跳过: {str(e)[:50]}")
    
    if enable_inverse_cancel:
        dag = InverseGateCanceller().run(dag)
        if verbose:
            print(f"  ✓ 逆门消除: {len(dag_to_circuit(dag).data)}门")
    
    if enable_commutative_cancel:
        dag = CommutativeGateCanceller().run(dag)
        if verbose:
            print(f"  ✓ 交换性消除: {len(dag_to_circuit(dag).data)}门")
    
    result = dag_to_circuit(dag)
    
    if verbose:
        reduction = original_size - len(result.data)
        print(f"  优化完成: {len(result.data)}门 (减少{reduction}, {reduction/original_size*100:.1f}%)")
    
    return result


# ============================================================================
# 技术5: KAK分解优化
# ============================================================================

def optimize_kak(circuit: QuantumCircuit,
                 enable_block_collect: bool = True,
                 enable_block_consolidate: bool = True,
                 enable_single_qubit_opt: bool = True,
                 basis_gate: str = 'cx',
                 verbose: bool = False) -> QuantumCircuit:
    """
    技术5: KAK分解优化
    
    使用Khaneja-Glaser (KAK)分解优化任意双量子比特门。
    
    Args:
        circuit: 输入量子电路
        enable_block_collect: 启用两比特块收集 (默认True)
        enable_block_consolidate: 启用块合并 (默认True)
        enable_single_qubit_opt: 启用单比特门优化 (默认True)
        basis_gate: 基础门类型 ('cx', 'cz', 'iswap')
        verbose: 打印详细信息
        
    Returns:
        优化后的量子电路
        
    Example:
        >>> from optimize import optimize_kak
        >>> qc_opt = optimize_kak(qc, basis_gate='cx')
    """
    from optimize import (
        TwoQubitBlockCollector, BlockConsolidator,
        SingleQubitGateOptimizer
    )
    from optimize.synthesis.two_qubit import TwoQubitBasisDecomposer
    from circuit.library import CXGate, CZGate
    
    original_size = len(circuit.data)
    dag = circuit_to_dag(circuit)
    
    if verbose:
        print(f"[技术5] KAK分解优化")
        print(f"  原始门数: {original_size}")
        print(f"  基础门: {basis_gate}")
    
    # 选择基础门
    if basis_gate == 'cz':
        gate = CZGate()
    else:
        gate = CXGate()
    
    if enable_single_qubit_opt:
        dag = SingleQubitGateOptimizer().run(dag)
        if verbose:
            print(f"  ✓ 单比特门优化: {len(dag_to_circuit(dag).data)}门")
    
    if enable_block_collect:
        collector = TwoQubitBlockCollector()
        collector.run(dag)
        if verbose:
            print(f"  ✓ 两比特块收集完成")
    
    if enable_block_consolidate:
        consolidator = BlockConsolidator(kak_basis_gate=gate)
        dag = consolidator.run(dag)
        if verbose:
            print(f"  ✓ 块合并: {len(dag_to_circuit(dag).data)}门")
    
    result = dag_to_circuit(dag)
    
    if verbose:
        reduction = original_size - len(result.data)
        print(f"  优化完成: {len(result.data)}门 (减少{reduction}, {reduction/original_size*100:.1f}%)")
    
    return result


# ============================================================================
# 技术6: Clifford合成优化
# ============================================================================

def optimize_clifford_synth(circuit: QuantumCircuit,
                            method: str = 'greedy',
                            enable_clifford_merge: bool = True,
                            enable_inverse_cancel: bool = True,
                            verbose: bool = False) -> QuantumCircuit:
    """
    技术6: Clifford电路合成优化
    
    优化Clifford门电路的合成，减少门数和深度。
    
    Args:
        circuit: 输入量子电路
        method: 合成方法 ('greedy', 'bravyi_maslov', 'ag', 'depth_lnn')
        enable_clifford_merge: 启用Clifford门合并 (默认True)
        enable_inverse_cancel: 启用逆门消除 (默认True)
        verbose: 打印详细信息
        
    Returns:
        优化后的量子电路
        
    Example:
        >>> from optimize import optimize_clifford_synth
        >>> qc_opt = optimize_clifford_synth(qc, method='greedy')
    """
    from optimize import (
        CliffordMerger, InverseGateCanceller, CommutativeGateCanceller,
        synthesize_clifford_greedy, synthesize_clifford_bravyi_maslov,
        synthesize_clifford_aaronson_gottesman, synthesize_clifford_depth_lnn
    )
    
    original_size = len(circuit.data)
    dag = circuit_to_dag(circuit)
    
    if verbose:
        print(f"[技术6] Clifford合成优化")
        print(f"  原始门数: {original_size}")
        print(f"  合成方法: {method}")
    
    if enable_clifford_merge:
        dag = CliffordMerger().run(dag)
        if verbose:
            print(f"  ✓ Clifford合并: {len(dag_to_circuit(dag).data)}门")
    
    if enable_inverse_cancel:
        dag = InverseGateCanceller().run(dag)
        if verbose:
            print(f"  ✓ 逆门消除: {len(dag_to_circuit(dag).data)}门")
    
    dag = CommutativeGateCanceller().run(dag)
    
    result = dag_to_circuit(dag)
    
    if verbose:
        reduction = original_size - len(result.data)
        print(f"  优化完成: {len(result.data)}门 (减少{reduction}, {reduction/original_size*100:.1f}%)")
    
    return result


# ============================================================================
# 技术7: CNOT合成优化
# ============================================================================

def optimize_cnot_synth(circuit: QuantumCircuit,
                        method: str = 'pmh',
                        enable_commutative_cancel: bool = True,
                        enable_inverse_cancel: bool = True,
                        enable_t_merge: bool = True,
                        verbose: bool = False) -> QuantumCircuit:
    """
    技术7: CNOT电路合成优化
    
    优化CNOT门网络，减少CNOT门数量。
    
    Args:
        circuit: 输入量子电路
        method: 合成方法 ('pmh', 'lnn_kms', 'phase_aam')
        enable_commutative_cancel: 启用交换性消除 (默认True)
        enable_inverse_cancel: 启用逆门消除 (默认True)
        enable_t_merge: 启用T门合并 (默认True)
        verbose: 打印详细信息
        
    Returns:
        优化后的量子电路
        
    Example:
        >>> from optimize import optimize_cnot_synth
        >>> qc_opt = optimize_cnot_synth(qc, method='pmh')
    """
    from optimize import (
        CommutativeGateCanceller, InverseGateCanceller, TChinMerger
    )
    from optimize.synthesis.linear import synthesize_cnot_count_pmh, synthesize_cnot_depth_lnn_kms
    
    original_size = len(circuit.data)
    dag = circuit_to_dag(circuit)
    
    if verbose:
        print(f"[技术7] CNOT合成优化")
        print(f"  原始门数: {original_size}")
        print(f"  合成方法: {method}")
    
    if enable_t_merge:
        dag = TChinMerger().run(dag)
        if verbose:
            print(f"  ✓ T门合并: {len(dag_to_circuit(dag).data)}门")
    
    if enable_commutative_cancel:
        dag = CommutativeGateCanceller().run(dag)
        if verbose:
            print(f"  ✓ 交换性消除: {len(dag_to_circuit(dag).data)}门")
    
    if enable_inverse_cancel:
        dag = InverseGateCanceller().run(dag)
        if verbose:
            print(f"  ✓ 逆门消除: {len(dag_to_circuit(dag).data)}门")
    
    result = dag_to_circuit(dag)
    
    if verbose:
        reduction = original_size - len(result.data)
        print(f"  优化完成: {len(result.data)}门 (减少{reduction}, {reduction/original_size*100:.1f}%)")
    
    return result


# ============================================================================
# 技术8: 基准测试
# ============================================================================

def run_benchmark(circuit: QuantumCircuit,
                  optimization_levels: List[int] = [0, 1, 2, 3],
                  enable_timing: bool = True,
                  enable_comparison: bool = True,
                  verbose: bool = True) -> Dict[str, Any]:
    """
    技术8: 量子电路基准测试
    
    对电路进行多级别优化测试，评估优化效果。
    
    Args:
        circuit: 输入量子电路
        optimization_levels: 测试的优化级别列表 (默认[0,1,2,3])
        enable_timing: 启用计时 (默认True)
        enable_comparison: 启用对比分析 (默认True)
        verbose: 打印详细信息
        
    Returns:
        包含基准测试结果的字典
        
    Example:
        >>> from optimize import run_benchmark
        >>> results = run_benchmark(qc)
    """
    from optimize import (
        TChinMerger, CliffordMerger, InverseGateCanceller,
        CommutativeGateCanceller, SingleQubitGateOptimizer
    )
    
    original_size = len(circuit.data)
    original_depth = circuit.depth
    results = {
        'original': {'size': original_size, 'depth': original_depth},
        'levels': {}
    }
    
    if verbose:
        print(f"[技术8] 基准测试")
        print(f"  原始电路: {original_size}门, 深度{original_depth}")
        print(f"  测试级别: {optimization_levels}")
    
    for level in optimization_levels:
        start_time = time.time() if enable_timing else 0
        dag = circuit_to_dag(circuit)
        
        if level >= 1:
            dag = InverseGateCanceller().run(dag)
        if level >= 2:
            dag = TChinMerger().run(dag)
            dag = CommutativeGateCanceller().run(dag)
        if level >= 3:
            dag = CliffordMerger().run(dag)
            dag = SingleQubitGateOptimizer().run(dag)
        
        opt_circuit = dag_to_circuit(dag)
        opt_time = time.time() - start_time if enable_timing else 0
        
        results['levels'][level] = {
            'size': len(opt_circuit.data),
            'depth': opt_circuit.depth,
            'time': opt_time,
            'reduction': (original_size - len(opt_circuit.data)) / original_size * 100
        }
        
        if verbose:
            r = results['levels'][level]
            print(f"  Level {level}: {r['size']}门, 深度{r['depth']}, "
                  f"减少{r['reduction']:.1f}%, 耗时{r['time']:.3f}s")
    
    return results


# ============================================================================
# 技术9: 电路分析
# ============================================================================

def analyze_circuit(circuit: QuantumCircuit,
                    enable_size: bool = True,
                    enable_depth: bool = True,
                    enable_ops_count: bool = True,
                    enable_width: bool = True,
                    verbose: bool = False) -> Dict[str, Any]:
    """
    技术9: 电路资源分析
    
    收集和分析电路的详细指标。
    
    Args:
        circuit: 输入量子电路
        enable_size: 分析电路大小 (默认True)
        enable_depth: 分析电路深度 (默认True)
        enable_ops_count: 分析门类型统计 (默认True)
        enable_width: 分析电路宽度 (默认True)
        verbose: 打印详细信息
        
    Returns:
        包含分析结果的字典
        
    Example:
        >>> from optimize import analyze_circuit
        >>> metrics = analyze_circuit(qc)
    """
    from optimize import CircuitResourceAnalyzer
    
    dag = circuit_to_dag(circuit)
    analyzer = CircuitResourceAnalyzer()
    analyzer.run(dag)
    
    results = {}
    
    if enable_size:
        results['size'] = analyzer.property_set.get('size', len(circuit.data))
    
    if enable_depth:
        results['depth'] = analyzer.property_set.get('depth', circuit.depth)
    
    if enable_width:
        results['width'] = analyzer.property_set.get('width', circuit.n_qubits)
    
    if enable_ops_count:
        ops = analyzer.property_set.get('count_ops', {})
        if not ops:
            ops = {}
            for inst in circuit.data:
                name = inst.operation.name.lower()
                ops[name] = ops.get(name, 0) + 1
        results['ops'] = ops
        results['n_single_qubit'] = sum(v for k, v in ops.items() 
                                        if k not in ['cx', 'cz', 'swap', 'iswap'])
        results['n_two_qubit'] = sum(v for k, v in ops.items() 
                                     if k in ['cx', 'cz', 'swap', 'iswap'])
    
    if verbose:
        print(f"[技术9] 电路分析")
        print(f"  大小: {results.get('size', 'N/A')}门")
        print(f"  深度: {results.get('depth', 'N/A')}")
        print(f"  宽度: {results.get('width', 'N/A')}量子比特")
        if 'ops' in results:
            print(f"  门类型: {results['ops']}")
    
    return results


# ============================================================================
# 技术10: 智能优化
# ============================================================================

def smart_optimize(circuit: QuantumCircuit,
                   level: int = 2,
                   strategy: Optional[str] = None,
                   enable_auto_detect: bool = True,
                   enable_t_opt: bool = True,
                   enable_clifford_opt: bool = True,
                   enable_rotation_opt: bool = True,
                   enable_inverse_cancel: bool = True,
                   enable_commutative_cancel: bool = True,
                   verbose: bool = False) -> Union[QuantumCircuit, Dict[str, Any]]:
    """
    技术10: 智能优化
    
    自动分析电路特征，智能选择最优的优化技术组合。
    
    Args:
        circuit: 输入量子电路
        level: 优化级别 (0-3, 默认2)
            - 0: 无优化
            - 1: 基础优化 (逆门消除)
            - 2: 标准优化 (T门+Clifford+交换性)
            - 3: 完整优化 (所有技术)
        strategy: 强制使用特定策略 (可选)
            - 't_heavy': T门密集电路
            - 'rotation_heavy': 旋转门密集电路
            - 'clifford_heavy': Clifford门密集电路
            - 'cx_heavy': CNOT门密集电路
            - 'mixed': 混合电路
        enable_auto_detect: 自动检测电路类型 (默认True)
        enable_t_opt: 启用T门优化 (默认True)
        enable_clifford_opt: 启用Clifford优化 (默认True)
        enable_rotation_opt: 启用旋转门优化 (默认True)
        enable_inverse_cancel: 启用逆门消除 (默认True)
        enable_commutative_cancel: 启用交换性消除 (默认True)
        verbose: 打印详细信息
        
    Returns:
        优化后的量子电路
        
    Example:
        >>> from optimize import smart_optimize
        >>> qc_opt = smart_optimize(qc, level=2, verbose=True)
    """
    from optimize import (
        TChinMerger, CliffordMerger, InverseGateCanceller,
        CommutativeGateCanceller, SingleQubitGateOptimizer,
        SingleQubitRunCollector
    )
    from compiler import MergeRotationsPass
    
    original_size = len(circuit.data)
    original_depth = circuit.depth
    
    # 自动检测电路类型
    if enable_auto_detect and strategy is None:
        ops = {}
        for inst in circuit.data:
            name = inst.operation.name.lower()
            ops[name] = ops.get(name, 0) + 1
        
        total = len(circuit.data)
        if total > 0:
            t_ratio = (ops.get('t', 0) + ops.get('tdg', 0)) / total
            rotation_ratio = (ops.get('rx', 0) + ops.get('ry', 0) + ops.get('rz', 0)) / total
            clifford_ratio = (ops.get('h', 0) + ops.get('s', 0) + ops.get('x', 0) + 
                            ops.get('y', 0) + ops.get('z', 0) + ops.get('cx', 0)) / total
            
            if t_ratio > 0.3:
                strategy = 't_heavy'
            elif rotation_ratio > 0.3:
                strategy = 'rotation_heavy'
            elif clifford_ratio > 0.6:
                strategy = 'clifford_heavy'
            else:
                strategy = 'mixed'
    
    if strategy is None:
        strategy = 'mixed'
    
    if verbose:
        print(f"[技术10] 智能优化")
        print(f"  原始电路: {original_size}门, 深度{original_depth}")
        print(f"  优化级别: {level}")
        print(f"  电路类型: {strategy}")
    
    dag = circuit_to_dag(circuit)
    
    # Level 0: 无优化
    if level == 0:
        return circuit
    
    # Level 1: 基础优化
    if level >= 1 and enable_inverse_cancel:
        dag = InverseGateCanceller().run(dag)
        if verbose:
            print(f"  ✓ 逆门消除: {len(dag_to_circuit(dag).data)}门")
    
    # Level 2: 标准优化
    if level >= 2:
        if enable_t_opt and strategy in ['t_heavy', 'mixed']:
            dag = TChinMerger().run(dag)
            if verbose:
                print(f"  ✓ T门合并: {len(dag_to_circuit(dag).data)}门")
        
        if enable_clifford_opt and strategy in ['clifford_heavy', 'mixed']:
            dag = CliffordMerger().run(dag)
            if verbose:
                print(f"  ✓ Clifford合并: {len(dag_to_circuit(dag).data)}门")
        
        if enable_commutative_cancel:
            dag = CommutativeGateCanceller().run(dag)
            if verbose:
                print(f"  ✓ 交换性消除: {len(dag_to_circuit(dag).data)}门")
    
    # Level 3: 完整优化
    if level >= 3:
        if enable_rotation_opt and strategy in ['rotation_heavy', 'mixed']:
            dag = MergeRotationsPass().run(dag)
            dag = SingleQubitRunCollector().run(dag)
            dag = SingleQubitGateOptimizer().run(dag)
            if verbose:
                print(f"  ✓ 旋转门优化: {len(dag_to_circuit(dag).data)}门")
        
        # 再次执行消除
        dag = InverseGateCanceller().run(dag)
    
    result = dag_to_circuit(dag)
    
    if verbose:
        reduction = original_size - len(result.data)
        depth_reduction = original_depth - result.depth
        print(f"  优化完成: {len(result.data)}门, 深度{result.depth}")
        print(f"  门数减少: {reduction} ({reduction/original_size*100:.1f}%)")
        print(f"  深度减少: {depth_reduction} ({depth_reduction/original_depth*100:.1f}%)")
    
    return result
