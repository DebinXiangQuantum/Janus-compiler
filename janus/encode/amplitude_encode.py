from math import log2, ceil, asin
from typing import List, Optional, Literal

from janus.circuit import Circuit


# 状态树节点类
class StateNode:
    """用于表示量子状态层次结构的树节点"""
    def __init__(self, index: int, level: int, amplitude: float, 
                 left: Optional['StateNode'] = None, 
                 right: Optional['StateNode'] = None):
        self.index = index          # 节点索引
        self.level = level          # 树的层级
        self.amplitude = amplitude  # 振幅
        self.left = left            # 左子节点
        self.right = right          # 右子节点


# 角度树节点类
class NodeAngleTree:
    """用于表示旋转角度树的节点"""
    def __init__(self, index: int, level: int, qubit_index: int, angle: float,
                 left: Optional['NodeAngleTree'] = None,
                 right: Optional['NodeAngleTree'] = None):
        self.index = index              # 节点索引
        self.level = level              # 树的层级
        self.qubit_index = qubit_index  # 对应的量子比特索引
        self.angle = angle              # RY旋转角度
        self.left = left                # 左子节点
        self.right = right              # 右子节点


def _check_normalized(data: List[float], epsilon: float = 1e-13) -> bool:
    """检查数据是否为归一化的"""
    norm_squared = sum(x * x for x in data)
    return abs(1.0 - norm_squared) < epsilon


def _state_decomposition(n_qubits: int, data: List[float]) -> StateNode:
    """将数据递归分解为状态树"""
    # 为底层创建叶节点
    new_nodes = [StateNode(i, n_qubits, data[i]) for i in range(len(data))]
    nodes = new_nodes
    current_level = n_qubits
    
    # 逐层向上构建树
    while current_level > 0:
        new_nodes = []
        current_level -= 1
        
        for k in range(0, len(nodes), 2):
            # 计算两个节点的振幅范数
            left_node = nodes[k]
            right_node = nodes[k + 1]
            mag = (left_node.amplitude**2 + right_node.amplitude**2)**0.5
            
            # 创建父节点
            parent = StateNode(k // 2, current_level, mag, left_node, right_node)
            new_nodes.append(parent)
        
        nodes = new_nodes
    
    return nodes[0]


def _create_angles_tree(state_tree: StateNode) -> NodeAngleTree:
    """从状态树创建角度树"""
    angle = 0.0
    
    if state_tree.right:
        if abs(state_tree.amplitude) > 1e-6:
            # 计算右子节点相对于总振幅的角度
            Amp = state_tree.right.amplitude / state_tree.amplitude
            # asin 的定义域为 [-1, 1]，数值误差时进行双侧截断
            Amp = max(min(Amp, 1.0), -1.0)
            angle = 2 * asin(Amp)
    
    # 创建当前节点的角度树表示
    node = NodeAngleTree(state_tree.index, state_tree.level, 0, angle)
    
    # 如果有子节点，递归创建子树
    if state_tree.right and state_tree.right.left and state_tree.right.right:
        node.right = _create_angles_tree(state_tree.right)
        node.left = _create_angles_tree(state_tree.left)
    
    return node


def _children(nodes: List[NodeAngleTree]) -> List[NodeAngleTree]:
    """获取所有节点的子节点"""
    children = []
    for node in nodes:
        if node.left:
            children.append(node.left)
        if node.right:
            children.append(node.right)
    return children


def _add_register(angle_tree: NodeAngleTree, start_level: int) -> int:
    """为节点分配量子比特索引，并返回所需的总量子比特数（与 Encode.cpp 对齐）"""
    # 建立树的层级结构
    level_nodes = []
    nodes = [angle_tree]
    
    while len(nodes) > 0:
        level_nodes.append(len(nodes))
        nodes = _children(nodes)
    
    # 计算所需的总量子比特数
    noutput = len(level_nodes)
    nqubits = 0
    
    # Encode.cpp 中假设 start_level 一定合法，这里也按同样公式计算
    for i in range(start_level):
        nqubits += level_nodes[i]
    nqubits += level_nodes[start_level] * (noutput - start_level)
    
    # 分配量子比特索引
    qubit_queue = list(range(nqubits))
    _assign_qubits(angle_tree, qubit_queue, start_level)
    
    return nqubits


def _assign_qubits(angle_tree: NodeAngleTree, qubit_queue: List[int], start_level: int):
    """递归地为树节点分配量子比特"""
    if angle_tree and qubit_queue:
        angle_tree.qubit_index = qubit_queue.pop(0)
        
        if angle_tree.level < start_level:
            if angle_tree.left:
                _assign_qubits(angle_tree.left, qubit_queue, start_level)
            if angle_tree.right:
                _assign_qubits(angle_tree.right, qubit_queue, start_level)
        else:
            if angle_tree.left:
                _assign_qubits(angle_tree.left, qubit_queue, start_level)
            else:
                if angle_tree.right:
                    _assign_qubits(angle_tree.right, qubit_queue, start_level)


def _top_down_tree_walk(circuit: Circuit, angle_tree: NodeAngleTree, qubits: List[int], 
                        start_level: int, control_nodes: Optional[List[NodeAngleTree]] = None,
                        target_nodes: Optional[List[NodeAngleTree]] = None):
    """自顶向下遍历树，在split点以上应用controlled旋转"""
    if control_nodes is None:
        control_nodes = []
    if target_nodes is None:
        target_nodes = []
    
    if angle_tree:
        if angle_tree.level < start_level:
            # 在 split 点以下，递归处理两个子树
            if angle_tree.left:
                _top_down_tree_walk(circuit, angle_tree.left, qubits, start_level, 
                                   control_nodes.copy(), target_nodes.copy())
            if angle_tree.right:
                _top_down_tree_walk(circuit, angle_tree.right, qubits, start_level,
                                   control_nodes.copy(), target_nodes.copy())
        else:
            # 在 split 点或以上
            if not target_nodes:
                target_nodes = [angle_tree]
            else:
                target_nodes = _children(target_nodes)
            
            # 只有当有目标节点时才继续
            if not target_nodes:
                return
            
            # 收集当前层的旋转角度
            angles = [node.angle for node in target_nodes]
            target_qubits_index = target_nodes[0].qubit_index
            
            # 获取控制比特索引
            control_qubits_index = [node.qubit_index for node in control_nodes]
            
            # 反转角度列表
            angles_reversed = list(reversed(angles))
            
            # 应用受控旋转门
            num_controls = len(control_qubits_index)
            control_qubits_list = [qubits[i] for i in control_qubits_index]
            
            for k, angle in enumerate(angles_reversed):
                # 计算控制状态
                _apply_index_gates(circuit, k, control_qubits_list, num_controls)
                
                if not control_qubits_list:
                    circuit.ry(angle, qubits[target_qubits_index])
                else:
                    circuit.mcry(angle, control_qubits_list, qubits[target_qubits_index])
                
                _apply_index_gates(circuit, k, control_qubits_list, num_controls)
            
            new_control_nodes = control_nodes + [angle_tree]
            if angle_tree.left:
                _top_down_tree_walk(circuit, angle_tree.left, qubits, start_level,
                                   new_control_nodes, target_nodes)


def _bottom_up_tree_walk(circuit: Circuit, angle_tree: NodeAngleTree, qubits: List[int], 
                         start_level: int):
    """自底向上遍历树，在split点以下应用旋转"""
    if angle_tree and angle_tree.level < start_level:
        # 应用RY旋转
        circuit.ry(angle_tree.angle, qubits[angle_tree.qubit_index])
        
        # 递归处理子节点
        if angle_tree.left:
            _bottom_up_tree_walk(circuit, angle_tree.left, qubits, start_level)
        
        if angle_tree.right:
            _bottom_up_tree_walk(circuit, angle_tree.right, qubits, start_level)
        
        # 应用controlled swaps
        _apply_cswaps(circuit, angle_tree, qubits)


def _apply_cswaps(circuit: Circuit, angle_tree: NodeAngleTree, qubits: List[int]):
    """应用受控SWAP操作"""
    # Encode.cpp：只有当 angle != 0 时才应用受控 SWAP
    if abs(angle_tree.angle) > 1e-12:
        left = angle_tree.left
        right = angle_tree.right
        
        while left and right:
            circuit.cswap(qubits[angle_tree.qubit_index], qubits[left.qubit_index], qubits[right.qubit_index])
            
            left = left.left if left.left else None
            if right.left:
                right = right.left
            else:
                right = right.right if right else None


def _apply_index_gates(circuit: Circuit, value: int, control_qubits: List[int], num_controls: int):
    """根据索引值应用X门到控制比特
    
    与 C++ Encode.cpp 中的 _index 函数对应：
    遍历顺序是从最高有效位到最低有效位，
    对应 control_qubits[0] 到 control_qubits[num_controls-1]
    """
    if not control_qubits or num_controls == 0:
        return
    
    # 获取value的二进制表示
    binary = format(value, f'0{num_controls}b')
    
    # 从高位到低位遍历（不使用 reversed！与 C++ 一致）
    for i, bit in enumerate(binary):
        if bit == '1':
            circuit.x(control_qubits[i])


def _output(angle_tree: NodeAngleTree, qubits: List[int], output_list: List[int]):
    """递归遍历树以确定输出量子比特（沿着 left 优先路径）"""
    if angle_tree:
        if angle_tree.left:
            _output(angle_tree.left, qubits, output_list)
        elif angle_tree.right:
            _output(angle_tree.right, qubits, output_list)
        
        output_list.append(angle_tree.qubit_index)




def bidrc_encode(
    data: List[float],
    split: int = 0,
    mode: Literal['full', 'top_down', 'bottom_up'] = 'full'
) -> Circuit:
    """
    双向振幅编码
    
    使用分裂的二进制树方法进行量子状态编码，相比于标准的振幅编码方法，
    这种方法在门数量和电路深度上更为高效。
    
    参数：
        data: 表示量子状态的归一化振幅列表
        split: 树分裂点（0表示自动计算）
    
    返回：
        Circuit: 包含编码操作的量子电路
    
    异常：
        ValueError: 如果数据未归一化或参数不合法
    """
    # 数据类型转换和归一化检查
    data_temp = list(data)
    
    if not _check_normalized(data_temp):
        raise ValueError("数据必须是归一化的")
    
    # 计算所需的量子比特数
    n_qubits = ceil(log2(len(data_temp)))
    split_temp = split
    
    # 如果split为0，自动计算最优分裂点
    if split == 0:
        if n_qubits & 1:
            split_temp = n_qubits // 2 + 1
        else:
            split_temp = n_qubits // 2
    if split <0:
        split_temp = n_qubits+split
    # 检查参数合法性
    if split_temp > ceil(log2(len(data_temp))):
        raise ValueError("Bid_Amplitude_encode 参数错误：split值过大")
    
    # 将数据填充到2的幂次
    while len(data_temp) < (1 << n_qubits):
        data_temp.append(0.0)
    
    # 构建状态树
    state_tree = _state_decomposition(n_qubits, data_temp)
    
    # 从状态树创建角度树
    angle_tree = _create_angles_tree(state_tree)
    
    # 分配量子比特（并获得所需 qubits 数，与 Encode.cpp 对齐）
    start_level = n_qubits - split_temp
    num_qubits_needed = _add_register(angle_tree, start_level)
    
    # 记录输出量子比特（在创建电路前先获取）
    output_qubits = []
    _output(angle_tree, list(range(num_qubits_needed)), output_qubits)
    
    # 创建量子电路（包含足够的经典比特用于测量）
    circuit = Circuit(num_qubits_needed, len(output_qubits))
    
    # 生成量子电路
    qubits_list = list(range(num_qubits_needed))
    
    if mode not in ('full', 'top_down', 'bottom_up'):
        raise ValueError("mode 必须为 'full'、'top_down' 或 'bottom_up'")
    
    # 自顶向下应用 controlled 旋转（在 split 点以上）
    if mode in ('full', 'top_down'):
        _top_down_tree_walk(circuit, angle_tree, qubits_list, start_level)
    
    # 自底向上应用旋转与受控 SWAP（在 split 点以下）
    if mode in ('full', 'bottom_up'):
        _bottom_up_tree_walk(circuit, angle_tree, qubits_list, start_level)
    # 在输出量子比特上添加测量
    print(output_qubits)
    for i, qubit in enumerate(output_qubits):
        circuit.measure(qubit, i)
    return circuit
