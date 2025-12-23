"""
Janus DAG (有向无环图) 电路表示

DAG 表示便于进行电路优化和分析
"""
from typing import List, Dict, Set, Optional, Iterator, Tuple
from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """DAG 节点类型"""
    INPUT = "input"      # 输入节点（量子比特初始状态）
    OUTPUT = "output"    # 输出节点（量子比特最终状态）
    OP = "op"            # 操作节点（量子门）


@dataclass
class DAGNode:
    """
    DAG 节点

    Attributes:
        node_id: 节点唯一标识
        node_type: 节点类型
        qubits: 关联的量子比特
        clbits: 关联的经典比特
        op: 操作（仅 OP 类型节点）
    """
    node_id: int
    node_type: NodeType
    qubits: List[int]
    clbits: List[int] = None
    op: 'Gate' = None

    def __post_init__(self):
        if self.clbits is None:
            self.clbits = []

    @property
    def qargs(self) -> List[int]:
        """Qiskit兼容性别名: qargs -> qubits"""
        return self.qubits

    @property
    def cargs(self) -> List[int]:
        """Qiskit兼容性别名: cargs -> clbits"""
        return self.clbits

    @property
    def name(self) -> str:
        if self.node_type == NodeType.INPUT:
            return f"input_q{self.qubits[0]}"
        elif self.node_type == NodeType.OUTPUT:
            return f"output_q{self.qubits[0]}"
        else:
            return self.op.name if self.op else "unknown"
    
    def __repr__(self) -> str:
        if self.node_type == NodeType.OP:
            return f"DAGOpNode({self.op}, qubits={self.qubits})"
        return f"DAGNode({self.node_type.value}, qubits={self.qubits})"
    
    def __hash__(self) -> int:
        return hash(self.node_id)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, DAGNode):
            return self.node_id == other.node_id
        return False


class DAGCircuit:
    """
    DAG 电路表示
    
    将量子电路表示为有向无环图，其中：
    - 节点表示量子操作
    - 边表示量子比特的数据流
    
    Attributes:
        n_qubits: 量子比特数
        n_clbits: 经典比特数
    """
    
    def __init__(self, n_qubits: int = 0, n_clbits: int = 0, name: str = None):
        self._n_qubits = n_qubits
        self._n_clbits = n_clbits
        self._name = name
        self._global_phase = 0.0  # 全局相位
        self._metadata = {}  # 元数据

        # 节点存储
        self._nodes: Dict[int, DAGNode] = {}
        self._next_node_id = 0

        # 边存储: node_id -> set of successor node_ids
        self._successors: Dict[int, Set[int]] = {}
        self._predecessors: Dict[int, Set[int]] = {}

        # 输入输出节点
        self._input_nodes: Dict[int, DAGNode] = {}   # qubit -> input node
        self._output_nodes: Dict[int, DAGNode] = {}  # qubit -> output node

        # 每个量子比特当前的最后一个节点
        self._qubit_last_node: Dict[int, int] = {}

        # 初始化输入输出节点
        self._init_io_nodes()
    
    def _init_io_nodes(self):
        """初始化输入输出节点"""
        for q in range(self._n_qubits):
            # 输入节点
            input_node = self._create_node(NodeType.INPUT, [q])
            self._input_nodes[q] = input_node
            self._qubit_last_node[q] = input_node.node_id
            
            # 输出节点
            output_node = self._create_node(NodeType.OUTPUT, [q])
            self._output_nodes[q] = output_node
    
    def _create_node(self, node_type: NodeType, qubits: List[int], 
                     clbits: List[int] = None, op=None) -> DAGNode:
        """创建新节点"""
        node = DAGNode(
            node_id=self._next_node_id,
            node_type=node_type,
            qubits=qubits,
            clbits=clbits,
            op=op
        )
        self._nodes[node.node_id] = node
        self._successors[node.node_id] = set()
        self._predecessors[node.node_id] = set()
        self._next_node_id += 1
        return node
    
    def _add_edge(self, from_node: int, to_node: int):
        """添加边"""
        self._successors[from_node].add(to_node)
        self._predecessors[to_node].add(from_node)
    
    def _remove_edge(self, from_node: int, to_node: int):
        """移除边"""
        self._successors[from_node].discard(to_node)
        self._predecessors[to_node].discard(from_node)
    
    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def n_clbits(self) -> int:
        return self._n_clbits

    @property
    def name(self) -> str:
        """电路名称"""
        return self._name

    @name.setter
    def name(self, value: str):
        """设置电路名称"""
        self._name = value

    @property
    def global_phase(self) -> float:
        """全局相位"""
        return self._global_phase

    @global_phase.setter
    def global_phase(self, value: float):
        """设置全局相位"""
        self._global_phase = value

    @property
    def metadata(self) -> dict:
        """元数据字典"""
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        """设置元数据"""
        self._metadata = value if value is not None else {}

    @property
    def qubits(self) -> List[int]:
        """返回所有量子比特列表(Qiskit兼容性)"""
        return list(range(self._n_qubits))

    @property
    def clbits(self) -> List[int]:
        """返回所有经典比特列表(Qiskit兼容性)"""
        return list(range(self._n_clbits))

    @property
    def qregs(self) -> Dict:
        """返回量子寄存器字典(Qiskit兼容性,返回空字典stub)"""
        return {}

    @property
    def cregs(self) -> Dict:
        """返回经典寄存器字典(Qiskit兼容性,返回空字典stub)"""
        return {}

    def num_qubits(self) -> int:
        """返回量子比特数量(方法形式,兼容 Qiskit)"""
        return self._n_qubits

    def num_clbits(self) -> int:
        """返回经典比特数量(方法形式,兼容 Qiskit)"""
        return self._n_clbits
    
    def apply_operation(self, op, qubits: List[int], clbits: List[int] = None) -> DAGNode:
        """
        添加一个操作到 DAG

        Args:
            op: 量子操作（Gate）
            qubits: 作用的量子比特
            clbits: 作用的经典比特

        Returns:
            创建的 DAGNode
        """
        # 创建操作节点
        node = self._create_node(NodeType.OP, qubits, clbits, op)

        # 连接前驱节点
        for q in qubits:
            last_node_id = self._qubit_last_node[q]
            self._add_edge(last_node_id, node.node_id)
            self._qubit_last_node[q] = node.node_id

        return node

    def apply_operation_back(self, op, qubits: List[int], clbits: List[int] = None) -> DAGNode:
        """
        添加一个操作到 DAG (在末尾)

        Qiskit兼容性方法: apply_operation_back 与 apply_operation 相同

        Args:
            op: 量子操作（Gate）
            qubits: 作用的量子比特
            clbits: 作用的经典比特

        Returns:
            创建的 DAGNode
        """
        return self.apply_operation(op, qubits, clbits)
    
    def finalize(self):
        """完成 DAG 构建，连接到输出节点"""
        for q in range(self._n_qubits):
            last_node_id = self._qubit_last_node[q]
            output_node = self._output_nodes[q]
            self._add_edge(last_node_id, output_node.node_id)
    
    def op_nodes(self) -> Iterator[DAGNode]:
        """迭代所有操作节点"""
        for node in self._nodes.values():
            if node.node_type == NodeType.OP:
                yield node
    
    def topological_op_nodes(self) -> Iterator[DAGNode]:
        """按拓扑顺序迭代操作节点"""
        # Kahn's algorithm
        in_degree = {nid: len(self._predecessors[nid]) for nid in self._nodes}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]

        while queue:
            node_id = queue.pop(0)
            node = self._nodes[node_id]

            if node.node_type == NodeType.OP:
                yield node

            for succ_id in self._successors[node_id]:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)

    def control_flow_op_nodes(self) -> Iterator[DAGNode]:
        """
        迭代控制流操作节点（for, while, if等）

        Returns:
            Iterator[DAGNode]: 控制流节点迭代器
        """
        # 目前Janus不支持控制流，返回空迭代器
        # 如果未来支持，需要检查node.op的类型
        return iter([])

    def layers(self) -> List[List[DAGNode]]:
        """
        获取 DAG 的分层表示
        
        Returns:
            每层包含可并行执行的操作节点
        """
        result = []
        qubit_layer = {q: -1 for q in range(self._n_qubits)}
        
        for node in self.topological_op_nodes():
            # 计算该节点应该在哪一层
            layer_idx = 0
            for q in node.qubits:
                layer_idx = max(layer_idx, qubit_layer[q] + 1)
            
            # 确保有足够的层
            while len(result) <= layer_idx:
                result.append([])
            
            result[layer_idx].append(node)
            
            # 更新量子比特的层
            for q in node.qubits:
                qubit_layer[q] = layer_idx
        
        return result
    
    def depth(self, recurse: bool = False) -> int:
        """
        获取 DAG 深度

        Args:
            recurse: 是否递归计算嵌套电路的深度 (暂时忽略,保持 Qiskit 兼容性)

        Returns:
            DAG 的深度(层数)
        """
        return len(self.layers())

    def width(self) -> int:
        """
        获取 DAG 宽度

        Returns:
            量子比特数 + 经典比特数
        """
        return self._n_qubits + self._n_clbits

    def size(self, recurse: bool = False) -> int:
        """
        获取 DAG 大小(操作节点数量)

        Args:
            recurse: 是否递归计算嵌套电路的大小 (暂时忽略,保持 Qiskit 兼容性)

        Returns:
            操作节点的数量
        """
        return sum(1 for _ in self.op_nodes())

    def count_ops(self, recurse: bool = False) -> Dict[str, int]:
        """
        统计各类操作的数量

        Args:
            recurse: 是否递归计算嵌套电路的操作 (暂时忽略,保持 Qiskit 兼容性)

        Returns:
            操作名称到数量的映射
        """
        counts = {}
        for node in self.op_nodes():
            name = node.op.name if node.op else "unknown"
            counts[name] = counts.get(name, 0) + 1
        return counts

    def num_tensor_factors(self) -> int:
        """
        计算 DAG 电路中的张量因子数量

        张量因子是指独立的量子子系统,即没有门连接的量子比特组。
        使用并查集(Union-Find)算法来找出连通的量子比特组。

        Returns:
            张量因子的数量
        """
        if self._n_qubits == 0:
            return 0

        # 并查集数据结构
        parent = list(range(self._n_qubits))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_x] = root_y

        # 遍历所有操作节点,连接相关的量子比特
        for node in self.op_nodes():
            if len(node.qubits) >= 2:
                # 多量子比特门,连接所有相关的量子比特
                for i in range(len(node.qubits) - 1):
                    union(node.qubits[i], node.qubits[i + 1])

        # 计算独立的组数
        return len(set(find(q) for q in range(self._n_qubits)))

    def collect_runs(self, gate_names: List[str]) -> List[List[DAGNode]]:
        """
        收集 DAG 中指定门类型的连续运行序列

        在拓扑顺序中,找出指定门类型在同一量子比特上的连续序列。

        Args:
            gate_names: 要收集的门名称列表

        Returns:
            连续运行的门节点列表的列表
        """
        runs = []
        # 对每个量子比特跟踪当前的运行序列
        qubit_runs = {q: [] for q in range(self._n_qubits)}

        # 按拓扑顺序遍历节点
        for node in self.topological_op_nodes():
            # 检查是否是目标门类型
            gate_name = node.op.name if node.op else ""
            if gate_name in gate_names and len(node.qubits) == 1:
                # 单量子比特门,添加到对应量子比特的运行序列
                qubit = node.qubits[0]
                qubit_runs[qubit].append(node)
            else:
                # 不是目标门或是多量子比特门,保存并清空相关量子比特的运行序列
                for q in node.qubits:
                    if qubit_runs[q]:
                        if len(qubit_runs[q]) > 0:
                            runs.append(qubit_runs[q])
                        qubit_runs[q] = []

        # 保存所有剩余的运行序列
        for q in range(self._n_qubits):
            if qubit_runs[q]:
                runs.append(qubit_runs[q])

        return runs

    def collect_1q_runs(self) -> List[List[DAGNode]]:
        """
        收集单量子比特门的连续序列

        Returns:
            单量子比特门序列列表,每个序列是连续作用在同一量子比特上的单量子比特门
        """
        runs = []
        qubit_runs = {q: [] for q in range(self._n_qubits)}

        for node in self.topological_op_nodes():
            # 只处理单量子比特门
            if len(node.qubits) == 1:
                q = node.qubits[0]
                qubit_runs[q].append(node)
            else:
                # 遇到多量子比特门,结束当前运行
                for q in node.qubits:
                    if qubit_runs[q]:
                        runs.append(qubit_runs[q])
                        qubit_runs[q] = []

        # 添加剩余的运行
        for q in range(self._n_qubits):
            if qubit_runs[q]:
                runs.append(qubit_runs[q])

        return runs

    def collect_2q_runs(self) -> List[List[DAGNode]]:
        """
        收集双量子比特门块

        Returns:
            双量子比特门块列表,每个块包含连续的双量子比特门
        """
        blocks = []
        current_block = []
        used_qubits = set()

        for node in self.topological_op_nodes():
            # 只处理双量子比特门
            if len(node.qubits) == 2:
                # 检查是否与当前块有量子比特冲突
                node_qubits = set(node.qubits)
                if not current_block or not (node_qubits & used_qubits):
                    # 可以添加到当前块
                    current_block.append(node)
                    used_qubits.update(node_qubits)
                else:
                    # 开始新块
                    if current_block:
                        blocks.append(current_block)
                    current_block = [node]
                    used_qubits = node_qubits
            else:
                # 遇到非双量子比特门,结束当前块
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                    used_qubits = set()

        # 添加最后一个块
        if current_block:
            blocks.append(current_block)

        return blocks

    def copy_empty_like(self) -> 'DAGCircuit':
        """
        创建一个空的DAG副本,保留量子比特和经典比特数量,但不包含操作节点

        Returns:
            空的DAGCircuit对象
        """
        return DAGCircuit(self._n_qubits, self._n_clbits)

    def predecessors(self, node: DAGNode) -> Iterator[DAGNode]:
        """获取节点的前驱"""
        for pred_id in self._predecessors[node.node_id]:
            yield self._nodes[pred_id]
    
    def successors(self, node: DAGNode) -> Iterator[DAGNode]:
        """获取节点的后继"""
        for succ_id in self._successors[node.node_id]:
            yield self._nodes[succ_id]
    
    def remove_op_node(self, node: DAGNode):
        """
        移除一个操作节点，重新连接其前驱和后继
        """
        if node.node_type != NodeType.OP:
            raise ValueError("Can only remove OP nodes")
        
        # 对于每个量子比特，连接前驱到后继
        for q in node.qubits:
            # 找到该量子比特上的前驱和后继
            pred = None
            succ = None
            
            for p in self.predecessors(node):
                if q in p.qubits:
                    pred = p
                    break
            
            for s in self.successors(node):
                if q in s.qubits:
                    succ = s
                    break
            
            if pred and succ:
                self._add_edge(pred.node_id, succ.node_id)
        
        # 移除所有边
        for pred_id in list(self._predecessors[node.node_id]):
            self._remove_edge(pred_id, node.node_id)
        for succ_id in list(self._successors[node.node_id]):
            self._remove_edge(node.node_id, succ_id)
        
        # 移除节点
        del self._nodes[node.node_id]
        del self._successors[node.node_id]
        del self._predecessors[node.node_id]
    
    def substitute_node(self, node: DAGNode, new_op, inplace: bool = False) -> DAGNode:
        """
        替换节点的操作

        Args:
            node: 要替换的节点
            new_op: 新的操作
            inplace: 是否原地替换(兼容参数,总是原地替换)

        Returns:
            替换后的节点
        """
        if node.node_type != NodeType.OP:
            raise ValueError("Can only substitute OP nodes")
        node.op = new_op
        return node
    
    def __repr__(self) -> str:
        op_count = sum(1 for _ in self.op_nodes())
        return f"DAGCircuit(n_qubits={self._n_qubits}, ops={op_count}, depth={self.depth()})"


def circuit_to_dag(circuit: 'Circuit') -> DAGCircuit:
    """
    将 Circuit 转换为 DAGCircuit
    
    Args:
        circuit: Janus Circuit
    
    Returns:
        DAGCircuit
    """
    dag = DAGCircuit(circuit.n_qubits, circuit.n_clbits)
    
    for inst in circuit.instructions:
        dag.apply_operation(inst.operation, inst.qubits, inst.clbits)
    
    dag.finalize()
    return dag


def dag_to_circuit(dag: DAGCircuit) -> 'Circuit':
    """
    将 DAGCircuit 转换为 Circuit

    Args:
        dag: DAGCircuit

    Returns:
        Janus Circuit
    """
    from .circuit import Circuit

    circuit = Circuit(dag.n_qubits, dag.n_clbits)

    for node in dag.topological_op_nodes():
        circuit.append(node.op.copy(), node.qubits, node.clbits)

    return circuit


# Compatibility aliases for Qiskit-style node types
class DAGOpNode(DAGNode):
    """Operation node in the DAG - alias for DAGNode with OP type"""
    def __init__(self, op, qubits, clbits=None, node_id=None):
        super().__init__(
            node_id=node_id if node_id is not None else id(self),
            node_type=NodeType.OP,
            qubits=qubits,
            clbits=clbits or [],
            op=op
        )


class DAGInNode(DAGNode):
    """Input node in the DAG - alias for DAGNode with INPUT type"""
    def __init__(self, qubit, node_id=None):
        super().__init__(
            node_id=node_id if node_id is not None else id(self),
            node_type=NodeType.INPUT,
            qubits=[qubit],
            clbits=[],
            op=None
        )


class DAGOutNode(DAGNode):
    """Output node in the DAG - alias for DAGNode with OUTPUT type"""
    def __init__(self, qubit, node_id=None):
        super().__init__(
            node_id=node_id if node_id is not None else id(self),
            node_type=NodeType.OUTPUT,
            qubits=[qubit],
            clbits=[],
            op=None
        )
