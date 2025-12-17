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
    
    def __init__(self, n_qubits: int = 0, n_clbits: int = 0):
        self._n_qubits = n_qubits
        self._n_clbits = n_clbits
        
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
    
    def depth(self) -> int:
        """获取 DAG 深度"""
        return len(self.layers())
    
    def count_ops(self) -> Dict[str, int]:
        """统计各类操作的数量"""
        counts = {}
        for node in self.op_nodes():
            name = node.op.name if node.op else "unknown"
            counts[name] = counts.get(name, 0) + 1
        return counts
    
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
    
    def substitute_node(self, node: DAGNode, new_op) -> DAGNode:
        """替换节点的操作"""
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
