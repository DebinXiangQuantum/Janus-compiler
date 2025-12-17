"""
Janus 编译器优化 Pass

基础优化 pass 实现
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from janus.circuit.dag import DAGCircuit, DAGNode, NodeType


class BasePass(ABC):
    """优化 Pass 基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Pass 名称"""
        pass
    
    @abstractmethod
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        执行优化
        
        Args:
            dag: 输入 DAG
        
        Returns:
            优化后的 DAG
        """
        pass


class RemoveIdentityPass(BasePass):
    """
    移除恒等门
    
    移除对电路没有影响的门，如 I 门
    """
    
    @property
    def name(self) -> str:
        return "remove_identity"
    
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = []
        
        for node in dag.op_nodes():
            if node.op and node.op.name.lower() in ('id', 'i', 'identity'):
                nodes_to_remove.append(node)
        
        for node in nodes_to_remove:
            dag.remove_op_node(node)
        
        return dag


class CancelInversesPass(BasePass):
    """
    消除相邻的逆门对
    
    例如: X-X, H-H, CX-CX (相同控制和目标) 等
    """
    
    # 自逆门（自己是自己的逆）
    SELF_INVERSE = {'x', 'y', 'z', 'h', 'cx', 'cz', 'swap'}
    
    @property
    def name(self) -> str:
        return "cancel_inverses"
    
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        changed = True
        
        while changed:
            changed = False
            nodes_to_remove = []
            
            for node in list(dag.op_nodes()):
                if node.node_id not in dag._nodes:
                    continue
                    
                name = node.op.name.lower() if node.op else ""
                
                if name not in self.SELF_INVERSE:
                    continue
                
                # 检查后继节点
                for succ in dag.successors(node):
                    if succ.node_type != NodeType.OP:
                        continue
                    
                    succ_name = succ.op.name.lower() if succ.op else ""
                    
                    # 相同门，相同量子比特
                    if (succ_name == name and 
                        succ.qubits == node.qubits and
                        self._is_direct_successor(dag, node, succ)):
                        nodes_to_remove.append((node, succ))
                        changed = True
                        break
            
            # 移除配对的门
            for node1, node2 in nodes_to_remove:
                if node1.node_id in dag._nodes:
                    dag.remove_op_node(node1)
                if node2.node_id in dag._nodes:
                    dag.remove_op_node(node2)
        
        return dag
    
    def _is_direct_successor(self, dag: DAGCircuit, node1: DAGNode, node2: DAGNode) -> bool:
        """检查 node2 是否是 node1 在所有相关量子比特上的直接后继"""
        for q in node1.qubits:
            # 找 node1 在量子比特 q 上的直接后继
            found = False
            for succ in dag.successors(node1):
                if q in succ.qubits and succ.node_id == node2.node_id:
                    found = True
                    break
            if not found:
                return False
        return True


class MergeRotationsPass(BasePass):
    """
    合并连续的旋转门
    
    例如: RZ(a) - RZ(b) -> RZ(a+b)
    """
    
    ROTATION_GATES = {'rx', 'ry', 'rz'}
    
    @property
    def name(self) -> str:
        return "merge_rotations"
    
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        changed = True
        
        while changed:
            changed = False
            
            for node in list(dag.op_nodes()):
                if node.node_id not in dag._nodes:
                    continue
                
                name = node.op.name.lower() if node.op else ""
                
                if name not in self.ROTATION_GATES:
                    continue
                
                # 检查后继
                for succ in dag.successors(node):
                    if succ.node_type != NodeType.OP:
                        continue
                    
                    succ_name = succ.op.name.lower() if succ.op else ""
                    
                    # 相同旋转门，相同量子比特，直接后继
                    if (succ_name == name and 
                        succ.qubits == node.qubits and
                        len(node.qubits) == 1):
                        
                        # 合并角度
                        angle1 = node.op.params[0] if node.op.params else 0
                        angle2 = succ.op.params[0] if succ.op.params else 0
                        new_angle = float(angle1) + float(angle2)
                        
                        # 归一化到 [-2π, 2π]
                        new_angle = new_angle % (2 * np.pi)
                        
                        # 如果角度接近 0，移除两个门
                        if abs(new_angle) < 1e-10 or abs(new_angle - 2*np.pi) < 1e-10:
                            dag.remove_op_node(succ)
                            dag.remove_op_node(node)
                        else:
                            # 更新第一个门的角度，移除第二个
                            node.op.params[0] = new_angle
                            dag.remove_op_node(succ)
                        
                        changed = True
                        break
        
        return dag
