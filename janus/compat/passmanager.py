"""
PassManager相关基础类
完全独立实现,不依赖qiskit
"""

from typing import Dict, Any, Optional


class PropertySet:
    """
    Pass执行过程中的属性集合
    用于在不同Pass之间共享信息
    """

    def __init__(self):
        self._properties: Dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        return self._properties[key]

    def __setitem__(self, key: str, value: Any):
        self._properties[key] = value

    def __delitem__(self, key: str):
        """删除属性"""
        if key in self._properties:
            del self._properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self._properties

    def get(self, key: str, default: Any = None) -> Any:
        return self._properties.get(key, default)

    def __repr__(self) -> str:
        return f"PropertySet({self._properties})"


class RunState:
    """Pass运行状态"""

    def __init__(self):
        self.property_set = PropertySet()


class PassManagerState:
    """PassManager状态"""

    def __init__(self):
        self.property_set = PropertySet()


class PassManagerIR:
    """PassManager中间表示基类"""
    pass


class GenericPass:
    """
    Pass基类
    所有优化Pass继承此类
    """

    def __init__(self):
        self.property_set = PropertySet()
        self.requires = []  # 依赖的其他passes

    def run(self, dag):
        """
        运行pass的主方法
        子类必须实现此方法
        """
        raise NotImplementedError("Subclass must implement run()")

    def __call__(self, dag):
        """使pass可以像函数一样调用"""
        return self.run(dag)


class AnalysisPass(GenericPass):
    """
    分析Pass基类
    不修改DAG,只分析并设置property_set
    """
    pass


class TransformationPass(GenericPass):
    """
    转换Pass基类
    修改DAG并返回新的DAG
    """
    pass


class PassManager:
    """
    Pass管理器
    按顺序执行一系列Pass
    """

    def __init__(self, passes=None):
        self.passes = passes or []
        self.property_set = PropertySet()

    def append(self, passes):
        """添加pass到管理器"""
        if isinstance(passes, list):
            self.passes.extend(passes)
        else:
            self.passes.append(passes)

    def run(self, dag):
        """
        依次执行所有passes

        Args:
            dag: 输入的DAG电路

        Returns:
            优化后的DAG电路
        """
        current_dag = dag

        for pass_instance in self.passes:
            # 共享property_set
            pass_instance.property_set = self.property_set

            # 执行pass
            result = pass_instance.run(current_dag)

            # TransformationPass返回新DAG,AnalysisPass返回None
            if result is not None:
                current_dag = result

        return current_dag

    def __repr__(self) -> str:
        return f"PassManager(passes={len(self.passes)})"
