"""Provides a general transpiler pass for collecting and consolidating blocks of nodes
in a circuit."""

from janus.optimize.basepasses import TransformationPass
from janus.compat.converters import dag_to_dagdependency, dagdependency_to_dag
from janus.optimize.collect_blocks import BlockCollector, BlockCollapser
from janus.compat.control_flow_utils import trivial_recurse
# STUB: control_flow utils


class BlockCollectCollapser(TransformationPass):
    """A general transpiler pass to collect and to consolidate blocks of nodes
    in a circuit.

    This transpiler pass depends on two functions: the collection function and the
    collapsing function. The collection function ``collect_function`` takes a DAG
    and returns a list of blocks. The collapsing function ``collapse_function``
    takes a DAG and a list of blocks, consolidates each block, and returns the modified
    DAG.

    The input and the output DAGs are of type :class:`~qiskit.dagcircuit.DAGCircuit`,
    however when exploiting commutativity analysis to collect blocks, the
    :class:`~qiskit.dagcircuit.DAGDependency` representation is used internally.
    To support this, the ``collect_function`` and ``collapse_function`` should work
    with both types of DAGs and DAG nodes.

    Other collection and consolidation transpiler passes, for instance
    :class:`~.CollectLinearFunctions`, may derive from this pass, fixing
    ``collect_function`` and ``collapse_function`` to specific functions.
    """

    def __init__(
        self,
        collect_function,
        collapse_function,
        do_commutative_analysis=False,
    ):
        """
        Args:
            collect_function (callable): a function that takes a DAG and returns a list
                of "collected" blocks of nodes
            collapse_function (callable): a function that takes a DAG and a list of
                "collected" blocks, and consolidates each block.
            do_commutative_analysis (bool): if True, exploits commutativity relations
                between nodes.
        """
        self.collect_function = collect_function
        self.collapse_function = collapse_function
        self.do_commutative_analysis = do_commutative_analysis

        super().__init__()

    @trivial_recurse
    def collect_and_collapse(self, dag):
        """Run the BlockCollectCollapser pass on `dag`.
        Args:
            dag (DAGCircuit): the DAG to be optimized.
        Returns:
            DAGCircuit: the optimized DAG.
        """

        # If the option commutative_analysis is set, construct DAGDependency from the given DAGCircuit.
        if self.do_commutative_analysis:
            dag = dag_to_dagdependency(dag)

        # call collect_function to collect blocks from DAG
        blocks = self.collect_function(dag)

        # call collapse_function to collapse each block in the DAG
        self.collapse_function(dag, blocks)

        # If the option commutative_analysis is set, construct back DAGCircuit from DAGDependency.
        if self.do_commutative_analysis:
            dag = dagdependency_to_dag(dag)

        return dag

    def run(self, dag):
        """Alias for collect_and_collapse() to maintain backward compatibility."""
        return self.collect_and_collapse(dag)


# Backward compatibility alias
CollectAndCollapse = BlockCollectCollapser


def collect_using_filter_function(
    dag,
    filter_function,
    split_blocks,
    min_block_size,
    split_layers=False,
    collect_from_back=False,
    max_block_width=None,
):
    """Corresponds to an important block collection strategy that greedily collects
    maximal blocks of nodes matching a given ``filter_function``.
    """
    return BlockCollector(dag).collect_all_matching_blocks(
        filter_fn=filter_function,
        split_blocks=split_blocks,
        min_block_size=min_block_size,
        split_layers=split_layers,
        collect_from_back=collect_from_back,
        max_block_width=max_block_width,
    )


def collapse_to_operation(dag, blocks, collapse_function):
    """Corresponds to an important block collapsing strategy that collapses every block
    to a specific object as specified by ``collapse_function``.
    """
    return BlockCollapser(dag).collapse_to_operation(blocks, collapse_function)
