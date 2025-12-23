"""Calculate the width of a DAG circuit."""

from janus.optimize.basepasses import AnalysisPass


class CircuitWidthAnalyzer(AnalysisPass):
    """Calculate the width of a DAG circuit.

    The result is saved in ``property_set['width']`` as an integer that
    contains the number of qubits + the number of clbits.
    """

    def analyze_width(self, dag):
        """Analyze the width of the circuit in the given DAG."""
        self.property_set["width"] = dag.width()

    def run(self, dag):
        """Alias for analyze_width() to maintain backward compatibility."""
        return self.analyze_width(dag)


# Backward compatibility alias
Width = CircuitWidthAnalyzer
