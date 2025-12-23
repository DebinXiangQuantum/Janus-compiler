"""Provides a commutation checker that caches the determined commutation results during this session """

# TODO: Implement proper commutation checker
# For now, provide stubs


class CommutationChecker:
    """Stub for commutation checking"""
    def __init__(self):
        self.cc = None

    def commute(self, op1, op2, qubits1, qubits2):
        """Check if two operations commute"""
        # Stub implementation - always return False (conservative)
        return False


def get_standard_commutation_checker():
    """Get standard commutation checker stub"""
    return CommutationChecker()


# Standard gates commutations dictionary (stub)
standard_gates_commutations = {}

StandardGateCommutations = standard_gates_commutations
SessionCommutationChecker = CommutationChecker()
SessionCommutationChecker.cc = get_standard_commutation_checker()
