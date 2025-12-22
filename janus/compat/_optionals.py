"""Optionals module - stub for optional dependencies checking"""


class OptionalDependencyChecker:
    """Stub for checking optional dependencies"""
    def __init__(self, name):
        self.name = name

    def require_in_call(self, feature_msg=""):
        """Decorator that allows function calls (always passes)"""
        def decorator(func):
            return func
        return decorator


# Create stub for SYMPY check
HAS_SYMPY = OptionalDependencyChecker("sympy")
HAS_MATPLOTLIB = OptionalDependencyChecker("matplotlib")
HAS_SKLEARN = OptionalDependencyChecker("sklearn")
