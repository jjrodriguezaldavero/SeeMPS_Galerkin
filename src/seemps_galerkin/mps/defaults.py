from seemps.state import Strategy, Simplification, Truncation

DEFAULT_TOLERANCE = 1e-10
DEFAULT_SVD_STRATEGY = Strategy(
    method=Truncation.RELATIVE_SINGULAR_VALUE,
    tolerance=DEFAULT_TOLERANCE,
    simplification_tolerance=DEFAULT_TOLERANCE,
    max_bond_dimension=1000,
    max_sweeps=16,
    simplify=Simplification.VARIATIONAL,
)
