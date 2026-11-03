import numpy as np

from seemps.operators import MPO, MPOList
from seemps.state import DEFAULT_STRATEGY, MPS, MPSSum, Strategy
from seemps.typing import Tensor4
from .transform import mpo_chebyshev_transform
from seemps.analysis.mesh import ChebyshevInterval


def _coefficient_derivative_mpo(n: int, strategy: Strategy) -> MPO:
    def core_at(site: int) -> Tensor4:
        weight = 2.0 ** (n - site)
        core = np.zeros((6, 2, 2, 6))
        core[0, 0, 0, 0] = 1.0
        core[0, 1, 1, 0] = 1.0
        core[0, 1, 0, 2] = 1.0
        core[0, 1, 0, 3] = weight
        core[0, 1, 1, 5] = weight
        core[1, :, 0, 1] = 1.0
        core[1, 1, 0, 4] = -0.5 * weight
        core[2, :, :, 2] = 1.0
        core[2, 1, :, 3] = weight
        core[3, :, :, 3] = 1.0
        core[4, :, 0, 4] = 1.0
        core[5, 0, 0, 5] = 1.0
        core[5, 1, 1, 5] = 1.0
        core[5, 1, 0, 3] = 1.0
        return core

    cores = [core_at(site) for site in range(n)]
    cores[0] = (cores[0][0] + cores[0][1])[None, ...]
    cores[-1] = (cores[-1][..., 3] + cores[-1][..., 4])[..., None]
    cores[-1] *= np.array([[0.0, 1.0], [1.0, 0.0]])[None, :, :, None]
    return MPO(cores, strategy).T


def chebyshev_derivative_mpo(
    order: int, interval: ChebyshevInterval, strategy: Strategy = DEFAULT_STRATEGY
) -> MPOList:
    """Construct a Chebyshev spectral differentiation operator.

    The operator transforms values on an ascending Chebyshev--Gauss grid to
    Chebyshev coefficients, applies the coefficient-space recurrence, and
    transforms the result back to the grid. The affine scaling from ``[-1, 1]``
    to ``interval`` is included.

    Parameters
    ----------
    order : int
        Nonnegative derivative order.
    interval : ChebyshevInterval
        Chebyshev--Gauss grid with a power-of-two size.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Simplification strategy used when applying the operator layers.

    Returns
    -------
    MPOList
        Transform, differentiation, and inverse-transform layers.
    """
    if not isinstance(order, int) or isinstance(order, bool) or order < 0:
        raise ValueError("The derivative order must be a nonnegative integer")
    if not (
        isinstance(interval, ChebyshevInterval)
        and not interval.endpoints
        and interval.size >= 2
        and interval.size & (interval.size - 1) == 0
        and interval.length() != 0
    ):
        raise ValueError("Invalid Chebyshev interval")

    n = interval.size.bit_length() - 1
    transform = mpo_chebyshev_transform(n, strategy=strategy)
    transform_inverse = mpo_chebyshev_transform(n, inverse=True, strategy=strategy)
    derivatives: list[MPO] = []
    if order:
        derivative = _coefficient_derivative_mpo(n, strategy).reverse()
        derivatives = [derivative] * order
        derivatives[0] = (2 / interval.length()) ** order * derivatives[0]
    return MPOList([*transform.mpos, *derivatives, *transform_inverse.mpos], strategy)


def chebyshev_derivative(
    mps: MPS | MPSSum,
    order: int,
    interval: ChebyshevInterval,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPS | MPSSum:
    """Differentiate an MPS on a power-of-two Chebyshev--Gauss grid.

    Parameters
    ----------
    mps : MPS or MPSSum
        Function values on the grid.
    order : int
        Nonnegative derivative order.
    interval : ChebyshevInterval
        Grid and physical domain of the function.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Simplification strategy used while applying the operator.

    Returns
    -------
    MPS or MPSSum
        Values of the differentiated function on the same grid.
    """
    if not isinstance(order, int) or isinstance(order, bool) or order < 0:
        raise ValueError("The derivative order must be a nonnegative integer")
    if not (
        isinstance(interval, ChebyshevInterval)
        and not interval.endpoints
        and interval.size >= 2
        and interval.size & (interval.size - 1) == 0
        and interval.length() != 0
    ):
        raise ValueError("Invalid Chebyshev interval")

    n = interval.size.bit_length() - 1
    if mps.size != n:
        raise ValueError("The state size does not match the Chebyshev grid")

    coefficients = mpo_chebyshev_transform(n, strategy=strategy).apply(mps, simplify=False)
    if order:
        derivative = _coefficient_derivative_mpo(n, strategy).reverse()
        derivatives = [derivative] * order
        derivatives[0] = (2 / interval.length()) ** order * derivatives[0]
        for derivative in derivatives:
            coefficients = derivative.apply(coefficients)
    return mpo_chebyshev_transform(n, inverse=True, strategy=strategy).apply(coefficients)


__all__ = [
    "chebyshev_derivative",
    "chebyshev_derivative_mpo",
]
