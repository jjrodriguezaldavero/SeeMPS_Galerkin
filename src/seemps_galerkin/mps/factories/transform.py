import numpy as np
from collections.abc import Sequence

from seemps.operators import MPO, MPOList, MPOSum
from seemps.register import mpo_weighted_shifts
from seemps.state import DEFAULT_STRATEGY, MPS, Strategy


def mpo_dct2(n: int, strategy: Strategy = DEFAULT_STRATEGY) -> MPO | MPOList:
    r"""Construct the orthonormal radix-two DCT-II.

    The transform uses 2n - 2 real MPO layers, each with bond dimension at most 10.
    The construction follows the radix-two DCT-II/DCT-IV factorization of
    S. M. Perera, arXiv:1503.04106 (2015).

    Parameters
    ----------
    n : int
        Number of binary sites.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Strategy used when subsequently applying the MPO layers.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("The DCT requires a positive integer number of sites")

    def _mpo_butterfly(n: int) -> MPO:
        hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
        if n == 1:
            return MPO([hadamard.reshape(1, 2, 2, 1)], strategy)

        operators = []
        for bit in range(2):
            first = np.zeros((2, 2))
            first[:, bit] = hadamard[:, bit]
            tail = [np.eye(2)[:, ::-1] if bit else np.eye(2)] * (n - 1)
            operators.append(
                MPO([matrix.reshape(1, 2, 2, 1) for matrix in [first, *tail]], strategy)
            )
        return MPOSum(operators, strategy=strategy).join()

    def _mpo_rotation(n: int) -> MPO:
        if n == 1:
            cosine, sine = np.cos(np.pi / 8), np.sin(np.pi / 8)
            matrix = np.array([[cosine, sine], [sine, -cosine]])
            return MPO([matrix.reshape(1, 2, 2, 1)], strategy)

        operators = []
        theta0 = np.pi / (4 * 2**n)
        for input_bit in range(2):
            for output_bit in range(2):
                first = np.zeros((1, 2, 2, 1))
                first[0, input_bit, output_bit, 0] = (
                    -1.0 if (input_bit, output_bit) == (1, 0) else 1.0
                )
                cores: list[np.ndarray] = [first]
                left = np.array([[np.cos(theta0), np.sin(theta0)]])
                right = np.eye(2)[:, [int(input_bit != output_bit)]]
                for site in range(1, n):
                    core = np.zeros((2, 2, 2, 2))
                    for bit in range(2):
                        theta = (bit ^ input_bit) * np.pi / 2 ** (site + 2)
                        cosine, sine = np.cos(theta), np.sin(theta)
                        target = bit ^ input_bit ^ output_bit
                        core[:, bit, target, :] = np.array([[cosine, sine], [-sine, cosine]])
                        if site == n - 1 and input_bit and bit:
                            core[:, bit, target, :] *= -1
                    if site == 1:
                        core = np.einsum("ab,bijd->aijd", left, core)
                    if site == n - 1:
                        core = np.einsum("aijb,bc->aijc", core, right)
                    cores.append(core)
                operators.append(MPO(cores, strategy))
        return MPOSum(operators, strategy=strategy).join()

    def _mpo_rotation_butterfly(n: int) -> MPO:
        tail = n - 1
        identity = MPO([np.eye(2).reshape(1, 2, 2, 1) for _ in range(tail)], strategy)
        zero = MPO([np.diag([1.0, 0.0]).reshape(1, 2, 2, 1) for _ in range(tail)], strategy)
        last = MPO([np.diag([0.0, 1.0]).reshape(1, 2, 2, 1) for _ in range(tail)], strategy)
        up = mpo_weighted_shifts(tail, [1.0], [1], periodic=False).reverse()
        down = up.T
        scale = 1 / np.sqrt(2.0)
        operators = []
        for input_bit, output_bit, weight, operator in (
            (0, 0, scale, identity),
            (0, 0, 1 - scale, zero),
            (0, 1, scale, up),
            (1, 0, scale, down),
            (1, 1, -scale, identity),
            (1, 1, -(1 - scale), last),
        ):
            first = np.zeros((1, 2, 2, 1))
            first[0, input_bit, output_bit, 0] = weight
            cores = list(operator)
            if output_bit:
                cores = [core[:, :, ::-1, :].copy() for core in cores]
                cores[0] *= np.array([-1.0, 1.0]).reshape(1, 1, 2, 1)
            operators.append(MPO([first, *cores], strategy))
        return MPOSum(operators, strategy=strategy).join()

    def _mpo_select(prefix: int, dct2: MPO, dct4: MPO) -> MPO:
        if prefix == 0:
            return dct2

        transition = np.zeros((2, 2, 2, 2))
        for state in range(2):
            for bit in range(2):
                transition[state, bit, bit, (1 - state) * bit] = 1.0
        cores = [transition[[0]]] + [transition] * (prefix - 1)

        for site, (a, b) in enumerate(zip(dct2, dct4)):
            left = a.shape[0] + b.shape[0]
            right = 1 if site == dct2.size - 1 else a.shape[3] + b.shape[3]
            core = np.zeros((left, 2, 2, right))
            core[: a.shape[0], :, :, : a.shape[3]] = a
            start = 0 if site == dct2.size - 1 else a.shape[3]
            core[a.shape[0] :, :, :, start:] = b
            cores.append(core)
        return MPO(cores, strategy)

    layers = [
        _mpo_select(depth, _mpo_butterfly(n - depth), _mpo_rotation(n - depth))
        for depth in range(n)
    ]
    for depth in range(n - 2, 0, -1):
        size = n - depth
        identity = MPO([np.eye(2).reshape(1, 2, 2, 1) for _ in range(size)], strategy)
        layers.append(_mpo_select(depth, identity, _mpo_rotation_butterfly(size)))
    return layers[0] if n == 1 else MPOList(layers, strategy)


def mpo_chebyshev_transform(
    n: int, inverse: bool = False, strategy: Strategy = DEFAULT_STRATEGY
) -> MPOList:
    """Construct the transform between grid values and Chebyshev coefficients.

    The direct transform maps values ordered on an ascending Chebyshev--Gauss
    grid to bit-reversed Chebyshev coefficients. The inverse transform maps
    those coefficients back to ascending grid values.

    Parameters
    ----------
    n : int
        Number of binary sites. The grid contains ``2**n`` points.
    inverse : bool, default = False
        Select the coefficient-to-values transform instead of the
        values-to-coefficients transform.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Simplification strategy used when applying the transform layers.

    Returns
    -------
    MPOList
        Radix-two DCT and coefficient-normalization layers.
    """
    dct = mpo_dct2(n, strategy)
    layers = list(dct.mpos) if isinstance(dct, MPOList) else [dct]
    layers[0] = MPO([core[:, :, ::-1, :].copy() for core in layers[0]], strategy)

    # Convert between the orthonormal DCT convention and the standard
    # Chebyshev coefficients. The all-zero bit string selects coefficient 0.
    sq2 = np.sqrt(2.0)
    sc = sq2 ** (1 - n)
    if n == 1:
        diagonal = [sq2 / sc, 1 / sc] if inverse else [sc / sq2, sc]
        normalization = MPO([np.diag(diagonal).reshape(1, 2, 2, 1)], strategy)
    else:
        identity = MPO([np.eye(2).reshape(1, 2, 2, 1) for _ in range(n)], strategy)
        zro = MPO([np.diag([1.0, 0.0]).reshape(1, 2, 2, 1) for _ in range(n)], strategy)
        weights = [1 / sc, (sq2 - 1) / sc] if inverse else [sc, sc * (1 / sq2 - 1)]
        normalization = MPOSum([identity, zro], weights, strategy).join()

    if inverse:
        return MPOList([normalization, *[layer.T for layer in reversed(layers)]], strategy)
    return MPOList([*layers, normalization], strategy)


def mps_chebyshev_transform(
    mps: MPS,
    sites_per_dimension: Sequence[int],
    dimensions: Sequence[int] | None = None,
    inverse: bool = False,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPS:
    """Transform selected axes between grid values and Chebyshev coefficients.

    Parameters
    ----------
    mps : MPS
        Multidimensional function in serial ordering.
    sites_per_dimension : Sequence[int]
        Number of consecutive binary sites assigned to each dimension.
    dimensions : Sequence[int] or None, default = None
        Dimensions to transform. ``None`` transforms every dimension.
    inverse : bool, default = False
        Select coefficient-to-values transforms instead of
        values-to-coefficients transforms.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Simplification strategy used while applying each transform.
    """
    sites = list(sites_per_dimension)
    if any(n < 1 for n in sites) or sum(sites) != mps.size:
        raise ValueError("The dimension sites must match the MPS size")
    axes = list(range(len(sites))) if dimensions is None else list(dimensions)
    if len(set(axes)) != len(axes) or any(axis < 0 or axis >= len(sites) for axis in axes):
        raise ValueError("Invalid dimensions")

    offsets = np.cumsum([0, *sites])
    state = mps
    for axis in axes:
        axis_sites = list(range(int(offsets[axis]), int(offsets[axis + 1])))
        transform = mpo_chebyshev_transform(sites[axis], inverse=inverse, strategy=strategy).extend(
            mps.size, sites=axis_sites
        )
        state = transform.apply(state)
    return state


__all__ = ["mpo_chebyshev_transform", "mpo_dct2", "mps_chebyshev_transform"]
