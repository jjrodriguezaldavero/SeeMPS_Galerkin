import numpy as np
from collections.abc import Sequence

from seemps.state import MPS
from seemps.operators import MPO, MPOList
from seemps.register.transforms import mpo_weighted_shifts

from .utils import mpo_identity


def mps_chebyshev_endpoint(n: int, is_left: bool = True) -> MPS:
    sign = np.array([1.0, -1.0] if is_left else [1.0, 1.0]).reshape(1, 2, 1)
    one = np.ones((1, 2, 1), dtype=np.float64)
    return MPS([sign] + [one] * (n - 1))


def mpo_face_trace(sites_per_dimension: Sequence[int], dim_face: int, is_left: bool) -> MPO:
    if not 0 <= dim_face < len(sites_per_dimension):
        raise IndexError

    cores = []
    for dim, n in enumerate(sites_per_dimension):
        if dim == dim_face:
            sign = np.array([1.0, -1.0] if is_left else [1.0, 1.0])
            zero = np.array([1.0, 0.0])
            cores.append(np.outer(zero, sign).reshape(1, 2, 2, 1))
            cores.extend([np.outer(zero, np.ones(2)).reshape(1, 2, 2, 1)] * (n - 1))
        else:
            cores.extend([np.eye(2).reshape(1, 2, 2, 1)] * n)
    return MPO(cores)


def mpo_column_projector(
    n: int, is_left_dirichlet: bool = False, is_right_dirichlet: bool = False
) -> MPO:
    constraints = int(is_left_dirichlet) + int(is_right_dirichlet)
    if constraints == 0:
        return mpo_identity(n)
    if n < 2:
        raise ValueError

    P_plus = mpo_weighted_shifts(n, [1.0], [constraints], periodic=False)
    P_minus = mpo_weighted_shifts(n, [1.0], [-constraints], periodic=False)
    return MPOList([P_plus, P_minus]).join().reverse()


def mpo_chebyshev_recombination(
    n: int, is_left_dirichlet: bool = False, is_right_dirichlet: bool = False
) -> MPO:
    match is_left_dirichlet, is_right_dirichlet:
        case True, True:
            ρ_Z = mpo_weighted_shifts(n, [-1.0, 1.0], [0, 2], periodic=False)
        case True, False:
            ρ_Z = mpo_weighted_shifts(n, [1.0, 1.0], [0, 1], periodic=False)
        case False, True:
            ρ_Z = mpo_weighted_shifts(n, [-1.0, 1.0], [0, 1], periodic=False)
        case False, False:
            return mpo_identity(n)

    ρ_P = mpo_column_projector(n, is_left_dirichlet, is_right_dirichlet).reverse()
    return MPOList([ρ_P, ρ_Z]).join().reverse()
