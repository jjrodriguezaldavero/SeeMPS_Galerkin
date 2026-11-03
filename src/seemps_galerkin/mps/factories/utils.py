import numpy as np
from collections.abc import Sequence

from seemps.state import DEFAULT_STRATEGY, MPS, MPSSum, Strategy, mps_tensor_product
from seemps.operators import MPO, MPOSum, simplify_mpo

from ...typing import Matrix


def checked_log2(N: int) -> int:
    if not (isinstance(N, int) and N > 0 and N & (N - 1) == 0):
        raise ValueError
    return N.bit_length() - 1


def checked_sites_per_dimension(shape: Sequence[int]) -> list[int]:
    return [checked_log2(N) for N in shape]


# -----------------------
# General MPS/MPO helpers
# -----------------------


def mps_identity(n: int) -> MPS:
    return MPS([np.array([1.0, 1.0]).reshape(1, 2, 1) for _ in range(n)])


def mps_full(n: int, v: float) -> MPS:
    return v * mps_identity(n)


def mps_zeros(n: int) -> MPS:
    return MPS([np.array([0.0, 0.0]).reshape(1, 2, 1) for _ in range(n)])


def mps_vector_field(components: Sequence[MPS | None], n: int) -> MPS:
    m = len(components)
    if m == 1:
        return mps_zeros(n) if components[0] is None else components[0]

    mps_list = []
    for μ, ψ in enumerate(components):
        if ψ is None:
            continue
        selector = np.zeros((1, m, 1), dtype=ψ[0].dtype)
        selector[0, μ, 0] = 1.0
        mps_list.append(MPS([selector, *ψ._data]))
    if mps_list:
        return MPSSum([1.0] * len(mps_list), mps_list).join()

    selector = np.zeros((1, m, 1), dtype=float)
    return MPS([selector, *mps_identity(n)._data])


def mps_vector_reverse(
    ψ: MPS, sites_per_dimension: Sequence[int], which_dimensions: Sequence[int], offset: int = 0
) -> MPS:
    data = list(ψ._data)
    pos = offset
    for dim, n in enumerate(sites_per_dimension):
        start, end = pos, pos + n
        pos += n
        if dim in which_dimensions:
            for site in range(start, end):
                data[site] = data[site][:, ::-1, :]
    return MPS(data)


def get_mps_constant_coefficient(
    ψ: MPS, rtol: float = 100 * np.finfo(float).eps
) -> float | complex | None:
    n = len(ψ)
    right_environments: list[Matrix | None] = [None] * (n + 1)
    right_environments[-1] = np.ones((1, 1), dtype=ψ[0].dtype)
    for site in range(n - 1, -1, -1):
        core = ψ[site]
        right_environment = right_environments[site + 1]
        assert right_environment is not None
        environment = np.zeros((core.shape[0], core.shape[0]), dtype=ψ[0].dtype)
        for value in range(core.shape[1]):
            branch = core[:, value, :]
            environment += branch @ right_environment @ branch.conj().T
        right_environments[site] = environment

    prefix = np.ones((1, 1), dtype=ψ[0].dtype)
    residual_squared = 0.0
    for site, core in enumerate(ψ):
        right_environment = right_environments[site + 1]
        assert right_environment is not None
        branch = prefix @ core[:, 1, :]
        contribution = (branch @ right_environment @ branch.conj().T).item()
        residual_squared += max(float(np.real(contribution)), 0.0)
        prefix = prefix @ core[:, 0, :]

    coefficient = prefix.item()
    residual = np.sqrt(residual_squared)
    norm = np.hypot(abs(coefficient), residual)
    if residual > rtol * norm:
        return None
    return np.real_if_close(coefficient).item()


def mps_kron_expand(ψ: MPS, sites_per_dimension: Sequence[int], dim: int) -> MPS:
    states = [ψ if axis == dim else mps_identity(n) for axis, n in enumerate(sites_per_dimension)]
    return mps_tensor_product(states)


def mps_delta(n: int, idx: int) -> MPS:
    N = 2**n
    if not -N <= idx < N:
        raise IndexError
    idx %= N
    cores = [np.eye(2)[(idx >> bit) & 1].reshape(1, 2, 1) for bit in range(n)]
    return MPS(cores[::-1])


def diagonal_mpo_from_mps(mps: MPS, strategy: Strategy = DEFAULT_STRATEGY) -> MPO:
    """Return an MPO that is equivalent to Hadamard multiplication by the MPS."""
    tensors = []
    for A in mps:
        a, i, b = A.shape
        tensors.append(A.reshape(a, i, 1, b) * np.eye(i, i).reshape(1, i, i, 1))
    return MPO(tensors, strategy)


def mpo_identity(n: int) -> MPO:
    return MPO([np.eye(2, dtype=np.float64).reshape(1, 2, 2, 1)] * n)


def mpo_zeros(n: int) -> MPO:
    return MPO([np.zeros((1, 2, 2, 1), dtype=float) for _ in range(n)])


def mpo_vector_field(components: Sequence[Sequence[MPO | None]], n: int) -> MPO:
    m = len(components)
    if m == 1:
        return mpo_zeros(n) if components[0][0] is None else components[0][0]

    mpos = []
    for μ, row in enumerate(components):
        for ν, ρ in enumerate(row):
            if ρ is None:
                continue
            selector = np.zeros((1, m, m, 1), dtype=ρ[0].dtype)
            selector[0, μ, ν, 0] = 1.0
            mpos.append(MPO([selector, *ρ._data]))
    if mpos:
        return MPOSum(mpos, [1.0] * len(mpos)).join()

    selector = np.zeros((1, m, m, 1), dtype=float)
    return MPO([selector, *mpo_identity(n)._data])


def mpo_tensor_product(mpos: Sequence[MPO]) -> MPO:
    return MPO([core for ρ in mpos for core in ρ._data])


def mpo_kron_expand(ρ: MPO, sites_per_dimension: Sequence[int], dim: int) -> MPO:
    mpos = [ρ if axis == dim else mpo_identity(n) for axis, n in enumerate(sites_per_dimension)]
    return mpo_tensor_product(mpos)


def mpo_sequential_simplify(mpos: Sequence[MPO], strategy: Strategy) -> MPO:
    ρ = mpos[0]
    for ρ_next in mpos[1:]:
        ρ = simplify_mpo(ρ_next @ ρ, strategy=strategy)
    return ρ


def mps_insert_zero_site(ψ: MPS, pos: int) -> MPS:
    data = list(ψ._data)
    rank = data[-1].shape[2] if pos == len(data) else data[pos].shape[0]
    core = np.zeros((rank, 2, rank), dtype=data[0].dtype)
    core[:, 0, :] = np.eye(rank, dtype=data[0].dtype)
    data.insert(pos, core)
    return MPS(data)


def mps_project_site(ψ: MPS, pos: int, idx: int) -> MPS:
    data = list(ψ._data)
    core = data.pop(pos)[:, idx, :]
    if pos < len(data):
        data[pos] = np.tensordot(core, data[pos], axes=(1, 0))
    else:
        data[pos - 1] = np.tensordot(data[pos - 1], core, axes=(2, 0))
    return MPS(data)
