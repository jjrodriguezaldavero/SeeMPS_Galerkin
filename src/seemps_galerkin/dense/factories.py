import numpy as np
import functools
from collections.abc import Sequence

from seemps.analysis.mesh import RegularInterval


from ..typing import Operator, Vector

# -------------------
# Chebyshev transform
# -------------------


def _check_which_dimensions(which_dimensions: int | Sequence[int] | None, ndim: int) -> list[int]:
    dims = (
        list(range(ndim))
        if which_dimensions is None
        else [which_dimensions]
        if isinstance(which_dimensions, int)
        else list(which_dimensions)
    )

    if not all(isinstance(d, int) and 0 <= d < ndim for d in dims):
        raise ValueError
    if len(set(dims)) != len(dims):
        raise ValueError
    return dims


def op_dct2(N: int, inverse: bool = False) -> Operator:
    j = k = np.arange(N)
    F = np.cos(np.pi * np.outer(j, 2 * k + 1) / (2 * N))
    if inverse:
        F_inv = (1.0 / N) * F.T
        F_inv[:, 0] *= 0.5
        return F_inv
    else:
        return 2.0 * F


def op_chebyshev_transform(N: int, inverse: bool = False) -> Operator:
    F = op_dct2(N)
    if inverse:
        C = (F / 2.0).T
        return C[::-1, :]  # Flip rows so output vector y is in -1 -> +1 order
    else:
        C = F[:, ::-1]  # Flip columns to handle -1 -> +1 input order
        C /= N
        C[0, :] *= 0.5
        return C


def vec_nd_chebyshev_transform(
    v: Vector,
    shape: Sequence[int],
    which_dimensions: int | Sequence[int] | None = None,
    inverse: bool = False,
) -> Vector:
    ndim = len(shape)
    which_dimensions = _check_which_dimensions(which_dimensions, ndim)
    V = np.asarray(v, dtype=np.float64).reshape(shape)
    for k in which_dimensions:
        N_k = shape[k]
        C_k = op_chebyshev_transform(N_k, inverse=inverse)
        V = np.tensordot(C_k, V, axes=([1], [k]))
        V = np.moveaxis(V, 0, k)
    return V.ravel()


def vec_nd_chebyshev_interpolation(
    v: Vector, old_dimensions: Sequence[int], new_dimensions: Sequence[int]
) -> Vector:
    s_old, s_new = old_dimensions, new_dimensions
    if not (len(s_old) == len(s_new) and all(N_new >= N_old for N_old, N_new in zip(s_old, s_new))):
        raise ValueError

    dims = [k for k, (N_old, N_new) in enumerate(zip(s_old, s_new)) if N_new > N_old]
    if not dims:
        return v

    u = vec_nd_chebyshev_transform(v, s_old, which_dimensions=dims, inverse=False)
    u_pad = np.zeros(s_new, dtype=v.dtype)
    u_pad[tuple(slice(n) for n in s_old)] = u.reshape(s_old)
    return vec_nd_chebyshev_transform(u_pad.ravel(), s_new, which_dimensions=dims, inverse=True)


def op_chebyshev_vandermonde(
    N_chebyshev: int, N_equispaced: int, start: float = -1.0, stop: float = 1.0
) -> Operator:
    interval = RegularInterval(start, stop, N_equispaced, endpoint_left=True, endpoint_right=True)
    x = interval.to_vector()
    x_map = (2.0 / (stop - start)) * (x - interval.start) - 1.0  # Map to [-1, 1]
    k = np.arange(N_chebyshev, dtype=float)
    return np.cos(np.outer(np.arccos(x_map), k))


# -------------------
# Operator transforms
# -------------------


def op_bit_reversal(n: int) -> Operator:
    N = 2**n
    B = np.zeros((N, N))
    for j in range(N):
        i = int(f"{j:0{n}b}"[::-1], 2)
        B[i, j] = 1
    return B


def op_tensor_product(operators: Sequence[Operator]) -> Operator:
    return functools.reduce(np.kron, operators)


def op_kron_expand(A: Operator, shape: Sequence[int], dim: int) -> Operator:
    if not (A.shape[1] == shape[dim]):
        raise ValueError
    operators = [A if k == dim else np.eye(N_k, dtype=A.dtype) for k, N_k in enumerate(shape)]
    return op_tensor_product(operators)


def op_tensor_sum(operators: Sequence[Operator]) -> Operator:
    shape = [A.shape[0] for A in operators]
    N = int(np.prod(shape))
    A_sum = np.zeros((N, N), dtype=operators[0].dtype)
    for k, A_k in enumerate(operators):
        A_sum += op_kron_expand(A_k, shape, k)
    return A_sum


# -------------------
# Derivative operator
# -------------------


def op_chebyshev_derivative(N: int) -> Operator:
    D = np.zeros((N, N), dtype=float)
    for k in reversed(range(N - 1)):
        D[k + 1, k] += 2.0 * (k + 1)
        if k + 2 < N - 1:
            D[:, k] += D[:, k + 2]
    D[:, 0] *= 0.5
    return D.T


def op_nd_chebyshev_derivative(shape: Sequence[int], dim: int) -> Operator:
    if not 0 <= dim < len(shape):
        raise IndexError
    operators = [
        op_chebyshev_derivative(N) if axis == dim else np.eye(N) for axis, N in enumerate(shape)
    ]
    return op_tensor_product(operators)


# ----------------------
# Weighted-Gram operator
# ----------------------


def get_constant_coefficient(v: Vector, rtol: float = 100.0 * np.finfo(float).eps) -> float | None:
    atol = rtol * np.max(np.abs(v), initial=0.0)
    if np.max(np.abs(v[1:]), initial=0.0) > atol:
        return None
    return v[0].item()


def op_mult(v: Vector, N: int) -> Operator:
    get = lambda k: v[k] if 0 <= k < v.size else 0.0
    Q = np.zeros((N, N), dtype=np.result_type(v, float))
    for i in range(N):
        for j in range(N):
            Q[i, j] = 0.5 * (get(i + j) + get(abs(i - j)) + (get(0) if i == j else 0.0))
            if i == 0:
                Q[i, j] -= 0.5 * get(j)
    return Q


@functools.cache
def op_mult_e_cached(N: int, k: int) -> Operator:
    e = np.zeros(N, dtype=float)
    e[k] = 1.0
    return op_mult(e, N)


def op_nd_mult(
    v: Vector, shape: Sequence[int], round_rtol: float = 100.0 * np.finfo(float).eps
) -> Operator:
    N = int(np.prod(shape))
    Q = np.zeros((N, N), dtype=np.result_type(v, float))
    V = v.reshape(shape)

    round_tol = round_rtol * np.max(np.abs(V), initial=0.0)
    kept_indices = np.abs(V) > round_tol

    if not np.any(kept_indices):
        return Q
    if bool(kept_indices.flat[0]) and np.count_nonzero(kept_indices) == 1:
        return V.flat[0] * np.eye(N, dtype=Q.dtype)

    for idx in zip(*np.nonzero(kept_indices)):
        blocks = [op_mult_e_cached(N_k, k) for k, N_k in zip(idx, shape)]
        Q += V[idx] * op_tensor_product(blocks)
    return Q


def op_gram(N: int) -> Operator:
    G = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if (i + j) % 2 == 0:
                G[i, j] = 1.0 / (1.0 - (i + j) ** 2) + 1.0 / (1.0 - (i - j) ** 2)
    return G


def op_nd_gram(shape: Sequence[int]) -> Operator:
    if len(shape) == 0:
        return np.ones((1, 1), dtype=float)
    return op_tensor_product([op_gram(N) for N in shape])


def op_weighted_gram(v: Vector, shape: Sequence[int]) -> Operator:
    # A coefficient varying along just one coordinate has an exactly separated
    # weighted mass matrix. Ignore only transform noise, as for constant fields.
    if len(shape) > 1:
        coefficients = v.reshape(shape)
        atol = 100.0 * np.finfo(float).eps * np.max(np.abs(v), initial=0.0)
        for axis in range(len(shape)):
            section = tuple(slice(None) if dim == axis else 0 for dim in range(len(shape)))
            remainder = coefficients.copy()
            remainder[section] = 0
            if np.max(np.abs(remainder), initial=0.0) <= atol:
                factors = [op_gram(N_k) for N_k in shape]
                factors[axis] = op_weighted_gram(coefficients[section], [shape[axis]])
                return op_tensor_product(factors)

    N = int(np.prod(shape))
    shape_doubled = tuple(2 * N_k for N_k in shape)
    interior = tuple(slice(N_k) for N_k in shape)
    v_doubled = np.zeros(shape_doubled, dtype=v.dtype)
    v_doubled[interior] = v.reshape(shape)
    GW_doubled = op_nd_gram(shape_doubled) @ op_nd_mult(v_doubled.ravel(), shape_doubled)
    return GW_doubled.reshape(shape_doubled + shape_doubled)[interior + interior].reshape(N, N)


# ------------------
# Boundary operators
# ------------------


def vec_chebyshev_endpoint(N: int, is_left: bool = True) -> Vector:
    return (-1.0) ** np.arange(N) if is_left else np.ones(N, dtype=float)


def op_face_trace(shape: Sequence[int], dim_face: int, is_left: bool) -> Operator:
    if not 0 <= dim_face < len(shape):
        raise IndexError
    operators = [
        vec_chebyshev_endpoint(N, is_left=is_left).reshape(1, N) if dim == dim_face else np.eye(N)
        for dim, N in enumerate(shape)
    ]
    return op_tensor_product(operators)


def op_chebyshev_recombination(
    N: int, is_left_dirichlet: bool = False, is_right_dirichlet: bool = False
) -> Operator:
    Z = np.zeros((N, N), dtype=float)
    match is_left_dirichlet, is_right_dirichlet:
        case True, True:
            k = np.arange(max(N - 2, 0), dtype=int)
            Z[k, k] = -1.0
            Z[k + 2, k] = 1.0
        case True, False:
            k = np.arange(max(N - 1, 0), dtype=int)
            Z[k, k] = 1.0
            Z[k + 1, k] = 1.0
        case False, True:
            k = np.arange(max(N - 1, 0), dtype=int)
            Z[k, k] = -1.0
            Z[k + 1, k] = 1.0
        case False, False:
            return np.eye(N)
    return Z


def op_column_projector(
    N: int, is_left_dirichlet: bool = False, is_right_dirichlet: bool = False
) -> Operator:
    constraints = int(is_left_dirichlet) + int(is_right_dirichlet)
    if constraints == 0:
        return np.eye(N)

    P = np.ones(N)
    P[-constraints:] = 0.0
    return np.diag(P)
