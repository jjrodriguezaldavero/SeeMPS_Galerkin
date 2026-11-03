from __future__ import annotations

import numpy as np
import warnings
import scipy.linalg
import opt_einsum
from opt_einsum.contract import ContractExpression
from functools import lru_cache, partial
from typing import Literal
from collections.abc import Callable
from scipy.sparse.linalg import LinearOperator, cg, gmres

from seemps.cython import (
    destructively_truncate_vector,
    _destructive_svd,
    _contract_last_and_first,
)
from seemps.operators import MPO
from .quadratic import QuadraticForm
from seemps.state import CanonicalMPS, MPS, Strategy, DEFAULT_STRATEGY, random_mps
from seemps.state.antilinear import AntilinearForm
from seemps.tools import make_logger
from seemps.typing import Matrix, Tensor3, Tensor4, Vector


@lru_cache(maxsize=256)
def _local_operator_expression(
    shapes: tuple[tuple[int, ...], ...],
) -> ContractExpression:
    return opt_einsum.contract_expression("ijk,jlmn,rns,klr->ims", *shapes)


def _apply_local_operator(L: Tensor3, W: Matrix, R: Matrix, v: Vector) -> Vector:
    """Apply the local operator ``einsum('ijk,jlmn,rns,klr->ims', L, W, R, v)``.

    ``W`` is the MPO core as the matrix ``W[j*l, m*n]``, ``R`` the right environment as
    ``R[r*n, s]`` and ``v`` the flattened local tensor ``v[k, l, r]``. The contraction
    path is optimized once per combination of shapes and then reused.
    """
    _, j, k = L.shape
    l = W.shape[0] // j
    r = v.size // (k * l)
    n = R.shape[0] // r
    m = W.shape[1] // n
    operands = (L, W.reshape(j, l, m, n), R.reshape(r, n, R.shape[1]), v.reshape(k, l, r))
    expression = _local_operator_expression(tuple(op.shape for op in operands))
    return expression(*operands).reshape(-1)


def _jacobi_preconditioner(
    A: Matrix, B: Tensor3, C: Matrix, method: str, dtype: np.dtype
) -> LinearOperator:
    diagonal = np.real_if_close(np.einsum("ja,jnp,nr->apr", A, B, C, optimize=True)).reshape(-1)
    if not np.isfinite(diagonal).all() or np.any(diagonal == 0):
        raise ValueError("Jacobi needs a finite nonzero local diagonal")
    if method == "cg":
        # CG needs a symmetric positive-definite preconditioner.
        if np.any(np.abs(np.imag(diagonal)) > 1e-10 * np.abs(diagonal)) or np.any(
            np.real(diagonal) <= 0
        ):
            raise ValueError("Jacobi with CG needs a real positive local diagonal")
        diagonal = np.real(diagonal)

    return LinearOperator(
        (diagonal.size, diagonal.size), matvec=lambda v: v / diagonal, dtype=dtype
    )


def _block_jacobi_preconditioner(
    L: Tensor3,
    W: Tensor4,
    R: Tensor3,
    side: Literal["left", "right"],
    method: str,
    dtype: np.dtype,
) -> LinearOperator:

    α, p, β = L.shape[0], W.shape[1], R.shape[0]
    if side == "left":
        blocks = np.einsum(
            "ajb,jikn,nc->caibk", L, W, np.diagonal(R, axis1=0, axis2=2), optimize=True
        ).reshape(β, α * p, α * p)
    else:
        blocks = np.einsum(
            "ja,jikn,cnd->aickd", np.diagonal(L, axis1=0, axis2=2), W, R, optimize=True
        ).reshape(α, p * β, p * β)
    identity = np.eye(blocks.shape[-1], dtype=dtype)
    if method == "cg":
        inverses = np.array(
            [
                scipy.linalg.cho_solve(scipy.linalg.cho_factor(B, lower=True), identity)
                for B in blocks
            ]
        )
    else:
        inverses = np.array([scipy.linalg.solve(B, identity) for B in blocks])

    def apply(v: Vector) -> Vector:
        rhs = v.reshape(α * p, β).T if side == "left" else v.reshape(α, p * β)
        out = (inverses @ rhs[..., None])[..., 0]
        return (out.T if side == "left" else out).reshape(-1)

    return LinearOperator((α * p * β, α * p * β), matvec=apply, dtype=dtype)


def amen_solve(
    A: MPO,
    b: MPS,
    guess: MPS | None = None,
    maxiter: int = 20,
    atol: float = 0,
    rtol: float = 1e-8,
    strategy: Strategy = DEFAULT_STRATEGY,
    method: str = "cg",
    enrichment_rank: int = 2,
    residual_rank: int = 8,
    preconditioner: Literal["jacobi", "block_jacobi"] | None = None,
) -> CanonicalMPS:
    if preconditioner not in (None, "jacobi", "block_jacobi"):
        raise ValueError(f"Unknown preconditioner {preconditioner!r}")
    norm_b = b.norm()
    if norm_b == 0:
        return CanonicalMPS(b.zero_state(), normalize=False)

    target = max(atol / norm_b, rtol)
    local_tol = target * 0.05 / np.sqrt(b.size)
    strategy = strategy.replace(normalize=False)
    guess = (1 / norm_b) * (b if guess is None else guess)
    b = (1 / norm_b) * b
    x = CanonicalMPS([c.copy() for c in guess], center=0, strategy=strategy)
    complex_data = any(np.iscomplexobj(c) for state in (A, b, x) for c in state)
    random = random_mps(
        b.physical_dimensions(),
        residual_rank,
        complex=complex_data,
        rng=np.random.default_rng(0),
    )
    z = CanonicalMPS(random, center=0, strategy=strategy.replace(max_bond_dimension=residual_rank))

    # Orthogonality center of x and z. The sweep moves it by rewriting the
    # cores directly, so it is tracked here instead of in the CanonicalMPS.
    center = 0

    def _shift(enrichment: np.ndarray, i: int, direction: int) -> None:
        nonlocal center
        matrix, extra, zmat = [
            t.reshape(-1, t.shape[2]) if direction > 0 else t.reshape(t.shape[0], -1).conj().T
            for t in (x[i], enrichment, z[i])
        ]
        # The shared SVD overwrites its input; preserve the current core.
        u, s, vh = _destructive_svd(matrix.copy(order="C"))
        destructively_truncate_vector(s, strategy)
        keep = len(s)
        q, transfer = u[:, :keep], s[:keep, None] * vh[:keep]
        max_rank = strategy.get_max_bond_dimension()
        room = min(enrichment_rank, max_rank - keep, matrix.shape[0] - keep)
        if room > 0:
            # Reproject twice, then append zero-weight directions to preserve x.
            extra = extra - q @ (q.conj().T @ extra)
            extra = extra - q @ (q.conj().T @ extra)
            uq, sq, _ = _destructive_svd(extra)
            room = min(room, int(np.count_nonzero(sq > np.finfo(sq.dtype).eps * sq[0])))
            q = np.concatenate((q, uq[:, :room]), axis=1)
            transfer = np.concatenate(
                (transfer, np.zeros((room, transfer.shape[1]), dtype=transfer.dtype))
            )
        zq, zr = scipy.linalg.qr(zmat, mode="economic", check_finite=False)
        j = i + direction
        for state, basis, weight in ((x, q, transfer), (z, zq, zr)):
            a, n, b = state[i].shape
            if direction > 0:
                state[i] = basis.reshape(a, n, -1)
                state[j] = _contract_last_and_first(weight, state[j])
            else:
                state[i] = basis.conj().T.reshape(-1, n, b)
                state[j] = state[j] @ weight.conj().T
        center = j

    xx, zx = QuadraticForm(A, x), QuadraticForm(A, z, ket=x)
    xb, zb = AntilinearForm(x, b), AntilinearForm(z, b)
    krylov: Callable[..., tuple[np.ndarray, int]]
    if method == "cg":
        krylov = cg
    elif method == "gmres":
        krylov = gmres
    else:
        raise ValueError(f"Unknown Krylov method {method!r}")
    error = endpoint = float("inf")
    with make_logger() as logger:
        for half in range(2 * maxiter):
            direction = 1 if half % 2 == 0 else -1
            sites = range(b.size) if direction > 0 else range(b.size - 1, -1, -1)
            for i in sites:
                # Pack once per site; reuse across Krylov and residual contractions.
                W = np.ascontiguousarray(A[i].transpose(0, 2, 1, 3)).reshape(
                    A[i].shape[0] * A[i].shape[2], -1
                )
                R = np.ascontiguousarray(xx.right_env[i].transpose(2, 1, 0)).reshape(
                    xx.right_env[i].shape[2] * xx.right_env[i].shape[1], -1
                )
                zR = np.ascontiguousarray(zx.right_env[i].transpose(2, 1, 0)).reshape(
                    zx.right_env[i].shape[2] * zx.right_env[i].shape[1], -1
                )
                op = partial(_apply_local_operator, xx.left_env[i], W, R)
                rhs, value = xb.tensor1site().reshape(-1), x[i].reshape(-1)
                applied = op(value)
                if scipy.linalg.norm(rhs - applied) > local_tol:
                    dtype = np.result_type(rhs, value, applied)
                    operator = LinearOperator((rhs.size, rhs.size), matvec=op, dtype=dtype)
                    M = None
                    if preconditioner == "jacobi":
                        M = _jacobi_preconditioner(
                            np.diagonal(xx.left_env[i], axis1=0, axis2=2),
                            np.diagonal(A[i], axis1=1, axis2=2),
                            np.diagonal(xx.right_env[i], axis1=0, axis2=2),
                            method,
                            dtype,
                        )
                    elif preconditioner == "block_jacobi":
                        M = _block_jacobi_preconditioner(
                            xx.left_env[i], A[i], xx.right_env[i], "left", method, dtype
                        )
                    value, info = krylov(operator, rhs, x0=value, rtol=0, atol=local_tol, M=M)
                    if info < 0 or not np.isfinite(value).all():
                        raise RuntimeError(f"AMEn local {method} failed at site {i}: {info}")
                    if info:
                        logger(f"AMEn site={i}: local {method} iteration limit reached")
                x[i] = value.reshape(x[i].shape)
                z[i] = zb.tensor1site() - _apply_local_operator(
                    zx.left_env[i], W, zR, value
                ).reshape(z[i].shape)
                endpoint = float(scipy.linalg.norm(z[i]))
                j = i + direction
                if 0 <= j < b.size:
                    if direction > 0:
                        L, R, LB, RB = xx.left_env[i], zR, xb.L[i], zb.R[i]
                    else:
                        L, LB, RB = zx.left_env[i], zb.L[i], xb.R[i]
                    enrichment = _contract_last_and_first(LB, b[i]) @ RB
                    enrichment -= _apply_local_operator(L, W, R, value).reshape(enrichment.shape)
                    _shift(enrichment, i, direction)
                    for form in (xx, zx):
                        (form.move_right if direction > 0 else form.move_left)(j)
                    xb.update(direction)
                    zb.update(direction)
            logger(
                f"AMEn sweep={(half + 1) / 2:g}: projected residual={endpoint:.3e}, rank={x.max_bond_dimension()}"
            )
            # Projections can trigger verification, but only the global norm can stop.
            if endpoint <= target or half + 1 == 2 * maxiter:
                # Global residual norm: untruncated QR avoids cancellation in ||Ax-b||.
                if b.size == 1:
                    carry = A[0][0, :, :, 0] @ x[0].reshape(-1) - b[0].reshape(-1)
                else:
                    residual = (A.apply(x, simplify=False) - b).join()
                    carry = residual[0]
                    for core in list(residual)[1:]:
                        matrix = carry.reshape(-1, carry.shape[-1])
                        (r,) = scipy.linalg.qr(matrix, mode="r", check_finite=False)
                        carry = _contract_last_and_first(r[: min(matrix.shape)], core)
                error = float(scipy.linalg.norm(carry)) / b.norm()
                logger(f"AMEn verified residual={error:.3e}, target={target:.3e}")
                if error <= target:
                    break
        else:
            warnings.warn(
                f"AMEn did not converge: relative residual {error:.3e} > {target:.3e}",
                RuntimeWarning,
                stacklevel=2,
            )
    x[center] = norm_b * x[center]
    return CanonicalMPS(
        list(x), center=center, strategy=strategy, is_canonical=True, error=x.error()
    )
