"""Condition numbers of the Chebyshev-Galerkin operator with and without preconditioning.

Reproduces the conditioning table of the preconditioning appendix of the paper: the
condition number of -div(a grad u) + a u on [-1, 1]^d with homogeneous Dirichlet conditions
and p parameters, for N = 16 modes per spatial coordinate, N_xi = 8 modes per parameter and
a = 1 + (1 / 2p) sum_q xi_q. The operators are assembled densely, without the padded
Dirichlet modes.

Run with ``uv run python conditioning.py`` from this folder.
"""

from functools import reduce

import numpy as np
import scipy.linalg as la

from seemps_galerkin.dense.factories import (
    op_chebyshev_derivative,
    op_chebyshev_recombination,
    op_gram,
    op_weighted_gram,
)

N = 16  # Chebyshev modes per spatial coordinate
N_XI = 8  # Chebyshev modes per parameter
TAU = 100.0  # shift of the reference stiffness form
LENGTH = 1.0  # physical length of each spatial coordinate
CASES = [(1, 0), (2, 0), (3, 0), (1, 1), (1, 2), (1, 3)]  # (d, p)


def kron(*matrices: np.ndarray) -> np.ndarray:
    return reduce(np.kron, matrices)


def spatial_blocks(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Gram and stiffness matrices in the basis vanishing at both endpoints."""
    G, D = op_gram(N), op_chebyshev_derivative(N)
    Z = op_chebyshev_recombination(N, True, True)[:, : N - 2]
    return Z.T @ G @ Z, Z.T @ D.T @ G @ D @ Z


def galerkin_operator(d: int, p: int) -> np.ndarray:
    M, K = spatial_blocks(N)
    spatial = sum(kron(*[K if α == β else M for β in range(d)]) for α in range(d)) + kron(*[M] * d)
    if p == 0:
        return spatial
    G = op_gram(N_XI)
    W = op_weighted_gram(np.eye(N_XI)[1], [N_XI])  # Gram matrix weighted by xi = T_1(xi)
    parametric = kron(*[G] * p) + sum(
        kron(*[W if q == r else G for r in range(p)]) for q in range(p)
    ) / (2 * p)
    return np.kron(spatial, parametric)


def cholesky_basis(H: np.ndarray) -> np.ndarray:
    """Return U^{-1} for H = U^T U, so that U^{-T} H U^{-1} = I."""
    return la.solve_triangular(la.cholesky(H, lower=False), np.eye(len(H)))


def jacobi(A: np.ndarray) -> np.ndarray:
    scale = 1 / np.sqrt(np.diag(A))
    return A * scale[:, None] * scale[None, :]


def condition_number(A: np.ndarray) -> float:
    eigenvalues = la.eigvalsh(A)
    return float(eigenvalues[-1] / eigenvalues[0])


def main() -> None:
    M, K = spatial_blocks(N)
    spatial_basis = cholesky_basis(K / LENGTH**2 + TAU * M)
    parameter_basis = cholesky_basis(op_gram(N_XI))
    print(f"{'d':>2} {'p':>2} {'None':>10} {'Jacobi':>10} {'Cholesky':>10} {'Chol.+Jac.':>10}")
    for d, p in CASES:
        A = galerkin_operator(d, p)
        basis = kron(*([spatial_basis] * d + [parameter_basis] * p))
        A_cholesky = basis.T @ A @ basis
        values = [A, jacobi(A), A_cholesky, jacobi(A_cholesky)]
        print(f"{d:>2} {p:>2} " + " ".join(f"{condition_number(B):>10.1e}" for B in values))


if __name__ == "__main__":
    main()
