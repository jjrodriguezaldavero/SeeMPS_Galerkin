"""Cholesky change of basis for parametric QTT Galerkin systems.

The basis along each coordinate is orthonormalized, which prevents the condition number of
the Galerkin operator from growing multiplicatively with the number of spatial and
parametric coordinates (see the preconditioning appendix of the paper). Parametric
coordinates use the Cholesky factor of the Chebyshev Gram matrix, and spatial coordinates
that of a shifted one-dimensional stiffness form. The transformed system is solved by a
wrapped linear solver, which may apply its own local Jacobi preconditioning, and the
solution is returned in the original coordinates.
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, overload

import numpy as np
import scipy.linalg as la
from seemps.analysis.evaluation import evaluate_mps
from seemps.operators import MPO, simplify_mpo
from seemps.state import MPS, Strategy, simplify

from ..dense.factories import op_chebyshev_derivative, op_chebyshev_recombination, op_gram
from ..logger import Logger
from .defaults import DEFAULT_SVD_STRATEGY
from .linear import AMEnSolver, LinearSolver


def _bit_reversal(n: int) -> np.ndarray:
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(2**n)])


@overload
def apply_group(
    state: MPS,
    start: int,
    n: int,
    B: np.ndarray,
    operator: Literal[False] = False,
    strategy: Strategy = ...,
) -> MPS: ...


@overload
def apply_group(
    state: MPO,
    start: int,
    n: int,
    B: np.ndarray,
    operator: Literal[True],
    strategy: Strategy = ...,
) -> MPO: ...


def apply_group(
    state: MPS | MPO,
    start: int,
    n: int,
    B: np.ndarray,
    operator: bool = False,
    strategy: Strategy = DEFAULT_SVD_STRATEGY,
) -> MPS | MPO:
    """Change the basis of the coordinate stored in sites ``start, ..., start + n - 1``.

    Returns ``B.T @ x`` along that coordinate for an MPS, or ``B.T @ A @ B`` for an MPO.
    The cores of the coordinate are contracted, transformed and split again by SVD.
    """
    if n < 1 or B.shape != (2**n, 2**n):
        raise ValueError("Expected a square basis matrix for a nonempty binary axis")
    cores = list(state)
    group = cores[start]
    for core in cores[start + 1 : start + n]:
        group = np.tensordot(group, core, axes=(-1, 0))
    left, right = group.shape[0], group.shape[-1]
    N = 2**n
    if operator:
        group = group.transpose(
            [0] + list(range(1, 2 * n, 2)) + list(range(2, 2 * n + 1, 2)) + [2 * n + 1]
        )
        group = group.reshape(left, N, N, right)
        group = np.einsum("ij,aikb,kl->ajlb", B, group, B, optimize=True)
        group = group.reshape([left] + [2] * (2 * n) + [right])
        axes = [0] + [j for i in range(n) for j in (1 + i, 1 + n + i)] + [2 * n + 1]
        group = group.transpose(axes).reshape([left] + [4] * n + [right])
        d = 4
    else:
        group = group.reshape(left, N, right)
        group = np.einsum("ij,aib->ajb", B, group, optimize=True).reshape(
            [left] + [2] * n + [right]
        )
        d = 2
    out = []
    rank = left
    for _ in range(n - 1):
        mat = group.reshape(rank * d, -1)
        U, s, V = la.svd(mat, full_matrices=False, check_finite=False)
        keep = max(1, int(np.count_nonzero(s > strategy.get_tolerance() * s[0])))
        out.append(
            U[:, :keep].reshape(rank, 2, 2, keep)
            if operator
            else U[:, :keep].reshape(rank, 2, keep)
        )
        group = s[:keep, None] * V[:keep]
        rank = keep
    out.append(group.reshape(rank, 2, 2, right) if operator else group.reshape(rank, 2, right))
    return (
        MPO(cores[:start] + out + cores[start + n :], strategy=strategy)
        if operator
        else MPS(cores[:start] + out + cores[start + n :])
    )


def mass_basis(n: int) -> np.ndarray:
    """Basis that orthonormalizes ``2**n`` Chebyshev modes in the unweighted Gram matrix.

    The rows and columns follow the bit-reversed (LSB-first) ordering of the QTT.
    """
    Q = la.solve_triangular(la.cholesky(op_gram(2**n), lower=False), np.eye(2**n))
    perm = _bit_reversal(n)
    return Q[np.ix_(perm, perm)]


def spatial_basis(n: int, length: float, dirichlet: bool, shift: float = 100.0) -> np.ndarray:
    """Basis that orthonormalizes a shifted one-dimensional stiffness form.

    The form is ``K / length**2 + shift * M`` on ``2**n`` Chebyshev modes, where ``K`` and
    ``M`` are the stiffness and Gram matrices of an axis of physical ``length``. If
    ``dirichlet`` is true, the modes are recombined to vanish at the left endpoint and the
    unused padded mode is kept at unit scale. The rows and columns follow the bit-reversed
    (LSB-first) ordering of the QTT.
    """
    N = 2**n
    G = op_gram(N)
    D = op_chebyshev_derivative(N)
    Z = op_chebyshev_recombination(N, dirichlet, False)
    M = Z.T @ G @ Z
    K = Z.T @ D.T @ G @ D @ Z
    if dirichlet:
        M[-1, -1] = 1.0
        K[-1, -1] = 1.0
    H = K / length**2 + shift * M
    Q = la.solve_triangular(la.cholesky(H, lower=False), np.eye(N))
    perm = _bit_reversal(n)
    return Q[np.ix_(perm, perm)]


def padding_projector(
    n_xi: int, spatial_bits: tuple[int, int, int], num_parameters: int, dirichlet_axis: int
) -> MPO:
    """Projector onto the padded mode of the Dirichlet axis, which carries no unknown."""
    first = sum(spatial_bits[:dirichlet_axis])
    last = first + spatial_bits[dirichlet_axis]
    return MPO(
        [np.eye(3).reshape(1, 3, 3, 1)]
        + [
            (np.diag([0.0, 1.0]) if first <= i < last else np.eye(2)).reshape(1, 2, 2, 1)
            for i in range(sum(spatial_bits) + num_parameters * n_xi)
        ],
        strategy=DEFAULT_SVD_STRATEGY,
    )


@dataclass
class ParametricBasisSolver:
    """Solve a parametric elasticity system after a Cholesky change of basis.

    The system has three displacement components, three spatial coordinates with
    ``spatial_bits`` binary cores and physical ``lengths``, and ``num_parameters``
    parametric coordinates with ``n_xi`` cores each. Dirichlet conditions are imposed on the
    left face of ``dirichlet_axis``. Parametric coordinates use :func:`mass_basis` and
    spatial coordinates :func:`spatial_basis`. The transformed system is rescaled by the
    mean of its diagonal and solved by ``solver``, and the solution is transformed back to
    the original coordinates.

    Requires the Dirichlet projector scale logged by
    :func:`seemps_galerkin.solve.solve_mps_bvp`, so the :class:`Logger` must be collecting.
    """

    n_xi: int
    spatial_bits: tuple[int, int, int]
    lengths: tuple[float, float, float]
    num_parameters: int
    dirichlet_axis: int
    solver: LinearSolver = field(default_factory=AMEnSolver)
    strategy: Strategy = field(default_factory=lambda: DEFAULT_SVD_STRATEGY)

    def solve(self, A: MPO, b: MPS, guess: MPS | None = None) -> MPS:
        tick = perf_counter()
        n = self.n_xi
        if n < 1 or b.physical_dimensions() != [3] + [2] * (
            sum(self.spatial_bits) + self.num_parameters * n
        ):
            raise ValueError("Unexpected parametric QTT dimensions")
        strategy = self.strategy
        self.original_A, self.original_b = A, b
        projector_scale = float(Logger.sink["solve.mps.projector_scale"][-1])
        P = padding_projector(n, self.spatial_bits, self.num_parameters, self.dirichlet_axis)
        A = simplify_mpo(A - projector_scale * P, strategy=strategy)
        parameter_start = 1 + sum(self.spatial_bits)
        parameter_basis = mass_basis(n)
        groups = [
            (start, n, parameter_basis)
            for start in range(parameter_start, parameter_start + self.num_parameters * n, n)
        ]
        start = 1
        for dim, width in enumerate(self.spatial_bits):
            basis = spatial_basis(width, self.lengths[dim], dim == self.dirichlet_axis)
            groups.append((start, width, basis))
            start += width
        for start, width, basis in groups:
            A = apply_group(A, start, width, basis, operator=True, strategy=strategy)
            b = apply_group(b, start, width, basis, strategy=strategy)

        # Rescale the system by the mean diagonal over the physical unknowns, and put the
        # padded Dirichlet modes back at unit scale. This does not change the solution.
        diagonal = MPS([np.einsum("aiib->aib", core) for core in A])
        rng = np.random.default_rng(510)
        indices = np.column_stack([rng.integers(d, size=1024) for d in b.physical_dimensions()])
        first = 1 + sum(self.spatial_bits[: self.dirichlet_axis])
        last = first + self.spatial_bits[self.dirichlet_axis]
        indices = indices[~np.all(indices[:, first:last] == 1, axis=1)]
        scale = float(np.mean(np.real(evaluate_mps(diagonal, indices))))
        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError("Invalid diagonal scale in the transformed system")
        A = simplify_mpo(A / scale + P, strategy=strategy)
        b = simplify(b / scale, strategy=strategy)
        if guess is not None:
            for start, width, basis in groups:
                guess = apply_group(guess, start, width, np.linalg.inv(basis).T, strategy=strategy)
        self.time_basis = perf_counter() - tick
        self.maxbond_A = A.max_bond_dimension()
        self.bond_dimensions_A = A.bond_dimensions()
        y = self.solver.solve(A, b, guess)
        tick = perf_counter()
        for start, width, basis in reversed(groups):
            y = apply_group(y, start, width, basis.T, strategy=strategy)
        self.x = simplify(y, strategy=strategy)
        self.time_basis += perf_counter() - tick
        return self.x
