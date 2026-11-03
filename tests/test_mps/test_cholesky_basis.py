"""Check the Cholesky change of basis against dense linear algebra."""

import numpy as np
import scipy.linalg as la
from numpy.testing import assert_allclose
from seemps.operators import MPO, mps_as_mpo
from seemps.state import MPS

from seemps_galerkin.dense.factories import op_gram
from seemps_galerkin.logger import Logger
from seemps_galerkin.mps.defaults import DEFAULT_SVD_STRATEGY
from seemps_galerkin.mps.preconditioning import (
    ParametricBasisSolver,
    apply_group,
    mass_basis,
    spatial_basis,
)


def test_group_congruence_and_parameter_mass():
    rng = np.random.default_rng(52)
    B = rng.normal(size=(4, 4)) + 4 * np.eye(4)
    full = np.kron(np.eye(2), np.kron(B, np.eye(2)))
    matrix = rng.normal(size=(16, 16))
    operator = mps_as_mpo(
        MPS.from_vector(
            matrix.reshape([2] * 8).transpose(0, 4, 1, 5, 2, 6, 3, 7).ravel(),
            [4] * 4,
            normalize=False,
            strategy=DEFAULT_SVD_STRATEGY.replace(tolerance=1e-14),
        ),
        mpo_strategy=None,
    )
    assert_allclose(
        apply_group(operator, 1, 2, B, operator=True).to_matrix(),
        full.T @ matrix @ full,
        atol=1e-10,
    )
    for n in (1, 2, 4):
        p = [int(f"{i:0{n}b}"[::-1], 2) for i in range(2**n)]
        Q = mass_basis(n)
        assert_allclose(Q.T @ op_gram(2**n)[np.ix_(p, p)] @ Q, np.eye(2**n), atol=1e-13)
    Q = spatial_basis(4, length=1.0, dirichlet=True)
    assert_allclose(Q[-1, :-1], 0, atol=1e-14)
    assert_allclose(Q[:-1, -1], 0, atol=1e-14)


def test_basis_solver_restores_original_coordinates_and_guess():
    dims = [3] + [2] * 9
    A = MPO([np.eye(d).reshape(1, d, d, 1) for d in dims])
    x = MPS([np.arange(1.0, d + 1).reshape(1, d, 1) for d in dims])
    x[1][:, 1, :] = 0  # unused left-face Dirichlet coordinate

    class DenseReference:
        def solve(self, A, b, guess=None):
            matrix, rhs = A.to_matrix(), b.to_vector()
            assert_allclose(matrix @ guess.to_vector(), rhs, rtol=1e-11, atol=1e-10)
            return MPS.from_vector(
                la.solve(matrix, rhs, assume_a="pos"),
                dims,
                normalize=False,
                strategy=DEFAULT_SVD_STRATEGY.replace(tolerance=1e-14),
            )

    Logger.activate(level=0, collect=True)
    try:
        Logger.dump("solve.mps.projector_scale", 1.0)
        solver = ParametricBasisSolver(
            1,
            spatial_bits=(1, 1, 1),
            lengths=(1.0, 0.2, 0.2),
            num_parameters=6,
            dirichlet_axis=0,
            solver=DenseReference(),
        )
        actual = solver.solve(A, x, guess=x)
        assert_allclose(actual.to_vector(), x.to_vector(), rtol=1e-10, atol=1e-10)
    finally:
        Logger.deactivate()
        Logger.clear()
