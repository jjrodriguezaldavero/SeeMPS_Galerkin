from __future__ import annotations

import importlib
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from seemps.operators import MPO
from seemps_galerkin.mps.factories.utils import diagonal_mpo_from_mps
from seemps.state import MPS, NO_TRUNCATION
from seemps_galerkin.mps.linear import AMEnSolver


def dense_mpo(matrix: np.ndarray, dims: list[int]) -> MPO:
    n = len(dims)
    vector = (
        matrix.reshape(tuple(dims) * 2)
        .transpose([j for k in range(n) for j in (k, k + n)])
        .reshape(-1)
    )
    cores = MPS.from_vector(vector, [d * d for d in dims], strategy=NO_TRUNCATION, normalize=False)
    return MPO([c.reshape(c.shape[0], d, d, c.shape[-1]) for c, d in zip(cores, dims)])


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("method", ["cg", "gmres"])
def test_jacobi_local_diagonal_and_original_residual(complex_data: bool, method: str) -> None:
    """Check actual local contractions, nonbinary cores, both directions, and guesses."""
    rng = np.random.default_rng(48)
    dims = [2, 3, 2, 2]
    N = int(np.prod(dims))
    matrix = rng.normal(size=(N, N))
    rhs = rng.normal(size=N)
    if complex_data:
        matrix = matrix + 1j * rng.normal(size=(N, N))
        rhs = rhs + 1j * rng.normal(size=N)
    matrix = matrix.conj().T @ matrix + 2 * np.eye(N)
    A = dense_mpo(matrix, dims)
    b = MPS.from_vector(rhs, dims, normalize=False, strategy=NO_TRUNCATION)
    saved = [core.copy() for state in (A, b) for core in state]
    calls = []
    krylov = getattr(spla, method)

    def checked(operator, v, **kwargs):
        local = operator.matmat(np.eye(len(v)))
        expected = 1 / np.diag(local)
        actual = kwargs["M"].matvec(np.ones(len(v)))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        calls.append(len(v))
        return krylov(operator, v, **kwargs)

    amen = importlib.import_module("seemps_galerkin.mps.amen.amen")
    with patch.object(amen, method, checked):
        u = AMEnSolver(method=method, rtol=1e-9, preconditioner="jacobi").solve(
            A, b, b.zero_state()
        )
    assert len(calls) > len(dims)
    assert np.linalg.norm(matrix @ u.to_vector() - rhs) / np.linalg.norm(rhs) < 1e-9
    for core, original in zip([c for state in (A, b) for c in state], saved):
        np.testing.assert_array_equal(core, original)


def test_jacobi_reduces_iterations_for_scaled_spd_system() -> None:
    diagonal = np.geomspace(1.0, 1e4, 32)
    dims = [2] * 5
    A = diagonal_mpo_from_mps(MPS.from_vector(diagonal, dims, normalize=False))
    b = MPS.from_vector(np.ones(32), dims, normalize=False)
    amen = importlib.import_module("seemps_galerkin.mps.amen.amen")
    counts = []
    for preconditioner in (None, "jacobi"):
        iterations = []

        def counted(operator, rhs, counter=iterations, **kwargs):
            return spla.cg(operator, rhs, callback=lambda x: counter.append(1), **kwargs)

        with patch.object(amen, "cg", counted):
            u = AMEnSolver(preconditioner=preconditioner).solve(A, b)
        assert np.linalg.norm(diagonal * u.to_vector() - 1) / np.sqrt(32) < 1e-9
        counts.append(len(iterations))
    assert counts[1] < counts[0]
