"""Independent stencil checks and manufactured problems for both FD backends."""

import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import patch
from seemps.operators import MPO
from seemps.analysis.mesh import RegularInterval, QuantizedInterval, Mesh
from seemps.state import MPS, Simplification, Strategy
from types import SimpleNamespace
import scipy.linalg

from seemps_galerkin.fd import solve_dense_bvp_fd, solve_mps_bvp_fd
from seemps_galerkin.mps.defaults import DEFAULT_SVD_STRATEGY
from seemps_galerkin.mps.linear import AMEnSolver, DMRGSolver, LinearSolver, ScipySolver
from seemps_galerkin.problem import (
    BoundaryFace,
    BoundaryProblem,
    BulkProblem,
    DirichletBoundaryCondition,
    RobinBoundaryCondition,
    TensorField,
    field_to_mps,
)
from seemps_galerkin.typing import Vector, Matrix


STRATEGY = DEFAULT_SVD_STRATEGY.replace(
    tolerance=1e-13, simplification_tolerance=1e-13, simplify=Simplification.CANONICAL_FORM
)


def as_mps(
    bulk: BulkProblem[Vector], boundary: BoundaryProblem[Vector]
) -> tuple[BulkProblem[MPS], BoundaryProblem[MPS]]:
    def field(f: TensorField[Vector]) -> TensorField[MPS]:
        return field_to_mps(f, STRATEGY)

    return BulkProblem(
        bulk.mesh, field(bulk.principal), field(bulk.reaction), field(bulk.source)
    ), BoundaryProblem(
        boundary.mesh,
        [
            DirichletBoundaryCondition(bc.face, field(bc.data))
            if isinstance(bc, DirichletBoundaryCondition)
            else RobinBoundaryCondition(bc.face, field(bc.reaction), field(bc.data))
            for bc in boundary.boundary_conditions
        ],
    )


def dense_values(values: Vector, shape: tuple[int, ...], m: int) -> Matrix:
    return values.reshape(m, -1)


def mps_values(values: MPS, shape: tuple[int, ...], m: int) -> Matrix:
    return values.to_vector().reshape(m, -1)


def affine_problem(
    shape: tuple[int, ...] = (8,), m: int = 1, dirichlet: bool = True, nonlinear: bool = False
) -> tuple[BulkProblem[Vector], BoundaryProblem[Vector], Matrix]:
    d = len(shape)
    mesh = Mesh(
        [
            RegularInterval(-0.7 + k, 1.3 + 1.5 * k, N, endpoint_right=True)
            for k, N in enumerate(shape)
        ]
    )
    coords = np.array(
        [x.ravel() for x in np.meshgrid(*[i.to_vector() for i in mesh.intervals], indexing="ij")]
    )
    gradient = np.arange(1, m * d + 1).reshape(m, d) * 0.08
    material = 2 * np.eye(d * m) + 0.1 * np.ones((d * m, d * m))
    tensor = material.reshape(d, m, d, m).transpose(0, 2, 1, 3)
    reaction = np.eye(m) + 0.05 * np.ones((m, m))
    robin = 0.3 * np.eye(m) + 0.04 * np.ones((m, m))

    def values(x: Matrix) -> Matrix:
        v = np.arange(1, m + 1)[:, None] + gradient @ x
        return np.exp(v) if nonlinear else v

    def derivatives(x: Matrix) -> np.ndarray:
        return gradient[:, :, None] * (
            values(x)[:, None, :] if nonlinear else np.ones((1, 1, x.shape[1]))
        )

    def flux(x: Matrix, dim: int) -> Matrix:
        return np.einsum("buv,vbq->uq", tensor[dim], derivatives(x)) * (1 + 0.1 * x.sum(axis=0))

    principal = TensorField((d, d, m, m), shape)
    R = TensorField((m, m), shape)
    source = TensorField((m,), shape)
    u = values(coords)
    variation = 0 if min(shape) == 2 else 0.05
    f = -0.1 * np.einsum("abuv,vbq->uq", tensor, derivatives(coords)) + (reaction @ u) * (
        1 + variation * coords.sum(axis=0)
    )
    if nonlinear:
        f -= np.einsum("abuv,va,vb,vq->uq", tensor, gradient, gradient, u) * (
            1 + 0.1 * coords.sum(axis=0)
        )
    for index in np.ndindex(principal.field_shape):
        principal[index] = tensor[index] * (1 + 0.1 * coords.sum(axis=0))
    for mu in range(m):
        source[mu] = f[mu]
        for nu in range(m):
            R[mu, nu] = reaction[mu, nu] * (1 + variation * coords.sum(axis=0))
    conditions = []
    for dim in range(d):
        face_shape = shape[:dim] + shape[dim + 1 :]
        for left in (True, False):
            points = [
                np.array([i.start if left else i.stop]) if k == dim else i.to_vector()
                for k, i in enumerate(mesh.intervals)
            ]
            x = np.array([v.ravel() for v in np.meshgrid(*points, indexing="ij")])
            data = TensorField((m,), face_shape)
            B = TensorField((m, m), face_shape)
            is_dirichlet = dirichlet and left and (not nonlinear or dim == 0)
            g = (
                values(x)
                if is_dirichlet
                else (-1 if left else 1) * flux(x, dim)
                + (robin @ values(x)) * (1 + 0.03 * x.sum(axis=0))
            )
            for mu in range(m):
                data[mu] = g[mu]
                for nu in range(m):
                    B[mu, nu] = robin[mu, nu] * (1 + 0.03 * x.sum(axis=0))
            face = BoundaryFace(dim, left)
            conditions.append(
                DirichletBoundaryCondition(face, data)
                if is_dirichlet
                else RobinBoundaryCondition(face, B, data)
            )
    return BulkProblem(mesh, principal, R, source), BoundaryProblem(mesh, conditions), u


@pytest.mark.parametrize("shape,m", [((8,), 1), ((8, 4), 2), ((4, 4, 4), 1)])
@pytest.mark.parametrize("dirichlet", [False, True])
def test_dense_affine_full_tensor(shape: tuple[int, ...], m: int, dirichlet: bool) -> None:
    bulk, boundary, expected = affine_problem(shape, m, dirichlet)
    solution = solve_dense_bvp_fd(bulk, boundary)
    npt.assert_allclose(dense_values(solution, shape, m), expected, atol=2e-11)


@pytest.mark.parametrize("shape,m", [((8,), 1), ((4, 4), 2), ((4, 4, 4), 1)])
@pytest.mark.parametrize("dirichlet", [False, True])
def test_mps_affine_full_tensor(shape: tuple[int, ...], m: int, dirichlet: bool) -> None:
    bulk, boundary, expected = affine_problem(shape, m, dirichlet)
    p, q = as_mps(bulk, boundary)
    solution = solve_mps_bvp_fd(p, q, solver=ScipySolver(), strategy=STRATEGY)
    npt.assert_allclose(mps_values(solution, shape, m), expected, atol=2e-8)


def test_robin_assembly_matches_notebook_ghost_elimination() -> None:
    p, q, _ = affine_problem((8,), dirichlet=False)
    N = 8
    h = (p.mesh.intervals[0].stop - p.mesh.intervals[0].start) / (N - 1)
    p.principal[0, 0, 0, 0] = np.full(N, 1.7)
    p.reaction[0, 0] = np.full(N, 0.8)
    p.source[0] = np.linspace(0, 1, N)
    expected = 1.7 / h**2 * (2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)) + 0.8 * np.eye(N)
    rhs = p.source[0].copy()
    for bc, index, neighbor in zip(q.boundary_conditions, [0, N - 1], [1, N - 2]):
        bc.reaction[0, 0] = np.array([0.4])
        bc.data[0] = np.array([0.7])
        expected[index, index] += 2 * 0.4 / h
        expected[index, neighbor] -= 1.7 / h**2
        rhs[index] += 2 * 0.7 / h
    with patch.object(scipy.linalg, "solve", wraps=scipy.linalg.solve) as solve:
        solve_dense_bvp_fd(p, q)
    A, b = solve.call_args.args
    weights = np.full(N, h)
    weights[[0, -1]] *= 0.5
    npt.assert_allclose(A / weights[:, None], expected, atol=1e-12)
    npt.assert_allclose(b / weights, rhs, atol=1e-12)


def test_mps_and_dense_discrete_systems_agree() -> None:
    p, q, _ = affine_problem((4, 8), 2, dirichlet=False)
    with patch.object(scipy.linalg, "solve", wraps=scipy.linalg.solve) as solve:
        solve_dense_bvp_fd(p, q)
    A, b = solve.call_args.args
    mp, mq = as_mps(p, q)

    def check(ma: MPO, mb: MPS, guess: MPS | None) -> MPS:
        npt.assert_allclose(ma.to_matrix(), A, atol=2e-10)
        npt.assert_allclose(mb.to_vector(), b, atol=2e-10)
        return ScipySolver().solve(ma, mb, guess)

    solve_mps_bvp_fd(mp, mq, strategy=STRATEGY, solver=SimpleNamespace(solve=check))
    npt.assert_allclose(A, A.T, atol=1e-12)
    assert np.linalg.eigvalsh(A).min() > 0


@pytest.mark.parametrize("dirichlet", [False, True])
@pytest.mark.parametrize("strategy", [STRATEGY, DEFAULT_SVD_STRATEGY])
def test_fd_mpo_symmetry_after_compression_and_boundary_elimination(
    dirichlet: bool, strategy: Strategy
) -> None:
    # Full variable spatial/component couplings, with both mixed and all-Robin BCs.
    p, q, _ = affine_problem((8, 8), 2, dirichlet, nonlinear=True)
    with patch.object(scipy.linalg, "solve", wraps=scipy.linalg.solve) as solve:
        reference = solve_dense_bvp_fd(p, q)
    dense, _ = solve.call_args.args
    npt.assert_allclose(dense, dense.T, atol=1e-12, rtol=0)
    p, q = as_mps(p, q)

    def check(operator: MPO, rhs: MPS, guess: MPS | None) -> MPS:
        matrix = operator.to_matrix()
        tolerance = max(1e-12, 10 * strategy.get_tolerance())
        assert np.linalg.norm(matrix - matrix.T) <= tolerance * np.linalg.norm(matrix)
        assert np.linalg.eigvalsh((matrix + matrix.T) / 2).min() > 0
        return ScipySolver().solve(operator, rhs, guess)

    actual = solve_mps_bvp_fd(p, q, strategy=strategy, solver=SimpleNamespace(solve=check))
    npt.assert_allclose(actual.to_vector(), reference, atol=2e-7, rtol=0)


@pytest.mark.parametrize("dirichlet", [False, True])
def test_second_order_convergence_variable_diffusion(dirichlet: bool) -> None:
    errors = []
    for N in (16, 32, 64):
        p, q, _ = affine_problem((N,), dirichlet=dirichlet)
        x = p.mesh.intervals[0].to_vector()
        u = np.exp(0.6 * x)
        p.principal[0, 0, 0, 0] = 2 + 0.2 * x
        p.reaction[0, 0] = np.ones(N)
        p.source[0] = (1 - 0.12 - 0.36 * (2 + 0.2 * x)) * u
        for bc in q.boundary_conditions:
            point = p.mesh.intervals[0].start if bc.face.is_left else p.mesh.intervals[0].stop
            value = np.exp(0.6 * point)
            if isinstance(bc, DirichletBoundaryCondition):
                bc.data[0] = np.array([value])
            else:
                bc.reaction[0, 0] = np.array([0.3])
                bc.data[0] = np.array(
                    [(((-1 if bc.face.is_left else 1) * (2 + 0.2 * point) * 0.6) + 0.3) * value]
                )
        solution = dense_values(solve_dense_bvp_fd(p, q), (N,), 1)[0]
        errors.append(np.max(np.abs(solution - u)))
    assert errors[0] / errors[1] > 3.5
    assert errors[1] / errors[2] > 3.5


@pytest.mark.parametrize(
    "solver", [DMRGSolver(rtol=1e-11, strategy=STRATEGY), AMEnSolver(rtol=1e-11)]
)
def test_compressed_solver_and_nodal_guess(solver: LinearSolver) -> None:
    p, q, expected = affine_problem((8,), 2)
    p, q = as_mps(p, q)
    guess = MPS.from_vector(
        0.9 * expected.ravel(), [2, 2, 2, 2], normalize=False, strategy=STRATEGY
    )
    original = guess.to_vector().copy()
    actual = solve_mps_bvp_fd(
        p,
        q,
        guess=guess,
        strategy=STRATEGY,
        solver=solver,
    )
    npt.assert_allclose(mps_values(actual, (8,), 2), expected, atol=2e-8)
    npt.assert_array_equal(guess.to_vector(), original)


@pytest.mark.parametrize("dirichlet", [False, True])
def test_second_order_nonlinear_anisotropic_coupled_problem(dirichlet: bool) -> None:
    errors = []
    for N in (8, 16, 32):
        p, q, expected = affine_problem((N, N), 2, dirichlet, nonlinear=True)
        actual = dense_values(solve_dense_bvp_fd(p, q), (N, N), 2)
        errors.append(np.max(np.abs(actual - expected)))
    assert errors[0] / errors[1] > 3.2
    assert errors[1] / errors[2] > 3.2
    p, q, _ = affine_problem((4, 8), 2, dirichlet, nonlinear=True)
    reference = dense_values(solve_dense_bvp_fd(p, q), (4, 8), 2)
    p, q = as_mps(p, q)
    actual = solve_mps_bvp_fd(p, q, strategy=STRATEGY, solver=ScipySolver())
    npt.assert_allclose(mps_values(actual, (4, 8), 2), reference, atol=3e-8)


def test_partial_dirichlet_and_absent_fields() -> None:
    p, q, _ = affine_problem((8,), 2)
    x = p.mesh.intervals[0].to_vector()
    expected = np.array([1 + x, 2 * np.ones(8)])
    for index in np.ndindex(p.principal.field_shape):
        p.principal[index] = np.ones(8) if index[-2] == index[-1] else None
    for mu in range(2):
        p.source[mu] = expected[mu]
        for nu in range(2):
            p.reaction[mu, nu] = np.ones(8) if mu == nu else None
    left, right = q.boundary_conditions
    left.data[0] = np.array([1 + p.mesh.intervals[0].start])
    left.data[1] = None  # Homogeneous natural boundary for component one.
    right.data[0] = np.array([1.0])
    right.data[1] = None
    for index in np.ndindex(right.reaction.field_shape):
        right.reaction[index] = None
    actual = dense_values(solve_dense_bvp_fd(p, q), (8,), 2)
    npt.assert_allclose(actual, expected, atol=1e-11)
    p, q = as_mps(p, q)
    solution = solve_mps_bvp_fd(p, q, strategy=STRATEGY, solver=ScipySolver())
    npt.assert_allclose(mps_values(solution, (8,), 2), expected, atol=2e-8)


def test_native_nodal_data_and_quantized_mesh() -> None:
    p, q, _ = affine_problem(dirichlet=False)
    mesh = Mesh([QuantizedInterval(-0.7, 1.3, 3, endpoint_right=True)])
    p = BulkProblem(mesh, p.principal, p.reaction, p.source)
    q = BoundaryProblem(mesh, q.boundary_conditions)
    expected = np.random.default_rng(18).normal(size=8)
    p.principal[0, 0, 0, 0] = None
    p.reaction[0, 0] = np.ones(8)
    p.source[0] = expected.copy()
    for bc in q.boundary_conditions:
        bc.reaction[0, 0] = None
        bc.data[0] = None
    with patch.object(
        TensorField, "chebyshev_transform", side_effect=AssertionError("CG transform")
    ):
        npt.assert_allclose(solve_dense_bvp_fd(p, q), expected, atol=1e-13)
        mp, mq = as_mps(p, q)
        actual = solve_mps_bvp_fd(mp, mq, strategy=STRATEGY, solver=ScipySolver())
    npt.assert_allclose(actual.to_vector(), expected, atol=1e-12)
