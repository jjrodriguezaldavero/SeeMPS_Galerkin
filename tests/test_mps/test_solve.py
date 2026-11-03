import numpy as np
import numpy.testing as npt
import pytest

from seemps.state import MPS, Simplification, Strategy, Truncation
from seemps.analysis.mesh import ChebyshevInterval, Mesh

from seemps_galerkin.mps.factories.transform import mps_chebyshev_transform
from seemps_galerkin.mps.factories import mpo_identity
from seemps_galerkin.mps.linear import (
    AMEnSolver,
    DMRGSolver,
    ScipySolver,
)
from seemps_galerkin.problem import (
    BoundaryFace,
    BoundaryProblem,
    BulkProblem,
    DirichletBoundaryCondition,
    RobinBoundaryCondition,
    TensorField,
)
from seemps_galerkin.logger import Logger as Log
from seemps_galerkin.solve import solve_mps_bvp


class TestMPSCoupledSolve:
    rtol = 0.0
    atol = 1.0e-7
    strategy = Strategy(
        method=Truncation.RELATIVE_SINGULAR_VALUE,
        tolerance=1.0e-12,
        simplification_tolerance=1.0e-12,
        max_bond_dimension=256,
        max_sweeps=8,
        simplify=Simplification.VARIATIONAL,
    )

    @pytest.mark.parametrize(
        "solver",
        [ScipySolver(), AMEnSolver(rtol=1e-11), AMEnSolver(rtol=1e-11, preconditioner="jacobi")],
    )
    def test_mixed_robin_dirichlet_vector_problem_recovers_prescribed_fields(self, solver):
        n = 3
        N = 2**n
        interval = ChebyshevInterval(-1.0, 1.0, N, endpoints=False)
        mesh = Mesh([interval])
        x = interval.to_vector()
        A_values = np.array([[2.0, 0.3], [0.3, 1.5]])
        R_values = np.array([[1.0, 0.2], [0.2, 0.8]])
        u = np.vstack([np.sin(1.1 * x) + 0.2 * x, np.cos(0.7 * x) - 0.1 * x])
        d2u = np.vstack([-(1.1**2) * np.sin(1.1 * x), -(0.7**2) * np.cos(0.7 * x)])
        source = -A_values @ d2u + R_values @ u
        principal = TensorField((1, 1, 2, 2), mesh.dimensions)
        reaction = TensorField((2, 2), mesh.dimensions)
        source_field = TensorField((2,), mesh.dimensions)
        for μ in range(2):
            source_field[μ] = MPS.from_vector(source[μ], [2] * n, normalize=False)
            for ν in range(2):
                principal[0, 0, μ, ν] = MPS.from_vector(
                    np.full(N, A_values[μ, ν]), [2] * n, normalize=False
                )
                reaction[μ, ν] = MPS.from_vector(
                    np.full(N, R_values[μ, ν]), [2] * n, normalize=False
                )

        x_boundary = np.array([interval.start, interval.stop])
        u_boundary = np.vstack(
            [
                np.sin(1.1 * x_boundary) + 0.2 * x_boundary,
                np.cos(0.7 * x_boundary) - 0.1 * x_boundary,
            ]
        )
        du_boundary = np.vstack(
            [1.1 * np.cos(1.1 * x_boundary) + 0.2, -0.7 * np.sin(0.7 * x_boundary) - 0.1]
        )
        data_left = TensorField((2,), ())
        data_right = TensorField((2,), ())
        flux_right = A_values @ du_boundary[:, 1]
        for μ in range(2):
            data_left[μ] = MPS.from_vector(np.array([u_boundary[μ, 0]]), [1], normalize=False)
            data_right[μ] = MPS.from_vector(np.array([flux_right[μ]]), [1], normalize=False)
        boundary = BoundaryProblem(
            mesh,
            [
                DirichletBoundaryCondition(BoundaryFace(0, True), data_left),
                RobinBoundaryCondition(BoundaryFace(0, False), TensorField((2, 2), ()), data_right),
            ],
        )
        ψ = solve_mps_bvp(
            BulkProblem(mesh, principal, reaction, source_field),
            boundary,
            strategy=self.strategy,
            solver=solver,
        )
        ψ = mps_chebyshev_transform(ψ, [1, n], dimensions=[1], inverse=True, strategy=self.strategy)
        npt.assert_allclose(ψ.to_vector().reshape(2, N), u, rtol=self.rtol, atol=self.atol)

    def test_two_dimensional_elasticity_recovers_prescribed_displacement(self):
        n = 2
        N = 2**n
        intervals = [
            ChebyshevInterval(-1.0, 1.0, N, endpoints=False),
            ChebyshevInterval(-2.0, 1.0, N, endpoints=False),
        ]
        mesh = Mesh(intervals)  # type: ignore
        x, y = [interval.to_vector() for interval in intervals]
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = np.stack(
            [
                1.0 + 0.2 * X + 0.1 * Y + 0.15 * X**2 + 0.05 * X * Y + 0.07 * Y**2,
                -0.4 + 0.1 * X - 0.3 * Y + 0.04 * X**2 - 0.08 * X * Y + 0.11 * Y**2,
            ]
        )
        H = np.array([[[0.30, 0.05], [0.05, 0.14]], [[0.08, -0.08], [-0.08, 0.22]]])
        λ = 1.3
        μ_shear = 0.8
        C = np.zeros((2, 2, 2, 2))
        for α in range(2):
            for β in range(2):
                for μ in range(2):
                    for ν in range(2):
                        C[α, β, μ, ν] = (
                            λ * (α == μ) * (β == ν)
                            + μ_shear * (α == β) * (μ == ν)
                            + μ_shear * (α == ν) * (β == μ)
                        )
        source = -np.einsum("abij,jab->i", C, H)
        principal = TensorField((2, 2, 2, 2), mesh.dimensions)
        reaction = TensorField((2, 2), mesh.dimensions)
        source_field = TensorField((2,), mesh.dimensions)
        for α in range(2):
            for β in range(2):
                for μ in range(2):
                    for ν in range(2):
                        if C[α, β, μ, ν] != 0.0:
                            principal[α, β, μ, ν] = MPS.from_vector(
                                np.full(N * N, C[α, β, μ, ν]), [2] * (2 * n), normalize=False
                            )
        for μ in range(2):
            source_field[μ] = MPS.from_vector(
                np.full(N * N, source[μ]), [2] * (2 * n), normalize=False
            )

        bcs = []
        for dim in range(2):
            face_shape = mesh.dimensions[:dim] + mesh.dimensions[dim + 1 :]
            for is_left in (True, False):
                value = intervals[dim].start if is_left else intervals[dim].stop
                data = TensorField((2,), face_shape)
                if dim == 0:
                    values = [
                        1.0
                        + 0.2 * value
                        + 0.1 * y
                        + 0.15 * value**2
                        + 0.05 * value * y
                        + 0.07 * y**2,
                        -0.4
                        + 0.1 * value
                        - 0.3 * y
                        + 0.04 * value**2
                        - 0.08 * value * y
                        + 0.11 * y**2,
                    ]
                else:
                    values = [
                        1.0
                        + 0.2 * x
                        + 0.1 * value
                        + 0.15 * x**2
                        + 0.05 * x * value
                        + 0.07 * value**2,
                        -0.4
                        + 0.1 * x
                        - 0.3 * value
                        + 0.04 * x**2
                        - 0.08 * x * value
                        + 0.11 * value**2,
                    ]
                for μ in range(2):
                    data[μ] = MPS.from_vector(values[μ], [2] * n, normalize=False)
                bcs.append(DirichletBoundaryCondition(BoundaryFace(dim, is_left), data))
        ψ = solve_mps_bvp(
            BulkProblem(mesh, principal, reaction, source_field),
            BoundaryProblem(mesh, bcs),
            strategy=self.strategy,
            solver=ScipySolver(),
        )
        ψ = mps_chebyshev_transform(
            ψ, [1, n, n], dimensions=[1, 2], inverse=True, strategy=self.strategy
        )
        npt.assert_allclose(ψ.to_vector().reshape(2, N, N), u, rtol=self.rtol, atol=self.atol)


class TestMPSLinearSolvers:
    rtol = 0.0
    atol = 1.0e-11

    def test_scipy_and_dmrg_solvers_recover_identity_solution(self):
        ψ_b = MPS.from_vector(np.arange(1.0, 9.0), [2, 2, 2], normalize=False)
        ρ_A = mpo_identity(3)
        solvers = [
            ScipySolver(verbose=True),
            DMRGSolver(maxiter=4, rtol=1.0e-12),
            AMEnSolver(rtol=1e-12),
        ]
        Log.activate(level=0, collect=True)
        try:
            for solver in solvers:
                ψ = solver.solve(ρ_A, ψ_b, None)
                npt.assert_allclose(ψ.to_vector(), ψ_b.to_vector(), rtol=self.rtol, atol=self.atol)
            condition_numbers = Log.sink["solve.scipy.condition_number"]
            bond_dimensions = Log.sink["solve.scipy.u.bond_dimensions"][0]
        finally:
            Log.deactivate()
            Log.clear()

        npt.assert_allclose(condition_numbers, [1.0], rtol=self.rtol, atol=self.atol)
        npt.assert_array_equal(bond_dimensions, ψ_b.bond_dimensions())


class TestMPSSolve:
    rtol = 0.0
    atol = 1.0e-10

    def test_reaction_problem(self):
        n = 3
        N = 2**n
        mesh = Mesh([ChebyshevInterval(-1.0, 1.0, N, endpoints=False)])
        strategy = Strategy(
            method=Truncation.RELATIVE_SINGULAR_VALUE,
            tolerance=1.0e-12,
            simplification_tolerance=1.0e-12,
            max_bond_dimension=256,
            max_sweeps=8,
            simplify=Simplification.VARIATIONAL,
        )
        P = TensorField((1, 1, 1, 1), mesh.dimensions)
        R = TensorField((1, 1), mesh.dimensions)
        f = TensorField((1,), mesh.dimensions)
        R[0, 0] = MPS.from_vector(np.ones(N), [2] * n, normalize=False)
        f[0] = MPS.from_vector(np.ones(N), [2] * n, normalize=False)
        boundary = BoundaryProblem(
            mesh,
            [
                RobinBoundaryCondition(
                    BoundaryFace(0, is_left), TensorField((1, 1), ()), TensorField((1,), ())
                )
                for is_left in (True, False)
            ],
        )

        ψ = solve_mps_bvp(
            BulkProblem(mesh, P, R, f),
            boundary,
            strategy=strategy,
            solver=ScipySolver(),
        )
        ψ = mps_chebyshev_transform(ψ, [n], inverse=True, strategy=strategy)
        npt.assert_allclose(ψ.to_vector(), np.ones(N), rtol=self.rtol, atol=self.atol)

    def test_dirichlet_problem(self):
        n = 3
        N = 2**n
        interval = ChebyshevInterval(-1.0, 1.0, N, endpoints=False)
        mesh = Mesh([interval])
        x = interval.to_vector()
        strategy = Strategy(
            method=Truncation.RELATIVE_SINGULAR_VALUE,
            tolerance=1.0e-12,
            simplification_tolerance=1.0e-12,
            max_bond_dimension=256,
            max_sweeps=8,
            simplify=Simplification.VARIATIONAL,
        )
        u = 1.0 + 0.5 * x + 0.2 * x**2
        material_scale = 1.0e8
        principal = material_scale * (1.2 + 0.2 * x)
        source = material_scale * (-0.58 - 0.16 * x)
        P = TensorField((1, 1, 1, 1), mesh.dimensions)
        R = TensorField((1, 1), mesh.dimensions)
        f = TensorField((1,), mesh.dimensions)
        P[0, 0, 0, 0] = MPS.from_vector(principal, [2] * n, normalize=False)
        f[0] = MPS.from_vector(source, [2] * n, normalize=False)

        bcs = []
        for is_left in (True, False):
            value = interval.start if is_left else interval.stop
            data = TensorField((1,), ())
            data[0] = MPS.from_vector(
                np.array([1.0 + 0.5 * value + 0.2 * value**2]), [1], normalize=False
            )
            bcs.append(DirichletBoundaryCondition(BoundaryFace(0, is_left), data))

        Log.activate(level=0, collect=True)
        try:
            ψ = solve_mps_bvp(
                BulkProblem(mesh, P, R, f),
                BoundaryProblem(mesh, bcs),
                strategy=strategy,
                solver=ScipySolver(),
            )
            projector_scale = Log.sink["solve.mps.projector_scale"][-1]
        finally:
            Log.deactivate()
            Log.clear()
        ψ = mps_chebyshev_transform(ψ, [n], inverse=True, strategy=strategy)
        assert projector_scale > 1.0
        npt.assert_allclose(ψ.to_vector(), u, rtol=self.rtol, atol=self.atol)
