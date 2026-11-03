import numpy as np
import numpy.testing as npt

from seemps.analysis.mesh import ChebyshevInterval, Mesh

from seemps_galerkin.dense.factories import vec_nd_chebyshev_transform
from seemps_galerkin.solve import solve_dense_bvp
from seemps_galerkin.problem import (
    BoundaryFace,
    BoundaryProblem,
    BulkProblem,
    DirichletBoundaryCondition,
    RobinBoundaryCondition,
    TensorField,
)


class TestDenseSolve:
    rtol = 0.0
    atol = 1.0e-10
    N = 8
    interval = ChebyshevInterval(-1.0, 1.0, N, endpoints=False)
    mesh = Mesh([interval])
    x = interval.to_vector()

    @classmethod
    def bulk_fields(cls, m=1):
        return (
            TensorField((1, 1, m, m), cls.mesh.dimensions),
            TensorField((m, m), cls.mesh.dimensions),
            TensorField((m,), cls.mesh.dimensions),
        )

    @classmethod
    def robin_boundary(cls, m=1, mesh=None):
        mesh = cls.mesh if mesh is None else mesh
        conditions = [
            RobinBoundaryCondition(
                BoundaryFace(0, is_left), TensorField((m, m), ()), TensorField((m,), ())
            )
            for is_left in (True, False)
        ]
        return BoundaryProblem(mesh, conditions)

    @classmethod
    def to_values(cls, coefficients, m=1):
        return np.vstack(
            [
                vec_nd_chebyshev_transform(row, [cls.N], inverse=True)
                for row in coefficients.reshape(m, cls.N)
            ]
        )

    def test_reaction_problem_recovers_constant_field(self):
        principal, reaction, source = self.bulk_fields()
        reaction[0, 0] = np.ones(self.N)
        source[0] = np.ones(self.N)

        coefficients = solve_dense_bvp(
            BulkProblem(self.mesh, principal, reaction, source), self.robin_boundary()
        )
        npt.assert_allclose(
            self.to_values(coefficients)[0], np.ones(self.N), rtol=self.rtol, atol=self.atol
        )

    def test_variable_dirichlet_problem_recovers_polynomial_field(self):
        u = 1.0 + 0.5 * self.x + 0.2 * self.x**2
        principal, reaction, source = self.bulk_fields()
        principal[0, 0, 0, 0] = 1.2 + 0.2 * self.x
        source[0] = -0.58 - 0.16 * self.x
        conditions = []
        for is_left in (True, False):
            point = self.interval.start if is_left else self.interval.stop
            data = TensorField((1,), ())
            data[0] = np.array([1.0 + 0.5 * point + 0.2 * point**2])
            conditions.append(DirichletBoundaryCondition(BoundaryFace(0, is_left), data))

        coefficients = solve_dense_bvp(
            BulkProblem(self.mesh, principal, reaction, source),
            BoundaryProblem(self.mesh, conditions),
        )
        npt.assert_allclose(self.to_values(coefficients)[0], u, rtol=self.rtol, atol=self.atol)

    def test_mixed_dirichlet_robin_problem_recovers_polynomial_field(self):
        u = 1.0 + 0.3 * self.x + 0.2 * self.x**2
        stiffness, bulk_reaction, boundary_reaction = 1.5, 0.8, 0.4
        principal, reaction, source = self.bulk_fields()
        principal[0, 0, 0, 0] = np.full(self.N, stiffness)
        reaction[0, 0] = np.full(self.N, bulk_reaction)
        source[0] = -0.4 * stiffness + bulk_reaction * u

        left = TensorField((1,), ())
        left[0] = np.array([1.0 - 0.3 + 0.2])
        right_reaction, right_data = TensorField((1, 1), ()), TensorField((1,), ())
        right_reaction[0, 0] = np.array([boundary_reaction])
        right_data[0] = np.array([stiffness * (0.3 + 0.4) + boundary_reaction * (1.0 + 0.3 + 0.2)])
        boundary = BoundaryProblem(
            self.mesh,
            [
                DirichletBoundaryCondition(BoundaryFace(0, True), left),
                RobinBoundaryCondition(BoundaryFace(0, False), right_reaction, right_data),
            ],
        )

        coefficients = solve_dense_bvp(
            BulkProblem(self.mesh, principal, reaction, source), boundary
        )
        npt.assert_allclose(self.to_values(coefficients)[0], u, rtol=self.rtol, atol=self.atol)

    def test_coupled_reaction_problem_recovers_both_components(self):
        m = 2
        expected = np.vstack((1.0 + 0.2 * self.x, -0.4 + 0.1 * self.x**2))
        coupling = np.array([[2.0, 0.3], [0.3, 1.4]])
        principal, reaction, source = self.bulk_fields(m)
        values = coupling @ expected
        for μ in range(m):
            source[μ] = values[μ]
            for ν in range(m):
                reaction[μ, ν] = np.full(self.N, coupling[μ, ν])

        coefficients = solve_dense_bvp(
            BulkProblem(self.mesh, principal, reaction, source), self.robin_boundary(m)
        )
        npt.assert_allclose(
            self.to_values(coefficients, m), expected, rtol=self.rtol, atol=self.atol
        )
