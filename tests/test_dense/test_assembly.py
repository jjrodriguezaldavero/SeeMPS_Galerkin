import numpy as np
import numpy.testing as npt

from seemps.analysis.mesh import ChebyshevInterval, Mesh

from seemps_galerkin.dense.assembly import (
    assemble_dense_bulk,
    assemble_dense_robin_inplace,
    assemble_dirichlet_operators,
)
from seemps_galerkin.dense.factories import (
    op_chebyshev_derivative,
    op_chebyshev_recombination,
    op_column_projector,
    op_face_trace,
    op_gram,
    op_nd_chebyshev_derivative,
    op_nd_gram,
    op_nd_mult,
    op_tensor_product,
    op_weighted_gram,
    vec_nd_chebyshev_transform,
)
from seemps_galerkin.problem import (
    BoundaryFace,
    BoundaryProblem,
    BulkProblem,
    DirichletBoundaryCondition,
    RobinBoundaryCondition,
    TensorField,
)


class TestDenseBulkAssembly:
    rtol = 0.0
    atol = 1.0e-12

    def test_variable_scalar_operator_matches_direct_galerkin_product(self):
        N = 8
        interval = ChebyshevInterval(-2.0, 1.0, N, endpoints=False)
        mesh = Mesh([interval])
        x = interval.to_vector()
        principal_values = 1.2 + 0.1 * x
        reaction_values = 0.7 - 0.2 * x
        source_values = 0.4 + 0.3 * x
        principal = TensorField((1, 1, 1, 1), mesh.dimensions)
        reaction = TensorField((1, 1), mesh.dimensions)
        source = TensorField((1,), mesh.dimensions)
        principal[0, 0, 0, 0] = principal_values
        reaction[0, 0] = reaction_values
        source[0] = source_values

        A, b = assemble_dense_bulk(
            BulkProblem(mesh, principal, reaction, source).chebyshev_transform()
        )
        principal_coefficients = vec_nd_chebyshev_transform(principal_values, [N])
        reaction_coefficients = vec_nd_chebyshev_transform(reaction_values, [N])
        source_coefficients = vec_nd_chebyshev_transform(source_values, [N])
        G = op_nd_gram([N])
        D = op_nd_chebyshev_derivative([N], 0)
        scale = 2.0 / (interval.stop - interval.start)
        J = 1.0 / scale
        A_expected = J * (
            scale**2 * D.T @ G @ op_nd_mult(principal_coefficients, [N]) @ D
            + op_weighted_gram(reaction_coefficients, [N])
        )
        b_expected = J * G @ source_coefficients
        npt.assert_allclose(A, A_expected, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(b, b_expected, rtol=self.rtol, atol=self.atol)

    def test_constant_multidimensional_operator_matches_kronecker_form(self):
        shape = (4, 3)
        intervals = [
            ChebyshevInterval(-2.0, 1.0, shape[0], endpoints=False),
            ChebyshevInterval(-1.0, 3.0, shape[1], endpoints=False),
        ]
        mesh = Mesh(intervals)
        N = int(np.prod(shape))
        C = np.array([[1.2, 0.3], [0.3, 0.8]])
        principal = TensorField((2, 2, 1, 1), mesh.dimensions)
        for α in range(2):
            for β in range(2):
                principal[α, β, 0, 0] = C[α, β] * np.ones(N)
        reaction = TensorField((1, 1), mesh.dimensions)
        reaction[0, 0] = 0.4 * np.ones(N)
        source = TensorField((1,), mesh.dimensions)
        source[0] = 0.7 * np.ones(N)

        A, b = assemble_dense_bulk(
            BulkProblem(mesh, principal, reaction, source).chebyshev_transform()
        )

        grams = [op_gram(N_k) for N_k in shape]
        derivatives = [op_chebyshev_derivative(N_k) for N_k in shape]
        scales = [2.0 / (interval.stop - interval.start) for interval in intervals]
        J = 1.0 / np.prod(scales)
        G = op_tensor_product(grams)
        A_expected = 0.4 * G
        for α in range(2):
            for β in range(2):
                factors = []
                for dim, (G_k, D_k) in enumerate(zip(grams, derivatives)):
                    left = D_k.T if dim == α else np.eye(G_k.shape[0])
                    right = D_k if dim == β else np.eye(G_k.shape[0])
                    factors.append(left @ G_k @ right)
                A_expected += scales[α] * scales[β] * C[α, β] * op_tensor_product(factors)
        source_coefficients = vec_nd_chebyshev_transform(source[0], shape)
        b_expected = J * G @ source_coefficients

        npt.assert_allclose(A, J * A_expected, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(b, b_expected, rtol=self.rtol, atol=self.atol)


class TestDenseBoundaryAssembly:
    rtol = 0.0
    atol = 1.0e-12

    def test_endpoint_robin_terms_match_trace_products(self):
        N = 4
        mesh = Mesh([ChebyshevInterval(-1.0, 1.0, N, endpoints=False)])
        bcs = []
        for is_left, reaction_value, data_value in ((True, 0.4, -0.2), (False, 0.7, 0.3)):
            reaction = TensorField((1, 1), ())
            data = TensorField((1,), ())
            reaction[0, 0] = np.array([reaction_value])
            data[0] = np.array([data_value])
            bcs.append(RobinBoundaryCondition(BoundaryFace(0, is_left), reaction, data))

        A, b = assemble_dense_robin_inplace(BoundaryProblem(mesh, bcs).chebyshev_transform())
        A_expected = np.zeros((N, N))
        b_expected = np.zeros(N)
        for is_left, reaction_value, data_value in ((True, 0.4, -0.2), (False, 0.7, 0.3)):
            T = op_face_trace([N], 0, is_left)
            A_expected += reaction_value * T.T @ T
            b_expected += data_value * T.ravel()
        npt.assert_allclose(A, A_expected, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(b, b_expected, rtol=self.rtol, atol=self.atol)

    def test_constant_multidimensional_robin_term_matches_kronecker_form(self):
        shape = (4, 3)
        intervals = [
            ChebyshevInterval(-2.0, 1.0, shape[0], endpoints=False),
            ChebyshevInterval(-1.0, 3.0, shape[1], endpoints=False),
        ]
        mesh = Mesh(intervals)
        bcs = []
        for dim in range(2):
            face_shape = mesh.dimensions[:dim] + mesh.dimensions[dim + 1 :]
            for is_left in (True, False):
                reaction = TensorField((1, 1), face_shape)
                data = TensorField((1,), face_shape)
                if dim == 0 and is_left:
                    reaction[0, 0] = 1.7 * np.ones(shape[1])
                bcs.append(RobinBoundaryCondition(BoundaryFace(dim, is_left), reaction, data))

        A, b = assemble_dense_robin_inplace(BoundaryProblem(mesh, bcs).chebyshev_transform())

        trace = np.outer(np.array([1.0, -1.0, 1.0, -1.0]), np.array([1.0, -1.0, 1.0, -1.0]))
        J_face = (intervals[1].stop - intervals[1].start) / 2.0
        A_expected = 1.7 * J_face * op_tensor_product([trace, op_gram(shape[1])])
        npt.assert_allclose(A, A_expected, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(b, np.zeros(np.prod(shape)), rtol=self.rtol, atol=self.atol)

    def test_lift_reproduces_all_prescribed_face_values(self):
        shape = [4, 4]
        intervals = [
            ChebyshevInterval(-1.0, 1.0, shape[0], endpoints=False),
            ChebyshevInterval(-2.0, 1.0, shape[1], endpoints=False),
        ]
        mesh = Mesh(intervals)
        bcs = []
        for dim in range(2):
            x_face = intervals[1 - dim].to_vector()
            face_shape = mesh.dimensions[:dim] + mesh.dimensions[dim + 1 :]
            for is_left in (True, False):
                value = intervals[dim].start if is_left else intervals[dim].stop
                values = (
                    1.0 + value + 0.2 * x_face + 0.1 * value * x_face
                    if dim == 0
                    else 1.0 + x_face + 0.2 * value + 0.1 * value * x_face
                )
                data = TensorField((1,), face_shape)
                data[0] = values
                bcs.append(DirichletBoundaryCondition(BoundaryFace(dim, is_left), data))
        boundary = BoundaryProblem(mesh, bcs).chebyshev_transform()

        _, _, lift = assemble_dirichlet_operators(boundary)
        for bc in boundary.boundary_conditions:
            T = op_face_trace(shape, bc.face.dim, bc.face.is_left)
            npt.assert_allclose(T @ lift, bc.data[0], rtol=self.rtol, atol=self.atol)

    def test_componentwise_dirichlet_traces_build_distinct_component_operators(self):
        N = 4
        mesh = Mesh([ChebyshevInterval(-1.0, 1.0, N, endpoints=False)])
        data_left = TensorField((2,), ())
        data_right = TensorField((2,), ())
        value_left = np.array([1.25])
        value_right = np.array([-0.75])
        data_left[0] = value_left
        data_right[1] = value_right
        boundary = BoundaryProblem(
            mesh,
            [
                DirichletBoundaryCondition(BoundaryFace(0, True), data_left),
                DirichletBoundaryCondition(BoundaryFace(0, False), data_right),
            ],
        ).chebyshev_transform()

        Z, P, lift = assemble_dirichlet_operators(boundary)
        Z_expected = np.zeros((2 * N, 2 * N))
        P_expected = np.zeros((2 * N, 2 * N))
        for μ, sides in enumerate(((True, False), (False, True))):
            rows = slice(μ * N, (μ + 1) * N)
            Z_expected[rows, rows] = op_chebyshev_recombination(N, *sides)
            P_expected[rows, rows] = np.eye(N) - op_column_projector(N, *sides)

        npt.assert_allclose(Z, Z_expected, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(P, P_expected, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(
            op_face_trace([N], 0, True) @ lift[:N], value_left, rtol=self.rtol, atol=self.atol
        )
        npt.assert_allclose(
            op_face_trace([N], 0, False) @ lift[N:], value_right, rtol=self.rtol, atol=self.atol
        )
