import numpy as np
import numpy.testing as npt
from unittest.mock import patch

from seemps.state import MPS, Simplification, Strategy, Truncation
from seemps.analysis.mesh import ChebyshevInterval, Mesh

from seemps_galerkin.dense.assembly import (
    assemble_dense_bulk,
    assemble_dense_robin_inplace,
    assemble_dirichlet_operators,
)
from seemps_galerkin.dense.factories import (
    op_bit_reversal,
    op_chebyshev_derivative,
    op_tensor_product,
    op_weighted_gram,
)
from seemps_galerkin.mps.assembly import (
    assemble_mps_bulk,
    assemble_mps_dirichlet_operators,
    assemble_mps_robin_correction,
)
from seemps_galerkin.mps.factories import mps_identity
from seemps_galerkin.problem import (
    BoundaryFace,
    BoundaryProblem,
    BulkProblem,
    DirichletBoundaryCondition,
    RobinBoundaryCondition,
    TensorField,
)


class TestMPSBulkAssembly:
    rtol = 0.0
    atol = 1.0e-9
    strategy = Strategy(
        method=Truncation.RELATIVE_SINGULAR_VALUE,
        tolerance=1.0e-12,
        simplification_tolerance=1.0e-12,
        max_bond_dimension=256,
        max_sweeps=8,
        simplify=Simplification.VARIATIONAL,
    )

    def test_coupled_variable_operator_and_load_match_dense(self):
        sites = [2, 2]
        intervals = [
            ChebyshevInterval(-2.0, 1.0, 4, endpoints=False),
            ChebyshevInterval(-1.0, 3.0, 4, endpoints=False),
        ]
        mesh = Mesh(intervals)
        x, y = [interval.to_vector() for interval in intervals]
        X, Y = np.meshgrid(x, y, indexing="ij")
        principal_dense = TensorField((2, 2, 2, 2), mesh.dimensions)
        reaction_dense = TensorField((2, 2), mesh.dimensions)
        source_dense = TensorField((2,), mesh.dimensions)
        principal_mps = TensorField((2, 2, 2, 2), mesh.dimensions)
        reaction_mps = TensorField((2, 2), mesh.dimensions)
        source_mps = TensorField((2,), mesh.dimensions)

        for μ in range(2):
            values = (0.4 + μ + 0.1 * X - 0.2 * Y).ravel()
            source_dense[μ] = values
            source_mps[μ] = MPS.from_vector(values, [2] * 4, normalize=False)
            for ν in range(2):
                if (μ, ν) != (1, 0):
                    values = (0.7 + 0.2 * μ - 0.1 * ν + 0.05 * X * Y).ravel()
                    reaction_dense[μ, ν] = values
                    reaction_mps[μ, ν] = MPS.from_vector(values, [2] * 4, normalize=False)
                for α in range(2):
                    for β in range(2):
                        if (α + β + μ + ν) % 3 == 0:
                            continue
                        values = (
                            1.0 + 0.1 * α + 0.2 * β + 0.1 * μ - 0.05 * ν + 0.03 * X + 0.04 * Y
                        ).ravel()
                        principal_dense[α, β, μ, ν] = values
                        principal_mps[α, β, μ, ν] = MPS.from_vector(
                            values, [2] * 4, normalize=False
                        )

        A_dense, b_dense = assemble_dense_bulk(
            BulkProblem(mesh, principal_dense, reaction_dense, source_dense).chebyshev_transform()
        )
        ρ_A, ψ_b = assemble_mps_bulk(
            BulkProblem(mesh, principal_mps, reaction_mps, source_mps).chebyshev_transform(
                strategy=self.strategy
            ),
            self.strategy,
        )

        P = op_tensor_product([op_bit_reversal(n) for n in sites])
        P_vector = np.kron(np.eye(2), P)
        npt.assert_allclose(
            P_vector @ ρ_A.to_matrix() @ P_vector.T, A_dense, rtol=self.rtol, atol=self.atol
        )
        npt.assert_allclose(P_vector @ ψ_b.to_vector(), b_dense, rtol=self.rtol, atol=self.atol)

    def test_constant_coefficients_use_direct_operators_and_match_dense(self):
        sites = [2, 2]
        N = 2 ** sum(sites)
        mesh = Mesh([ChebyshevInterval(-1.0, 1.0, 4, endpoints=False)] * 2)
        principal_dense = TensorField((2, 2, 1, 1), mesh.dimensions)
        reaction_dense = TensorField((1, 1), mesh.dimensions)
        principal_mps = TensorField((2, 2, 1, 1), mesh.dimensions)
        reaction_mps = TensorField((1, 1), mesh.dimensions)
        for α in range(2):
            principal_dense[α, α, 0, 0] = np.full(N, 1.0 + α)
            principal_mps[α, α, 0, 0] = (1.0 + α) * mps_identity(sum(sites))
        reaction_dense[0, 0] = np.full(N, 0.5)
        reaction_mps[0, 0] = 0.5 * mps_identity(sum(sites))

        A_dense, _ = assemble_dense_bulk(
            BulkProblem(
                mesh, principal_dense, reaction_dense, TensorField((1,), mesh.dimensions)
            ).chebyshev_transform()
        )
        bulk_mps = BulkProblem(
            mesh, principal_mps, reaction_mps, TensorField((1,), mesh.dimensions)
        ).chebyshev_transform(strategy=self.strategy)
        with patch(
            "seemps_galerkin.mps.assembly.mpo_weighted_gram",
            side_effect=AssertionError("constant coefficients used weighted assembly"),
        ):
            ρ_A, _ = assemble_mps_bulk(bulk_mps, self.strategy)

        P = op_tensor_product([op_bit_reversal(n) for n in sites])
        npt.assert_allclose(P @ ρ_A.to_matrix() @ P.T, A_dense, rtol=self.rtol, atol=self.atol)

    def test_high_degree_stiffness_is_symmetric_and_matches_dense(self):
        n = 2
        N = 2**n
        mesh = Mesh([ChebyshevInterval(-1.0, 1.0, N, endpoints=False)])
        q = np.array([2.0, 0.1, 0.2, 0.3])
        P = op_bit_reversal(n)
        principal_dense = TensorField((1, 1, 1, 1), mesh.dimensions)
        principal_dense[0, 0, 0, 0] = q
        principal_mps = TensorField((1, 1, 1, 1), mesh.dimensions)
        principal_mps[0, 0, 0, 0] = MPS.from_vector(P @ q, [2] * n, normalize=False)
        A_dense, _ = assemble_dense_bulk(
            BulkProblem(
                mesh,
                principal_dense,
                TensorField((1, 1), mesh.dimensions),
                TensorField((1,), mesh.dimensions),
            )
        )
        ρ_A, _ = assemble_mps_bulk(
            BulkProblem(
                mesh,
                principal_mps,
                TensorField((1, 1), mesh.dimensions),
                TensorField((1,), mesh.dimensions),
            ),
            self.strategy,
        )
        A = P @ ρ_A.to_matrix() @ P.T
        D = op_chebyshev_derivative(N)
        A_expected = D.T @ op_weighted_gram(q, [N]) @ D

        npt.assert_allclose(A_dense, A_expected, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(A, A_expected, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(A, A.T, rtol=self.rtol, atol=self.atol)


class TestMPSRobinAssembly:
    rtol = 0.0
    atol = 1.0e-10
    strategy = TestMPSBulkAssembly.strategy

    def test_coupled_endpoint_terms_from_both_faces_match_dense(self):
        n = 2
        N = 2**n
        mesh = Mesh([ChebyshevInterval(-2.0, 1.0, N, endpoints=False)])
        bcs_dense = []
        bcs_mps = []
        for is_left in (True, False):
            reaction_dense = TensorField((2, 2), ())
            data_dense = TensorField((2,), ())
            reaction_mps = TensorField((2, 2), ())
            data_mps = TensorField((2,), ())
            side = -1.0 if is_left else 1.0
            for μ in range(2):
                value = np.array([0.5 + 0.2 * μ + 0.1 * side])
                data_dense[μ] = value
                data_mps[μ] = MPS.from_vector(value, [1], normalize=False)
                for ν in range(2):
                    value = np.array([0.3 + 0.1 * μ - 0.05 * ν + 0.02 * side])
                    reaction_dense[μ, ν] = value
                    reaction_mps[μ, ν] = MPS.from_vector(value, [1], normalize=False)
            face = BoundaryFace(0, is_left)
            bcs_dense.append(RobinBoundaryCondition(face, reaction_dense, data_dense))
            bcs_mps.append(RobinBoundaryCondition(face, reaction_mps, data_mps))

        A_dense, b_dense = assemble_dense_robin_inplace(
            BoundaryProblem(mesh, bcs_dense).chebyshev_transform()
        )
        ρ_A, ψ_b = assemble_mps_robin_correction(
            BoundaryProblem(mesh, bcs_mps).chebyshev_transform(strategy=self.strategy),
            self.strategy,
        )

        P = op_bit_reversal(n)
        P_vector = np.kron(np.eye(2), P)
        npt.assert_allclose(
            P_vector @ ρ_A.to_matrix() @ P_vector.T, A_dense, rtol=self.rtol, atol=self.atol
        )
        npt.assert_allclose(P_vector @ ψ_b.to_vector(), b_dense, rtol=self.rtol, atol=self.atol)


class TestMPSSpectralLift:
    rtol = 0.0
    atol = 1.0e-9
    strategy = TestMPSBulkAssembly.strategy

    def test_three_dimensional_transfinite_lift_matches_dense(self):
        sites = [2, 2, 2]
        intervals = [
            ChebyshevInterval(-1.0, 1.0, 4, endpoints=False),
            ChebyshevInterval(-2.0, 1.0, 4, endpoints=False),
            ChebyshevInterval(0.0, 2.0, 4, endpoints=False),
        ]
        mesh = Mesh(intervals)  # type: ignore
        coordinates = [interval.to_vector() for interval in intervals]
        bcs_dense = []
        bcs_mps = []
        for dim in range(3):
            free = [axis for axis in range(3) if axis != dim]
            face_shape = tuple(mesh.dimensions[axis] for axis in free)
            X, Y = np.meshgrid(coordinates[free[0]], coordinates[free[1]], indexing="ij")
            for is_left in (True, False):
                point = intervals[dim].start if is_left else intervals[dim].stop
                xyz = [None, None, None]
                xyz[dim] = point
                xyz[free[0]] = X
                xyz[free[1]] = Y
                x, y, z = xyz
                values = (
                    1.0
                    + 0.2 * x
                    - 0.1 * y
                    + 0.3 * z
                    + 0.05 * x * y
                    - 0.04 * x * z
                    + 0.02 * y * z
                    + 0.01 * x * y * z
                ).ravel()
                data_dense = TensorField((1,), face_shape)
                data_mps = TensorField((1,), face_shape)
                data_dense[0] = values
                data_mps[0] = MPS.from_vector(values, [2] * 4, normalize=False)
                face = BoundaryFace(dim, is_left)
                bcs_dense.append(DirichletBoundaryCondition(face, data_dense))
                bcs_mps.append(DirichletBoundaryCondition(face, data_mps))

        Z, P_columns, lift = assemble_dirichlet_operators(
            BoundaryProblem(mesh, bcs_dense).chebyshev_transform()
        )
        ρ_Z, ρ_P, ψ_lift = assemble_mps_dirichlet_operators(
            BoundaryProblem(mesh, bcs_mps).chebyshev_transform(strategy=self.strategy),
            self.strategy,
        )

        P = op_tensor_product([op_bit_reversal(n) for n in sites])
        npt.assert_allclose(P @ ρ_Z.to_matrix() @ P.T, Z, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(P @ ρ_P.to_matrix() @ P.T, P_columns, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(P @ ψ_lift.to_vector(), lift, rtol=self.rtol, atol=self.atol)


class TestMPSBoundaryAssembly:
    rtol = 0.0
    atol = 1.0e-9

    def test_robin_terms_match_dense(self):
        sites = [2, 2]
        intervals = [
            ChebyshevInterval(-1.0, 1.0, 4, endpoints=False),
            ChebyshevInterval(-2.0, 1.0, 4, endpoints=False),
        ]
        mesh = Mesh(intervals)
        strategy = Strategy(
            method=Truncation.RELATIVE_SINGULAR_VALUE,
            tolerance=1.0e-12,
            simplification_tolerance=1.0e-12,
            max_bond_dimension=256,
            max_sweeps=8,
            simplify=Simplification.VARIATIONAL,
        )
        bcs_dense = []
        bcs_mps = []
        for dim in range(2):
            x_face = intervals[1 - dim].to_vector()
            face_shape = mesh.dimensions[:dim] + mesh.dimensions[dim + 1 :]
            for is_left in (True, False):
                R_dense = TensorField((1, 1), face_shape)
                f_dense = TensorField((1,), face_shape)
                R_mps = TensorField((1, 1), face_shape)
                f_mps = TensorField((1,), face_shape)
                if dim == 0 and is_left:
                    R_dense[0, 0] = 0.5 + 0.1 * x_face
                    f_dense[0] = 1.0 + 0.2 * x_face
                    R_mps[0, 0] = MPS.from_vector(R_dense[0, 0], [2] * sites[1], normalize=False)
                    f_mps[0] = MPS.from_vector(f_dense[0], [2] * sites[1], normalize=False)
                face = BoundaryFace(dim, is_left)
                bcs_dense.append(RobinBoundaryCondition(face, R_dense, f_dense))
                bcs_mps.append(RobinBoundaryCondition(face, R_mps, f_mps))

        A_dense, b_dense = assemble_dense_robin_inplace(
            BoundaryProblem(mesh, bcs_dense).chebyshev_transform()
        )
        ρ_A, ψ_b = assemble_mps_robin_correction(
            BoundaryProblem(mesh, bcs_mps).chebyshev_transform(strategy=strategy),
            strategy,
        )

        P = op_tensor_product([op_bit_reversal(n) for n in sites])
        npt.assert_allclose(P @ ρ_A.to_matrix() @ P.T, A_dense, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(P @ ψ_b.to_vector(), b_dense, rtol=self.rtol, atol=self.atol)

    def test_dirichlet_operators_and_lift_match_dense(self):
        sites = [2, 2]
        intervals = [
            ChebyshevInterval(-1.0, 1.0, 4, endpoints=False),
            ChebyshevInterval(-2.0, 1.0, 4, endpoints=False),
        ]
        mesh = Mesh(intervals)
        strategy = Strategy(
            method=Truncation.RELATIVE_SINGULAR_VALUE,
            tolerance=1.0e-12,
            simplification_tolerance=1.0e-12,
            max_bond_dimension=256,
            max_sweeps=8,
            simplify=Simplification.VARIATIONAL,
        )

        def u(x, y):
            return 1.0 + 0.2 * x + 0.3 * y + 0.1 * x * y

        bcs_dense = []
        bcs_mps = []
        for dim in range(2):
            x_face = intervals[1 - dim].to_vector()
            face_shape = mesh.dimensions[:dim] + mesh.dimensions[dim + 1 :]
            for is_left in (True, False):
                value = intervals[dim].start if is_left else intervals[dim].stop
                values = u(value, x_face) if dim == 0 else u(x_face, value)
                data_dense = TensorField((1,), face_shape)
                data_mps = TensorField((1,), face_shape)
                data_dense[0] = values
                data_mps[0] = MPS.from_vector(values, [2] * sites[1 - dim], normalize=False)
                face = BoundaryFace(dim, is_left)
                bcs_dense.append(DirichletBoundaryCondition(face, data_dense))
                bcs_mps.append(DirichletBoundaryCondition(face, data_mps))

        Z, P_dirichlet, lift = assemble_dirichlet_operators(
            BoundaryProblem(mesh, bcs_dense).chebyshev_transform()
        )
        ρ_Z, ρ_P, ψ_lift = assemble_mps_dirichlet_operators(
            BoundaryProblem(mesh, bcs_mps).chebyshev_transform(strategy=strategy), strategy
        )

        P_bits = op_tensor_product([op_bit_reversal(n) for n in sites])
        npt.assert_allclose(P_bits @ ρ_Z.to_matrix() @ P_bits.T, Z, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(
            P_bits @ ρ_P.to_matrix() @ P_bits.T, P_dirichlet, rtol=self.rtol, atol=self.atol
        )
        npt.assert_allclose(P_bits @ ψ_lift.to_vector(), lift, rtol=self.rtol, atol=self.atol)

    def test_componentwise_dirichlet_operators_and_lift_match_dense(self):
        n = 2
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
        data_left_dense = TensorField((3,), ())
        data_right_dense = TensorField((3,), ())
        data_left_mps = TensorField((3,), ())
        data_right_mps = TensorField((3,), ())
        value_left = np.array([1.25])
        value_right = np.array([-0.75])
        data_left_dense[0] = value_left
        data_right_dense[1] = value_right
        data_left_mps[0] = MPS.from_vector(value_left, [1], normalize=False)
        data_right_mps[1] = MPS.from_vector(value_right, [1], normalize=False)
        bcs_dense = [
            DirichletBoundaryCondition(BoundaryFace(0, True), data_left_dense),
            DirichletBoundaryCondition(BoundaryFace(0, False), data_right_dense),
        ]
        bcs_mps = [
            DirichletBoundaryCondition(BoundaryFace(0, True), data_left_mps),
            DirichletBoundaryCondition(BoundaryFace(0, False), data_right_mps),
        ]

        Z, P, lift = assemble_dirichlet_operators(
            BoundaryProblem(mesh, bcs_dense).chebyshev_transform()
        )
        ρ_Z, ρ_P, ψ_lift = assemble_mps_dirichlet_operators(
            BoundaryProblem(mesh, bcs_mps).chebyshev_transform(strategy=strategy), strategy
        )

        P_bits = np.kron(np.eye(3), op_bit_reversal(n))
        npt.assert_allclose(P_bits @ ρ_Z.to_matrix() @ P_bits.T, Z, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(P_bits @ ρ_P.to_matrix() @ P_bits.T, P, rtol=self.rtol, atol=self.atol)
        npt.assert_allclose(P_bits @ ψ_lift.to_vector(), lift, rtol=self.rtol, atol=self.atol)
