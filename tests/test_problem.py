import numpy as np
import numpy.testing as npt
import pytest
from typing import ClassVar

from seemps.analysis.mesh import ChebyshevInterval, Interval, Mesh
from seemps.state import MPS, Simplification, Strategy, Truncation

from seemps_galerkin.dense.factories import op_bit_reversal, vec_nd_chebyshev_transform
from seemps_galerkin.problem import (
    BoundaryFace,
    BoundaryProblem,
    BulkProblem,
    DirichletBoundaryCondition,
    RobinBoundaryCondition,
    TensorField,
    field_to_mps,
)


class TestTensorField:
    shape = (4, 2)
    size = int(np.prod(shape))
    values = np.random.default_rng(0).normal(size=size)
    strategy = Strategy(
        method=Truncation.RELATIVE_SINGULAR_VALUE,
        tolerance=1.0e-13,
        simplification_tolerance=1.0e-13,
        max_bond_dimension=64,
        max_sweeps=4,
        simplify=Simplification.VARIATIONAL,
    )

    def test_metadata_assignment_and_flat_iteration(self):
        field = TensorField((2, 2), self.shape)
        ψ = MPS.from_vector(self.values, [2] * 3, normalize=False)
        field[0, 1], field[1, 0] = self.values, ψ

        assert field.field_shape == (2, 2)
        assert field.field_ndim == field.mesh_ndim == 2
        assert field.mesh_shape == self.shape
        assert field.mesh_size == self.size
        assert field.is_quantizable
        entries = list(field.flat)
        assert entries[0] is entries[3] is None
        assert entries[1] is self.values
        assert entries[2] is ψ

    @pytest.mark.parametrize("as_mps", [False, True], ids=["dense", "mps"])
    def test_transform_matches_dense_and_round_trips(self, as_mps):
        entry = MPS.from_vector(self.values, [2] * 3, normalize=False) if as_mps else self.values
        field = TensorField((2,), self.shape)
        field[0] = entry

        coefficients = field.chebyshev_transform(inverse=False, strategy=self.strategy)
        transformed = coefficients[0]
        assert coefficients.field_shape == field.field_shape
        assert coefficients.mesh_shape == field.mesh_shape
        assert coefficients[1] is None

        expected = vec_nd_chebyshev_transform(self.values, self.shape)
        if isinstance(transformed, MPS):
            reversal = np.kron(op_bit_reversal(2), op_bit_reversal(1))
            npt.assert_allclose(reversal @ transformed.to_vector(), expected, atol=1.0e-11)
        else:
            npt.assert_allclose(transformed, expected, atol=1.0e-12)

        restored = coefficients.chebyshev_transform(inverse=True, strategy=self.strategy)[0]
        actual = restored.to_vector() if isinstance(restored, MPS) else restored
        npt.assert_allclose(actual, self.values, atol=1.0e-11)

    def test_scalar_mesh_mps_is_preserved(self):
        ψ = MPS.from_vector(np.array([2.5]), [1], normalize=False)
        field = TensorField((1,), ())
        field[0] = ψ
        assert field.chebyshev_transform(False, self.strategy)[0] is ψ

    def test_field_to_mps_preserves_layout_and_entries(self):
        field = TensorField((2,), self.shape)
        field[1] = self.values

        field_mps = field_to_mps(field, self.strategy)

        assert field_mps.field_shape == field.field_shape
        assert field_mps.mesh_shape == field.mesh_shape
        assert field_mps[0] is None
        ψ = field_mps[1]
        assert isinstance(ψ, MPS)
        npt.assert_allclose(ψ.to_vector(), self.values)


class TestProblems:
    intervals: ClassVar[list[Interval]] = [
        ChebyshevInterval(-2.0, 1.0, 4, endpoints=False),
        ChebyshevInterval(-1.0, 3.0, 2, endpoints=False),
    ]
    mesh = Mesh(intervals)
    shape = mesh.dimensions
    size = int(np.prod(shape))

    @classmethod
    def bulk(cls, m=2):
        return BulkProblem(
            cls.mesh,
            TensorField((2, 2, m, m), cls.shape),
            TensorField((m, m), cls.shape),
            TensorField((m,), cls.shape),
        )

    @classmethod
    def boundary_conditions(cls, m=2):
        conditions = []
        for dim in range(2):
            face_shape = cls.shape[:dim] + cls.shape[dim + 1 :]
            for is_left in (True, False):
                face = BoundaryFace(dim, is_left)
                data = TensorField((m,), face_shape)
                if dim == 0:
                    conditions.append(DirichletBoundaryCondition(face, data))
                else:
                    conditions.append(
                        RobinBoundaryCondition(face, TensorField((m, m), face_shape), data)
                    )
        return conditions

    def test_bulk_metadata_and_transform(self):
        bulk = self.bulk()
        values = np.arange(self.size, dtype=float)
        bulk.source[1] = values

        transformed = bulk.chebyshev_transform()
        assert (bulk.d, bulk.m, bulk.mesh_shape, bulk.mesh_size) == (2, 2, self.shape, self.size)
        assert transformed.mesh is self.mesh
        npt.assert_allclose(
            transformed.source[1], vec_nd_chebyshev_transform(values, self.shape), atol=1.0e-12
        )
        assert transformed.source[0] is None

    def test_boundary_metadata_lookup_and_transform(self):
        conditions = self.boundary_conditions()
        boundary = BoundaryProblem(self.mesh, conditions)

        assert isinstance(boundary.boundary_conditions, tuple)
        assert (boundary.d, boundary.m) == (2, 2)
        assert boundary.mesh_shape == self.shape
        assert boundary.mesh_size == self.size
        for condition in conditions:
            assert (
                boundary.get_boundary_condition(condition.face.dim, condition.face.is_left)
                is condition
            )

        transformed = boundary.chebyshev_transform()
        assert transformed.mesh_shape == self.shape
        assert [bc.face for bc in transformed.boundary_conditions] == [bc.face for bc in conditions]
        with pytest.raises(IndexError):
            boundary.get_boundary_condition(2, True)
