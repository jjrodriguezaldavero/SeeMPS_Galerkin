from itertools import pairwise

import numpy as np
import numpy.testing as npt
import pytest
from seemps.analysis.mesh import ChebyshevInterval, Mesh
from seemps.state import MPS

from seemps_galerkin.dense.assembly import vec_spectral_lift
from seemps_galerkin.mps.factories.lift import mps_chebyshev_blending
from seemps_galerkin.mps.factories.lift import mps_spectral_lift
from seemps_galerkin.problem import (
    BoundaryFace,
    BoundaryProblem,
    DirichletBoundaryCondition,
    RobinBoundaryCondition,
    TensorField,
)


def bit_order(sites):
    offsets = np.cumsum([0, *sites])
    axes = [i for start, stop in pairwise(offsets) for i in range(stop - 1, start - 1, -1)]
    return np.arange(2 ** sum(sites)).reshape([2] * sum(sites)).transpose(axes).ravel()


def boundary_pair(sites, prescribed=True):
    """Compatible complex traces of one tensor, with different faces per component."""
    rng = np.random.default_rng(12)
    shape = tuple(2**n for n in sites)
    mesh = Mesh([ChebyshevInterval(-1, 1, N, endpoints=False) for N in shape])
    coefficients = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    dense, quantized = [], []
    for dim in range(len(sites)):
        tangential = sites[:dim] + sites[dim + 1 :]
        face_shape = shape[:dim] + shape[dim + 1 :]
        for left in (True, False):
            data = TensorField((3,), face_shape)
            data_mps = TensorField((3,), face_shape)
            e = (-1.0 if left else 1.0) ** np.arange(shape[dim])
            values = np.tensordot(e, coefficients, axes=(0, dim)).reshape(-1)
            if prescribed:
                for component in range(2 if left else 1):
                    data[component] = values
                    data_mps[component] = MPS.from_vector(
                        values[bit_order(tangential)] if tangential else values,
                        [2] * sum(tangential) if tangential else [1],
                        normalize=False,
                    )
            face = BoundaryFace(dim, left)
            if prescribed and (left or dim % 2):
                dense.append(DirichletBoundaryCondition(face, data))
                quantized.append(DirichletBoundaryCondition(face, data_mps))
            else:
                dense.append(RobinBoundaryCondition(face, TensorField((3, 3), face_shape), data))
                quantized.append(
                    RobinBoundaryCondition(face, TensorField((3, 3), face_shape), data_mps)
                )
    order = rng.permutation(len(dense))
    return (
        BoundaryProblem(mesh, [dense[i] for i in order]),
        BoundaryProblem(mesh, [quantized[i] for i in order]),
    )


@pytest.mark.parametrize("sites", [(2,), (2, 3, 2)])
@pytest.mark.parametrize("prescribed", [False, True])
def test_lift_matches_dense_for_complex_componentwise_data(sites, prescribed):
    dense, boundary = boundary_pair(sites, prescribed)
    states = [
        state for bc in boundary.boundary_conditions for state in bc.data.flat if state is not None
    ]
    before = [[c.copy() for c in state] for state in states]
    expected = vec_spectral_lift(dense).reshape(3, -1)
    result = mps_spectral_lift(boundary).to_vector().reshape(3, -1)
    npt.assert_allclose(result[:, bit_order(sites)], expected, rtol=1e-12, atol=1e-12)
    for state, old in zip(states, before):
        for core, saved in zip(state, old):
            npt.assert_array_equal(core, saved)


@pytest.mark.parametrize("n", [2, 3, 6])
@pytest.mark.parametrize("left", [False, True])
def test_boundary_vector_coefficients_and_endpoint_derivatives(n, left):
    coefficients = mps_chebyshev_blending(n, left).to_vector()[bit_order((n,))]
    sign = -1 if left else 1
    expected = np.zeros(2**n)
    expected[:4] = [0.5, sign * 9 / 16, 0, -sign / 16]
    npt.assert_array_equal(coefficients, expected)
    npt.assert_allclose(
        np.polynomial.chebyshev.chebval([-1, 1], coefficients), [int(left), int(not left)]
    )
    npt.assert_allclose(
        np.polynomial.chebyshev.chebval([-1, 1], np.polynomial.chebyshev.chebder(coefficients)),
        0,
        atol=1e-15,
    )
