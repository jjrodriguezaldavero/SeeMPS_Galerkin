import numpy as np
import numpy.testing as npt
import pytest
from numpy.polynomial.chebyshev import chebder, chebint, chebmul, chebval

from seemps.analysis.mesh import RegularInterval

from seemps_galerkin.dense.factories import (
    get_constant_coefficient,
    op_bit_reversal,
    op_chebyshev_derivative,
    op_chebyshev_recombination,
    op_chebyshev_transform,
    op_chebyshev_vandermonde,
    op_column_projector,
    op_dct2,
    op_face_trace,
    op_gram,
    op_kron_expand,
    op_mult,
    op_nd_chebyshev_derivative,
    op_nd_gram,
    op_nd_mult,
    op_tensor_product,
    op_tensor_sum,
    op_weighted_gram,
    vec_chebyshev_endpoint,
    vec_nd_chebyshev_interpolation,
    vec_nd_chebyshev_transform,
)


def integrate_chebyshev(coefficients):
    antiderivative = chebint(coefficients)
    return chebval(1.0, antiderivative) - chebval(-1.0, antiderivative)


class TestDenseChebyshevFactories:
    rtol = 0.0
    atol = 1.0e-12
    shape = (4, 8)
    values = np.random.default_rng(0).normal(size=int(np.prod(shape)))

    @pytest.mark.parametrize("N", [1, 2, 5, 8])
    def test_transform_matrices_are_mutual_inverses(self, N):
        identity = np.eye(N)
        npt.assert_allclose(op_dct2(N, True) @ op_dct2(N), identity, atol=self.atol)
        npt.assert_allclose(
            op_chebyshev_transform(N, True) @ op_chebyshev_transform(N), identity, atol=self.atol
        )

    @pytest.mark.parametrize("dimensions", [None, 0, [1], []])
    def test_transform_round_trips_selected_dimensions(self, dimensions):
        coefficients = vec_nd_chebyshev_transform(
            self.values, self.shape, which_dimensions=dimensions
        )
        actual = vec_nd_chebyshev_transform(
            coefficients, self.shape, which_dimensions=dimensions, inverse=True
        )
        npt.assert_allclose(actual, self.values, rtol=self.rtol, atol=self.atol)

    def test_partial_transform_acts_only_on_selected_axis(self):
        actual = vec_nd_chebyshev_transform(self.values, self.shape, which_dimensions=1)
        expected = np.vstack(
            [
                vec_nd_chebyshev_transform(row, [self.shape[1]])
                for row in self.values.reshape(self.shape)
            ]
        ).ravel()
        npt.assert_allclose(actual, expected, rtol=self.rtol, atol=self.atol)

    @pytest.mark.parametrize("new_shape", [(4, 8), (8, 8), (4, 16), (8, 16)])
    def test_interpolation_matches_zero_padded_coefficients(self, new_shape):
        actual = vec_nd_chebyshev_interpolation(self.values, self.shape, new_shape)
        changed = [k for k, pair in enumerate(zip(self.shape, new_shape)) if pair[0] != pair[1]]
        if not changed:
            assert actual is self.values
            return

        coefficients = vec_nd_chebyshev_transform(self.values, self.shape, which_dimensions=changed)
        padded = np.zeros(new_shape)
        padded[tuple(slice(N) for N in self.shape)] = coefficients.reshape(self.shape)
        expected = vec_nd_chebyshev_transform(
            padded.ravel(), new_shape, which_dimensions=changed, inverse=True
        )
        npt.assert_allclose(actual, expected, rtol=self.rtol, atol=self.atol)

    @pytest.mark.parametrize("N", [1, 2, 4, 8])
    def test_derivative_and_multiplication_match_polynomial_algebra(self, N):
        coefficients = np.random.default_rng(N).normal(size=N)
        multiplier = np.random.default_rng(N + 1).normal(size=N)
        derivative = np.pad(chebder(coefficients), (0, 1))[:N]
        product = chebmul(multiplier, coefficients)[:N]
        npt.assert_allclose(
            op_chebyshev_derivative(N) @ coefficients, derivative, rtol=self.rtol, atol=self.atol
        )
        npt.assert_allclose(
            op_mult(multiplier, N) @ coefficients, product, rtol=self.rtol, atol=self.atol
        )

    @pytest.mark.parametrize("dim", [0, 1])
    def test_multidimensional_derivative_acts_on_selected_axis(self, dim):
        shape = (4, 3)
        coefficients = np.random.default_rng(2).normal(size=shape)
        actual = (op_nd_chebyshev_derivative(shape, dim) @ coefficients.ravel()).reshape(shape)
        expected = np.apply_along_axis(
            lambda row: np.pad(chebder(row), (0, 1))[: row.size], dim, coefficients
        )
        npt.assert_allclose(actual, expected, rtol=self.rtol, atol=self.atol)

    def test_multidimensional_multiplication_handles_zero_constant_and_separable_fields(self):
        shape = (3, 4)
        size = int(np.prod(shape))
        npt.assert_array_equal(op_nd_mult(np.zeros(size), shape), np.zeros((size, size)))

        constant = np.zeros(size)
        constant[0] = 1.7
        npt.assert_allclose(op_nd_mult(constant, shape), 1.7 * np.eye(size), atol=self.atol)

        q0, q1 = np.array([1.0, 0.2, -0.1]), np.array([0.7, 0.3, -0.2, 0.1])
        q = np.outer(q0, q1).ravel()
        expected = np.kron(op_mult(q0, shape[0]), op_mult(q1, shape[1]))
        npt.assert_allclose(op_nd_mult(q, shape), expected, atol=self.atol)

    @pytest.mark.parametrize("N", [1, 2, 5, 8])
    def test_gram_matrix_integrates_chebyshev_products_exactly(self, N):
        left = np.random.default_rng(N).normal(size=N)
        right = np.random.default_rng(N + 20).normal(size=N)
        expected = integrate_chebyshev(chebmul(left, right))
        npt.assert_allclose(left @ op_gram(N) @ right, expected, atol=self.atol)

    def test_weighted_gram_integrates_three_polynomials_exactly(self):
        N = 5
        rng = np.random.default_rng(4)
        weight, left, right = (rng.normal(size=N) for _ in range(3))
        expected = integrate_chebyshev(chebmul(weight, chebmul(left, right)))
        actual = left @ op_weighted_gram(weight, [N]) @ right
        npt.assert_allclose(actual, expected, rtol=self.rtol, atol=2.0e-12)

    def test_nd_gram_is_the_tensor_product_of_one_dimensional_grams(self):
        shape = (2, 4, 3)
        expected = op_tensor_product([op_gram(N) for N in shape])
        npt.assert_allclose(op_nd_gram(shape), expected, atol=self.atol)
        npt.assert_array_equal(op_nd_gram(()), np.ones((1, 1)))

    @pytest.mark.parametrize("axis", [0, 1, 2, None])
    def test_weighted_gram_matches_full_multidimensional_construction(self, axis):
        shape = (3, 4, 2)
        rng = np.random.default_rng(41)
        if axis is None:
            weight = rng.normal(size=shape)
        else:
            weight = 1e-16 * rng.normal(size=shape)
            section = tuple(slice(None) if dim == axis else 0 for dim in range(3))
            weight[section] = rng.normal(size=shape[axis])
        expanded = tuple(2 * n for n in shape)
        interior = tuple(slice(n) for n in shape)
        padded = np.zeros(expanded)
        padded[interior] = weight
        full = op_nd_gram(expanded) @ op_nd_mult(padded.ravel(), expanded)
        expected = full.reshape(expanded + expanded)[interior + interior].reshape(24, 24)
        npt.assert_allclose(
            op_weighted_gram(weight.ravel(), shape), expected, rtol=self.rtol, atol=self.atol
        )

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            (np.array([0.0, 0.0, 0.0]), 0.0),
            (np.array([1.5, 0.0, 0.0]), 1.5),
            (np.array([1.5, 1.0e-8, 0.0]), None),
        ],
    )
    def test_constant_coefficient_detection(self, values, expected):
        assert get_constant_coefficient(values) == expected

    def test_vandermonde_evaluates_on_a_physical_interval(self):
        coefficients = np.array([1.0, -0.3, 0.2, 0.1])
        interval = RegularInterval(-2.0, 3.0, 7, endpoint_left=True, endpoint_right=True)
        x_mapped = 2.0 * (interval.to_vector() + 2.0) / 5.0 - 1.0
        actual = op_chebyshev_vandermonde(4, 7, -2.0, 3.0) @ coefficients
        npt.assert_allclose(actual, chebval(x_mapped, coefficients), atol=self.atol)


class TestDenseTensorAndBoundaryFactories:
    rtol = 0.0
    atol = 1.0e-12

    @pytest.mark.parametrize("n", [1, 3])
    def test_bit_reversal_is_a_symmetric_permutation(self, n):
        reversal = op_bit_reversal(n)
        npt.assert_array_equal(reversal, reversal.T)
        npt.assert_array_equal(reversal @ reversal, np.eye(2**n))

    def test_tensor_expansion_and_sum_have_kronecker_semantics(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.5, -1.0], [2.0, 0.0]])
        npt.assert_array_equal(op_tensor_product([A, B]), np.kron(A, B))
        npt.assert_array_equal(op_kron_expand(A, (3, 2), 1), np.kron(np.eye(3), A))
        npt.assert_array_equal(op_tensor_sum([A, B]), np.kron(A, np.eye(2)) + np.kron(np.eye(2), B))

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("is_left", [False, True])
    def test_face_trace_evaluates_the_normal_chebyshev_axis(self, dim, is_left):
        shape = (4, 3)
        coefficients = np.random.default_rng(8).normal(size=shape)
        endpoint = vec_chebyshev_endpoint(shape[dim], is_left)
        expected = endpoint @ coefficients if dim == 0 else coefficients @ endpoint
        actual = op_face_trace(shape, dim, is_left) @ coefficients.ravel()
        npt.assert_allclose(actual, np.ravel(expected), atol=self.atol)

    @pytest.mark.parametrize("is_left", [False, True])
    @pytest.mark.parametrize("is_right", [False, True])
    def test_recombination_and_projector_encode_each_constraint_pair(self, is_left, is_right):
        N = 8
        recombination = op_chebyshev_recombination(N, is_left, is_right)
        if is_left:
            npt.assert_allclose(vec_chebyshev_endpoint(N, True) @ recombination, 0.0)
        if is_right:
            npt.assert_allclose(vec_chebyshev_endpoint(N, False) @ recombination, 0.0)

        constraints = int(is_left) + int(is_right)
        expected = np.eye(N)
        if constraints:
            expected[-constraints:, -constraints:] = 0.0
        npt.assert_array_equal(op_column_projector(N, is_left, is_right), expected)
