import numpy as np
import numpy.testing as npt
import pytest

from seemps_galerkin.mps.factories.transform import mpo_chebyshev_transform
from seemps_galerkin.mps.factories.differentiation import _coefficient_derivative_mpo
from seemps.state import (
    MPS,
    MPSSum,
    Simplification,
    Strategy,
    Truncation,
    mps_tensor_product,
    simplify,
)

from seemps_galerkin.dense.factories import (
    get_constant_coefficient,
    op_bit_reversal,
    op_chebyshev_recombination,
    op_column_projector,
    op_face_trace,
    op_nd_gram,
    op_nd_mult,
    op_tensor_product,
    op_weighted_gram,
    vec_chebyshev_endpoint,
    vec_nd_chebyshev_transform,
)
from seemps_galerkin.mps.factories import (
    get_mps_constant_coefficient,
    mps_chebyshev_endpoint,
    mps_delta,
    mps_full,
    mps_gram_weights,
    mps_identity,
    mps_zeros,
    mpo_chebyshev_recombination,
    mpo_column_projector,
    mpo_face_trace,
    mpo_gram,
    mpo_identity,
    mpo_kron_expand,
    mpo_nd_gram,
    mpo_nd_mult,
    mpo_sequential_simplify,
    mpo_tensor_product,
    mpo_weighted_gram,
    mpo_zeros,
)


def bit_reversal(sites):
    return op_tensor_product([op_bit_reversal(n) for n in sites])


def mps_from_dense(values, sites):
    return MPS.from_vector(bit_reversal(sites) @ values, [2] * sum(sites), normalize=False)


def mpo_to_dense(operator, sites):
    reversal = bit_reversal(sites)
    return reversal @ operator.to_matrix() @ reversal.T


class TestMPSUtilityFactories:
    rtol = 0.0
    atol = 1.0e-12
    strategy = Strategy(
        method=Truncation.RELATIVE_SINGULAR_VALUE,
        tolerance=1.0e-13,
        simplification_tolerance=1.0e-13,
        max_bond_dimension=128,
        max_sweeps=6,
        simplify=Simplification.VARIATIONAL,
    )

    @pytest.mark.parametrize(
        ("factory", "args", "expected"),
        [
            (mps_identity, (3,), np.ones(8)),
            (mps_full, (3, -2.5), np.full(8, -2.5)),
            (mps_zeros, (3,), np.zeros(8)),
            (mps_delta, (3, 0), np.eye(8)[0]),
            (mps_delta, (3, 5), np.eye(8)[5]),
            (mps_delta, (3, -1), np.eye(8)[-1]),
        ],
    )
    def test_state_factories_have_dense_vector_semantics(self, factory, args, expected):
        npt.assert_array_equal(factory(*args).to_vector(), expected)

    @pytest.mark.parametrize(
        "values",
        [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([1.5, 0.0, 0.0, 0.0]),
            np.array([1.0 + 2.0j, 0.0, 0.0, 0.0]),
            np.array([1.5, 1.0e-8, 0.0, 0.0]),
        ],
    )
    def test_constant_detector_matches_dense(self, values):
        expected = get_constant_coefficient(values)
        actual = get_mps_constant_coefficient(MPS.from_vector(values, [2, 2], normalize=False))
        if expected is None:
            assert actual is None
        else:
            npt.assert_allclose(actual, expected, rtol=self.rtol, atol=self.atol)

    @pytest.mark.parametrize(
        ("factory", "expected"), [(mpo_identity, np.eye(8)), (mpo_zeros, np.zeros((8, 8)))]
    )
    def test_basic_operator_factories_have_dense_matrix_semantics(self, factory, expected):
        npt.assert_array_equal(factory(3).to_matrix(), expected)

    def test_tensor_product_expansion_and_sequential_product_match_dense(self):
        derivative_1 = _coefficient_derivative_mpo(1, self.strategy)
        derivative_2 = _coefficient_derivative_mpo(2, self.strategy)
        tensor_product = mpo_tensor_product([derivative_1, derivative_2])
        npt.assert_allclose(
            tensor_product.to_matrix(),
            np.kron(derivative_1.to_matrix(), derivative_2.to_matrix()),
            atol=self.atol,
        )

        expanded = mpo_kron_expand(derivative_1, [2, 1], 1)
        npt.assert_allclose(
            expanded.to_matrix(), np.kron(np.eye(4), derivative_1.to_matrix()), atol=self.atol
        )

        sequential = mpo_sequential_simplify([derivative_2, derivative_2.T], self.strategy)
        npt.assert_allclose(
            sequential.to_matrix(),
            derivative_2.T.to_matrix() @ derivative_2.to_matrix(),
            atol=self.atol,
        )


class TestMPSDifferentialAndGramFactories:
    rtol = 0.0
    atol = 1.0e-11
    sites_cases = ([1], [3], [2, 2], [1, 2, 2])

    @pytest.mark.parametrize("n", [1, 3, 7])
    def test_one_dimensional_gram_matches_dense(self, n):
        actual = mpo_to_dense(mpo_gram(n), [n])
        npt.assert_allclose(actual, op_nd_gram([2**n]), rtol=self.rtol, atol=self.atol)

    @pytest.mark.parametrize("sites", sites_cases)
    def test_every_nd_gram_matches_dense(self, sites):
        shape = [2**n for n in sites]
        actual = mpo_to_dense(mpo_nd_gram(sites), sites)
        npt.assert_allclose(actual, op_nd_gram(shape), rtol=self.rtol, atol=self.atol)

    @pytest.mark.parametrize("n", [12])
    def test_gram_bond_dimension_is_bounded(self, n):
        χ_w = mps_gram_weights(n + 1).max_bond_dimension()
        assert max(mpo_gram(n).bond_dimensions()) <= 6 * χ_w + 1

    @pytest.mark.parametrize("sites", sites_cases)
    def test_every_nd_multiplication_operator_matches_dense(self, sites):
        shape = [2**n for n in sites]
        values = np.random.default_rng(sum(sites)).normal(size=int(np.prod(shape)))
        actual = mpo_to_dense(mpo_nd_mult(mps_from_dense(values, sites), sites), sites)
        npt.assert_allclose(actual, op_nd_mult(values, shape), rtol=self.rtol, atol=self.atol)


class TestMPSWeightedGram:
    rtol = 0.0
    atol = 1.0e-9
    strategy = TestMPSUtilityFactories.strategy

    @pytest.mark.parametrize("sites", [[1], [2], [1, 2], [2, 2]])
    def test_weighted_gram_matches_dense(self, sites):
        shape = [2**n for n in sites]
        values = np.random.default_rng(20 + sum(sites)).normal(size=int(np.prod(shape)))
        state = mps_from_dense(values, sites)
        actual = mpo_to_dense(mpo_weighted_gram(state, sites, self.strategy), sites)
        npt.assert_allclose(actual, op_weighted_gram(values, shape), rtol=self.rtol, atol=self.atol)

    def test_small_affine_coupling_survives_expanded_projection(self):
        n, N, modes = 3, 8, 10
        strategy = self.strategy.replace(tolerance=1e-10, simplification_tolerance=1e-10)
        zero, one = np.eye(N)[0], np.eye(N)[1]
        spatial = [
            np.array([1.0, 0.3, 0.05, 0.005, 0.001, 0.0, 0.0, 0.0]),
            np.array([1.0, -0.2, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.15, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]

        def state(values):
            return mps_from_dense(values, [n])

        terms = [mps_tensor_product([state(zero)] * (3 + modes))]
        weights = [1.0]
        for r in range(modes):
            values = spatial.copy()
            values[0] = values[0].copy()
            values[0][1] += 0.01 * r
            terms.append(
                mps_tensor_product(
                    [state(v) for v in values]
                    + [state(one if k == r else zero) for k in range(modes)]
                )
            )
            weights.append(0.05 / (r + 1))
        coefficient = simplify(MPSSum(weights, terms), strategy=strategy)
        operator = mpo_weighted_gram(coefficient, [n] * (3 + modes), strategy)

        out_space, in_space = [6, 1, 6], [4, 0, 0]
        out_param, in_param = [0] * modes, [1] + [0] * (modes - 1)
        permutation = [int(f"{k:0{n}b}"[::-1], 2) for k in range(N)]
        output = [int(bit) for k in out_space + out_param for bit in f"{permutation[k]:0{n}b}"]
        input_ = [int(bit) for k in in_space + in_param for bit in f"{permutation[k]:0{n}b}"]
        value = np.ones(1)
        for core, i, j in zip(operator, output, input_):
            value = value @ core[:, i, j, :]

        exact = (
            weights[1]
            * np.prod(
                [op_weighted_gram(v, [N])[i, j] for v, i, j in zip(spatial, out_space, in_space)]
            )
            * op_weighted_gram(one, [N])[0, 1]
            * 2 ** (modes - 1)
        )
        npt.assert_allclose(value[0], exact, rtol=0, atol=1e-11)


class TestMPSComponentTransformLayout:
    @pytest.mark.parametrize("sites", [[3], [2, 3]])
    @pytest.mark.parametrize("components", [2, 3])
    def test_inverse_transform_preserves_component_site(self, sites, components):
        shape = [2**n for n in sites]
        N = int(np.prod(shape))
        values = np.random.default_rng(40).normal(size=(components, N))
        reversal = bit_reversal(sites)
        coefficients = np.vstack(
            [reversal @ vec_nd_chebyshev_transform(row, shape) for row in values]
        )
        state = MPS.from_vector(
            coefficients.ravel(), [components] + [2] * sum(sites), normalize=False
        )
        dimensions = state.physical_dimensions()
        start = 1
        for n in sites:
            end = start + n
            transform = mpo_chebyshev_transform(
                n, inverse=True, strategy=TestMPSUtilityFactories.strategy
            ).extend(
                state.size,
                sites=list(range(start, end)),
                dimensions=dimensions[:start] + dimensions[end:],
            )
            state = transform.apply(state)
            start = end
        npt.assert_allclose(state.to_vector().reshape(components, N), values, atol=1e-10)


class TestMPSBoundaryFactories:
    rtol = 0.0
    atol = 1.0e-12

    @pytest.mark.parametrize("n", [1, 4])
    @pytest.mark.parametrize("is_left", [False, True])
    def test_endpoints_match_dense(self, n, is_left):
        actual = op_bit_reversal(n) @ mps_chebyshev_endpoint(n, is_left).to_vector()
        npt.assert_allclose(
            actual, vec_chebyshev_endpoint(2**n, is_left), rtol=self.rtol, atol=self.atol
        )

    @pytest.mark.parametrize("sites", [[3], [2, 3], [1, 2, 2]])
    def test_every_face_trace_matches_dense(self, sites):
        shape = [2**n for n in sites]
        for dim in range(len(sites)):
            embedding = op_tensor_product(
                [np.eye(N)[:, [0]] if axis == dim else np.eye(N) for axis, N in enumerate(shape)]
            )
            for is_left in (False, True):
                actual = mpo_to_dense(mpo_face_trace(sites, dim, is_left), sites)
                expected = embedding @ op_face_trace(shape, dim, is_left)
                npt.assert_allclose(actual, expected, rtol=self.rtol, atol=self.atol)

    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("is_left", [False, True])
    @pytest.mark.parametrize("is_right", [False, True])
    def test_every_dirichlet_operator_matches_dense(self, n, is_left, is_right):
        N = 2**n
        actual = mpo_to_dense(mpo_chebyshev_recombination(n, is_left, is_right), [n])
        npt.assert_allclose(
            actual, op_chebyshev_recombination(N, is_left, is_right), rtol=self.rtol, atol=self.atol
        )

        actual = mpo_to_dense(mpo_column_projector(n, is_left, is_right), [n])
        npt.assert_allclose(
            actual, op_column_projector(N, is_left, is_right), rtol=self.rtol, atol=self.atol
        )
