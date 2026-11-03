import numpy as np

from seemps.state import MPS, MPSSum, Strategy, simplify
from seemps.operators import MPO, MPOSum, simplify_mpo
from .factories.differentiation import _coefficient_derivative_mpo

from ..problem import (
    BoundaryProblem,
    BulkProblem,
    DirichletBoundaryCondition,
    RobinBoundaryCondition,
)
from .factories.boundary import mpo_chebyshev_recombination, mpo_column_projector, mpo_face_trace
from .factories.gram import mpo_nd_gram, mpo_weighted_gram
from .factories.lift import mps_spectral_lift
from .factories.utils import (
    checked_sites_per_dimension,
    mps_vector_field,
    mpo_identity,
    mpo_kron_expand,
    mpo_sequential_simplify,
    mpo_tensor_product,
    mpo_vector_field,
    get_mps_constant_coefficient,
)


def assemble_mps_bulk(bulk_chebyshev: BulkProblem[MPS], svd_strategy: Strategy) -> tuple[MPO, MPS]:
    p = bulk_chebyshev
    sites = checked_sites_per_dimension(p.mesh_shape)
    n = sum(sites)

    scales = [2.0 / (interval.stop - interval.start) for interval in p.mesh.intervals]
    J = float(1.0 / np.prod(scales))
    constant_rtol = max(100 * np.finfo(float).eps, svd_strategy.get_tolerance())

    # Locate the constant reaction terms
    reaction_constants = {}
    has_variable_reaction = False
    for μ in range(p.m):
        for ν in range(p.m):
            ψ_R = p.reaction[μ, ν]
            if ψ_R is None:
                continue
            coefficient = get_mps_constant_coefficient(ψ_R, rtol=constant_rtol)
            if coefficient is None:
                has_variable_reaction = True
            else:
                reaction_constants[μ, ν] = coefficient

    # Locate the constant principal terms
    principal_constants = {}
    has_variable_principal = False
    for α in range(p.d):
        for β in range(p.d):
            for μ in range(p.m):
                for ν in range(p.m):
                    ψ_P = p.principal[α, β, μ, ν]
                    if ψ_P is None:
                        continue
                    coefficient = get_mps_constant_coefficient(ψ_P, rtol=constant_rtol)
                    if coefficient is None:
                        has_variable_principal = True
                    else:
                        principal_constants[α, β, μ, ν] = coefficient

    has_source = any(ψ is not None for ψ in p.source.flat)
    has_constant = bool(reaction_constants or principal_constants)
    ρ_G = mpo_nd_gram(sites, svd_strategy) if has_source or has_constant else None

    ψ_b: MPS
    if has_source:
        assert ρ_G is not None
        b_blocks = [None if ψ is None else ρ_G.apply(ψ, svd_strategy) for ψ in p.source.flat]
        ψ_b = simplify(J * mps_vector_field(b_blocks, n), strategy=svd_strategy)
    else:
        ψ_b = mps_vector_field([None] * p.m, n)

    has_reaction = bool(reaction_constants) or has_variable_reaction
    has_principal = bool(principal_constants) or has_variable_principal
    if not has_reaction and not has_principal:
        return mpo_vector_field([[None] * p.m for _ in range(p.m)], n), ψ_b

    has_variable = has_variable_reaction or has_variable_principal
    if has_variable:
        sites_expand = [n_k + 1 for n_k in sites]
        ρ_G_expand = mpo_nd_gram(sites_expand, svd_strategy)
    else:
        ρ_G_expand = None

    # Galerkin assembly needs coefficient derivatives in LSB-first ordering,
    # not SeeMPS's public nodal differentiation operator.
    diffs = (
        [
            mpo_kron_expand(_coefficient_derivative_mpo(sites[α], svd_strategy).reverse(), sites, α)
            for α in range(p.d)
        ]
        if has_principal
        else []
    )
    diffs_T = [ρ_D.T for ρ_D in diffs]
    stiffness_constants = {}
    weighted_grams: dict[int, MPO] = {}

    def weighted_gram(coefficient: MPS) -> MPO:
        # Constitutive components often share the very same material field.
        key = id(coefficient)
        if key not in weighted_grams:
            weighted_grams[key] = mpo_weighted_gram(coefficient, sites, svd_strategy, ρ_G_expand)
        return weighted_grams[key]

    A_blocks = []
    for μ in range(p.m):
        A_row = []
        for ν in range(p.m):
            terms = []
            weights = []

            ψ_R = p.reaction[μ, ν]
            if ψ_R is not None:
                coefficient = reaction_constants.get((μ, ν))
                if coefficient is not None:
                    assert ρ_G is not None
                    terms.append(ρ_G)
                    weights.append(coefficient)
                else:
                    assert ρ_G_expand is not None
                    terms.append(weighted_gram(ψ_R))
                    weights.append(1.0)

            for α in range(p.d):
                for β in range(p.d):
                    ψ_P = p.principal[α, β, μ, ν]
                    if ψ_P is None:
                        continue

                    coefficient = principal_constants.get((α, β, μ, ν))
                    if coefficient is not None:
                        assert ρ_G is not None
                        key = (α, β)
                        if key not in stiffness_constants:
                            stiffness_constants[key] = mpo_sequential_simplify(
                                [diffs[β], ρ_G, diffs_T[α]], svd_strategy
                            )
                        terms.append(stiffness_constants[key])
                        weights.append(scales[α] * scales[β] * coefficient)
                    else:
                        assert ρ_G_expand is not None
                        ρ = weighted_gram(ψ_P)
                        terms.append(
                            mpo_sequential_simplify([diffs[β], ρ, diffs_T[α]], svd_strategy)
                        )
                        weights.append(scales[α] * scales[β])

            if not terms:
                ρ_block = None
            elif len(terms) == 1:
                ρ_block = weights[0] * terms[0]
            else:
                ρ_block = simplify_mpo(MPOSum(terms, weights), strategy=svd_strategy)

            A_row.append(ρ_block)

        A_blocks.append(A_row)

    ρ_A = simplify_mpo(J * mpo_vector_field(A_blocks, n), strategy=svd_strategy)
    return ρ_A, ψ_b


def assemble_mps_robin_correction(
    boundary_chebyshev: BoundaryProblem[MPS], svd_strategy: Strategy
) -> tuple[MPO, MPS]:
    p = boundary_chebyshev
    sites = checked_sites_per_dimension(p.mesh_shape)
    n = sum(sites)

    scales = [2.0 / (interval.stop - interval.start) for interval in p.mesh.intervals]
    J = float(1.0 / np.prod(scales))

    A_corrections: list[MPO] = []
    b_corrections: list[MPS] = []

    for dim in range(p.d):
        n_face = sites[:dim] + sites[dim + 1 :]
        n_start = sum(sites[:dim])
        n_stop = n_start + sites[dim]
        face_positions = [*range(n_start), *range(n_stop, n)]
        face_dimensions = [2] * sites[dim]
        J_face = scales[dim] * J

        for is_left in (True, False):
            bc = p.get_boundary_condition(dim, is_left)
            if not isinstance(bc, RobinBoundaryCondition):
                continue

            has_source = any(ψ is not None for ψ in bc.data.flat)
            has_reaction = any(ψ is not None for ψ in bc.reaction.flat)
            if not has_source and not has_reaction:
                continue

            ρ_T = mpo_face_trace(sites, dim, is_left)
            ρ_T_T = ρ_T.T

            if has_source:
                ρ_G = mpo_nd_gram(n_face, svd_strategy) if n_face else None
                b_blocks: list[MPS | None] = []
                for ψ_f in bc.data.flat:
                    if ψ_f is None:
                        b_blocks.append(None)
                        continue
                    if n_face:
                        assert ρ_G is not None
                        ψ_face = ρ_G.apply(ψ_f, svd_strategy).extend(
                            n, sites=face_positions, dimensions=face_dimensions
                        )
                    else:
                        value = ψ_f.to_vector().item()
                        core = np.array([1.0, 0.0], dtype=ψ_f[0].dtype).reshape(1, 2, 1)
                        ψ_face = value * MPS([core] * n)
                    ψ_b_block = ρ_T_T.apply(ψ_face, strategy=svd_strategy)
                    b_blocks.append(ψ_b_block)
                ψ_b = J_face * simplify(mps_vector_field(b_blocks, n), strategy=svd_strategy)
                b_corrections.append(ψ_b)

            if has_reaction:
                sites_expand = [n_k + 1 for n_k in n_face]
                ρ_G_expand = mpo_nd_gram(sites_expand, svd_strategy) if n_face else None
                ρ_I = mpo_identity(n) if not n_face else None
                A_blocks: list[list[MPO | None]] = []
                for μ in range(p.m):
                    A_row: list[MPO | None] = []
                    for ν in range(p.m):
                        ψ_R = bc.reaction[μ, ν]
                        if ψ_R is None:
                            A_row.append(None)
                            continue
                        if n_face:
                            assert ρ_G_expand is not None
                            ρ_face = mpo_weighted_gram(
                                ψ_R, n_face, svd_strategy, ρ_G_expand
                            ).extend(n, sites=face_positions, dimensions=face_dimensions)
                        else:
                            assert ρ_I is not None
                            ρ_face = ψ_R.to_vector().item() * ρ_I
                        ρ_B = mpo_sequential_simplify([ρ_T, ρ_face, ρ_T_T], strategy=svd_strategy)
                        A_row.append(ρ_B)
                    A_blocks.append(A_row)
                ρ_A = J_face * simplify_mpo(mpo_vector_field(A_blocks, n), strategy=svd_strategy)
                A_corrections.append(ρ_A)

    ρ_A = (
        mpo_vector_field([[None] * p.m for _ in range(p.m)], n)
        if not A_corrections
        else A_corrections[0]
        if len(A_corrections) == 1
        else simplify_mpo(MPOSum(A_corrections, [1.0] * len(A_corrections)), strategy=svd_strategy)
    )

    ψ_b = (
        mps_vector_field([None] * p.m, n)
        if not b_corrections
        else b_corrections[0]
        if len(b_corrections) == 1
        else simplify(MPSSum([1.0] * len(b_corrections), b_corrections), strategy=svd_strategy)
    )

    return ρ_A, ψ_b


def assemble_mps_dirichlet_operators(
    boundary_chebyshev: BoundaryProblem[MPS], svd_strategy: Strategy
) -> tuple[MPO, MPO, MPS]:
    p = boundary_chebyshev
    sites = checked_sites_per_dimension(p.mesh_shape)
    n = sum(sites)

    ρ_Z_components: list[list[MPO | None]] = []
    ρ_P_components: list[list[MPO | None]] = []
    for μ in range(p.m):
        ρ_Z_terms = []
        ρ_Q_terms = []
        for dim, n_k in enumerate(sites):
            bc_left = p.get_boundary_condition(dim, True)
            bc_right = p.get_boundary_condition(dim, False)
            is_left_dirichlet = (
                isinstance(bc_left, DirichletBoundaryCondition) and bc_left.data[μ] is not None
            )
            is_right_dirichlet = (
                isinstance(bc_right, DirichletBoundaryCondition) and bc_right.data[μ] is not None
            )
            ρ_Z_terms.append(
                mpo_chebyshev_recombination(n_k, is_left_dirichlet, is_right_dirichlet)
            )
            ρ_Q_terms.append(mpo_column_projector(n_k, is_left_dirichlet, is_right_dirichlet))

        ρ_Z_μ = mpo_tensor_product(ρ_Z_terms)
        ρ_P_μ = (mpo_identity(n) - mpo_tensor_product(ρ_Q_terms)).join()
        ρ_Z_components.append([ρ_Z_μ if μ == ν else None for ν in range(p.m)])
        ρ_P_components.append([ρ_P_μ if μ == ν else None for ν in range(p.m)])

    ρ_Z = simplify_mpo(mpo_vector_field(ρ_Z_components, n), strategy=svd_strategy)
    ρ_P = simplify_mpo(mpo_vector_field(ρ_P_components, n), strategy=svd_strategy)
    ψ_lift = simplify(mps_spectral_lift(p), strategy=svd_strategy)
    return ρ_Z, ρ_P, ψ_lift
