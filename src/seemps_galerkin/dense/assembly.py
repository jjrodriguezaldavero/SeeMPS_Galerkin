import numpy as np
import itertools

from .factories import (
    get_constant_coefficient,
    op_chebyshev_derivative,
    op_chebyshev_recombination,
    op_column_projector,
    op_face_trace,
    op_gram,
    op_nd_chebyshev_derivative,
    op_nd_gram,
    op_tensor_product,
    op_weighted_gram,
    vec_chebyshev_endpoint,
)
from ..problem import (
    BoundaryFace,
    BulkProblem,
    BoundaryProblem,
    DirichletBoundaryCondition,
    RobinBoundaryCondition,
)
from ..typing import Vector, Operator


def assemble_dense_bulk(bulk_chebyshev: BulkProblem[Vector]) -> tuple[Operator, Vector]:
    p = bulk_chebyshev
    m, d, N, shape = p.m, p.d, p.mesh_size, p.mesh_shape

    A = np.zeros((m * N, m * N), dtype=float, order="F")  # SciPy's faster with Fortran order
    b = np.zeros(m * N, dtype=float)

    scales = [2.0 / (interval.stop - interval.start) for interval in p.mesh.intervals]
    diffs = [op_chebyshev_derivative(N_k) for N_k in shape]
    grams = [op_gram(N_k) for N_k in shape]
    G = op_tensor_product(grams)

    # Assemble load and reaction terms
    for μ in range(m):
        rows = slice(μ * N, (μ + 1) * N)
        entry = p.source[μ]
        if entry is not None:
            b[rows] = G @ entry
        for ν in range(m):
            cols = slice(ν * N, (ν + 1) * N)
            entry = p.reaction[μ, ν]
            if entry is None:
                continue
            if (c := get_constant_coefficient(entry)) is not None:
                A[rows, cols] += c * G
            else:
                A[rows, cols] += op_weighted_gram(entry, shape)

    # Assemble stiffness terms one derivative pair at a time
    for α in range(d):
        for β in range(d):
            scale = scales[α] * scales[β]
            constant_entries = []
            variable_entries = []
            for μ in range(m):
                for ν in range(m):
                    entry = p.principal[α, β, μ, ν]
                    if entry is None:
                        continue
                    if (c := get_constant_coefficient(entry)) is not None:
                        constant_entries.append((μ, ν, c))
                    else:
                        variable_entries.append((μ, ν, entry))

            if constant_entries:  # Assemble constant stiffness components
                K = _assemble_constant_stiffness(grams, diffs, α, β)
                for μ, ν, c in constant_entries:
                    rows = slice(μ * N, (μ + 1) * N)
                    cols = slice(ν * N, (ν + 1) * N)
                    A[rows, cols] += scale * c * K
                del K  # Deallocate to reduce peak memory usage

            if variable_entries:  # Assemble variable stiffness components
                D_α = op_nd_chebyshev_derivative(shape, dim=α)
                D_β = D_α if β == α else op_nd_chebyshev_derivative(shape, dim=β)
                for μ, ν, entry in variable_entries:
                    rows = slice(μ * N, (μ + 1) * N)
                    cols = slice(ν * N, (ν + 1) * N)
                    W = op_weighted_gram(entry, shape)
                    A[rows, cols] += scale * (D_α.T @ W @ D_β)
                del D_α, D_β  # Deallocate to reduce peak memory usage

    J = 1.0 / np.prod(scales)
    A *= J
    b *= J
    return A, b


def _assemble_constant_stiffness(
    grams: list[Operator], derivatives: list[Operator], α: int, β: int
) -> Operator:
    operators = []
    for dim, (G, D) in enumerate(zip(grams, derivatives)):
        op = D.T @ G if dim == α else G
        op = op @ D if dim == β else op
        operators.append(op)
    return op_tensor_product(operators)


def assemble_dense_robin_inplace(
    boundary_chebyshev: BoundaryProblem[Vector], A: Operator | None = None, b: Vector | None = None
) -> tuple[Operator, Vector]:
    p = boundary_chebyshev
    m, d, N, shape = p.m, p.d, p.mesh_size, p.mesh_shape

    A = np.zeros((m * N, m * N), dtype=float, order="F") if A is None else A
    b = np.zeros(m * N, dtype=float) if b is None else b

    scales = [2.0 / (interval.stop - interval.start) for interval in p.mesh.intervals]
    component_slices = [slice(μ * N, (μ + 1) * N) for μ in range(m)]

    for dim in range(d):
        shape_face = tuple(N_k for k, N_k in enumerate(shape) if k != dim)
        G_face = op_nd_gram(shape_face)
        J_face = 1.0 / np.prod([scale for k, scale in enumerate(scales) if k != dim])

        for is_left in (True, False):
            bc = p.get_boundary_condition(dim, is_left)
            if not isinstance(bc, RobinBoundaryCondition):
                continue
            T = op_face_trace(shape, dim, is_left)
            M: Operator | None = None  # Mass operator shared by every constant (μ, ν) face entry.

            for μ, rows in enumerate(component_slices):
                # Assemble boundary load
                entry = bc.data[μ]
                if entry is not None:
                    b[rows] += J_face * (T.T @ G_face @ entry)

                # Assemble boundary mass
                for ν, cols in enumerate(component_slices):
                    entry = bc.reaction[μ, ν]
                    if entry is None:
                        continue
                    if (c := get_constant_coefficient(entry)) is not None:
                        if M is None:
                            M = T.T @ G_face @ T
                        A[rows, cols] += J_face * c * M
                    else:
                        W_face = op_weighted_gram(entry, shape_face)
                        A[rows, cols] += J_face * (T.T @ W_face @ T)

    return A, b


def assemble_dirichlet_operators(
    boundary_chebyshev: BoundaryProblem[Vector],
) -> tuple[Operator, Operator, Vector]:
    p = boundary_chebyshev
    m, N, shape = p.m, p.mesh_size, p.mesh_shape
    Z = np.zeros((m * N, m * N), dtype=float)
    P = np.zeros((m * N, m * N), dtype=float)

    for μ in range(m):
        Z_blocks = []
        Q_blocks = []
        for dim, N_k in enumerate(shape):
            bc_left = p.get_boundary_condition(dim, True)
            bc_right = p.get_boundary_condition(dim, False)
            is_left = (
                isinstance(bc_left, DirichletBoundaryCondition) and bc_left.data[μ] is not None
            )
            is_right = (
                isinstance(bc_right, DirichletBoundaryCondition) and bc_right.data[μ] is not None
            )
            Z_blocks.append(op_chebyshev_recombination(N_k, is_left, is_right))
            Q_blocks.append(op_column_projector(N_k, is_left, is_right))

        rows = slice(μ * N, (μ + 1) * N)
        Z[rows, rows] = op_tensor_product(Z_blocks)
        P[rows, rows] = np.eye(N) - op_tensor_product(Q_blocks)

    lift = vec_spectral_lift(boundary_chebyshev)
    return Z, P, lift


def vec_spectral_lift(boundary_chebyshev: BoundaryProblem[Vector]) -> Vector:
    def vec_hermite_coeffs(N: int, is_left: bool) -> Vector:
        if N <= 3:
            raise ValueError
        θ = np.zeros(N)
        sign = -1.0 if is_left else 1.0
        θ[:4] = [0.5, sign * 9.0 / 16.0, 0.0, -sign * 1.0 / 16.0]
        return θ

    def collect_face_data(boundary: BoundaryProblem[Vector], face: BoundaryFace, μ: int) -> Vector:
        bc = boundary.get_boundary_condition(face.dim, face.is_left)
        if not isinstance(bc, DirichletBoundaryCondition):
            raise TypeError

        shape_face = [N_k for k, N_k in enumerate(boundary.mesh_shape) if k != face.dim]
        f_μ = bc.data[μ]
        if f_μ is None:
            raise TypeError
        return f_μ.reshape(shape_face)

    p = boundary_chebyshev
    m, d, N, shape = p.m, p.d, p.mesh_size, p.mesh_shape

    lift = np.zeros(m * N, dtype=complex)
    for μ in range(m):
        faces = tuple(
            BoundaryFace(dim, is_left)
            for dim in range(d)
            for is_left in (True, False)
            if isinstance(bc := p.get_boundary_condition(dim, is_left), DirichletBoundaryCondition)
            and bc.data[μ] is not None
        )
        if not faces:
            continue

        θ_dict = {face: vec_hermite_coeffs(shape[face.dim], face.is_left) for face in faces}
        e_dict = {face: vec_chebyshev_endpoint(shape[face.dim], face.is_left) for face in faces}
        u_μ = np.zeros(shape, dtype=complex)
        h_dict = {face: collect_face_data(p, face, μ) for face in faces}

        for degree in range(1, len(faces) + 1):
            for S in itertools.combinations(faces, degree):
                if len({face.dim for face in S}) != len(S):
                    continue

                h_S = h_dict[S[0]]
                free_dims = [η for η in range(d) if η != S[0].dim]

                for face in S[1:]:
                    η = face.dim
                    idx = free_dims.index(η)
                    e = e_dict[face]
                    h_S = np.tensordot(e, np.atleast_1d(h_S), axes=([0], [idx]))
                    free_dims.pop(idx)

                if h_S.ndim == 0:
                    h_tfi = h_S.reshape((1,) * d)
                else:
                    reshape_dims = [1] * d
                    for η, size in zip(free_dims, h_S.shape):
                        reshape_dims[η] = size
                    h_tfi = h_S.reshape(reshape_dims)

                for face in S:
                    η = face.dim
                    θ = θ_dict[face]
                    reshape_dims = [1] * d
                    reshape_dims[η] = θ.size
                    h_tfi = h_tfi * θ.reshape(reshape_dims)

                u_μ += (-1.0) ** (len(S) - 1) * h_tfi

        for face, expected in h_dict.items():
            η = face.dim
            actual = np.tensordot(e_dict[face], u_μ, axes=([0], [η]))
            error = np.linalg.norm(np.ravel(actual - expected))
            scale = np.linalg.norm(np.ravel(expected))
            if error > 1e-9 * (1.0 + scale):
                raise ValueError

        rows = slice(μ * N, (μ + 1) * N)
        lift[rows] = u_μ.ravel()

    return np.real_if_close(lift)
