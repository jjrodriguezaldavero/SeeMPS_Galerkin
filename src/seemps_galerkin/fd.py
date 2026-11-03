import numpy as np
import scipy.linalg
from math import prod, sqrt
from time import perf_counter
from collections.abc import Sequence

from seemps.analysis.mesh import RegularInterval
from seemps.operators import MPO, MPOSum, mpo_as_mps, simplify_mpo
from .mps.factories.utils import diagonal_mpo_from_mps
from seemps.register.transforms import mpo_weighted_shifts
from seemps.state import MPS, MPSSum, Strategy, mps_tensor_product, simplify

from .dense.factories import op_kron_expand, op_tensor_product
from .logger import Logger
from .mps.defaults import DEFAULT_SVD_STRATEGY
from .mps.factories import utils as tt
from .mps.linear import DEFAULT_SOLVER, LinearSolver
from .problem import (
    BoundaryProblem,
    BulkProblem,
    DirichletBoundaryCondition,
    RobinBoundaryCondition,
)
from .typing import Vector


def solve_dense_bvp_fd(bulk: BulkProblem[Vector], boundary: BoundaryProblem[Vector]) -> Vector:
    """Solve on the supplied uniform mesh; return component-major nodal values."""
    if bulk.mesh_shape != boundary.mesh_shape or bulk.m != boundary.m:
        raise ValueError("Bulk and boundary shapes and component counts must agree")
    for mesh in (bulk.mesh, boundary.mesh):
        for axis in mesh.intervals:
            if not isinstance(axis, RegularInterval) or not (
                axis.endpoint_left and axis.endpoint_right
            ):
                raise ValueError("FD requires endpoint-inclusive equispaced intervals")
            if axis.size < 2 or not np.isfinite(axis.step) or axis.step <= 0:
                raise ValueError("FD needs at least two points on each finite increasing interval")
    if any(
        (a.start, a.stop) != (b.start, b.stop)
        for a, b in zip(bulk.mesh.intervals, boundary.mesh.intervals)
    ):
        raise ValueError("Bulk and boundary domains must agree")
    h = [(a.stop - a.start) / (a.size - 1) for a in bulk.mesh.intervals]
    shape, N, m, d = bulk.mesh_shape, bulk.mesh_size, bulk.m, bulk.d
    fields = [bulk.principal, bulk.reaction, bulk.source] + [
        bc.data for bc in boundary.boundary_conditions
    ]
    fields += [
        bc.reaction for bc in boundary.boundary_conditions if isinstance(bc, RobinBoundaryCondition)
    ]
    dtype = np.result_type(float, *[v.dtype for f in fields for v in f.flat if v is not None])
    A, b = np.zeros((m * N, m * N), dtype=dtype), np.zeros((m, N), dtype=dtype)
    endpoints = [[np.eye(1, K, index).ravel() for index in (0, K - 1)] for K in shape]
    weights = [np.ones(K) - 0.5 * (left + right) for K, (left, right) in zip(shape, endpoints)]
    volume = prod(h) * op_tensor_product(weights)
    edges, averages, centered, edge_weights = [], [], [], []
    for dim, K in enumerate(shape):
        I, S = np.eye(K), np.eye(K, k=1)
        L, R = (np.diag(v) for v in endpoints[dim])
        edges.append(op_kron_expand((S - I + R) / h[dim], shape, dim))
        averages.append(op_kron_expand(0.5 * (I + S), shape, dim))
        C = (S - S.T + L @ S - 2 * L + 2 * R - R @ S.T) / (2 * h[dim])
        centered.append(op_kron_expand(C, shape, dim))
        edge_weights.append(
            prod(h)
            * op_tensor_product(
                [np.ones(K) - endpoints[dim][1] if k == dim else w for k, w in enumerate(weights)]
            )
        )
    for μ in range(m):
        rows = slice(μ * N, (μ + 1) * N)
        if (source := bulk.source[μ]) is not None:
            b[μ] = volume * source
        for ν in range(m):
            block = A[rows, ν * N : (ν + 1) * N]
            if (reaction := bulk.reaction[μ, ν]) is not None:
                block += np.diag(volume * reaction)
            for α in range(d):
                for β in range(d):
                    if (coefficient := bulk.principal[α, β, μ, ν]) is None:
                        continue
                    if α == β:
                        w = edge_weights[α] * (averages[α] @ coefficient)
                        left = right = edges[α]
                    else:
                        w, left, right = volume * coefficient, centered[α], centered[β]
                    block += left.T @ (w[:, None] * right)
    mask, lift = np.ones((m, N)), np.zeros((m, N), dtype=dtype)
    for bc in boundary.boundary_conditions:
        dim, index = bc.face.dim, 0 if bc.face.is_left else -1
        face_indices: list[slice | int] = [slice(None)] * d
        face_indices[dim] = index
        nodes = np.arange(N).reshape(shape)[tuple(face_indices)].ravel()
        fw = (
            prod(h[:dim] + h[dim + 1 :])
            * op_tensor_product(
                [
                    endpoints[dim][0 if bc.face.is_left else 1] if k == dim else w
                    for k, w in enumerate(weights)
                ]
            )[nodes]
        )
        for μ in range(m):
            data = bc.data[μ]
            if isinstance(bc, RobinBoundaryCondition):
                if data is not None:
                    b[μ, nodes] += fw * data
                for ν in range(m):
                    if (reaction := bc.reaction[μ, ν]) is not None:
                        A[μ * N + nodes, ν * N + nodes] += fw * reaction
            elif data is not None:
                fixed = mask[μ, nodes] == 0
                if np.linalg.norm((lift[μ, nodes] - data)[fixed]) > 1e-9 * (
                    1 + np.linalg.norm(data)
                ):
                    raise ValueError("Incompatible Dirichlet data at intersecting faces")
                lift[μ, nodes], mask[μ, nodes] = data, 0
    free = np.flatnonzero(mask.ravel())
    u = lift.ravel()
    if free.size:
        rhs = (b.ravel() - A @ u)[free]
        u[free] = scipy.linalg.solve(A[np.ix_(free, free)], rhs, check_finite=False)
    return np.real_if_close(u)


def solve_mps_bvp_fd(
    bulk: BulkProblem[MPS],
    boundary: BoundaryProblem[MPS],
    guess: MPS | None = None,
    strategy: Strategy = DEFAULT_SVD_STRATEGY,
    solver: LinearSolver = DEFAULT_SOLVER,
) -> MPS:
    """Solve directly in nodal QTT form; guesses use the same equispaced mesh."""

    def _mps_sum(terms: Sequence[MPS]) -> MPS:
        # SeeMPS sum compression needs this case for a single physical site.
        if terms[0].size == 1:
            return MPS([np.add.reduce([v[0] for v in terms])])
        return simplify(MPSSum([1.0] * len(terms), terms), strategy=strategy)

    def _mpo_sum(terms: Sequence[MPO]) -> MPO:
        if terms[0].size == 1:
            return MPO([np.add.reduce([v[0] for v in terms])], strategy=strategy)
        return simplify_mpo(MPOSum(terms), strategy=strategy)

    def _embed_face(values: MPS, dim: int, endpoint: MPS) -> MPS:
        if len(sites) == 1:
            return values[0].item() * endpoint
        start = sum(sites[:dim])
        cores = list(values)
        rank = cores[start].shape[0] if start < len(cores) else 1
        inserted = [np.kron(core, np.eye(rank).reshape(rank, 1, rank)) for core in endpoint]
        return MPS(cores[:start] + inserted + cores[start:])

    start = perf_counter()
    if bulk.mesh_shape != boundary.mesh_shape or bulk.m != boundary.m:
        raise ValueError("Bulk and boundary shapes and component counts must agree")
    for mesh in (bulk.mesh, boundary.mesh):
        for axis in mesh.intervals:
            if not isinstance(axis, RegularInterval) or not (
                axis.endpoint_left and axis.endpoint_right
            ):
                raise ValueError("FD requires endpoint-inclusive equispaced intervals")
            if axis.size < 2 or not np.isfinite(axis.step) or axis.step <= 0:
                raise ValueError("FD needs at least two points on each finite increasing interval")
    if any(
        (a.start, a.stop) != (b.start, b.stop)
        for a, b in zip(bulk.mesh.intervals, boundary.mesh.intervals)
    ):
        raise ValueError("Bulk and boundary domains must agree")
    h = [(a.stop - a.start) / (a.size - 1) for a in bulk.mesh.intervals]
    shape, m, d = bulk.mesh_shape, bulk.m, bulk.d
    sites = tt.checked_sites_per_dimension(shape)
    n = sum(sites)
    endpoints = [[tt.mps_delta(k, index) for index in (0, -1)] for k in sites]
    weights = [
        _mps_sum([tt.mps_identity(k), -0.5 * L, -0.5 * R]) for k, (L, R) in zip(sites, endpoints)
    ]
    volume = prod(h) * mps_tensor_product(weights)
    edges, averages, centered, edge_weights = [], [], [], []
    for dim, k in enumerate(sites):
        I = tt.mpo_identity(k)
        S = mpo_weighted_shifts(k, [1.0], [-1], periodic=False)
        L, R = (diagonal_mpo_from_mps(v) for v in endpoints[dim])
        G = (1 / h[dim]) * _mpo_sum([S, -1 * I, R])
        C = (1 / (2 * h[dim])) * _mpo_sum(
            [S, -1 * S.T, (L @ S).join(), -2 * L, 2 * R, -1 * (R @ S.T).join()]
        )
        edges.append(tt.mpo_kron_expand(G, sites, dim))
        centered.append(tt.mpo_kron_expand(C, sites, dim))
        averages.append(tt.mpo_kron_expand(0.5 * _mpo_sum([I, S]), sites, dim))
        edge_weights.append(
            prod(h)
            * mps_tensor_product(
                [
                    _mps_sum([tt.mps_identity(k), -1 * endpoints[dim][1]]) if axis == dim else w
                    for axis, w in enumerate(weights)
                ]
            )
        )
    faces = []
    for bc in boundary.boundary_conditions:
        dim = bc.face.dim
        endpoint = endpoints[dim][0 if bc.face.is_left else 1]
        fw = prod(h[:dim] + h[dim + 1 :]) * mps_tensor_product(
            [endpoint if axis == dim else w for axis, w in enumerate(weights)]
        )
        fm = mps_tensor_product(
            [endpoint if axis == dim else tt.mps_identity(k) for axis, k in enumerate(sites)]
        )
        faces.append((bc, endpoint, fw, fm))
    blocks: list[list[MPO | None]] = []
    loads, masks, lifts = [], [], []
    constrained = False
    for μ in range(m):
        load = [tt.mps_zeros(n)]
        if (source := bulk.source[μ]) is not None:
            load.append(simplify(volume * source, strategy=strategy))
        mask, lift = tt.mps_identity(n), tt.mps_zeros(n)
        for bc, endpoint, fw, fm in faces:
            if (data := bc.data[μ]) is None:
                continue
            value = _embed_face(data, bc.face.dim, endpoint)
            if isinstance(bc, DirichletBoundaryCondition):
                fresh = simplify(mask * value, strategy=strategy)
                error = _mps_sum([simplify(fm * lift, strategy=strategy), -1 * value, fresh])
                if error.norm() > max(1e-9, 100 * strategy.get_tolerance()) * (1 + value.norm()):
                    raise ValueError("Incompatible Dirichlet data at intersecting faces")
                lift = _mps_sum([lift, fresh])
                mask = _mps_sum([mask, -1 * simplify(mask * fm, strategy=strategy)])
                constrained = True
            else:
                load.append(simplify(fw * value, strategy=strategy))
        loads.append(_mps_sum(load))
        masks.append(diagonal_mpo_from_mps(mask))
        lifts.append(lift)
        row: list[MPO | None] = []
        for ν in range(m):
            terms = [tt.mpo_zeros(n)]
            if (reaction := bulk.reaction[μ, ν]) is not None:
                terms.append(diagonal_mpo_from_mps(simplify(volume * reaction, strategy=strategy)))
            for α in range(d):
                for β in range(d):
                    if (coefficient := bulk.principal[α, β, μ, ν]) is None:
                        continue
                    if α == β:
                        w = edge_weights[α] * averages[α].apply(coefficient, strategy=strategy)
                        left = right = edges[α]
                    else:
                        w, left, right = volume * coefficient, centered[α], centered[β]
                    W = diagonal_mpo_from_mps(simplify(w, strategy=strategy))
                    terms.append(tt.mpo_sequential_simplify([right, W, left.T], strategy))
            for bc, endpoint, fw, _ in faces:
                if (
                    isinstance(bc, RobinBoundaryCondition)
                    and (reaction := bc.reaction[μ, ν]) is not None
                ):
                    value = _embed_face(reaction, bc.face.dim, endpoint)
                    terms.append(diagonal_mpo_from_mps(simplify(fw * value, strategy=strategy)))
            row.append(_mpo_sum(terms))
        blocks.append(row)
    A = simplify_mpo(tt.mpo_vector_field(blocks, n), strategy=strategy)
    b: MPS = simplify(tt.mps_vector_field(loads, n), strategy=strategy)
    Q = tt.mpo_vector_field([[masks[μ] if μ == ν else None for ν in range(m)] for μ in range(m)], n)
    lift = simplify(tt.mps_vector_field(lifts, n), strategy=strategy)
    if constrained:
        I = tt.mpo_vector_field(
            [[tt.mpo_identity(n) if μ == ν else None for ν in range(m)] for μ in range(m)], n
        )
        scale = max(mpo_as_mps(A).norm() / sqrt(m * prod(shape)), 1.0)
        b = Q.apply(_mps_sum([b, -1 * A.apply(lift, strategy=strategy)]), strategy=strategy)
        A = _mpo_sum([tt.mpo_sequential_simplify([Q, A, Q], strategy), scale * I, -scale * Q])
        if guess is not None:
            guess = Q.apply(_mps_sum([guess, -1 * lift]), strategy=strategy)
    Logger.dump("solve.mps.A.bond_dimensions", A.bond_dimensions())
    Logger.dump("solve.mps.b.bond_dimensions", b.bond_dimensions())
    Logger.dump("solve.mps.time_assembly", perf_counter() - start)
    start = perf_counter()
    u = solver.solve(A, b, guess)
    Logger.dump("solve.mps.time_solver", perf_counter() - start)
    start = perf_counter()
    if constrained:
        u = _mps_sum([Q.apply(u, strategy=strategy), lift])
    Logger.dump("solve.mps.time_reconstruction", perf_counter() - start)
    Logger.dump("solve.mps.u.bond_dimensions", u.bond_dimensions())
    return u


__all__ = ["solve_dense_bvp_fd", "solve_mps_bvp_fd"]
