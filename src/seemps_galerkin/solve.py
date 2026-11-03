from __future__ import annotations

import numpy as np
import scipy.linalg
from math import prod
from time import perf_counter

from seemps.state import MPS, Strategy, simplify
from seemps.operators import mpo_as_mps, simplify_mpo

from .problem import BoundaryProblem, BulkProblem, DirichletBoundaryCondition
from .dense.assembly import (
    assemble_dense_bulk,
    assemble_dense_robin_inplace,
    assemble_dirichlet_operators,
)
from .mps.assembly import (
    assemble_mps_bulk,
    assemble_mps_dirichlet_operators,
    assemble_mps_robin_correction,
)
from .mps.defaults import DEFAULT_SVD_STRATEGY
from .mps.linear import DEFAULT_SOLVER, LinearSolver
from .logger import Logger
from .typing import Vector


def solve_dense_bvp(bulk: BulkProblem[Vector], boundary: BoundaryProblem[Vector]) -> Vector:
    if bulk.mesh_shape != boundary.mesh_shape:
        raise ValueError
    if bulk.m != boundary.m:
        raise ValueError

    bulk_chebyshev = bulk.chebyshev_transform()
    boundary_chebyshev = boundary.chebyshev_transform()

    A, b = assemble_dense_bulk(bulk_chebyshev)
    assemble_dense_robin_inplace(boundary_chebyshev, A, b)

    if not any(
        isinstance(bc, DirichletBoundaryCondition) for bc in boundary_chebyshev.boundary_conditions
    ):
        u = scipy.linalg.solve(
            A,
            b,
            assume_a="pos",
            check_finite=False,
            overwrite_a=True,
            overwrite_b=True,
        )
        return np.real_if_close(u)

    Z, P, lift = assemble_dirichlet_operators(boundary_chebyshev)
    Z_T = Z.T
    b -= A @ lift
    b = Z_T @ b
    AZ = A @ Z
    del A  # Reduce peak memory

    A_dirichlet = np.empty(P.shape, dtype=np.result_type(AZ, P), order="F")
    np.matmul(Z_T, AZ, out=A_dirichlet)
    A_dirichlet += P
    del AZ, P

    u_dirichlet = scipy.linalg.solve(
        A_dirichlet,
        b,
        assume_a="pos",
        lower=False,
        check_finite=False,
        overwrite_a=True,
        overwrite_b=True,
    )
    u = Z @ u_dirichlet
    u += lift
    return np.real_if_close(u)


def solve_mps_bvp(
    bulk: BulkProblem[MPS],
    boundary: BoundaryProblem[MPS],
    guess: MPS | None = None,
    strategy: Strategy = DEFAULT_SVD_STRATEGY,
    solver: LinearSolver = DEFAULT_SOLVER,
) -> MPS:
    start = perf_counter()
    if bulk.mesh_shape != boundary.mesh_shape:
        raise ValueError
    if bulk.m != boundary.m:
        raise ValueError

    bulk_chebyshev = bulk.chebyshev_transform(strategy=strategy)
    boundary_chebyshev = boundary.chebyshev_transform(strategy=strategy)

    # Assemble the unconstrained Galerkin system
    Logger.log("[MPS] assemble")
    ρ_A, ψ_b = assemble_mps_bulk(bulk_chebyshev, strategy)
    ρ_A_robin, ψ_b_robin = assemble_mps_robin_correction(boundary_chebyshev, strategy)
    ρ_A = simplify_mpo(ρ_A + ρ_A_robin, strategy=strategy)
    ψ_b = simplify(ψ_b + ψ_b_robin, strategy=strategy)
    Logger.dump("solve.mps.A.bond_dimensions", ρ_A.bond_dimensions())
    Logger.dump("solve.mps.b.bond_dimensions", ψ_b.bond_dimensions())
    Logger.log(
        f"[MPS] χ(A)={max(ρ_A.bond_dimensions(), default=1)}, χ(b)={ψ_b.max_bond_dimension()}"
    )

    if not any(
        isinstance(bc, DirichletBoundaryCondition) for bc in boundary_chebyshev.boundary_conditions
    ):
        Logger.dump("solve.mps.time_assembly", perf_counter() - start)
        start = perf_counter()
        ψ_u = solver.solve(ρ_A, ψ_b, guess)
        Logger.dump("solve.mps.time_solver", perf_counter() - start)
        Logger.dump("solve.mps.u.bond_dimensions", ψ_u.bond_dimensions())
        return ψ_u

    # Assemble the Dirichlet operators
    ρ_Z, ρ_P, ψ_lift = assemble_mps_dirichlet_operators(boundary_chebyshev, strategy)
    # This is uncompressed to avoid instabilities.
    ρ_A_dirichlet = (ρ_Z.T.conj() @ ρ_A @ ρ_Z).join()
    norm_A = float(np.real_if_close(mpo_as_mps(ρ_A_dirichlet).norm()))
    N = prod(ρ_A_dirichlet.physical_dimensions())
    scale = max(norm_A / N, 1.0)
    ρ_A_dirichlet = simplify_mpo(ρ_A_dirichlet + scale * ρ_P, strategy=strategy)
    ψ_b_dirichlet: MPS = simplify(ψ_b - ρ_A.apply(ψ_lift, strategy=strategy), strategy=strategy)
    ψ_b_dirichlet = ρ_Z.T.conj().apply(ψ_b_dirichlet, strategy=strategy)
    Logger.dump("solve.mps.A_dirichlet.bond_dimensions", ρ_A_dirichlet.bond_dimensions())
    Logger.dump("solve.mps.b_dirichlet.bond_dimensions", ψ_b_dirichlet.bond_dimensions())
    Logger.dump("solve.mps.lift.bond_dimensions", ψ_lift.bond_dimensions())
    Logger.dump("solve.mps.projector_scale", scale)
    Logger.log(
        f"[MPS] χ(A_D)={max(ρ_A_dirichlet.bond_dimensions(), default=1)}, "
        f"χ(b_D)={ψ_b_dirichlet.max_bond_dimension()}, s(P)={scale:.2e}"
    )

    # Solve the constrained Galerkin system
    Logger.dump("solve.mps.time_assembly", perf_counter() - start)
    start = perf_counter()
    ψ_u_dirichlet = solver.solve(ρ_A_dirichlet, ψ_b_dirichlet, guess)
    Logger.dump("solve.mps.time_solver", perf_counter() - start)

    # Undo the Dirichlet transformation
    start = perf_counter()
    ψ_u = ρ_Z.apply(ψ_u_dirichlet, strategy=strategy)
    ψ_u = simplify(ψ_u + ψ_lift, strategy=strategy)
    Logger.dump("solve.mps.time_reconstruction", perf_counter() - start)
    Logger.dump("solve.mps.u.bond_dimensions", ψ_u.bond_dimensions())
    return ψ_u


__all__ = ["solve_dense_bvp", "solve_mps_bvp"]
