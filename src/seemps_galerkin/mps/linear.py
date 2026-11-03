from __future__ import annotations

import numpy as np
import scipy.linalg
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol

from seemps.state import CanonicalMPS, MPS, Strategy
from seemps.operators import MPO
from seemps.solve import dmrg_solve

from .amen import amen_solve
from ..logger import Logger
from .defaults import DEFAULT_SVD_STRATEGY, DEFAULT_TOLERANCE


class LinearSolver(Protocol):
    def solve(self, ρ_A: MPO, ψ_b: MPS, ψ_guess: MPS | None) -> MPS: ...


@dataclass
class ScipySolver:
    verbose: bool = False

    def solve(self, ρ_A: MPO, ψ_b: MPS, ψ_guess: MPS | None) -> MPS:
        max_dimension = 2**14
        dimension = int(np.prod(ψ_b.physical_dimensions()))
        if dimension > max_dimension:
            raise ValueError(f"Refusing dense solve: matrix dimension is {dimension}.")

        A_dense = ρ_A.to_matrix()
        b_dense = ψ_b.to_vector()

        u = scipy.linalg.solve(A_dense, b_dense)
        ψ = MPS.from_vector(
            u, dimensions=ψ_b.physical_dimensions(), normalize=False, strategy=DEFAULT_SVD_STRATEGY
        )

        Logger.dump("solve.scipy.dimension", dimension)
        Logger.dump("solve.scipy.u.bond_dimensions", ψ.bond_dimensions())
        if self.verbose or Logger.is_active():
            cond = np.linalg.cond(A_dense)
            Logger.dump("solve.scipy.condition_number", cond)
            warning = ""
            if cond > 1e14:
                warning = ", singular"
            elif cond > 1e6:
                warning = ", ill-conditioned"
            level = 0 if self.verbose else 1
            Logger.log(
                f"[SciPy] N={dimension}, κ={cond:.2e}, χ={ψ.max_bond_dimension()}{warning}",
                level=level,
            )

        return ψ


@dataclass
class DMRGSolver:
    maxiter: int = 10
    atol: float = 0.0
    rtol: float = DEFAULT_TOLERANCE
    strategy: Strategy = DEFAULT_SVD_STRATEGY
    method: Literal["cg", "bicg", "bicgstab"] = "cg"

    def solve(self, ρ_A: MPO, ψ_b: MPS, ψ_guess: MPS | None) -> MPS:
        # Start on the left: the upstream right-start branch misaligns its RHS form.
        initial = ψ_b if ψ_guess is None else ψ_guess
        guess = CanonicalMPS([core.copy() for core in initial], center=0, normalize=False)
        ψ, residual = dmrg_solve(
            ρ_A,
            ψ_b,
            guess,
            maxiter=self.maxiter,
            atol=self.atol,
            rtol=self.rtol,
            strategy=self.strategy,
            method=self.method,
        )
        Logger.dump("solve.dmrg.residual", residual)
        Logger.dump("solve.dmrg.u.bond_dimensions", ψ.bond_dimensions())
        Logger.log(f"[DMRG] χ={ψ.max_bond_dimension()}, r={residual:.2e}")
        return ψ


@dataclass
class AMEnSolver:
    maxiter: int = 20
    atol: float = 0.0
    rtol: float = DEFAULT_TOLERANCE
    strategy: Strategy = field(
        default_factory=lambda: DEFAULT_SVD_STRATEGY.replace(
            tolerance=1e-14, max_bond_dimension=128
        )
    )
    method: Literal["cg", "gmres"] = "cg"
    enrichment_rank: int = 2
    residual_rank: int = 8
    preconditioner: Literal["jacobi", "block_jacobi"] | None = "jacobi"

    def solve(self, ρ_A: MPO, ψ_b: MPS, ψ_guess: MPS | None = None) -> MPS:
        ψ = amen_solve(
            ρ_A,
            ψ_b,
            ψ_guess,
            maxiter=self.maxiter,
            atol=self.atol,
            rtol=self.rtol,
            strategy=self.strategy,
            method=self.method,
            enrichment_rank=self.enrichment_rank,
            residual_rank=self.residual_rank,
            preconditioner=self.preconditioner,
        )
        Logger.dump("solve.amen.u.bond_dimensions", ψ.bond_dimensions())
        return ψ


DEFAULT_SOLVER: Final[LinearSolver] = AMEnSolver()


__all__ = [
    "DEFAULT_SOLVER",
    "AMEnSolver",
    "DMRGSolver",
    "LinearSolver",
    "ScipySolver",
]
