"""Notebook diagnostics around the library AMEn solver (no solver implementation)."""

import re
from time import perf_counter, process_time
from typing import Self
from unittest.mock import patch

import numpy as np
from scipy.linalg import norm, qr
from seemps.operators import MPO
from seemps.state import MPS
from seemps_galerkin.logger import Logger
from seemps_galerkin.mps.linear import AMEnSolver


def relative_residual(A: MPO, b: MPS, x: MPS) -> float:
    """Untruncated QR norm of Ax-b, avoiding cancellation of squared norms."""
    residual = (A.apply(x, simplify=False) - b).join()
    carry = residual[0]
    for core in list(residual)[1:]:
        matrix = carry.reshape(-1, carry.shape[-1])
        r = qr(matrix, mode="r", check_finite=False)[0][: min(matrix.shape)]
        carry = np.tensordot(r, core, axes=(-1, 0))
    return float(norm(carry) / b.norm())


class SolverDiagnostics:
    """Time native sweeps/checks and optionally recompute the final residual.

    SeeMPS currently has no callback. Only its logger factory is temporarily
    replaced; all numerical work is delegated to the supplied AMEnSolver.
    Logged residuals have the precision of SeeMPS's printed values. With
    ``verify=True``, ``residual`` is independently recomputed instead. Native
    verification is timed between its preceding sweep log and its result log;
    neither the numerical algorithm nor its stopping criterion is changed.
    """

    def __init__(self, solver: AMEnSolver, verify: bool = True) -> None:
        self.solver = solver
        self.verify = verify
        self.sweeps: list[float] = []
        self.projected_residuals: list[float] = []
        self.time_native_residual = 0.0
        self.time_native_residual_cpu = 0.0
        self.time_final_residual = 0.0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def __call__(self, message: str) -> None:
        now = perf_counter()
        match = re.search(r"AMEn sweep=([\d.]+): projected residual=([\deE+.-]+)", message)
        if match:
            self.sweeps.append(float(match[1]))
            self.projected_residuals.append(float(match[2]))
            self._last_sweep_time = now
            self._last_sweep_cpu = process_time()
        match = re.search(r"AMEn verified residual=([\deE+.-]+)", message)
        if match:
            self.native_residual = float(match[1])
            self.time_native_residual += now - self._last_sweep_time
            self.time_native_residual_cpu += process_time() - self._last_sweep_cpu
        Logger.log(message)

    def solve(self, A: MPO, b: MPS, guess: MPS | None = None) -> MPS:
        self.sweeps.clear()
        self.projected_residuals.clear()
        self.time_native_residual = self.time_final_residual = 0.0
        self.time_native_residual_cpu = 0.0
        self.native_residual = float("nan")
        self.A, self.b = A, b
        tick = perf_counter()
        with patch("seemps_galerkin.mps.amen.amen.make_logger", return_value=self):
            self.x = self.solver.solve(A, b, guess)
        self.time_solver = perf_counter() - tick
        self.time_algorithm = self.time_solver - self.time_native_residual
        self.residual = self.native_residual
        if self.verify:
            tick = perf_counter()
            self.residual = relative_residual(A, b, self.x)
            self.time_final_residual = perf_counter() - tick
        Logger.dump("solve.amen.residual", self.residual)
        Logger.dump("solve.amen.sweeps", self.sweeps.copy())
        Logger.dump("solve.amen.projected_residual_history", self.projected_residuals.copy())
        return self.x
