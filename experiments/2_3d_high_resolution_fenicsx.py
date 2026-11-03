from __future__ import annotations

import numpy as np
import json
import pickle
import resource
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter


# ====================
# Benchmark parameters
# ====================

WAVENUMBERS = np.pi * np.array([8.0, 9.0, 10.0])
WEIGHTS = np.array([1.0, -0.6, 0.3])

κ = np.max(WAVENUMBERS) ** 2
E = 1.0
ν_P = 0.3
λ_1 = E * ν_P / ((1.0 + ν_P) * (1.0 - 2.0 * ν_P))
λ_2 = E / (2.0 * (1.0 + ν_P))

N_elements_range = {
    2: np.arange(2, 60 + 1, step=2),
    4: np.arange(2, 20 + 1, step=2),
}

N_CORES = 24
TARGET_DOFS_PER_RANK = 100_000
RANK_CHOICES = (1, 2, 4, 8, 12, 24)
RHS_QUADRATURE_DEGREE = 24
KSP_RTOL = 1e-8

WORKER_FLAG = "--worker"
RESULT_PREFIX = "FENICSX_RESULT="
IMAGE = "seemps-fenicsx:0.11.0"
JIT_CACHE_VOLUME = "seemps-fenicsx-0.11-jit-cache"

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENT_DIR.parent
SCRIPT_NAME = Path(__file__).name
DOCKERFILE = ROOT_DIR / "packages/fenicsx.Dockerfile"
DATA_PATH = EXPERIMENT_DIR / "data/2_results_fenicsx.pkl"


# Import FEniCSx only inside the Docker worker.
if WORKER_FLAG in sys.argv:
    import ufl
    from basix import LagrangeVariant
    from basix.ufl import element
    from dolfinx import default_scalar_type, fem, mesh
    from dolfinx.fem import petsc
    from mpi4py import MPI
    from petsc4py import PETSc


def expected_unknowns(N_elements: int, degree: int) -> int:
    # Continuous tensor-product Q_p space on an N^3 structured hexahedral mesh.
    return 3 * (1 + degree * N_elements) ** 3


def choose_mpi_ranks(N_elements: int, degree: int) -> int:
    """Use at most the useful amount of MPI parallelism for this problem size."""
    unknowns = expected_unknowns(N_elements, degree)
    useful_cap = max(1, unknowns // TARGET_DOFS_PER_RANK)
    cap = min(N_CORES, useful_cap)
    return max(rank for rank in RANK_CHOICES if rank <= cap)


def build_elasticity_near_nullspace(V):
    """Rigid-body near-nullspace used by PETSc GAMG for 3D elasticity."""
    gdim = V.mesh.geometry.dim
    coords = V.tabulate_dof_coordinates()[: V.dofmap.index_map.size_local]
    coordinate_vec = PETSc.Vec().createWithArray(
        coords.reshape(-1),
        bsize=gdim,
        comm=V.mesh.comm,
    )
    return PETSc.NullSpace().createRigidBody(coordinate_vec)


def run_fenicsx(N_elements: int, degree: int) -> dict[str, float]:
    comm = MPI.COMM_WORLD
    comm.barrier()
    total_tick = perf_counter()

    # ====================
    # Assemble the problem
    # ====================

    domain = mesh.create_box(
        comm,
        [np.full(3, -1.0), np.full(3, 1.0)],
        [N_elements] * 3,
        cell_type=mesh.CellType.hexahedron,
    )

    options = {"lagrange_variant": LagrangeVariant.gll_warped} if degree == 4 else {}
    V = fem.functionspace(
        domain,
        element("Lagrange", domain.basix_cell(), degree, shape=(3,), **options),
    )

    # Compute the separable modes
    x = ufl.SpatialCoordinate(domain)
    sines = [ufl.sin(k * x_i) for k, x_i in zip(WAVENUMBERS, x)]
    cosines = [ufl.cos(k * x_i) for k, x_i in zip(WAVENUMBERS, x)]
    modes = [
        sines[0] * cosines[1] * cosines[2],
        cosines[0] * sines[1] * cosines[2],
        cosines[0] * cosines[1] * sines[2],
    ]

    # Compute the exact solution
    k = WAVENUMBERS
    amplitudes = (λ_2 * np.dot(k, k) + κ) * np.eye(3) + (λ_1 + λ_2) * np.outer(k, k)
    a = np.linalg.solve(amplitudes, WEIGHTS)

    u_exact = ufl.as_vector([a[μ] * modes[μ] for μ in range(3)])
    f = ufl.as_vector([WEIGHTS[μ] * modes[μ] for μ in range(3)])

    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    ε_u = ufl.sym(ufl.grad(u))
    ε_v = ufl.sym(ufl.grad(v))
    σ_u = λ_1 * ufl.tr(ε_u) * ufl.Identity(3) + 2.0 * λ_2 * ε_u

    # The operator is polynomial. On this affine structured box, degree 2p is
    # sufficient for exact integration of the Q_p mass and elasticity terms.
    # The manufactured trigonometric forcing/error integrals keep the original
    # high quadrature degree so the mathematical benchmark is unchanged there.
    dx_operator = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * degree})
    dx_accurate = ufl.Measure(
        "dx", domain=domain, metadata={"quadrature_degree": RHS_QUADRATURE_DEGREE}
    )

    jit_options = {"cffi_extra_compile_args": ["-O3", "-march=native", "-g0"]}

    lhs = fem.form(
        (ufl.inner(σ_u, ε_v) + κ * ufl.inner(u, v)) * dx_operator,
        jit_options=jit_options,
    )
    rhs = fem.form(ufl.inner(f, v) * dx_accurate, jit_options=jit_options)

    # =============================
    # Assemble the boundary problem
    # =============================

    bcs = []
    face_dim = domain.topology.dim - 1

    for dim in range(3):
        facets = mesh.locate_entities_boundary(
            domain,
            face_dim,
            lambda x, dim=dim: np.isclose(np.abs(x[dim]), 1.0),
        )
        dofs = fem.locate_dofs_topological(V.sub(dim), face_dim, facets)
        bcs.append(
            fem.dirichletbc(
                default_scalar_type(0.0),
                dofs,
                V.sub(dim),
            )
        )

    # ========
    # Assembly
    # ========

    comm.barrier()
    tick = perf_counter()

    A = petsc.assemble_matrix(lhs, bcs=bcs)
    A.assemble()

    b = petsc.assemble_vector(rhs)
    # All Dirichlet values are exactly zero, so apply_lifting would only add
    # zeros. The reverse ghost update is still required after vector assembly.
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    petsc.set_bc(b, bcs)

    time_assembly = float(comm.allreduce(perf_counter() - tick, op=MPI.MAX))

    # ============
    # Solver setup
    # ============

    u_num = fem.Function(V)

    comm.barrier()
    tick = perf_counter()

    # Tell GAMG that this is a 3-component elasticity operator and provide the
    # six rigid-body modes. This is the standard PETSc/DOLFINx setup for
    # smoothed-aggregation AMG on elasticity problems.
    A.setBlockSize(V.dofmap.index_map_bs)
    A.setOption(PETSc.Mat.Option.SPD, True)
    near_nullspace = build_elasticity_near_nullspace(V)
    A.setNearNullSpace(near_nullspace)

    solver = PETSc.KSP().create(comm)
    solver.setOptionsPrefix("bench_")
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.GAMG)
    solver.setTolerances(rtol=KSP_RTOL, atol=0.0, max_it=4000)

    # PETSc/FEniCSx-recommended GAMG ingredients. The larger coarse/process
    # equation limits reduce communication and excessive tiny coarse levels.
    options = PETSc.Options()
    options["bench_pc_gamg_coarse_eq_limit"] = 1000
    options["bench_pc_gamg_process_eq_limit"] = 1000
    options["bench_mg_levels_ksp_type"] = "chebyshev"
    options["bench_mg_levels_pc_type"] = "jacobi"
    options["bench_mg_levels_esteig_ksp_type"] = "cg"
    options["bench_matptap_via"] = "scalable"
    solver.setFromOptions()
    solver.setUp()

    time_setup = float(comm.allreduce(perf_counter() - tick, op=MPI.MAX))

    # =====
    # Solve
    # =====

    comm.barrier()
    tick = perf_counter()

    solver.solve(b, u_num.x.petsc_vec)
    u_num.x.scatter_forward()

    time_solve = float(comm.allreduce(perf_counter() - tick, op=MPI.MAX))

    if solver.getConvergedReason() <= 0:
        raise RuntimeError(f"PETSc failed after {solver.getIterationNumber()} iterations")

    # Setup belongs to the solve cost for the benchmark.
    time = time_assembly + time_setup + time_solve

    # Peak memory of the whole container, before error post-processing
    cgroup_peak = Path("/sys/fs/cgroup/memory.peak")

    if cgroup_peak.exists():
        memory = int(cgroup_peak.read_text().strip()) / 2**30
    else:
        memory_local = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        memory = float(comm.allreduce(memory_local, op=MPI.SUM))

    # =================
    # Compute the errors
    # =================

    error = u_num - u_exact

    error_L2_squared = fem.assemble_scalar(
        fem.form(ufl.inner(error, error) * dx_accurate, jit_options=jit_options)
    )
    norm_L2_squared = fem.assemble_scalar(
        fem.form(ufl.inner(u_exact, u_exact) * dx_accurate, jit_options=jit_options)
    )

    error_L2_squared = comm.allreduce(error_L2_squared, op=MPI.SUM)
    norm_L2_squared = comm.allreduce(norm_L2_squared, op=MPI.SUM)

    error_L2 = np.sqrt(error_L2_squared / norm_L2_squared)

    ε_error = ufl.sym(ufl.grad(error))
    σ_error = λ_1 * ufl.tr(ε_error) * ufl.Identity(3) + 2.0 * λ_2 * ε_error

    ε_exact = ufl.sym(ufl.grad(u_exact))
    σ_exact = λ_1 * ufl.tr(ε_exact) * ufl.Identity(3) + 2.0 * λ_2 * ε_exact

    error_energy_squared = fem.assemble_scalar(
        fem.form(
            (ufl.inner(σ_error, ε_error) + κ * ufl.inner(error, error)) * dx_accurate,
            jit_options=jit_options,
        )
    )

    norm_energy_squared = fem.assemble_scalar(
        fem.form(
            (ufl.inner(σ_exact, ε_exact) + κ * ufl.inner(u_exact, u_exact)) * dx_accurate,
            jit_options=jit_options,
        )
    )

    error_energy_squared = comm.allreduce(
        error_energy_squared,
        op=MPI.SUM,
    )
    norm_energy_squared = comm.allreduce(
        norm_energy_squared,
        op=MPI.SUM,
    )

    error_energy = np.sqrt(error_energy_squared / norm_energy_squared)

    unknowns = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    time_total_worker = float(comm.allreduce(perf_counter() - total_tick, op=MPI.MAX))

    return {
        "N_elements": float(N_elements),
        "N_eff": float(1 + degree * N_elements),
        "unknowns": float(unknowns),
        "mpi_ranks": float(comm.size),
        "iterations": float(solver.getIterationNumber()),
        "time": float(time),
        "time_assembly": float(time_assembly),
        "time_setup": float(time_setup),
        "time_solve": float(time_solve),
        "time_total_worker": float(time_total_worker),
        "memory": float(memory),
        "error_L2": float(error_L2),
        "error_energy": float(error_energy),
    }


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        index = sys.argv.index(WORKER_FLAG)
        N_elements = int(sys.argv[index + 1])
        degree = int(sys.argv[index + 2])

        data = run_fenicsx(N_elements, degree)

        if MPI.COMM_WORLD.rank == 0:
            print(
                f"{RESULT_PREFIX}{json.dumps(data)}",
                flush=True,
            )

    else:
        subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(DOCKERFILE),
                "-t",
                IMAGE,
                str(DOCKERFILE.parent),
            ],
            check=True,
        )

        results_fenicsx = {degree: defaultdict(list) for degree in N_elements_range}

        for degree, elements in N_elements_range.items():
            for N_elements in elements:
                n_ranks = choose_mpi_ranks(N_elements, degree)
                process_tick = perf_counter()
                completed = subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--cpus",
                        str(N_CORES),
                        "-e",
                        "OMP_NUM_THREADS=1",
                        "-e",
                        "OPENBLAS_NUM_THREADS=1",
                        "-e",
                        "MKL_NUM_THREADS=1",
                        "-v",
                        f"{ROOT_DIR}:/work",
                        "-v",
                        f"{JIT_CACHE_VOLUME}:/root/.cache/fenics",
                        "-w",
                        f"/work/{EXPERIMENT_DIR.name}",
                        IMAGE,
                        "mpirun",
                        "-n",
                        str(n_ranks),
                        "python3",
                        SCRIPT_NAME,
                        WORKER_FLAG,
                        str(N_elements),
                        str(degree),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                if completed.returncode != 0:
                    print(completed.stdout, end="")
                    print(
                        completed.stderr,
                        file=sys.stderr,
                        end="",
                    )
                    raise RuntimeError(f"FEniCSx Q{degree}, N_elements={N_elements} failed")

                line = next(
                    line.removeprefix(RESULT_PREFIX)
                    for line in completed.stdout.splitlines()
                    if line.startswith(RESULT_PREFIX)
                )
                data = json.loads(line)
                data["time_end_to_end"] = perf_counter() - process_tick

                print(
                    f"Q{degree}, N_elements={N_elements}, N_eff={int(data['N_eff'])}, "
                    f"ranks={n_ranks}, data={data}",
                    flush=True,
                )

                for key, value in data.items():
                    results_fenicsx[degree][key].append(value)

                DATA_PATH.parent.mkdir(exist_ok=True)

                with DATA_PATH.open("wb") as file:
                    pickle.dump(results_fenicsx, file)
