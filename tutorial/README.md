# Tutorials

Hands-on introduction to `seemps_galerkin`, for using the QTT Chebyshev–Galerkin method on your
own problems. The notebooks increase in complexity and are meant to be read in order.

| Notebook | Problem | You learn |
|---|---|---|
| `1_poisson_1d.ipynb` | Poisson equation in 1D | The problem classes, dense and QTT solvers, reading the solution, spectral convergence |
| `2_poisson_2d_boundary_conditions.ipynb` | Poisson equation in 2D | Loading data with tensor cross-interpolation (TCI); Dirichlet, Neumann and Robin conditions with arbitrary data on each face |
| `3_anisotropic_diffusion_2d.ipynb` | Anisotropic, heterogeneous diffusion–reaction | Variable principal tensors, checking against the dense solver, accuracy controls, bond dimensions and cost |
| `4_elasticity_3d.ipynb` | 3D linear elasticity | Vector problems, elasticity tensors, component-wise Dirichlet conditions, tractions |
| `5_parametric.ipynb` | Bar with two parameters | Parameters as coordinates, one solve for all parameter values, working with the compressed solution |

Run them with the environment of the repository (`uv sync` at its root), for example with
`uv run jupyter lab`.

## Quick reference

**Problem.** For a field $u$ with $m$ components on a box in $d$ coordinates,

$$
-\partial_\alpha\big(C_{\mu\alpha\nu\beta}\,\partial_\beta u_\nu\big) + R_{\mu\nu}\,u_\nu = f_\mu,
$$

with a Dirichlet condition $u_\mu=g_\mu$ or a Robin condition
$(C\nabla u)_\mu\cdot n + S_{\mu\nu}u_\nu = g_\mu$ on each face, where $n$ is the outward normal.

| Object | Fields and shapes |
|---|---|
| `Mesh([ChebyshevInterval(a, b, N, endpoints=False), ...])` | the box $[a,b]\times\cdots$; $N=2^n$ for the QTT solver |
| `BulkProblem(mesh, principal, reaction, source)` | `principal[α, β, μ, ν]` $=C_{\mu\alpha\nu\beta}$, shape `(d, d, m, m)`; `reaction` `(m, m)`; `source` `(m,)` |
| `DirichletBoundaryCondition(BoundaryFace(dim, is_left), data)` | `data` `(m,)` on the face mesh |
| `RobinBoundaryCondition(BoundaryFace(dim, is_left), reaction, data)` | $S$ `(m, m)` and $g$ `(m,)` on the face mesh |
| `BoundaryProblem(mesh, conditions)` | exactly one condition per face, $2d$ in total |

**Conventions.**

- Every field is a `TensorField` of values on the Chebyshev–Gauss nodes: NumPy vectors for
  `solve_dense_bvp`, MPS with binary sites for `solve_mps_bvp`. Multidimensional data are ordered
  as `(x, y, ...)`, with the sites of each coordinate consecutive.
- Empty entries are zero. A Neumann or traction-free face is a Robin condition with empty fields.
  In a Dirichlet condition, an empty component is left free, so constraints can be set component
  by component.
- Boundary data live on the face mesh, the mesh without the normal coordinate. The face of a 1D
  domain is a point: a NumPy vector of length one, or an MPS with one site of dimension one.
- Both solvers return the Chebyshev coefficients of the solution. `vec_nd_chebyshev_transform`
  and `mps_chebyshev_transform` (or `mpo_chebyshev_transform` per coordinate, for vector fields)
  give back nodal values. QTT solutions of vector problems have a first site of dimension $m$.
- Functions are loaded as QTTs with SeeMPS's `cross_interpolation`, using
  `BlackBoxLoadMPS(func, mesh, map_matrix=mps_to_mesh_matrix(sites), physical_dimensions=[2] * n)`.
- Parameters are extra mesh coordinates with no derivatives; their faces take empty Robin
  conditions.
- `solve_mps_bvp(..., strategy=..., solver=...)` sets the truncation tolerance and the linear
  solver (`AMEnSolver` by default). `Logger.activate(collect=True)` records the bond dimensions.

## Going further

- **Curved domains.** Smoothly mapped geometries are handled by pulling the problem back to the
  box, which turns the mapping into variable coefficients. `experiments/1_2d_feature_complete.ipynb`
  does this for a tapered cantilever.
- **Hard parametric problems.** `seemps_galerkin.mps.preconditioning.ParametricBasisSolver` wraps
  the linear solver with a Cholesky change of basis that keeps parametric systems well
  conditioned (`experiments/3_3d_parametric.ipynb` and `4_3d_stochastic.ipynb`).
- **Finite-difference reference.** `seemps_galerkin.fd` solves the same problem classes with
  second-order finite differences on uniform grids that include the endpoints, in dense or QTT
  form.
