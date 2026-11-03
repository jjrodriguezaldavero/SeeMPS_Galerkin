# seemps_galerkin

Code and data for *A Tensor-Network Chebyshev-Galerkin Method for Boundary-Value Problems in
Computational Mechanics* (J. J. Rodríguez-Aldavero, Á. de la Rosa Velasco, J. J. García-Ripoll).

The package implements a QTT Chebyshev-Galerkin method on top of
[SeeMPS](https://github.com/juanjosegarciaripoll/seemps2), together with dense Chebyshev-Galerkin
and QTT finite-difference references. The `experiments/` folder contains the four numerical
experiments of the paper and the conditioning study of its appendix, with the results used
for the figures.

## Tutorials

The `tutorial/` folder contains five notebooks of increasing complexity, from a 1D Poisson
equation to 3D elasticity and parametric problems, which show how to set up and solve your own
problems with the method. Start with `tutorial/README.md`.

## Installation

Requires Python 3.12–3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

This installs the package with SeeMPS 3.0.0 from PyPI, plus the dependencies needed for the
notebooks and the tests.

## Layout

```
src/seemps_galerkin/       QTT and dense Chebyshev-Galerkin assembly and solvers
src/seemps_galerkin/mps/amen/
                           AMEn linear solver
src/seemps_galerkin/mps/preconditioning.py
                           Cholesky change of basis for parametric problems
tests/                     test suite
tutorial/                  tutorial notebooks on using the package
experiments/               notebooks for the four experiments of the paper
experiments/conditioning.py
                           condition numbers of the preconditioning appendix
experiments/data/          stored results of the experiments
experiments/tools/         helpers shared by the notebooks
packages/fenicsx.Dockerfile
                           FEniCSx image for the finite-element benchmark of experiment 2
```

## Experiments

| Notebook | Paper section |
|---|---|
| `1_2d_feature_complete.ipynb` | Manufactured verification |
| `2_3d_high_resolution.ipynb` | Three-dimensional benchmark against QTT finite differences and FEniCSx |
| `3_3d_parametric.ipynb` | Six-parameter cantilever |
| `4_3d_stochastic.ipynb` | High-dimensional intrusive stochastic elasticity, including the comparison with Monte Carlo sampling |
| `conditioning.py` | Conditioning table of the preconditioning appendix |

Run the notebooks from the `experiments/` folder, for example with

```bash
cd experiments
uv run jupyter nbconvert --to notebook --execute --inplace 1_2d_feature_complete.ipynb
```

`conditioning.py` prints its table directly (`uv run python conditioning.py`). Each notebook
loads its stored results from `experiments/data/` when they are present and
writes its figures to `experiments/figures/`. Deleting a results file makes the notebook
recompute it. The finite-element benchmark of experiment 2 runs FEniCSx inside Docker
(`experiments/2_3d_high_resolution_fenicsx.py` builds the image from
`packages/fenicsx.Dockerfile`).

The stored results were computed on an AMD Ryzen 9 5900X (12 cores) with 64 GB of RAM, using
the SeeMPS development branch from which the modules below were taken, in which the AMEn
local-operator contraction is a compiled Cython kernel. Here it is an `opt_einsum`
contraction, so recomputed timings may differ slightly from the stored ones.

## Modules adapted from SeeMPS

Some of the SeeMPS functionality used here is not part of SeeMPS 3.0.0. It is included in this
package, adapted from the SeeMPS branch `feature/jacobi-hermite` (commit `44b9430`):

| Module | Origin in SeeMPS |
|---|---|
| `mps/factories/transform.py` | `seemps.analysis.dct` (Chebyshev transforms) |
| `mps/factories/differentiation.py` | `seemps.analysis.derivatives.chebyshev_differentiation` |
| `mps/amen/amen.py` | `seemps.solve.amen` (AMEn linear solver) |
| `mps/amen/quadratic.py` | `seemps.operators.quadratic` |
| `diagonal_mpo_from_mps` in `mps/factories/utils.py` | `seemps.operators.projectors` |

## License

MIT (see `LICENSE`). The modules adapted from SeeMPS are also MIT-licensed, Copyright (c) 2019
Juan José García Ripoll.
