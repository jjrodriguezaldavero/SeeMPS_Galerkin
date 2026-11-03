import numpy as np
from itertools import combinations

from seemps.state import MPS, MPSSum

from ...problem import BoundaryProblem, DirichletBoundaryCondition
from .boundary import mps_chebyshev_endpoint
from .utils import checked_sites_per_dimension, mps_vector_field


def mps_chebyshev_blending(n: int, is_left: bool = True) -> MPS:
    """Coefficients of 1/2 ± (3x-x³)/4, with least-significant bits first."""
    if n < 2:
        raise ValueError("A Dirichlet lift needs at least four coefficients per normal axis")
    sign = -1.0 if is_left else 1.0
    coefficients = np.array([[0.5, 0.0], [sign * 9 / 16, -sign / 16]])
    zero = np.array([1.0, 0.0]).reshape(1, 2, 1)
    return MPS([np.eye(2).reshape(1, 2, 2), coefficients.reshape(2, 2, 1)] + [zero] * (n - 2))


def mps_spectral_lift(boundary: BoundaryProblem[MPS]) -> MPS:
    """Sum face extensions minus edge extensions plus corner extensions, etc."""

    sites = checked_sites_per_dimension(boundary.mesh_shape)
    offsets = np.cumsum([0, *sites]).tolist()
    data = {
        bc.face: bc.data
        for bc in boundary.boundary_conditions
        if isinstance(bc, DirichletBoundaryCondition)
    }
    vectors = {face: mps_chebyshev_blending(sites[face.dim], face.is_left) for face in data}

    components = []
    for μ in range(boundary.m):
        faces = [face for face in data if data[face][μ] is not None]
        terms, weights = [], []
        for degree in range(1, min(boundary.d, len(faces)) + 1):
            for subset in combinations(faces, degree):
                if len({face.dim for face in subset}) != degree:
                    continue
                values = data[subset[0]][μ]
                if not isinstance(values, MPS):
                    raise TypeError("Dirichlet data must be an MPS")
                scalar = values.to_vector().item() if values.physical_dimensions() == [1] else 1
                cores = [] if values.physical_dimensions() == [1] else list(values)
                for k, face in enumerate(subset):
                    start = offsets[face.dim]
                    stop = offsets[face.dim + 1] if k else start
                    rank = cores[start].shape[0] if start < len(cores) else 1
                    intersection = np.eye(rank)
                    for core, e in zip(
                        cores[start:stop], mps_chebyshev_endpoint(sites[face.dim], face.is_left)
                    ):
                        intersection = intersection @ np.tensordot(core, e.ravel(), axes=(1, 0))
                    # First θ ⊗ data; on further axes use θ ⊗ trace(data).
                    rank = min(intersection.shape)
                    identity = np.eye(rank).reshape(rank, 1, rank)
                    extension = [np.kron(v, identity) for v in vectors[face]]
                    if intersection.shape[0] == rank:
                        extension[-1] = extension[-1] @ intersection
                    else:
                        extension[0] = np.tensordot(intersection, extension[0], axes=1)
                    cores[start:stop] = extension
                terms.append(MPS(cores))
                weights.append(scalar * (-1.0) ** (degree - 1))
        components.append(MPSSum(weights, terms).join() if terms else None)
    return mps_vector_field(components, offsets[-1])
