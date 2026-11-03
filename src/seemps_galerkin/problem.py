from __future__ import annotations

import numpy as np
import math
import dataclasses
from typing import cast
from collections.abc import Iterator, Sequence

from seemps.state import MPS, Strategy
from seemps.analysis.mesh import Mesh
from .mps.factories.transform import mps_chebyshev_transform

from .dense.factories import vec_nd_chebyshev_transform
from .mps.factories import checked_sites_per_dimension
from .mps.defaults import DEFAULT_SVD_STRATEGY
from .typing import Vector


class TensorField[T: Vector | MPS]:
    _data: np.typing.NDArray[np.object_]

    def __init__(self, field_shape: Sequence[int], mesh_shape: Sequence[int]) -> None:
        if any(d <= 0 for d in field_shape):
            raise ValueError
        self.field_shape = tuple(field_shape)
        self.field_ndim = len(self.field_shape)

        if any(N <= 0 for N in mesh_shape):
            raise ValueError
        self.mesh_shape = tuple(mesh_shape)
        self.mesh_size = math.prod(self.mesh_shape)
        self.mesh_ndim = len(self.mesh_shape)
        self.is_quantizable = all(N & (N - 1) == 0 for N in self.mesh_shape)

        self._data = np.full(self.field_shape, None, dtype=object)

    def __getitem__(self, key: int | tuple[int, ...]) -> T | None:
        return cast(T | None, self._data[key])

    @property
    def flat(self) -> Iterator[T | None]:
        for entry in self._data.flat:
            yield cast(T | None, entry)

    def __setitem__(self, key: int | tuple[int, ...], entry: T | None) -> None:
        if isinstance(entry, np.ndarray):
            if entry.ndim != 1 or entry.size != self.mesh_size:
                raise ValueError

        elif isinstance(entry, MPS):
            dims = entry.physical_dimensions()
            if self.mesh_size == 1 and dims == [1]:
                pass
            elif (
                not self.is_quantizable
                or not all(s == 2 for s in dims)
                or math.prod(dims) != self.mesh_size
            ):
                raise ValueError

        elif entry is not None:
            raise TypeError

        self._data[key] = entry

    def chebyshev_transform(self, inverse: bool, strategy: Strategy | None) -> TensorField[T]:
        out = TensorField[T](self.field_shape, self.mesh_shape)
        transformed_mps: dict[int, MPS] = {}

        for idx in np.ndindex(self.field_shape):
            entry = self[idx]
            if entry is None:
                continue

            if isinstance(entry, np.ndarray):
                out[idx] = cast(
                    T, vec_nd_chebyshev_transform(entry, self.mesh_shape, inverse=inverse)
                )
            elif isinstance(entry, MPS):
                if id(entry) in transformed_mps:
                    out[idx] = cast(T, transformed_mps[id(entry)])
                    continue
                if self.mesh_shape == ():
                    out[idx] = entry
                    continue
                sites_per_dimension = [N.bit_length() - 1 for N in self.mesh_shape]
                strategy = DEFAULT_SVD_STRATEGY if strategy is None else strategy
                out[idx] = cast(
                    T,
                    mps_chebyshev_transform(
                        entry, sites_per_dimension, inverse=inverse, strategy=strategy
                    ),
                )
                transformed_mps[id(entry)] = cast(MPS, out[idx])
            else:
                raise TypeError
        return out


def field_to_mps(field: TensorField[Vector], strategy: Strategy) -> TensorField[MPS]:
    sites = checked_sites_per_dimension(field.mesh_shape)
    dimensions = [2] * sum(sites) if sites else [1]
    field_mps = TensorField[MPS](field.field_shape, field.mesh_shape)
    for idx, entry in zip(np.ndindex(field.field_shape), field.flat):
        if entry is not None:
            field_mps[idx] = MPS.from_vector(entry, dimensions, normalize=False, strategy=strategy)
    return field_mps


@dataclasses.dataclass(frozen=True)
class BulkProblem[T: Vector | MPS]:
    mesh: Mesh
    principal: TensorField[T]  # (d, d, m, m)
    reaction: TensorField[T]  # (m, m)
    source: TensorField[T]  # (m,)

    mesh_shape: tuple[int, ...] = dataclasses.field(init=False)
    mesh_size: int = dataclasses.field(init=False)
    d: int = dataclasses.field(init=False)
    m: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        mesh_shape = self.mesh.dimensions
        mesh_size = math.prod(mesh_shape)
        d = self.mesh.dimension

        # Check source
        if self.source.field_ndim != 1 or self.source.mesh_shape != mesh_shape:
            raise ValueError
        m = self.source.field_shape[0]

        # Check reaction
        if self.reaction.field_shape != (m, m) or self.reaction.mesh_shape != mesh_shape:
            raise ValueError

        # Check principal
        if self.principal.field_shape != (d, d, m, m) or self.principal.mesh_shape != mesh_shape:
            raise ValueError

        object.__setattr__(self, "mesh_shape", mesh_shape)
        object.__setattr__(self, "mesh_size", mesh_size)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "m", m)

    def chebyshev_transform(
        self, inverse: bool = False, strategy: Strategy | None = None
    ) -> BulkProblem[T]:
        return BulkProblem(
            mesh=self.mesh,
            principal=self.principal.chebyshev_transform(inverse, strategy),
            reaction=self.reaction.chebyshev_transform(inverse, strategy),
            source=self.source.chebyshev_transform(inverse, strategy),
        )


@dataclasses.dataclass(frozen=True)
class BoundaryFace:
    dim: int
    is_left: bool


@dataclasses.dataclass(frozen=True)
class DirichletBoundaryCondition[T: Vector | MPS]:
    face: BoundaryFace
    data: TensorField[T]  # (m,)

    def __post_init__(self) -> None:
        if self.data.field_ndim != 1:
            raise ValueError

    def chebyshev_transform(
        self, inverse: bool = False, strategy: Strategy | None = None
    ) -> DirichletBoundaryCondition[T]:
        return DirichletBoundaryCondition(
            face=self.face, data=self.data.chebyshev_transform(inverse, strategy)
        )


@dataclasses.dataclass(frozen=True)
class RobinBoundaryCondition[T: Vector | MPS]:
    face: BoundaryFace
    reaction: TensorField[T]  # (m, m)
    data: TensorField[T]  # (m,)

    def __post_init__(self) -> None:
        if self.data.field_ndim != 1:
            raise ValueError
        m = self.data.field_shape[0]

        if self.reaction.field_shape != (m, m) or self.reaction.mesh_shape != self.data.mesh_shape:
            raise ValueError

    def chebyshev_transform(
        self, inverse: bool = False, strategy: Strategy | None = None
    ) -> RobinBoundaryCondition[T]:
        return RobinBoundaryCondition(
            face=self.face,
            reaction=self.reaction.chebyshev_transform(inverse, strategy),
            data=self.data.chebyshev_transform(inverse, strategy),
        )


type BoundaryCondition[T: Vector | MPS] = RobinBoundaryCondition[T] | DirichletBoundaryCondition[T]


@dataclasses.dataclass(frozen=True)
class BoundaryProblem[T: Vector | MPS]:
    mesh: Mesh
    boundary_conditions: Sequence[BoundaryCondition[T]]

    mesh_shape: tuple[int, ...] = dataclasses.field(init=False)
    mesh_size: int = dataclasses.field(init=False)
    d: int = dataclasses.field(init=False)
    m: int = dataclasses.field(init=False)

    _boundary_conditions_by_face: dict[BoundaryFace, BoundaryCondition[T]] = dataclasses.field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        boundary_conditions = tuple(self.boundary_conditions)
        mesh_shape = self.mesh.dimensions
        mesh_size = math.prod(mesh_shape)

        d = self.mesh.dimension
        if len(boundary_conditions) != 2 * d:
            raise ValueError

        m: int | None = None
        boundary_conditions_by_face: dict[BoundaryFace, BoundaryCondition[T]] = {}
        for bc in boundary_conditions:
            if not isinstance(bc, (DirichletBoundaryCondition, RobinBoundaryCondition)):
                raise TypeError

            dim = bc.face.dim
            if not 0 <= dim < d:
                raise IndexError

            if bc.face in boundary_conditions_by_face:
                raise ValueError

            # A boundary field lives on the mesh with the normal dimension removed
            face_mesh_shape = self.mesh.dimensions[:dim] + self.mesh.dimensions[dim + 1 :]
            if bc.data.mesh_shape != face_mesh_shape:
                raise ValueError

            # Infer m from the first boundary condition and enforce it for all remaining conditions
            m_bc = bc.data.field_shape[0]
            if m is None:
                m = m_bc
            elif m_bc != m:
                raise ValueError

            # Populate the lookup dictionary
            boundary_conditions_by_face[bc.face] = bc

        if m is None:
            raise ValueError

        object.__setattr__(self, "boundary_conditions", boundary_conditions)
        object.__setattr__(self, "mesh_shape", mesh_shape)
        object.__setattr__(self, "mesh_size", mesh_size)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "_boundary_conditions_by_face", boundary_conditions_by_face)

    def get_boundary_condition(self, dim: int, is_left: bool) -> BoundaryCondition[T]:
        if not 0 <= dim < self.d:
            raise IndexError
        return self._boundary_conditions_by_face[BoundaryFace(dim=dim, is_left=is_left)]

    def chebyshev_transform(
        self, inverse: bool = False, strategy: Strategy | None = None
    ) -> BoundaryProblem[T]:
        bcs = tuple(bc.chebyshev_transform(inverse, strategy) for bc in self.boundary_conditions)
        return BoundaryProblem(mesh=self.mesh, boundary_conditions=bcs)
