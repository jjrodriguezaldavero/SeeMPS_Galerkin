import numpy as np
from collections.abc import Sequence

from seemps.state import MPS
from seemps.operators import MPO, MPOSum

from ...typing import Tensor3, Tensor4

type MPSSlice = Sequence[Tensor3]
type MPOSlice = list[Tensor4]


def _mpo_slice_add(ψ: MPSSlice) -> MPOSlice:
    mpo_cores = []
    for site, core in enumerate(ψ):
        carry_in = [0] if site == 0 else [0, 1]
        carry_out = [0] if site == len(ψ) - 1 else [0, 1]
        χ_left, _, χ_right = core.shape
        mpo_core = np.zeros(
            (len(carry_in) * χ_left, 2, 2, len(carry_out) * χ_right), dtype=core.dtype
        )
        for a, carry in enumerate(carry_in):
            for j_bit in (0, 1):
                for i_bit in (0, 1):
                    total = i_bit + j_bit + carry
                    q_bit = total % 2
                    carry_next = total // 2
                    if carry_next not in carry_out:
                        continue
                    b = carry_out.index(carry_next)
                    mpo_core[
                        a * χ_left : (a + 1) * χ_left, i_bit, j_bit, b * χ_right : (b + 1) * χ_right
                    ] = core[:, q_bit, :]
        mpo_cores.append(mpo_core)
    return mpo_cores


def _mpo_slice_subtract(ψ: MPSSlice, is_upper: bool) -> MPOSlice:
    mpo_cores = []
    for k, core in enumerate(ψ):
        borrow_in = [0] if k == 0 else [0, 1]
        borrow_out = [0] if k == len(ψ) - 1 else [0, 1]
        χ_left, _, χ_right = core.shape

        mpo_core = np.zeros(
            (len(borrow_in) * χ_left, 2, 2, len(borrow_out) * χ_right), dtype=core.dtype
        )
        for a, borrow in enumerate(borrow_in):
            for j_bit in [0, 1]:
                for i_bit in [0, 1]:
                    difference = j_bit - i_bit - borrow if is_upper else i_bit - j_bit - borrow
                    if difference >= 0:
                        q_bit = difference
                        borrow_next = 0
                    else:
                        q_bit = difference + 2
                        borrow_next = 1
                    if borrow_next not in borrow_out:
                        continue
                    b = borrow_out.index(borrow_next)
                    mpo_core[
                        a * χ_left : (a + 1) * χ_left, i_bit, j_bit, b * χ_right : (b + 1) * χ_right
                    ] = core[:, q_bit, :]
        mpo_cores.append(mpo_core)
    return mpo_cores


def _mpo_slice_choose(ψ: MPSSlice) -> MPOSlice:
    mpo_cores = []
    for core in ψ:
        χ_left, _, χ_right = core.shape
        mpo_core = np.zeros((χ_left, 2, 2, χ_right), dtype=core.dtype)
        for j in [0, 1]:
            mpo_core[:, 0, j, :] = core[:, j, :]
        mpo_cores.append(mpo_core)
    return mpo_cores


def _mpo_slice_mult(ψ: MPSSlice) -> MPOSlice:
    if len(ψ) == 1:
        q_0 = ψ[0][:, 0, :]
        q_1 = ψ[0][:, 1, :]
        core = np.empty((q_0.shape[0], 2, 2, q_0.shape[1]), dtype=ψ[0].dtype)
        core[:, 0, 0, :] = q_0
        core[:, 0, 1, :] = 0.5 * q_1
        core[:, 1, 0, :] = q_1
        core[:, 1, 1, :] = q_0
        return [core]

    mpos = [
        MPO(_mpo_slice_subtract(ψ, is_upper=False)),
        MPO(_mpo_slice_subtract(ψ, is_upper=True)),
        MPO(_mpo_slice_add(ψ)),
        MPO(_mpo_slice_choose(ψ)),
    ]
    return MPOSum(mpos=mpos, weights=[0.5, 0.5, 0.5, -0.5]).join()._data


def mpo_nd_mult(ψ: MPS, sites_per_dimension: Sequence[int]) -> MPO:
    if sum(sites_per_dimension) != ψ.size:
        raise ValueError

    start = 0
    mpo_cores: list[Tensor4] = []
    for n_k in sites_per_dimension:
        stop = start + n_k
        mpo_cores.extend(_mpo_slice_mult(ψ[start:stop]))
        start = stop
    return MPO(mpo_cores)
