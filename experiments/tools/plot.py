import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Any, cast
from collections.abc import Iterator


DEFAULT_DPI = 100
DEFAULT_PLOT_STYLE: dict[str, Any] = {
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 16,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
    "lines.markersize": 7,
    "lines.linewidth": 2.2,
    "grid.color": "grey",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "text.usetex": True,
    "font.family": "serif",
}


class AxesArray2D:
    def __init__(self, axes: npt.NDArray[np.object_]) -> None:
        if axes.ndim != 2:
            raise ValueError("Expected a two-dimensional axes array")
        self._axes = axes

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self._axes.shape
        return rows, cols

    def __getitem__(self, key: tuple[int, int]) -> Axes:
        return cast(Axes, self._axes[key])

    @property
    def flat(self) -> Iterator[Axes]:
        return (cast(Axes, ax) for ax in self._axes.flat)


def create_subplots(
    rows: int = 1,
    cols: int = 1,
    width_px: int = 600,
    height_px: int = 400,
) -> tuple[Figure, AxesArray2D]:
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            cols * width_px / DEFAULT_DPI,
            rows * height_px / DEFAULT_DPI,
        ),
        dpi=DEFAULT_DPI,
        squeeze=False,
    )

    return fig, AxesArray2D(axes)


def set_plot_styles(**kwargs: Any) -> None:
    style: dict[str, Any] = DEFAULT_PLOT_STYLE | kwargs

    for key, value in style.items():
        plt.rcParams[key] = value  # type: ignore
