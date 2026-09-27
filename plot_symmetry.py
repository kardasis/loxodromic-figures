"""Render the order-four Möbius symmetry of the seam-corrected panel field.

The horizontal drawing coordinate is Im(p), matching the panel's long axis.
The vertical drawing coordinate is Re(p).  The four-point orbits are exact
images under T = M^{-1} o (z -> i z) o M, not Euclidean quarter-turns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch


PAPER = np.array(to_rgb("#fffefa"))
ROSE = np.array(to_rgb("#ffb7b5"))
BLUE = np.array(to_rgb("#a8d0f7"))
INK = "#35434a"
GAMMA = 1.9 * np.exp(1j * np.pi / 3)
LAMBDA = np.log(GAMMA)
A = (1 / LAMBDA).real
B = (1 / LAMBDA).imag


def m(p: np.ndarray) -> np.ndarray:
    return (p + 1j) / (p - 1j)


def m_inverse(z: np.ndarray) -> np.ndarray:
    return 1j * (z + 1) / (z - 1)


def height(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.log(m(p)) / LAMBDA
        return np.sin(8 * w.imag / A) + 0.5 * np.sin(12 * w.real / B)


def orbit(radius: float, angle: float, phases: np.ndarray) -> np.ndarray:
    return m_inverse(radius * np.exp(1j * (angle + phases)))


def draw_orbit(ax) -> None:
    """Mark four equal-height bulges and the discrete map between them."""
    points = orbit(2.0, 0.50, np.arange(4) * np.pi / 2)
    for k, point in enumerate(points):
        ax.scatter(point.imag, point.real, s=210, facecolor="#f08f91",
                   edgecolor=INK, linewidth=1.1, zorder=6)
        ax.text(point.imag, point.real, str(k), ha="center", va="center",
                color=INK, fontsize=9, weight="medium", zorder=7)

    # These arrows denote point correspondences, not continuous trajectories.
    # Small gaps at their ends keep them visually distinct from an orbit circle.
    bends = (0.06, 0.05, -0.08, -0.13)
    for k, bend in enumerate(bends):
        start, end = points[k], points[(k + 1) % 4]
        arrow = FancyArrowPatch((start.imag, start.real), (end.imag, end.real),
                                arrowstyle="-|>", mutation_scale=13,
                                connectionstyle=f"arc3,rad={bend}",
                                shrinkA=16, shrinkB=16, linewidth=1.25,
                                color=INK, zorder=5)
        ax.add_patch(arrow)


def render(output: Path) -> None:
    horizontal = np.linspace(0.12, 2.40, 1200)  # Im(p), near p=i
    vertical = np.linspace(-1.43, 1.53, 1500)   # Re(p)
    X, Y = np.meshgrid(horizontal, vertical)
    p = Y + 1j * X
    h = height(p)
    distance = np.abs(p - 1j)
    h[distance < 0.045] = np.nan  # The ideal field has no value at either pole.

    magnitude = np.clip(np.abs(h) / 1.5, 0, 1)
    base = np.where((h >= 0)[..., None], ROSE, BLUE)
    wash = (0.08 + 0.20 * (1 - magnitude))[..., None]
    rgb = base * (1 - wash) + PAPER * wash
    rgb[~np.isfinite(h)] = PAPER

    fig, ax = plt.subplots(figsize=(8, 9.5), dpi=220, facecolor="#fffefa")
    ax.set_facecolor("#fffefa")
    ax.imshow(rgb, origin="lower", interpolation="bilinear",
              extent=(horizontal[0], horizontal[-1], vertical[0], vertical[-1]),
              aspect="equal", zorder=1)
    ax.contour(X, Y, np.ma.masked_invalid(h), levels=[0], colors=[INK],
               linewidths=0.43, alpha=0.31, zorder=2)

    draw_orbit(ax)
    ax.scatter(1, 0, s=65, facecolor=INK, edgecolor="#fffefa",
               linewidth=1.1, zorder=8)
    ax.text(1.08, -0.06, r"$p=i$", ha="left", va="top", color=INK,
            fontsize=11, bbox={"facecolor": "#fffefa", "edgecolor": "none",
                             "alpha": 0.86, "pad": 1.8}, zorder=9)

    ax.set_xlim(horizontal[0], horizontal[-1])
    ax.set_ylim(vertical[0], vertical[-1])
    ax.set_xticks([0.5, 1, 1.5, 2])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"$\operatorname{Im}p$", fontsize=11, color=INK, labelpad=8)
    ax.set_ylabel(r"$\operatorname{Re}p$", fontsize=11, color=INK, labelpad=8)
    ax.tick_params(colors="#66747a", labelsize=8, length=2.5, width=0.5)
    for spine in ax.spines.values():
        spine.set_color("#84929a")
        spine.set_linewidth(0.65)

    fig.text(0.5, 0.94, r"$T(p)=M^{-1}(iM(p)),\qquad T^4(p)=p$",
             ha="center", color=INK, fontsize=15)
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.08, top=0.89)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="#fffefa")
    fig.savefig(output.with_suffix(".svg"), facecolor="#fffefa")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("figures/08-fourfold-panel-closeup.png"))
    args = parser.parse_args()
    render(args.output)
