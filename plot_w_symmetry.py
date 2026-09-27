"""Lift the panel's fourfold symmetry to translations of the w-plane.

The five markers are continuous logarithmic lifts of the four panel markers.
w_4 differs from w_0 by 2*pi*i/lambda, so they coincide only after the
exponential map. Axes use actual u and v coordinates with equal unit lengths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from plot_symmetry import A, B, BLUE, INK, LAMBDA, PAPER, ROSE


P_U = -np.pi * B / 6
P_V = np.pi * A / 4
TAU = np.pi * 1j / (2 * LAMBDA)
W_0 = (np.log(2) + 0.50j) / LAMBDA


def height_w(w: np.ndarray) -> np.ndarray:
    return np.sin(8 * w.imag / A) + 0.5 * np.sin(12 * w.real / B)


def render(output: Path) -> None:
    points = W_0 + np.arange(5) * TAU
    u = np.linspace(points[0].real - 0.52, points[-1].real + 0.43, 1800)
    v = np.linspace(points[0].imag - 0.43, points[-1].imag + 0.48, 1200)
    U, V = np.meshgrid(u, v)
    h = height_w(U + 1j * V)
    base = np.where((h >= 0)[..., None], ROSE, BLUE)
    wash = (0.08 + 0.20 * (1 - np.clip(np.abs(h) / 1.5, 0, 1)))[..., None]
    rgb = base * (1 - wash) + PAPER * wash

    fig, ax = plt.subplots(figsize=(11.4, 8), dpi=250, facecolor="#fffefa")
    ax.imshow(rgb, origin="lower", interpolation="bilinear", aspect="equal",
              extent=(u[0], u[-1], v[0], v[-1]), zorder=1)
    ax.contour(U, V, h, levels=[0], colors=[INK], linewidths=0.43,
               alpha=0.31, zorder=2)

    # Resolve one translation into three horizontal and two vertical periods.
    start, end = points[:2]
    corner = end.real + 1j * start.imag
    ax.plot([start.real, corner.real, end.real],
            [start.imag, corner.imag, end.imag], color=INK,
            linewidth=0.75, alpha=0.68, zorder=3)
    box = {"facecolor": "#fffefa", "edgecolor": "none", "alpha": 0.92, "pad": 2}
    ax.text((start.real + end.real) / 2, start.imag - 0.10,
            r"$3P_u$", ha="center", va="top", fontsize=11, color=INK,
            bbox=box, zorder=8)
    ax.text(end.real + 0.08, (start.imag + end.imag) / 2,
            r"$2P_v$", ha="left", va="center", fontsize=11, color=INK,
            bbox=box, zorder=8)

    for k, point in enumerate(points):
        ax.scatter(point.real, point.imag, s=220, facecolor="#f08f91",
                   edgecolor=INK, linewidth=1.1, zorder=6)
        ax.text(point.real, point.imag, str(k), ha="center", va="center",
                color=INK, fontsize=9, zorder=7)
    for start, end in zip(points[:-1], points[1:]):
        ax.add_patch(FancyArrowPatch(
            (start.real, start.imag), (end.real, end.imag),
            arrowstyle="-|>", mutation_scale=13, shrinkA=14, shrinkB=14,
            linewidth=1.15, color=INK, zorder=5,
        ))

    ax.text(points[-1].real - 0.03, points[-1].imag + 0.20,
            "Same panel point as 0", ha="right", va="bottom", fontsize=10,
            color=INK, bbox=box, zorder=8)
    ax.set_xlim(u[0], u[-1])
    ax.set_ylim(v[0], v[-1])
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel(r"$u=\operatorname{Re}w$", fontsize=12, color=INK, labelpad=8)
    ax.set_ylabel(r"$v=\operatorname{Im}w$", fontsize=12, color=INK, labelpad=8)
    ax.tick_params(colors="#66747a", labelsize=9, length=2.5, width=0.5)
    for spine in ax.spines.values():
        spine.set_color("#84929a")
        spine.set_linewidth(0.65)
    fig.text(0.5, 0.94, r"$\tau=3P_u+2iP_v=\frac{\pi i}{2\lambda}$",
             ha="center", va="center", fontsize=17, color=INK)
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.10, top=0.87)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="#fffefa")
    fig.savefig(output.with_suffix(".svg"), facecolor="#fffefa")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("figures/09-fourfold-w-plane.png"))
    render(parser.parse_args().output)
