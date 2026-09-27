"""Construction diagrams for the loxodromic-panel essay.

The flat glue plane is z=0. Rose denotes walnut and blue denotes maple,
matching the sign colors in plot_fields.py. Each view retains the complete
blank footprint. Layer dimensions are illustrative.

Run: python plot_construction.py --output-dir figures/construction
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from plot_fields import BLUE, INK, PAPER, ROSE, h_w


TEXT = "#34414a"
MUTED = "#76828a"
EDGE = "#687780"
LIGHT = np.array([-0.35, -0.55, 0.76])
LIGHT /= np.linalg.norm(LIGHT)


@dataclass(frozen=True)
class Dimensions:
    width: float = 8.0
    depth: float = 5.2
    upper: float = 0.43
    lower: float = 0.72
    height_scale: float = 0.21


class Camera:
    """One fixed perspective camera shared by all construction stages."""

    def __init__(self, azimuth: float = -58, elevation: float = 32):
        az, el = np.deg2rad([azimuth, elevation])
        self.view = np.array([np.cos(el) * np.cos(az),
                              np.cos(el) * np.sin(az), np.sin(el)])
        self.right = np.array([-np.sin(az), np.cos(az), 0])
        self.up = np.cross(self.view, self.right)
        self.distance = 22.0
        self.target = np.array([0.0, 0.0, -0.12])

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(points) - self.target
        depth = self.distance - p @ self.view
        xy = np.stack((p @ self.right, p @ self.up), axis=-1)
        return xy * (self.distance / depth)[..., None], depth


def height(x: np.ndarray, y: np.ndarray, d: Dimensions) -> np.ndarray:
    """Use the essay's initial field on a readable, rescaled test specimen.

    Across the full blank: three cycles in v (x direction) and two in u
    (y direction). This is a local construction illustration, not a plot of
    the final panel's Möbius pullback or a fabrication drawing.
    """
    u = np.pi * np.asarray(y) / (3 * d.depth)
    v = 3 * np.pi * np.asarray(x) / (4 * d.width)
    return d.height_scale * h_w(u + 1j * v)


def clip_at_glue(vertices: np.ndarray, upper: bool) -> np.ndarray:
    """Clip a polygon exactly at z=0, so material colors meet geometrically."""
    out = []
    sign = 1 if upper else -1
    prev = vertices[-1]
    prev_inside = sign * prev[2] >= 0
    for point in vertices:
        inside = sign * point[2] >= 0
        if inside != prev_inside:
            fraction = -prev[2] / (point[2] - prev[2])
            crossing = prev + fraction * (point - prev)
            crossing[2] = 0
            out.append(crossing)
        if inside:
            out.append(point)
        prev, prev_inside = point, inside
    return np.asarray(out)


class Solid:
    """A single depth-sorted mesh avoids separate surface/wall draw orders."""

    def __init__(self):
        self.faces: list[np.ndarray] = []
        self.colors: list[np.ndarray] = []
        self.normals: list[np.ndarray] = []

    def add(self, vertices, *, normal=None, surface=False):
        vertices = np.asarray(vertices, dtype=float)
        if normal is None:
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            if normal[2] < 0:
                normal = -normal
        normal = np.asarray(normal, dtype=float)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            return
        normal /= norm
        illumination = 0.79 + 0.21 * max(0, normal @ LIGHT)
        for upper, base in ((False, BLUE), (True, ROSE)):
            clipped = clip_at_glue(vertices, upper)
            if len(clipped) < 3:
                continue
            area = sum(np.linalg.norm(np.cross(clipped[k] - clipped[0],
                                                clipped[k + 1] - clipped[0]))
                       for k in range(1, len(clipped) - 1))
            if area < 1e-12:
                continue
            color = np.array(base) * illumination
            if surface:
                # A restrained contour at the material boundary, on the
                # surface itself, is naturally hidden by nearer geometry.
                mean_z = abs(np.mean(clipped[:, 2]))
                line = 0.26 * np.exp(-(mean_z / 0.005) ** 2)
                color = (1 - line) * color + line * INK
            self.faces.append(clipped)
            self.colors.append(color)
            self.normals.append(normal)

    def draw(self, ax, camera: Camera):
        eye = camera.target + camera.distance * camera.view
        visible = [i for i, face in enumerate(self.faces)
                   if self.normals[i] @ (eye - face.mean(axis=0)) > 0]
        projected = [camera.project(self.faces[i]) for i in visible]
        order = np.argsort([np.mean(depth) for _, depth in projected])[::-1]
        polygons = [projected[i][0] for i in order]
        colors = [self.colors[visible[i]] for i in order]
        artist = PolyCollection(polygons, facecolors=colors, edgecolors="face",
                                linewidths=0.10, antialiaseds=False, zorder=3)
        ax.add_collection(artist)


def patches(d: Dimensions, stage: str, resolution: int):
    """Separate patches preserve the actual vertical step in the half-carved view."""
    if stage == "half-carved":
        spans = [(-d.depth / 2, 0, True), (0, d.depth / 2, False)]
    else:
        spans = [(-d.depth / 2, d.depth / 2, stage == "carved")]
    x = np.linspace(-d.width / 2, d.width / 2, resolution + 1)
    for start, end, carved in spans:
        count = max(2, round(resolution * (end - start) / d.width))
        y = np.linspace(start, end, count + 1)
        X, Y = np.meshgrid(x, y)
        Z = height(X, Y, d) if carved else np.full_like(X, d.upper)
        yield x, y, X, Y, Z


def make_solid(d: Dimensions, stage: str, resolution: int) -> Solid:
    solid = Solid()
    for x, y, X, Y, Z in patches(d, stage, resolution):
        # Tessellate flat stock as well as relief for consistent depth sorting.
        for j in range(len(y) - 1):
            for i in range(len(x) - 1):
                p = [(X[j, i], Y[j, i], Z[j, i]),
                     (X[j, i + 1], Y[j, i + 1], Z[j, i + 1]),
                     (X[j + 1, i + 1], Y[j + 1, i + 1], Z[j + 1, i + 1]),
                     (X[j + 1, i], Y[j + 1, i], Z[j + 1, i])]
                solid.add([p[0], p[1], p[2]], surface=True)
                solid.add([p[0], p[2], p[3]], surface=True)

        # Only exterior walls: no fictitious wall through the center of the block.
        for ys, zs, normal in ((y[0], Z[0], [0, -1, 0]),
                               (y[-1], Z[-1], [0, 1, 0])):
            if abs(ys) < 1e-12:
                continue
            for i in range(len(x) - 1):
                solid.add([[x[i], ys, -d.lower], [x[i + 1], ys, -d.lower],
                           [x[i + 1], ys, zs[i + 1]], [x[i], ys, zs[i]]], normal=normal)
        for xs, zs, normal in ((x[0], Z[:, 0], [-1, 0, 0]),
                               (x[-1], Z[:, -1], [1, 0, 0])):
            for j in range(len(y) - 1):
                solid.add([[xs, y[j], -d.lower], [xs, y[j + 1], -d.lower],
                           [xs, y[j + 1], zs[j + 1]], [xs, y[j], zs[j]]], normal=normal)

    if stage == "half-carved":
        # The machining boundary is a vertical riser, not a ramp or missing half.
        z = height(x, 0, d)
        for i in range(len(x) - 1):
            solid.add([[x[i], 0, z[i]], [x[i + 1], 0, z[i + 1]],
                       [x[i + 1], 0, d.upper], [x[i], 0, d.upper]], normal=[0, -1, 0])
    solid.add([[-d.width / 2, -d.depth / 2, -d.lower],
               [d.width / 2, -d.depth / 2, -d.lower],
               [d.width / 2, d.depth / 2, -d.lower],
               [-d.width / 2, d.depth / 2, -d.lower]], normal=[0, 0, -1])
    return solid


def line(ax, camera: Camera, points, *, color=EDGE, lw=0.65, alpha=1, zorder=4):
    xy, _ = camera.project(np.asarray(points))
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=lw, alpha=alpha,
            solid_capstyle="round", solid_joinstyle="round", zorder=zorder)


def outline(ax, camera: Camera, d: Dimensions, stage: str):
    x = np.linspace(-d.width / 2, d.width / 2, 600)
    front = -d.depth / 2
    hx = height(x, front, d) if stage != "blank" else np.full_like(x, d.upper)
    line(ax, camera, np.column_stack((x, np.full_like(x, front), hx)), lw=0.8)
    line(ax, camera, [[-d.width / 2, front, hx[0]], [-d.width / 2, front, -d.lower],
                      [d.width / 2, front, -d.lower], [d.width / 2, d.depth / 2, -d.lower]], lw=0.65)
    line(ax, camera, [[d.width / 2, front, hx[-1]], [d.width / 2, front, -d.lower]], lw=0.6)
    # Only draw the glue line where the remaining material reaches z=0.
    glue = np.zeros_like(x)
    glue[hx < 0] = np.nan
    line(ax, camera, np.column_stack((x, np.full_like(x, front), glue)), color=TEXT, lw=0.9)
    for _, y, _, _, Z in patches(d, stage, 600):
        hy = Z[:, -1]
        line(ax, camera, np.column_stack((np.full_like(y, d.width / 2), y, hy)), lw=0.6)
        side_glue = np.zeros_like(y)
        side_glue[hy < 0] = np.nan
        line(ax, camera, np.column_stack((np.full_like(y, d.width / 2), y, side_glue)),
             color=TEXT, lw=0.7)
    if stage == "half-carved":
        z = height(x, 0, d)
        line(ax, camera, [[x[0], 0, d.upper], [x[-1], 0, d.upper]], lw=0.65)
        line(ax, camera, np.column_stack((x, np.zeros_like(x), z)), lw=0.6)
        line(ax, camera, [[x[-1], 0, z[-1]], [x[-1], 0, d.upper]], lw=0.6)
        # On the riser, glue is visible precisely where carving goes below it.
        glue = np.zeros_like(x)
        glue[z > 0] = np.nan
        line(ax, camera, np.column_stack((x, np.zeros_like(x), glue)), lw=0.7, color=TEXT)


def callout(ax, camera: Camera, target, text, position, *, align="left"):
    xy, _ = camera.project(np.asarray(target))
    ax.annotate(text, xy=xy, xytext=position, textcoords="axes fraction",
                ha=align, va="center", color=TEXT, fontsize=11.5,
                linespacing=1.55,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.65,
                            "shrinkA": 7, "shrinkB": 3},
                zorder=9)
    ax.plot(*xy, marker="o", markersize=2.4, markeredgewidth=0,
            color=TEXT, zorder=10)


def render(ax, camera: Camera, d: Dimensions, *, stage: str, resolution: int):
    ax.set_aspect("equal")
    ax.set_xlim(-6.4, 6.4)
    ax.set_ylim(-4.1, 4.0)
    ax.axis("off")
    make_solid(d, stage, resolution).draw(ax, camera)
    outline(ax, camera, d, stage)

    if stage == "carved":
        xp, yp = d.width / 12, 0.8
        callout(ax, camera, (xp, yp, height(xp, yp, d)),
                "Walnut", (0.69, 0.84))
    else:
        callout(ax, camera, (0.8, 1.4, d.upper), "Walnut", (0.69, 0.84))
    if stage != "blank":
        xn, yn = d.width / 4, -1.1
        callout(ax, camera, (xn, yn, height(xn, yn, d)),
                "Maple", (0.80, 0.22))
    else:
        callout(ax, camera, (d.width / 2, 0.6, -d.lower * 0.55),
                "Maple", (0.80, 0.22))
    callout(ax, camera, (-0.30 * d.width, -d.depth / 2, 0),
            "Glue plane\n" + r"$z=0$", (0.045, 0.64))


def save_figure(out: Path, d: Dimensions, *, stage: str, dpi: int, resolution: int):
    fig = plt.figure(figsize=(10, 7.8), dpi=dpi, facecolor=PAPER)
    ax = fig.add_axes([0.035, 0.14, 0.93, 0.74], facecolor=PAPER)
    render(ax, Camera(), d, stage=stage, resolution=resolution)
    title, subtitle = {
        "blank": ("Laminating the blank", "Walnut over maple, joined at a flat glue plane."),
        "carved": ("Carving the height field", "Carving below the glue plane exposes maple."),
        "half-carved": ("From blank to carved surface", "Front half carved; back half left at full thickness."),
    }[stage]
    fig.text(0.5, 0.935, title, ha="center", fontsize=23, color=TEXT)
    fig.text(0.5, 0.884, subtitle, ha="center", fontsize=12, color=MUTED)
    fig.text(0.5, 0.088, "Rose: walnut     ·     Blue: maple",
             ha="center", fontsize=10.5, color=MUTED)
    fig.text(0.5, 0.046, "Schematic · illustrative thicknesses and relief",
             ha="center", fontsize=9.5, color=MUTED)
    fig.savefig(out.with_suffix(".png"), facecolor=PAPER, dpi=dpi)
    fig.savefig(out.with_suffix(".svg"), facecolor=PAPER)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/construction"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--resolution", type=int, default=180)
    parser.add_argument("--upper-thickness", type=float, default=0.43)
    parser.add_argument("--lower-thickness", type=float, default=0.72)
    parser.add_argument("--height-scale", type=float, default=0.21)
    args = parser.parse_args()
    d = Dimensions(upper=args.upper_thickness, lower=args.lower_thickness,
                   height_scale=args.height_scale)
    if min(d.upper, d.lower) <= 1.5 * d.height_scale or d.height_scale <= 0:
        parser.error("Both stock layers must contain the full ±1.5 × height-scale relief.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavuserif",
                         "svg.fonttype": "path"})
    for stage, name in (("blank", "05-laminated-blank"), ("carved", "06-carved-lamination"),
                         ("half-carved", "07-half-carved-lamination")):
        save_figure(args.output_dir / name, d, stage=stage,
                    dpi=args.dpi, resolution=args.resolution)
    print(f"Wrote three PNGs and three SVGs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
