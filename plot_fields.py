"""Render one height field in four related coordinate spaces.

Run: python plot_fields.py --output-dir figures
The principal logarithm is used throughout; its cut is deliberately left open.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection


GAMMA = 1.9 * np.exp(1j * np.pi / 3)
LAMBDA = np.log(GAMMA)
ROSE = np.asarray(to_rgb("#ffb7b5"))
BLUE = np.asarray(to_rgb("#a8d0f7"))
INK = np.asarray(to_rgb("#50606a"))
SEAM_INK = "#34414a"
PAPER = "#fffefa"
SPHERE_DISPLACEMENT = 0.07  # Maximum radius change; not part of the height function.
PLANE_HEIGHT_SCALE = 1 / 6  # Visual height only; h_w, h_z, and h_p are unchanged.


def h_w(w: np.ndarray) -> np.ndarray:
    """The initial (not yet seam-corrected) field in the w-plane."""
    return np.sin(8 * w.imag) + 0.5 * np.sin(12 * w.real)


def m(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (p + 1j) / (p - 1j)


def h_z(z: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return h_w(np.log(z) / LAMBDA)


def h_p(p: np.ndarray) -> np.ndarray:
    return h_z(m(p))


def colors(h: np.ndarray, grid: np.ndarray | None = None) -> np.ndarray:
    """Pastel sign colors, with restrained zero contour and coordinate grid."""
    rgb = np.where((h >= 0)[..., None], ROSE, BLUE).astype(float)
    if grid is not None:
        rgb = rgb * (1 - grid[..., None]) + INK * grid[..., None]
    zero = 0.31 * np.exp(-((h / 0.035) ** 2))
    rgb = rgb * (1 - zero[..., None]) + INK * zero[..., None]
    return np.dstack((rgb, np.ones(h.shape)))


def grid_opacity(x: np.ndarray, y: np.ndarray, step: float) -> np.ndarray:
    def distance(t: np.ndarray) -> np.ndarray:
        return np.abs((t + step / 2) % step - step / 2)

    d = np.minimum(distance(x), distance(y))
    return 0.10 * np.exp(-((d / (step / 35)) ** 2))


def plane_figure(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    name: str,
    xlabel: str,
    ylabel: str,
    out: Path,
    *,
    grid_step: float,
    azim: float = -58,
    seam: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    X, Y = np.meshgrid(x, y)
    # Avoid joining distinct banks of the principal-logarithm cut.
    masked = np.ma.masked_invalid(h * PLANE_HEIGHT_SCALE)
    rgba = colors(h, grid_opacity(X, Y, grid_step))
    fig = plt.figure(figsize=(9, 9), dpi=180, facecolor=PAPER)
    ax = fig.add_subplot(111, projection="3d", facecolor=PAPER, computed_zorder=False)
    ax.plot_surface(
        X, Y, masked, facecolors=rgba, rstride=1, cstride=1,
        linewidth=0, antialiased=True, shade=True, zorder=1,
    )
    if seam is not None:
        sx, sy, sh = seam
        # Trace only one bank of the cut; a second edge reads as a dashed pair.
        ax.plot(sx, sy, PLANE_HEIGHT_SCALE * sh + 0.002,
                color=SEAM_INK, linewidth=1.15, alpha=0.88, zorder=5)
    ax.set_box_aspect((np.ptp(x), np.ptp(y), 3 * PLANE_HEIGHT_SCALE), zoom=0.91)
    ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]),
           zlim=(-1.65 * PLANE_HEIGHT_SCALE, 1.65 * PLANE_HEIGHT_SCALE))
    ax.set_xlabel(xlabel, labelpad=7, fontsize=13, color="#394850")
    ax.set_ylabel(ylabel, labelpad=7, fontsize=13, color="#394850")
    ax.set_zticks([])
    ax.set_xticks([round(x[0], 1), 0, round(x[-1], 1)])
    ax.set_yticks([round(y[0], 1), 0, round(y[-1], 1)])
    ax.tick_params(colors="#67737a", labelsize=9, pad=0)
    ax.view_init(elev=32, azim=azim)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#e3e5e5")
        axis.line.set_color("#a3aaad")
    fig.text(0.5, 0.93, name, ha="center", color="#263238", fontsize=20)
    fig.text(
        0.5, 0.075, "rose: positive    ·    blue: negative",
        ha="center", color="#637078", fontsize=11,
    )
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.08, top=0.94)
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


def render_sphere(out: Path) -> None:
    # Inverse stereographic projection: z=cot(theta/2) exp(i*phi).
    # The north and south poles represent infinity and zero, respectively.
    theta = np.linspace(0.025, np.pi - 0.025, 361)
    phi = np.linspace(-np.pi + 0.012, np.pi - 0.012, 641)
    PHI, THETA = np.meshgrid(phi, theta)
    log_z = np.log(1 / np.tan(THETA / 2)) + 1j * PHI
    h = h_w(log_z / LAMBDA)
    radius = 1 + SPHERE_DISPLACEMENT * h / 1.5
    X = radius * np.sin(THETA) * np.cos(PHI)
    Y = radius * np.sin(THETA) * np.sin(PHI)
    Z = radius * np.cos(THETA)
    rgba = colors(h, grid_opacity(PHI, THETA, np.pi / 12))
    # Shade a narrow band on the near bank itself.  Unlike an added wall,
    # this remains on the true radial surface and is occluded correctly.
    bank_shade = 0.48 * np.exp(-((PHI - phi[0]) / 0.047) ** 2)
    rgba[..., :3] = (rgba[..., :3] * (1 - bank_shade[..., None])
                     + np.asarray(to_rgb("#26343d")) * bank_shade[..., None])
    d_theta = np.stack([np.gradient(a, theta, axis=0) for a in (X, Y, Z)], axis=-1)
    d_phi = np.stack([np.gradient(a, phi, axis=1) for a in (X, Y, Z)], axis=-1)
    normals = np.cross(d_theta, d_phi)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    light = np.array([-0.55, -0.40, 0.72])
    light /= np.linalg.norm(light)
    illumination = 0.82 + 0.18 * np.clip(normals @ light, 0, 1)
    rgba[..., :3] *= illumination[..., None]
    fig = plt.figure(figsize=(9, 9), dpi=180, facecolor=PAPER)
    ax = fig.add_subplot(111, projection="3d", facecolor=PAPER, computed_zorder=False)
    ax.plot_surface(
        X, Y, Z, facecolors=rgba, rstride=1, cstride=1,
        linewidth=0, antialiased=False, shade=False, zorder=1,
    )
    # The farther bank gets a fine, pale trace. Fade it smoothly as that edge
    # turns behind the sphere instead of painting a strong line through it.
    edge_visible = (theta > 0.13) & (theta < np.pi - 0.13)
    far_points = np.column_stack((X[edge_visible, -1], Y[edge_visible, -1], Z[edge_visible, -1]))
    far_segments = np.stack((far_points[:-1], far_points[1:]), axis=1)
    view_direction = np.array([
        np.cos(np.deg2rad(18)) * np.cos(np.deg2rad(205)),
        np.cos(np.deg2rad(18)) * np.sin(np.deg2rad(205)),
        np.sin(np.deg2rad(18)),
    ])
    outward = far_points / np.linalg.norm(far_points, axis=1, keepdims=True)
    visibility = np.clip((outward @ view_direction + 0.05) / 0.35, 0, 1)
    visibility = (visibility[:-1] + visibility[1:]) / 2
    far_rgb = np.asarray(to_rgb("#74818a"))
    far_rgba = np.column_stack((np.tile(far_rgb, (len(far_segments), 1)),
                                0.20 + 0.38 * visibility))
    ax.add_collection3d(Line3DCollection(
        far_segments, colors=far_rgba, linewidths=0.34 + 0.34 * visibility, zorder=4,
    ))
    # The near edge remains a single darker line.
    ax.plot(X[edge_visible, 0], Y[edge_visible, 0], Z[edge_visible, 0], color=SEAM_INK,
            linewidth=1.05, alpha=0.92, zorder=5)
    ax.set_box_aspect((1, 1, 1), zoom=1.30)
    ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.12), zlim=(-1.12, 1.12))
    ax.view_init(elev=18, azim=205)
    ax.set_axis_off()
    fig.text(0.5, 0.93, "Riemann sphere", ha="center", color="#263238", fontsize=20)
    fig.text(
        0.5, 0.075, "radial displacement: ±0.07    ·    rose: positive    ·    blue: negative",
        ha="center", color="#637078", fontsize=11,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95)
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--planes-only", action="store_true",
                           help="Regenerate the three plane views without touching the sphere image")
    selection.add_argument("--sphere-only", action="store_true",
                           help="Regenerate the sphere view without touching the plane images")
    selection.add_argument("--p-only", action="store_true",
                           help="Regenerate the p-plane view without touching the other images")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.sphere_only and not args.p_only:
        u = np.linspace(-1.6, 1.6, 321)
        v = np.linspace(-1.6, 1.6, 321)
        U, V = np.meshgrid(u, v)
        plane_figure(u, v, h_w(U + 1j * V), "w-plane", r"$u$", r"$v$",
                     args.output_dir / "01-w-plane.png", grid_step=0.4)

        z_axis = np.linspace(-3.5, 3.5, 401)
        X, Y = np.meshgrid(z_axis, z_axis)
        Z = X + 1j * Y
        hz = h_z(Z)
        hz[(np.abs(Z) < 0.12) | ((X < 0) & (np.abs(Y) < 0.025))] = np.nan
        seam_x = np.linspace(-3.5, -0.13, 500)
        seam_y = np.full_like(seam_x, 0.03)
        seam_z = h_z(seam_x + 1j * seam_y)
        plane_figure(z_axis, z_axis, hz, "z-plane", r"$\mathrm{Re}\,z$", r"$\mathrm{Im}\,z$",
                     args.output_dir / "02-z-plane.png", grid_step=0.7, azim=-65,
                     seam=(seam_x, seam_y, seam_z))

    if not args.sphere_only:
        p_real = np.linspace(-3, 3, 301)
        p_imag = np.linspace(-6, 6, 601)
        X, Y = np.meshgrid(p_real, p_imag)
        P = X + 1j * Y
        hp = h_p(P)
        hp[(np.abs(P - 1j) < 0.09) | (np.abs(P + 1j) < 0.09)
           | ((np.abs(X) < 0.025) & (np.abs(Y) < 1))] = np.nan
        seam_y = np.linspace(-0.90, 0.90, 500)
        seam_x = np.full_like(seam_y, 0.03)
        seam_p = h_p(seam_x + 1j * seam_y)
        plane_figure(p_real, p_imag, hp, "p-plane", r"$\mathrm{Re}\,p$", r"$\mathrm{Im}\,p$",
                     args.output_dir / "03-p-plane.png", grid_step=0.5, azim=-60,
                     seam=(seam_x, seam_y, seam_p))

    if not args.planes_only and not args.p_only:
        render_sphere(args.output_dir / "04-riemann-sphere.png")
    count = ("p-plane figure" if args.p_only else "three plane figures" if args.planes_only
             else "sphere figure" if args.sphere_only else "four figures")
    print(f"Wrote {count} to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
