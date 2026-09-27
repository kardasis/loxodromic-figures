# Loxodromic height-field figures

Reproducible plots for Ari Kardasis's `/loxodromy` write-up. The four images show the **same initial field**

\[
h_W(u+iv)=\sin(8v)+\tfrac12\sin(12u)
\]

in the \(w\)-plane, \(z\)-plane, \(p\)-plane, and on the Riemann sphere. The coordinate maps are

\[
\gamma=1.9e^{i\pi/3},\quad \lambda=\log\gamma,\quad
z=e^{\lambda w},\quad z=M(p)=\frac{p+i}{p-i}.
\]

To reproduce the figures:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python plot_fields.py --output-dir figures
```

The program uses the principal logarithm in the inverse map \(w=\operatorname{Log}(z)/\lambda\). It deliberately leaves the branch cut open; the plotted field has **not** been adjusted to close the seam. The sphere uses inverse stereographic projection from the \(z\)-plane, with a visual radial displacement of at most 0.07. This displacement is only a display choice, not part of the field's definition. Both fixed points are excluded from the domain: the oscillations accumulate there without a unique limiting height.

The three plane plots display height at one sixth of the field's numerical value. This changes only the visual vertical scale, not the function or its sign colors. The panel-coordinate view spans \(\operatorname{Re}p\in[-3,3]\) and \(\operatorname{Im}p\in[-6,6]\). To regenerate only the plane plots, run `python plot_fields.py --planes-only --output-dir figures`; use `--p-only` or `--sphere-only` to change just one view.

The site stores exported copies under `public/projects/loxodromic-panel/`. After regenerating, copy the four PNGs there to update the page.

## Construction diagrams

`plot_construction.py` renders three matching perspective diagrams:

1. **Laminating the blank** — walnut (rose) over maple (blue), with their glue plane at `z=0`.
2. **Carving the height field** — the same stock after carving. Where the height is positive, walnut remains; where it is negative, maple is exposed.
3. **From blank to carved surface** — the front half is carved while the back half remains at the original stock height. A vertical riser at the midpoint shows the material removed by carving, without a ramp between the two states.

All three views retain the complete rectangular footprint: no half is omitted. The side walls show the glue line only where material remains above it. On the machining riser, the glue line is visible where the carved surface is below it. Mesh polygons are clipped at `z=0`, so the material boundary is geometric rather than a sampled color threshold.

```sh
python plot_construction.py --output-dir figures/construction
```

This exports `05-laminated-blank`, `06-carved-lamination`, and `07-half-carved-lamination` as 3000 × 2340 PNGs and vector SVGs. The shared perspective, rose/blue palette, and light line weights are coordinated with the mathematical figures. For a quick draft, add `--dpi 140 --resolution 96`.

These are **schematic construction studies**. The walnut-over-maple order is confirmed; the thicknesses remain illustrative, at 0.43 units of walnut and 0.72 units of maple. The test surface uses the same initial `h_W` as the mathematical figures, multiplied by 0.21, with three periods across the blank and two front-to-back. It is a readable periodic specimen, not the final panel's Möbius pullback. The machining boundary in the third view lies at the midpoint of the specimen and is an explanatory comparison, not a claim about the actual toolpath sequence.

Thicknesses and carving depth are parameters:

```sh
python plot_construction.py --upper-thickness 0.43 --lower-thickness 0.72 --height-scale 0.21
```

The stock must enclose the entire height range `±1.5 * height-scale`. The diagrams note that thicknesses and relief are illustrative; replacing those parameters can follow the construction photographs and measurements.

## Fourfold symmetry in panel coordinates

`plot_symmetry.py` draws a top-down close-up of the **seam-corrected** field
around `p=i`, with `Im(p)` along the panel's horizontal axis. Its four marked
bulges are an exact orbit of the order-four Möbius map

\[
T(p)=M^{-1}(iM(p)),\qquad T^4(p)=p.
\]

The four markers have equal height, so they retain the same rose sign color.
Arrows indicate the discrete map from one bulge to the next, **not** a
continuous trajectory or a physical rotation of the panel. Their endpoints
are computed by conjugating a quarter-turn in the `z`-coordinate.

```sh
python plot_symmetry.py --output figures/08-fourfold-panel-closeup.png
```

This repository contains generators for the article's mathematical and
construction **figures**. It does not contain the CNC mesh generator for the
physical panel.
