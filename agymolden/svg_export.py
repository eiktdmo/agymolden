"""
Scalable Vector Graphics (SVG) Publication Exporter for AgyMolden.

Projects 3D molecular coordinates onto the current 2D camera viewport, sorts
elements by depth (painter's algorithm), and renders infinite-resolution XML vector
graphics with radial sphere gradients and cylinder bonds for Adobe Illustrator,
Inkscape, LaTeX/TikZ, and journal submissions.
"""

import math
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from .elements import ELEMENT_DATA
from .svg_exporter import export_molecule_svg, ELEMENT_COLORS


def _color_to_hex(val: Any) -> str:
    """Normalize tuple (r,g,b in 0..1) or hex string to '#rrggbb'."""
    if isinstance(val, (tuple, list)) and len(val) >= 3:
        r = int(min(1.0, max(0.0, val[0])) * 255)
        g = int(min(1.0, max(0.0, val[1])) * 255)
        b = int(min(1.0, max(0.0, val[2])) * 255)
        return f"#{r:02x}{g:02x}{b:02x}"
    if isinstance(val, str) and val.startswith("#"):
        return val
    return "#94a3b8"


def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    h = _color_to_hex(hex_str).lstrip("#")
    if len(h) == 6:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (180, 180, 180)


def _adjust_color(hex_str: str, factor: float) -> str:
    r, g, b = _hex_to_rgb(hex_str)
    r = min(255, max(0, int(r * factor)))
    g = min(255, max(0, int(g * factor)))
    b = min(255, max(0, int(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


def export_molecule_to_svg(
    symbols: List[str],
    coords: np.ndarray,
    bonds: List[Tuple[int, int]],
    view_matrix: Optional[np.ndarray] = None,
    width: int = 800,
    height: int = 800,
    background: str = "transparent",
    ball_scale: float = 0.35,
    bond_width: float = 6.0,
    show_labels: bool = False,
) -> str:
    """
    Export 3D molecular structure to an SVG XML string.
    
    Args:
        symbols: Atomic element symbols.
        coords: (N, 3) Cartesian coordinates in Angstroms.
        bonds: List of (atom_i, atom_j) bond tuples.
        view_matrix: Optional (3, 3) rotation matrix from viewer camera.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        background: "transparent", "white", or hex color.
        ball_scale: Atom sphere radius scaling factor.
        bond_width: Width of bond lines in pixels.
        show_labels: Whether to overlay element symbol labels.
    """
    n = len(symbols)
    if n == 0 or len(coords) != n:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'

    # 1. Center coordinates
    center = np.mean(coords, axis=0)
    c_coords = coords - center

    # 2. Apply camera orientation
    if view_matrix is not None and view_matrix.shape == (3, 3):
        t_coords = np.dot(c_coords, view_matrix.T)
    else:
        t_coords = c_coords.copy()

    # 3. Compute orthographic scale
    xs = t_coords[:, 0]
    ys = t_coords[:, 1]
    zs = t_coords[:, 2]

    span_x = max(1.0, float(np.max(xs) - np.min(xs)))
    span_y = max(1.0, float(np.max(ys) - np.min(ys)))
    max_span = max(span_x, span_y)

    margin = 80
    scale = (min(width, height) - 2 * margin) / max_span

    # Screen projection: X maps to screen X, Y maps to screen Y (inverted)
    screen_x = width / 2.0 + xs * scale
    screen_y = height / 2.0 - ys * scale  # SVG y is top-down

    # 4. Gather render elements: atoms and bonds with depth z
    elements_to_render = []

    # Atom radius lookup
    radii = []
    colors = []
    for s in symbols:
        data = ELEMENT_DATA.get(s, {})
        cov = data.get("cov_radius", 0.77)
        raw_c = data.get("color", (0.58, 0.64, 0.72))
        hex_c = _color_to_hex(raw_c)
        radii.append(max(10.0, cov * 55.0 * ball_scale))
        colors.append(hex_c)

    # Bonds: split each bond into two half-cylinders colored by their respective atom
    for u, v in bonds:
        if u < 0 or u >= n or v < 0 or v >= n:
            continue
        z_mid = (zs[u] + zs[v]) / 2.0 - 0.05  # Bonds sit slightly behind sphere centers
        elements_to_render.append({
            "type": "bond",
            "z": z_mid,
            "u": u,
            "v": v,
            "x1": screen_x[u],
            "y1": screen_y[u],
            "x2": screen_x[v],
            "y2": screen_y[v],
            "color_u": colors[u],
            "color_v": colors[v],
        })

    # Atoms
    for i in range(n):
        elements_to_render.append({
            "type": "atom",
            "z": zs[i],
            "idx": i,
            "symbol": symbols[i],
            "x": screen_x[i],
            "y": screen_y[i],
            "r": radii[i],
            "color": colors[i],
        })

    # 5. Sort by z ascending (Painter's algorithm: draw farthest back first)
    elements_to_render.sort(key=lambda item: item["z"])

    # 6. Build SVG XML
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '  <defs>',
    ]

    # Generate radial gradients for each unique element color for 3D sphere shading
    unique_colors = set(colors)
    for col in unique_colors:
        grad_id = f"sphere_{col.replace('#', '')}"
        hi = _adjust_color(col, 1.45)
        lo = _adjust_color(col, 0.65)
        svg_lines.append(
            f'    <radialGradient id="{grad_id}" cx="35%" cy="35%" r="65%">'
            f'<stop offset="0%" stop-color="{hi}" />'
            f'<stop offset="55%" stop-color="{col}" />'
            f'<stop offset="100%" stop-color="{lo}" />'
            f'</radialGradient>'
        )

    svg_lines.append('  </defs>')

    # Background
    if background and background.lower() != "transparent":
        svg_lines.append(f'  <rect width="100%" height="100%" fill="{background}" />')

    # Render sorted elements
    for el in elements_to_render:
        if el["type"] == "bond":
            x1, y1, x2, y2 = el["x1"], el["y1"], el["x2"], el["y2"]
            xm = (x1 + x2) / 2.0
            ym = (y1 + y2) / 2.0
            col_u = el["color_u"]
            col_v = el["color_v"]

            if col_u == col_v:
                svg_lines.append(
                    f'  <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                    f'stroke="{col_u}" stroke-width="{bond_width:.2f}" stroke-linecap="round" />'
                )
            else:
                svg_lines.append(
                    f'  <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{xm:.2f}" y2="{ym:.2f}" '
                    f'stroke="{col_u}" stroke-width="{bond_width:.2f}" stroke-linecap="round" />'
                )
                svg_lines.append(
                    f'  <line x1="{xm:.2f}" y1="{ym:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                    f'stroke="{col_v}" stroke-width="{bond_width:.2f}" stroke-linecap="round" />'
                )
        elif el["type"] == "atom":
            x, y, r = el["x"], el["y"], el["r"]
            col = el["color"]
            grad_id = f"sphere_{col.replace('#', '')}"
            svg_lines.append(
                f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="url(#{grad_id})" '
                f'stroke="#0f172a" stroke-width="0.75" />'
            )
            if show_labels:
                font_sz = max(9.0, r * 0.85)
                svg_lines.append(
                    f'  <text x="{x:.2f}" y="{y + font_sz*0.35:.2f}" text-anchor="middle" '
                    f'font-family="system-ui, -apple-system, sans-serif" font-weight="bold" '
                    f'font-size="{font_sz:.1f}px" fill="#ffffff" stroke="#0f172a" stroke-width="0.5">'
                    f'{el["symbol"]}</text>'
                )

    svg_lines.append('</svg>')
    return "\n".join(svg_lines)
