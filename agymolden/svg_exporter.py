"""
Vector SVG & High-DPI Publication Graphics Exporter for AgyMolden.

Provides publication-grade 2D vector graphic generation (SVG) for molecular graphs,
Z-matrix reference trees, and spectroscopic profiles with:
- True resolution-independent scalable vector rendering (<svg>)
- Standard CPK element colors and customizable bond stroke widths
- Optional atomic index / symbol labels with font embedding
- Transparent or crisp publication-white backgrounds
- Integration with LaTeX, Inkscape, Adobe Illustrator, and Word
"""

import math
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from .elements import ELEMENT_DATA, get_cov_radius

# Publication color palette (clean presentation / journal standards)
ELEMENT_COLORS: Dict[str, str] = {
    "H": "#FFFFFF",
    "C": "#333333",
    "N": "#2563EB",
    "O": "#DC2626",
    "F": "#10B981",
    "P": "#EA580C",
    "S": "#EAB308",
    "Cl": "#059669",
    "Br": "#991B1B",
    "I": "#7C3AED",
    "Si": "#D97706",
    "B": "#F59E0B",
    "Fe": "#B91C1C",
    "Cu": "#C2410C",
    "Zn": "#4B5563",
    "Pt": "#475569",
    "Au": "#EAB308",
}


def export_molecule_svg(
    symbols: List[str],
    coords: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]] = None,
    width: int = 800,
    height: int = 600,
    padding: int = 60,
    show_labels: bool = True,
    show_hydrogens: bool = True,
    dark_mode: bool = False,
    background: Optional[str] = None,
) -> str:
    """
    Export current molecular geometry and connectivity to a standalone, publication-ready SVG.
    Orthogonally projects 3D coordinates onto the 2D SVG canvas with automatic centering,
    scaling, and depth-based z-sorting (painter's algorithm).
    """
    n = len(symbols)
    if n == 0 or len(coords) != n:
        bg = background if background else ("#18181b" if dark_mode else "#ffffff")
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="{bg}"/></svg>'

    # Filter out hydrogens if requested
    visible_indices = [i for i in range(n) if show_hydrogens or symbols[i] != "H"]
    if not visible_indices:
        visible_indices = list(range(n))

    sub_coords = coords[visible_indices]
    sub_symbols = [symbols[i] for i in visible_indices]
    idx_map = {old: new for new, old in enumerate(visible_indices)}

    # Map bonds
    valid_bonds: List[Tuple[int, int]] = []
    if bonds:
        for u, v in bonds:
            if u in idx_map and v in idx_map:
                valid_bonds.append((idx_map[u], idx_map[v]))

    # Center coordinates around COM
    com = np.mean(sub_coords, axis=0)
    shifted = sub_coords - com

    # Project onto XY plane
    xs = shifted[:, 0]
    ys = -shifted[:, 1]  # Invert Y for SVG coordinates
    zs = shifted[:, 2]

    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))

    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)

    usable_w = width - 2 * padding
    usable_h = height - 2 * padding

    scale = min(usable_w / span_x, usable_h / span_y)
    cx_screen = width / 2.0
    cy_screen = height / 2.0

    screen_x = cx_screen + xs * scale
    screen_y = cy_screen + ys * scale

    # Depth sorting (draw back to front)
    depth_order = np.argsort(zs)

    # Theme colors
    if background is not None:
        bg_fill = background
    elif dark_mode:
        bg_fill = "#111827"
    else:
        bg_fill = "#ffffff"

    stroke_bond = "#4b5563" if dark_mode else "#374151"
    text_color = "#f3f4f6" if dark_mode else "#111827"

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <defs>',
        f'    <filter id="atom-shadow" x="-20%" y="-20%" width="140%" height="140%">',
        f'      <feDropShadow dx="1" dy="2" stdDeviation="2" flood-opacity="0.25"/>',
        f'    </filter>',
        f'  </defs>',
    ]

    if bg_fill.lower() != "transparent":
        svg_lines.append(f'  <rect width="100%" height="100%" fill="{bg_fill}"/>')

    # Draw bonds
    svg_lines.append('  <g id="bonds" stroke-linecap="round">')
    for u, v in valid_bonds:
        x1, y1 = screen_x[u], screen_y[u]
        x2, y2 = screen_x[v], screen_y[v]
        b_width = max(2.5, min(6.0, scale * 0.12))
        svg_lines.append(
            f'    <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke_bond}" stroke-width="{b_width:.1f}" opacity="0.85"/>'
        )
    svg_lines.append('  </g>')

    # Draw atoms in depth order
    svg_lines.append('  <g id="atoms" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif">')
    for idx in depth_order:
        sym = sub_symbols[idx]
        x = screen_x[idx]
        y = screen_y[idx]
        cov_r = get_cov_radius(sym)
        # Scaled circle radius in SVG points
        r_circle = max(10.0, min(28.0, cov_r * scale * 0.28))
        color = ELEMENT_COLORS.get(sym, "#9CA3AF")
        border_color = "#1F2937" if (color.upper() == "#FFFFFF" or not dark_mode) else "#E5E7EB"

        svg_lines.append(
            f'    <circle cx="{x:.1f}" cy="{y:.1f}" r="{r_circle:.1f}" fill="{color}" '
            f'stroke="{border_color}" stroke-width="1.8" filter="url(#atom-shadow)"/>'
        )

        if show_labels:
            font_size = max(9, int(r_circle * 0.95))
            txt_fill = "#111827" if (color.upper() in ("#FFFFFF", "#EAB308", "#F59E0B", "#10B981")) else "#FFFFFF"
            svg_lines.append(
                f'    <text x="{x:.1f}" y="{y + font_size * 0.35:.1f}" font-size="{font_size}" '
                f'font-weight="bold" text-anchor="middle" fill="{txt_fill}">{sym}</text>'
            )

    svg_lines.append('  </g>')
    svg_lines.append('</svg>')
    return "\n".join(svg_lines)
