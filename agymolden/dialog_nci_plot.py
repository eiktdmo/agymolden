"""
Interactive Non-Covalent Interactions (NCI / RDG) 2D Scatter Plot Dialog.
Plots the Reduced Density Gradient s(r) versus sign(lambda_2)*rho(r).
Interactive identification of Hydrogen bonding, van der Waals interactions, and steric strain.
"""

import os
from typing import Optional
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .workers import run_in_background


class NCIRdgPlotDialog(QtWidgets.QDialog):
    """Interactive NCI / RDG 2D scatter plot dialog."""

    def __init__(self, parent: QtWidgets.QWidget = None, molden_path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Non-Covalent Interactions (NCI) — RDG vs sign(λ₂)ρ Plot")
        self.resize(780, 580)
        self.molden_path = molden_path

        self._init_ui()
        if self.molden_path and os.path.isfile(self.molden_path):
            self._compute_rdg_scatter()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # Header Description
        header = QtWidgets.QLabel(
            "<b>Reduced Density Gradient (RDG) vs. sign(λ₂)ρ Scatter Analysis</b><br>"
            "<span style='color: #64748b; font-size: 11px;'>"
            "Reveals non-covalent interactions in real space: "
            "<b style='color: #3b82f6;'>Blue (sign(λ₂)ρ &lt; -0.01)</b> = Attractive H-Bonds &amp; Halogen bonds; "
            "<b style='color: #22c55e;'>Green (sign(λ₂)ρ ≈ 0.0)</b> = van der Waals dispersion; "
            "<b style='color: #ef4444;'>Red (sign(λ₂)ρ &gt; +0.01)</b> = Non-bonded steric clash &amp; ring strain."
            "</span>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Controls row
        ctrl_row = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #0284c7; font-weight: bold;")
        ctrl_row.addWidget(self.lbl_status)
        ctrl_row.addStretch()

        btn_recalc = QtWidgets.QPushButton("🔄 Recompute NCI")
        btn_recalc.setStyleSheet("background-color: #0d9488; color: #fff; font-weight: bold; padding: 4px 10px;")
        btn_recalc.clicked.connect(self._compute_rdg_scatter)
        ctrl_row.addWidget(btn_recalc)

        layout.addLayout(ctrl_row)

        # Matplotlib Canvas
        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        # Bottom Bar
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)

    def _compute_rdg_scatter(self):
        if not self.molden_path or not os.path.isfile(self.molden_path):
            QtWidgets.QMessageBox.information(self, "No Molden File", "Please load a .molden file first.")
            return

        self.lbl_status.setText("Computing RDG grid & density Hessian in background (PySCF)...")

        def _worker():
            from .volumetric import _load_pyscf_molden, BOHR_TO_ANGSTROM
            from pyscf.dft import numint
            from pyscf.tools import cubegen
            import contextlib, io

            with contextlib.redirect_stdout(io.StringIO()):
                res = _load_pyscf_molden(self.molden_path)
            if not res:
                return None
            mol = res[0]
            mo_coeff = np.asarray(res[2])
            mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else np.zeros(mo_coeff.shape[-1])

            if mo_coeff.ndim == 3:
                dm = np.sum([c @ np.diag(occ) @ c.T for c, occ in zip(mo_coeff, mo_occ)], axis=0)
            else:
                dm = mo_coeff @ np.diag(mo_occ) @ mo_coeff.T

            # Medium grid for responsive scatter plot
            c = cubegen.Cube(mol, nx=32, ny=32, nz=32, margin=2.5)
            coords = c.get_coords()
            ni = numint.NumInt()
            chunk = 1500
            n_pts = len(coords)

            cf = 2.0 * (3.0 * np.pi**2) ** (1.0 / 3.0)
            slr_list = []
            rdg_list = []

            for i in range(0, n_pts, chunk):
                sub = coords[i:i + chunk]
                ao_2 = ni.eval_ao(mol, sub, deriv=2)
                rho_p = np.einsum("pi,ij,pj->p", ao_2[0], dm, ao_2[0])

                # Discard low density (vacuum) and high density (core) points
                mask = (rho_p > 1e-4) & (rho_p < 0.07)
                if not np.any(mask):
                    continue

                sub_n = len(sub)
                gx = 2.0 * np.einsum("pi,ij,pj->p", ao_2[1], dm, ao_2[0])
                gy = 2.0 * np.einsum("pi,ij,pj->p", ao_2[2], dm, ao_2[0])
                gz = 2.0 * np.einsum("pi,ij,pj->p", ao_2[3], dm, ao_2[0])
                norm_g = np.sqrt(gx**2 + gy**2 + gz**2)
                rdg = norm_g / (cf * np.maximum(rho_p, 1e-10) ** (4.0 / 3.0))

                # Hessian
                hess = np.zeros((sub_n, 3, 3), dtype=np.float64)
                hess[:, 0, 0] = 2.0 * (np.einsum("pi,ij,pj->p", ao_2[4], dm, ao_2[0]) + np.einsum("pi,ij,pj->p", ao_2[1], dm, ao_2[1]))
                hess[:, 1, 1] = 2.0 * (np.einsum("pi,ij,pj->p", ao_2[7], dm, ao_2[0]) + np.einsum("pi,ij,pj->p", ao_2[2], dm, ao_2[2]))
                hess[:, 2, 2] = 2.0 * (np.einsum("pi,ij,pj->p", ao_2[9], dm, ao_2[0]) + np.einsum("pi,ij,pj->p", ao_2[3], dm, ao_2[3]))
                h_xy = 2.0 * (np.einsum("pi,ij,pj->p", ao_2[5], dm, ao_2[0]) + np.einsum("pi,ij,pj->p", ao_2[1], dm, ao_2[2]))
                hess[:, 0, 1] = hess[:, 1, 0] = h_xy
                h_xz = 2.0 * (np.einsum("pi,ij,pj->p", ao_2[6], dm, ao_2[0]) + np.einsum("pi,ij,pj->p", ao_2[1], dm, ao_2[3]))
                hess[:, 0, 2] = hess[:, 2, 0] = h_xz
                h_yz = 2.0 * (np.einsum("pi,ij,pj->p", ao_2[8], dm, ao_2[0]) + np.einsum("pi,ij,pj->p", ao_2[2], dm, ao_2[3]))
                hess[:, 1, 2] = hess[:, 2, 1] = h_yz

                eig_v = np.linalg.eigvalsh(hess)
                lam2 = eig_v[:, 1]
                slr = np.sign(lam2) * rho_p

                slr_list.append(slr[mask])
                rdg_list.append(rdg[mask])

            if not slr_list:
                return None
            return np.concatenate(slr_list), np.concatenate(rdg_list)

        def _on_done(data):
            if data is None:
                self.lbl_status.setText("Failed to evaluate RDG scatter points.")
                return
            slr_arr, rdg_arr = data
            self._plot_scatter(slr_arr, rdg_arr)
            self.lbl_status.setText(f"Computed {len(slr_arr):,} non-covalent grid points.")

        def _on_error(err):
            self.lbl_status.setText(f"Error: {err}")
            QtWidgets.QMessageBox.critical(self, "NCI Calculation Error", str(err))

        run_in_background(self, "NCI Scatter Plot", "Evaluating Reduced Density Gradient (PySCF)...", _worker, _on_done, _on_error)

    def _plot_scatter(self, slr: np.ndarray, rdg: np.ndarray):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Filter out extreme RDG spikes for clean visualization
        mask = rdg < 2.0
        x = slr[mask]
        y = rdg[mask]

        # Color mapping: Blue (attractive H-bonds) -> Green (vdW) -> Red (steric strain)
        norm_val = np.clip((x - (-0.03)) / 0.06, 0.0, 1.0)
        cmap = matplotlib.colormaps.get("bwr_r", matplotlib.colormaps["coolwarm"])
        colors = cmap(norm_val)

        sc = ax.scatter(x, y, c=colors, s=3.5, alpha=0.6, edgecolors="none")
        ax.set_xlim(-0.04, 0.04)
        ax.set_ylim(0.0, 1.5)
        ax.set_xlabel("sign(λ₂)ρ (a.u.)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Reduced Density Gradient s(r)", fontsize=11, fontweight="bold")
        ax.set_title("Non-Covalent Interactions (NCI / RDG)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)

        # Annotations
        ax.axvline(0.0, color="#64748b", linestyle=":", linewidth=1.2)
        ax.axhline(0.5, color="#f59e0b", linestyle="--", linewidth=1.0, label="Standard s = 0.5 Cutoff")
        ax.text(-0.035, 1.35, "◀ Attractive (H-Bonds)", color="#2563eb", fontweight="bold", fontsize=10)
        ax.text(-0.005, 1.35, "van der Waals", color="#16a34a", fontweight="bold", fontsize=10)
        ax.text(0.015, 1.35, "Steric Clash ▶", color="#dc2626", fontweight="bold", fontsize=10)
        ax.legend(loc="upper right", framealpha=0.85)

        self.figure.tight_layout()
        self.canvas.draw()
