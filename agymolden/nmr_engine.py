"""Nuclear Magnetic Resonance (NMR) Chemical Shielding Referencing and Spectrum Synthesis Module for AgyMolden.

Computes:
1. Chemical Shift Referencing (delta in ppm) from isotropic shielding (sigma_iso):
     Linear regression: delta = (intercept - sigma_iso) / (-slope)
     or standard direct referencing: delta = sigma_ref - sigma_iso
2. 1H, 13C, 19F, 31P standard reference parameters (TMS, CFCl3, H3PO4) across DFT levels.
3. Continuous synthetic NMR spectra with Lorentzian / Gaussian line broadening.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

# Standard referencing standards: (slope, intercept, nucleus_name)
# delta = (intercept - sigma_iso) / (-slope)  or  delta = intercept - sigma_iso (when slope = -1.0)
NMR_REFERENCE_STANDARDS: Dict[str, Dict[str, Dict[str, float]]] = {
    "1H": {
        "B3LYP/6-31G(d) [Direct TMS]": {"intercept": 31.88, "slope": -1.00},
        "B3LYP/def2-TZVP [Direct TMS]": {"intercept": 31.75, "slope": -1.00},
        "wB97X-D/6-311+G(2d,p) [Direct TMS]": {"intercept": 31.62, "slope": -1.00},
        "PBE0/def2-TZVP [Direct TMS]": {"intercept": 31.55, "slope": -1.00},
        "Empirical Scaled (Tantillo Model)": {"intercept": 31.80, "slope": -1.05},
    },
    "13C": {
        "B3LYP/6-31G(d) [Direct TMS]": {"intercept": 189.67, "slope": -1.00},
        "B3LYP/def2-TZVP [Direct TMS]": {"intercept": 185.50, "slope": -1.00},
        "wB97X-D/6-311+G(2d,p) [Direct TMS]": {"intercept": 188.10, "slope": -1.00},
        "PBE0/def2-TZVP [Direct TMS]": {"intercept": 186.20, "slope": -1.00},
        "Empirical Scaled (Tantillo Model)": {"intercept": 181.24, "slope": -0.96},
    },
    "19F": {
        "B3LYP/6-311+G(2d,p) [CFCl3]": {"intercept": 162.00, "slope": -1.00},
    },
    "31P": {
        "B3LYP/6-311+G(2d,p) [85% H3PO4]": {"intercept": 298.50, "slope": -1.00},
    },
}


class NMRPeak(NamedTuple):
    atom_index: int
    symbol: str
    sigma_iso: float
    anisotropy: float
    chemical_shift_ppm: float


class NMRSpectrum(NamedTuple):
    nucleus: str
    standard_name: str
    peaks: List[NMRPeak]
    ppm_grid: np.ndarray
    intensity_grid: np.ndarray
    fwhm_ppm: float


def compute_chemical_shifts(
    shieldings: List[Dict[str, float]],
    nucleus: str = "1H",
    standard_key: Optional[str] = None,
    custom_intercept: Optional[float] = None,
    custom_slope: Optional[float] = None,
) -> List[NMRPeak]:
    """Computes referenced chemical shifts (ppm) for the selected nucleus."""
    element_sym = "".join(c for c in nucleus if c.isalpha())
    stds = NMR_REFERENCE_STANDARDS.get(nucleus, NMR_REFERENCE_STANDARDS["1H"])

    if custom_intercept is not None:
        intercept = custom_intercept
        slope = custom_slope if custom_slope is not None else -1.00
    elif standard_key and standard_key in stds:
        intercept = stds[standard_key]["intercept"]
        slope = stds[standard_key]["slope"]
    else:
        first_key = next(iter(stds))
        intercept = stds[first_key]["intercept"]
        slope = stds[first_key]["slope"]

    peaks: List[NMRPeak] = []
    for item in shieldings:
        sym = item.get("symbol", "")
        if sym.upper() == element_sym.upper():
            at_idx = int(item.get("atom_index", 0))
            s_iso = float(item.get("sigma_iso", 0.0))
            aniso = float(item.get("anisotropy", 0.0))

            # Referencing formula: delta = (intercept - sigma_iso) / (-slope)
            # or if slope is -1.0: delta = intercept - sigma_iso
            if abs(slope) < 1e-4:
                shift = intercept - s_iso
            else:
                shift = (intercept - s_iso) / (-slope)

            peaks.append(
                NMRPeak(
                    atom_index=at_idx,
                    symbol=sym,
                    sigma_iso=s_iso,
                    anisotropy=aniso,
                    chemical_shift_ppm=float(shift),
                )
            )

    # Sort peaks by chemical shift
    peaks.sort(key=lambda p: p.chemical_shift_ppm)
    return peaks


def synthesize_nmr_spectrum(
    peaks: List[NMRPeak],
    nucleus: str = "1H",
    fwhm_ppm: float = 0.05,
    ppm_range: Optional[Tuple[float, float]] = None,
    num_points: int = 1500,
) -> NMRSpectrum:
    """Synthesizes a continuous 1D NMR spectrum using Lorentzian line broadening."""
    if not peaks:
        # Default empty range
        p_min, p_max = (-1.0, 14.0) if nucleus == "1H" else (-10.0, 230.0)
        grid = np.linspace(p_min, p_max, num_points)
        return NMRSpectrum(nucleus, "None", [], grid, np.zeros_like(grid), fwhm_ppm)

    shifts = [p.chemical_shift_ppm for p in peaks]
    if ppm_range is None:
        if nucleus == "1H":
            p_min = min(-0.5, min(shifts) - 1.0)
            p_max = max(12.0, max(shifts) + 1.0)
        elif nucleus == "13C":
            p_min = min(-5.0, min(shifts) - 5.0)
            p_max = max(215.0, max(shifts) + 5.0)
        else:
            p_min = min(shifts) - 10.0
            p_max = max(shifts) + 10.0
    else:
        p_min, p_max = ppm_range

    ppm_grid = np.linspace(p_min, p_max, num_points)
    intensity = np.zeros_like(ppm_grid)

    gamma = max(0.005, fwhm_ppm / 2.0)
    gamma_sq = gamma ** 2

    # Lorentzian peak summation: L(x) = (1/pi) * (gamma / ((x - x0)^2 + gamma^2))
    for p in peaks:
        diff_sq = (ppm_grid - p.chemical_shift_ppm) ** 2
        lorentz = (1.0 / math.pi) * (gamma / (diff_sq + gamma_sq))
        intensity += lorentz

    # Normalize max intensity to 100.0
    max_i = np.max(intensity) if len(intensity) > 0 else 1.0
    if max_i > 1e-6:
        intensity = (intensity / max_i) * 100.0

    return NMRSpectrum(
        nucleus=nucleus,
        standard_name="Referenced",
        peaks=peaks,
        ppm_grid=ppm_grid,
        intensity_grid=intensity,
        fwhm_ppm=fwhm_ppm,
    )
