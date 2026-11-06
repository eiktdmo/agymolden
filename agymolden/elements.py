"""Elements and periodic table database with covalent / VDW radii and CPK coloring."""

from typing import Dict, Tuple, List, Optional

ELEMENT_DATA: Dict[str, Dict] = {
    "X":  {"name": "Dummy",     "atomic_number": 0,  "period": 0, "group": 0,  "mass": 0.0,    "cov_radius": 0.3,  "vdw_radius": 0.5,  "color": (0.6, 0.6, 0.6)},
    "H":  {"name": "Hydrogen",  "atomic_number": 1,  "period": 1, "group": 1,  "mass": 1.008,  "cov_radius": 0.37, "vdw_radius": 1.20, "color": (0.95, 0.95, 0.95)},
    "He": {"name": "Helium",    "atomic_number": 2,  "period": 1, "group": 18, "mass": 4.0026, "cov_radius": 0.32, "vdw_radius": 1.40, "color": (0.85, 1.0, 1.0)},
    "Li": {"name": "Lithium",   "atomic_number": 3,  "period": 2, "group": 1,  "mass": 6.94,   "cov_radius": 1.34, "vdw_radius": 1.82, "color": (0.8, 0.5, 1.0)},
    "Be": {"name": "Beryllium", "atomic_number": 4,  "period": 2, "group": 2,  "mass": 9.0122, "cov_radius": 0.90, "vdw_radius": 1.53, "color": (0.76, 1.0, 0.0)},
    "B":  {"name": "Boron",     "atomic_number": 5,  "period": 2, "group": 13, "mass": 10.81,  "cov_radius": 0.82, "vdw_radius": 1.92, "color": (1.0, 0.7, 0.7)},
    "C":  {"name": "Carbon",    "atomic_number": 6,  "period": 2, "group": 14, "mass": 12.011, "cov_radius": 0.77, "vdw_radius": 1.70, "color": (0.50, 0.50, 0.52)},
    "N":  {"name": "Nitrogen",  "atomic_number": 7,  "period": 2, "group": 15, "mass": 14.007, "cov_radius": 0.75, "vdw_radius": 1.55, "color": (0.2, 0.4, 0.9)},
    "O":  {"name": "Oxygen",    "atomic_number": 8,  "period": 2, "group": 16, "mass": 15.999, "cov_radius": 0.73, "vdw_radius": 1.52, "color": (0.9, 0.15, 0.15)},
    "F":  {"name": "Fluorine",  "atomic_number": 9,  "period": 2, "group": 17, "mass": 18.998, "cov_radius": 0.71, "vdw_radius": 1.47, "color": (0.4, 0.85, 0.3)},
    "Ne": {"name": "Neon",      "atomic_number": 10, "period": 2, "group": 18, "mass": 20.180, "cov_radius": 0.69, "vdw_radius": 1.54, "color": (0.7, 0.89, 0.96)},
    "Na": {"name": "Sodium",    "atomic_number": 11, "period": 3, "group": 1,  "mass": 22.990, "cov_radius": 1.54, "vdw_radius": 2.27, "color": (0.67, 0.36, 0.95)},
    "Mg": {"name": "Magnesium", "atomic_number": 12, "period": 3, "group": 2,  "mass": 24.305, "cov_radius": 1.30, "vdw_radius": 1.73, "color": (0.54, 1.0, 0.0)},
    "Al": {"name": "Aluminum",  "atomic_number": 13, "period": 3, "group": 13, "mass": 26.982, "cov_radius": 1.18, "vdw_radius": 1.84, "color": (0.75, 0.65, 0.65)},
    "Si": {"name": "Silicon",   "atomic_number": 14, "period": 3, "group": 14, "mass": 28.085, "cov_radius": 1.11, "vdw_radius": 2.10, "color": (0.94, 0.78, 0.63)},
    "P":  {"name": "Phosphorus","atomic_number": 15, "period": 3, "group": 15, "mass": 30.974, "cov_radius": 1.06, "vdw_radius": 1.80, "color": (1.0, 0.5, 0.0)},
    "S":  {"name": "Sulfur",    "atomic_number": 16, "period": 3, "group": 16, "mass": 32.06,  "cov_radius": 1.02, "vdw_radius": 1.80, "color": (0.9, 0.85, 0.15)},
    "Cl": {"name": "Chlorine",  "atomic_number": 17, "period": 3, "group": 17, "mass": 35.45,  "cov_radius": 0.99, "vdw_radius": 1.75, "color": (0.12, 0.94, 0.12)},
    "Ar": {"name": "Argon",     "atomic_number": 18, "period": 3, "group": 18, "mass": 39.948, "cov_radius": 0.97, "vdw_radius": 1.88, "color": (0.5, 0.82, 0.89)},
    "K":  {"name": "Potassium", "atomic_number": 19, "period": 4, "group": 1,  "mass": 39.098, "cov_radius": 1.96, "vdw_radius": 2.75, "color": (0.56, 0.25, 0.83)},
    "Ca": {"name": "Calcium",   "atomic_number": 20, "period": 4, "group": 2,  "mass": 40.078, "cov_radius": 1.74, "vdw_radius": 2.31, "color": (0.24, 1.0, 0.0)},
    "Sc": {"name": "Scandium",  "atomic_number": 21, "period": 4, "group": 3,  "mass": 44.956, "cov_radius": 1.44, "vdw_radius": 2.11, "color": (0.9, 0.9, 0.9)},
    "Ti": {"name": "Titanium",  "atomic_number": 22, "period": 4, "group": 4,  "mass": 47.867, "cov_radius": 1.36, "vdw_radius": 2.00, "color": (0.75, 0.76, 0.78)},
    "V":  {"name": "Vanadium",  "atomic_number": 23, "period": 4, "group": 5,  "mass": 50.942, "cov_radius": 1.25, "vdw_radius": 2.00, "color": (0.65, 0.65, 0.67)},
    "Cr": {"name": "Chromium",  "atomic_number": 24, "period": 4, "group": 6,  "mass": 51.996, "cov_radius": 1.27, "vdw_radius": 2.00, "color": (0.54, 0.6, 0.78)},
    "Mn": {"name": "Manganese", "atomic_number": 25, "period": 4, "group": 7,  "mass": 54.938, "cov_radius": 1.39, "vdw_radius": 2.00, "color": (0.61, 0.48, 0.78)},
    "Fe": {"name": "Iron",      "atomic_number": 26, "period": 4, "group": 8,  "mass": 55.845, "cov_radius": 1.25, "vdw_radius": 2.00, "color": (0.88, 0.4, 0.2)},
    "Co": {"name": "Cobalt",    "atomic_number": 27, "period": 4, "group": 9,  "mass": 58.933, "cov_radius": 1.26, "vdw_radius": 2.00, "color": (0.94, 0.56, 0.63)},
    "Ni": {"name": "Nickel",    "atomic_number": 28, "period": 4, "group": 10, "mass": 58.693, "cov_radius": 1.21, "vdw_radius": 1.63, "color": (0.31, 0.82, 0.31)},
    "Cu": {"name": "Copper",    "atomic_number": 29, "period": 4, "group": 11, "mass": 63.546, "cov_radius": 1.38, "vdw_radius": 1.40, "color": (0.78, 0.5, 0.2)},
    "Zn": {"name": "Zinc",      "atomic_number": 30, "period": 4, "group": 12, "mass": 65.38,  "cov_radius": 1.31, "vdw_radius": 1.39, "color": (0.49, 0.5, 0.69)},
    "Ga": {"name": "Gallium",   "atomic_number": 31, "period": 4, "group": 13, "mass": 69.723, "cov_radius": 1.26, "vdw_radius": 1.87, "color": (0.76, 0.56, 0.56)},
    "Ge": {"name": "Germanium", "atomic_number": 32, "period": 4, "group": 14, "mass": 72.630, "cov_radius": 1.22, "vdw_radius": 2.11, "color": (0.4, 0.56, 0.56)},
    "As": {"name": "Arsenic",   "atomic_number": 33, "period": 4, "group": 15, "mass": 74.922, "cov_radius": 1.19, "vdw_radius": 1.85, "color": (0.74, 0.5, 0.89)},
    "Se": {"name": "Selenium",  "atomic_number": 34, "period": 4, "group": 16, "mass": 78.971, "cov_radius": 1.16, "vdw_radius": 1.90, "color": (1.0, 0.63, 0.0)},
    "Br": {"name": "Bromine",   "atomic_number": 35, "period": 4, "group": 17, "mass": 79.904, "cov_radius": 1.14, "vdw_radius": 1.85, "color": (0.65, 0.16, 0.16)},
    "Kr": {"name": "Krypton",   "atomic_number": 36, "period": 4, "group": 18, "mass": 83.798, "cov_radius": 1.10, "vdw_radius": 2.02, "color": (0.36, 0.72, 0.82)},
    "Rb": {"name": "Rubidium",  "atomic_number": 37, "period": 5, "group": 1,  "mass": 85.468, "cov_radius": 2.11, "vdw_radius": 3.03, "color": (0.44, 0.18, 0.69)},
    "Sr": {"name": "Strontium", "atomic_number": 38, "period": 5, "group": 2,  "mass": 87.62,  "cov_radius": 1.92, "vdw_radius": 2.49, "color": (0.0, 1.0, 0.0)},
    "Y":  {"name": "Yttrium",   "atomic_number": 39, "period": 5, "group": 3,  "mass": 88.906, "cov_radius": 1.62, "vdw_radius": 2.00, "color": (0.58, 1.0, 1.0)},
    "Zr": {"name": "Zirconium", "atomic_number": 40, "period": 5, "group": 4,  "mass": 91.224, "cov_radius": 1.48, "vdw_radius": 2.00, "color": (0.0, 1.0, 1.0)},
    "Nb": {"name": "Niobium",   "atomic_number": 41, "period": 5, "group": 5,  "mass": 92.906, "cov_radius": 1.37, "vdw_radius": 2.00, "color": (0.45, 0.76, 0.79)},
    "Mo": {"name": "Molybdenum","atomic_number": 42, "period": 5, "group": 6,  "mass": 95.95,  "cov_radius": 1.45, "vdw_radius": 2.00, "color": (0.33, 0.71, 0.71)},
    "Tc": {"name": "Technetium","atomic_number": 43, "period": 5, "group": 7,  "mass": 98.0,   "cov_radius": 1.56, "vdw_radius": 2.00, "color": (0.23, 0.62, 0.62)},
    "Ru": {"name": "Ruthenium", "atomic_number": 44, "period": 5, "group": 8,  "mass": 101.07, "cov_radius": 1.26, "vdw_radius": 2.00, "color": (0.14, 0.56, 0.56)},
    "Rh": {"name": "Rhodium",   "atomic_number": 45, "period": 5, "group": 9,  "mass": 102.91, "cov_radius": 1.35, "vdw_radius": 2.00, "color": (0.04, 0.49, 0.55)},
    "Pd": {"name": "Palladium", "atomic_number": 46, "period": 5, "group": 10, "mass": 106.42, "cov_radius": 1.28, "vdw_radius": 1.63, "color": (0.0, 0.41, 0.52)},
    "Ag": {"name": "Silver",    "atomic_number": 47, "period": 5, "group": 11, "mass": 107.87, "cov_radius": 1.53, "vdw_radius": 1.72, "color": (0.75, 0.75, 0.75)},
    "Cd": {"name": "Cadmium",   "atomic_number": 48, "period": 5, "group": 12, "mass": 112.41, "cov_radius": 1.48, "vdw_radius": 1.58, "color": (1.0, 0.85, 0.56)},
    "In": {"name": "Indium",    "atomic_number": 49, "period": 5, "group": 13, "mass": 114.82, "cov_radius": 1.44, "vdw_radius": 1.93, "color": (0.65, 0.46, 0.45)},
    "Sn": {"name": "Tin",       "atomic_number": 50, "period": 5, "group": 14, "mass": 118.71, "cov_radius": 1.41, "vdw_radius": 2.17, "color": (0.4, 0.5, 0.5)},
    "Sb": {"name": "Antimony",  "atomic_number": 51, "period": 5, "group": 15, "mass": 121.76, "cov_radius": 1.38, "vdw_radius": 2.00, "color": (0.62, 0.39, 0.71)},
    "Te": {"name": "Tellurium", "atomic_number": 52, "period": 5, "group": 16, "mass": 127.60, "cov_radius": 1.35, "vdw_radius": 2.06, "color": (0.83, 0.48, 0.0)},
    "I":  {"name": "Iodine",    "atomic_number": 53, "period": 5, "group": 17, "mass": 126.90, "cov_radius": 1.33, "vdw_radius": 1.98, "color": (0.58, 0.0, 0.58)},
    "Xe": {"name": "Xenon",     "atomic_number": 54, "period": 5, "group": 18, "mass": 131.29, "cov_radius": 1.30, "vdw_radius": 2.16, "color": (0.26, 0.58, 0.65)},
    "Cs": {"name": "Cesium",    "atomic_number": 55, "period": 6, "group": 1,  "mass": 132.91, "cov_radius": 2.25, "vdw_radius": 3.43, "color": (0.34, 0.09, 0.56)},
    "Ba": {"name": "Barium",    "atomic_number": 56, "period": 6, "group": 2,  "mass": 137.33, "cov_radius": 1.98, "vdw_radius": 2.68, "color": (0.0, 0.79, 0.0)},
    "Pt": {"name": "Platinum",  "atomic_number": 78, "period": 6, "group": 10, "mass": 195.08, "cov_radius": 1.28, "vdw_radius": 1.75, "color": (0.82, 0.82, 0.88)},
    "Au": {"name": "Gold",      "atomic_number": 79, "period": 6, "group": 11, "mass": 196.97, "cov_radius": 1.44, "vdw_radius": 1.66, "color": (1.0, 0.82, 0.14)},
    "Hg": {"name": "Mercury",   "atomic_number": 80, "period": 6, "group": 12, "mass": 200.59, "cov_radius": 1.49, "vdw_radius": 1.55, "color": (0.72, 0.72, 0.82)},
    "Pb": {"name": "Lead",      "atomic_number": 82, "period": 6, "group": 14, "mass": 207.2,  "cov_radius": 1.47, "vdw_radius": 2.02, "color": (0.34, 0.35, 0.38)},
    "Bi": {"name": "Bismuth",   "atomic_number": 83, "period": 6, "group": 15, "mass": 208.98, "cov_radius": 1.46, "vdw_radius": 2.07, "color": (0.62, 0.31, 0.71)},
    "U":  {"name": "Uranium",   "atomic_number": 92, "period": 7, "group": 3,  "mass": 238.03, "cov_radius": 1.96, "vdw_radius": 1.86, "color": (0.0, 0.56, 1.0)},
}

# Molden standard layout elements in order:
MOLDEN_PERIODIC_TABLE = [
    # (symbol, row, col)  0-indexed in a 18-col grid
    ("H",  0, 0), ("He", 0, 17),
    ("Li", 1, 0), ("Be", 1, 1), ("B", 1, 12), ("C", 1, 13), ("N", 1, 14), ("O", 1, 15), ("F", 1, 16), ("Ne", 1, 17),
    ("Na", 2, 0), ("Mg", 2, 1), ("Al", 2, 12), ("Si", 2, 13), ("P", 2, 14), ("S", 2, 15), ("Cl", 2, 16), ("Ar", 2, 17),
    ("K",  3, 0), ("Ca", 3, 1), ("Sc", 3, 2), ("Ti", 3, 3), ("V", 3, 4), ("Cr", 3, 5), ("Mn", 3, 6), ("Fe", 3, 7),
    ("Co", 3, 8), ("Ni", 3, 9), ("Cu", 3, 10), ("Zn", 3, 11), ("Ga", 3, 12), ("Ge", 3, 13), ("As", 3, 14), ("Se", 3, 15), ("Br", 3, 16), ("Kr", 3, 17),
    ("Rb", 4, 0), ("Sr", 4, 1), ("Y",  4, 2), ("Zr", 4, 3), ("Nb", 4, 4), ("Mo", 4, 5), ("Tc", 4, 6), ("Ru", 4, 7),
    ("Rh", 4, 8), ("Pd", 4, 9), ("Ag", 4, 10), ("Cd", 4, 11), ("In", 4, 12), ("Sn", 4, 13), ("Sb", 4, 14), ("Te", 4, 15), ("I", 4, 16), ("Xe", 4, 17),
    ("Cs", 5, 0), ("Ba", 5, 1), ("Pt", 5, 9), ("Au", 5, 10), ("Hg", 5, 11), ("Pb", 5, 13), ("Bi", 5, 14), ("U", 6, 2),
    ("X",  6, 9) # Dummy atom
]

def normalize_symbol(symbol: str) -> str:
    if not symbol or not isinstance(symbol, str):
        return "C"
    s = symbol.strip()
    if not s:
        return "C"
    if s.upper() == "X":
        return "X"
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:].lower()

def get_cov_radius(symbol: str) -> float:
    norm = normalize_symbol(symbol)
    return ELEMENT_DATA.get(norm, {}).get("cov_radius", 0.77)

def get_vdw_radius(symbol: str) -> float:
    norm = normalize_symbol(symbol)
    return ELEMENT_DATA.get(norm, {}).get("vdw_radius", 1.50)

def get_color(symbol: str, bg_mode: str = "dark") -> Tuple[float, float, float]:
    """Return RGB (0.0 - 1.0) for an element, optimized for dark or white viewport backgrounds."""
    norm = normalize_symbol(symbol)
    base = ELEMENT_DATA.get(norm, {}).get("color", (0.7, 0.7, 0.7))
    if bg_mode == "white":
        if norm == "H":
            # Silver/platinum grey with subtle tint so hydrogen has crisp contrast against pure white
            return (0.84, 0.86, 0.88)
        elif norm == "C":
            # Deeper obsidian slate grey for rich contrast against white
            return (0.32, 0.34, 0.36)
        elif norm == "S":
            # Deeper golden amber so yellow doesn't get washed out on white
            return (0.82, 0.72, 0.10)
    else:  # dark background
        if norm == "H":
            # Crisp luminous white/ivory
            return (0.95, 0.95, 0.97)
        elif norm == "C":
            # Modern balanced graphite/charcoal
            return (0.48, 0.50, 0.53)
    return base

def get_mass(symbol: str) -> float:
    norm = normalize_symbol(symbol)
    return ELEMENT_DATA.get(norm, {}).get("mass", 12.011)

def get_atomic_number(symbol: str) -> int:
    norm = normalize_symbol(symbol)
    return ELEMENT_DATA.get(norm, {}).get("atomic_number", 0)

STANDARD_BONDS: Dict[Tuple[str, str], float] = {
    ("C", "H"): 1.090,
    ("H", "C"): 1.090,
    ("O", "H"): 0.960,
    ("H", "O"): 0.960,
    ("N", "H"): 1.010,
    ("H", "N"): 1.010,
    ("S", "H"): 1.340,
    ("H", "S"): 1.340,
    ("P", "H"): 1.420,
    ("H", "P"): 1.420,
    ("C", "C"): 1.540,
    ("C", "O"): 1.430,
    ("O", "C"): 1.430,
    ("C", "N"): 1.470,
    ("N", "C"): 1.470,
    ("C", "F"): 1.350,
    ("F", "C"): 1.350,
    ("C", "Cl"): 1.770,
    ("Cl", "C"): 1.770,
    ("C", "Br"): 1.940,
    ("Br", "C"): 1.940,
}

STANDARD_MULTIPLE_BONDS: Dict[Tuple[str, str, int], float] = {
    ("C", "C", 1): 1.540,
    ("C", "C", 2): 1.335,
    ("C", "C", 3): 1.205,
    ("C", "O", 1): 1.430,
    ("C", "O", 2): 1.220,
    ("C", "O", 3): 1.128, # carbon monoxide
    ("C", "N", 1): 1.470,
    ("C", "N", 2): 1.280,
    ("C", "N", 3): 1.157,
    ("N", "N", 1): 1.450,
    ("N", "N", 2): 1.250,
    ("N", "N", 3): 1.100,
    ("N", "O", 1): 1.400,
    ("N", "O", 2): 1.210,
    ("O", "O", 1): 1.480,
    ("O", "O", 2): 1.207,
    ("C", "S", 1): 1.815,
    ("C", "S", 2): 1.620,
    ("C", "P", 1): 1.850,
    ("C", "P", 2): 1.670,
    ("C", "P", 3): 1.540,
    ("P", "O", 1): 1.600,
    ("P", "O", 2): 1.480,
    ("S", "O", 1): 1.580,
    ("S", "O", 2): 1.440,
}

def estimate_bond_length(sym1: str, sym2: str, bond_order: int = 1) -> float:
    """Estimate equilibrium bond distance for a given bond order (1, 2, or 3)."""
    n1 = normalize_symbol(sym1)
    n2 = normalize_symbol(sym2)
    order = max(1, min(3, int(bond_order)))

    # Check explicit multiple bond tables
    if (n1, n2, order) in STANDARD_MULTIPLE_BONDS:
        return STANDARD_MULTIPLE_BONDS[(n1, n2, order)]
    if (n2, n1, order) in STANDARD_MULTIPLE_BONDS:
        return STANDARD_MULTIPLE_BONDS[(n2, n1, order)]

    # Standard single bond lookup
    if order == 1:
        if (n1, n2) in STANDARD_BONDS:
            return STANDARD_BONDS[(n1, n2)]
        r1 = get_cov_radius(n1)
        r2 = get_cov_radius(n2)
        return round(float(r1 + r2), 3)

    # For orders 2 and 3 when not explicitly tabulated, scale using standard covalent contraction factors
    # (Double bond ~ 86-88% of single bond; Triple bond ~ 78-80% of single bond)
    single_len = estimate_bond_length(n1, n2, bond_order=1)
    if order == 2:
        return round(single_len * 0.867, 3)
    elif order == 3:
        return round(single_len * 0.785, 3)
    return single_len


def get_formula_and_weight(symbols: List[str]) -> Tuple[str, float]:
    """
    Compute Hill-system empirical formula (e.g. C6H6, H2O, C2H5NO2) and total molecular weight (g/mol).
    Hill system convention: Carbon first, Hydrogen second, then remaining elements alphabetically.
    If no Carbon present, all elements are listed strictly alphabetically.
    """
    counts: Dict[str, int] = {}
    total_mw: float = 0.0

    for s in symbols:
        sym = normalize_symbol(s)
        if not sym or sym == "X":
            continue
        counts[sym] = counts.get(sym, 0) + 1
        m = ELEMENT_DATA.get(sym, {}).get("mass", 0.0)
        total_mw += float(m)

    if not counts:
        return "Empty", 0.0

    # Hill system sorting
    sorted_syms = []
    if "C" in counts:
        sorted_syms.append("C")
        if "H" in counts:
            sorted_syms.append("H")
        for sym in sorted(counts.keys()):
            if sym not in ("C", "H"):
                sorted_syms.append(sym)
    else:
        sorted_syms = sorted(counts.keys())

    parts = []
    for sym in sorted_syms:
        cnt = counts[sym]
        parts.append(f"{sym}{cnt}" if cnt > 1 else sym)

    formula = "".join(parts)
    return formula, round(total_mw, 3)


