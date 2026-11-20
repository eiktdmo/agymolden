"""
Chemical Identifiers & Formats Converter.

Converts molecular structures (symbols, 3D coordinates, and bonds)
into standard chemical identifiers:
- Cartesian XYZ text
- SMILES (Simplified Molecular Input Line Entry System)
- InChI (IUPAC International Chemical Identifier)
- InChIKey

Uses RDKit when available, with a pure-Python fallback for standalone Cartesian XYZ.
"""

import logging
from typing import Any, List, Tuple, Optional, Dict
import numpy as np

log = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


def format_xyz_string(symbols: List[str], coords: np.ndarray, comment: str = "AgyMolden structure") -> str:
    """Format molecule into standard Cartesian XYZ text."""
    n = len(symbols)
    lines = [str(n), comment]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym:<2}  {x:12.6f} {y:12.6f} {z:12.6f}")
    return "\n".join(lines) + "\n"


def format_coords_block_string(symbols: List[str], coords: np.ndarray) -> str:
    """Format strictly the aligned Cartesian coordinates block (without atom-count or comment header) for ORCA / Gaussian / Q-Chem input files."""
    lines = []
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym:<2}  {x:12.6f} {y:12.6f} {z:12.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


_COMMON_CHEMICAL_NAMES_SMILES: Dict[str, str] = {
    "water": "O",
    "methane": "C",
    "ethane": "CC",
    "propane": "CCC",
    "butane": "CCCC",
    "ethylene": "C=C",
    "ethene": "C=C",
    "acetylene": "C#C",
    "ethyne": "C#C",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "phenol": "Oc1ccccc1",
    "aniline": "Nc1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrimidine": "c1cncnc1",
    "pyrrole": "c1cc[nH]c1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "imidazole": "c1c[nH]cn1",
    "indole": "c1ccc2[nH]ccc2c1",
    "naphthalene": "c1ccc2ccccc2c1",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "methanol": "CO",
    "ethanol": "CCO",
    "isopropanol": "CC(C)O",
    "acetone": "CC(=O)C",
    "formaldehyde": "C=O",
    "acetaldehyde": "CC=O",
    "formic acid": "C(=O)O",
    "acetic acid": "CC(=O)O",
    "benzoic acid": "O=C(O)c1ccccc1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "dopamine": "NCCc1ccc(O)c(O)c1",
    "serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "adrenaline": "CNC[C@H](O)c1ccc(O)c(O)c1",
    "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "urea": "NC(=O)N",
    "ammonia": "N",
    "carbon dioxide": "O=C=O",
    "carbon monoxide": "[C-]#[O+]",
    "acetonitrile": "CC#N",
    "dmso": "CS(=O)C",
    "dimethyl sulfoxide": "CS(=O)C",
    "dmf": "CN(C)C=O",
    "thf": "C1CCOC1",
    "tetrahydrofuran": "C1CCOC1",
    "dichloromethane": "ClCCl",
    "chloroform": "ClC(Cl)Cl",
    "cyclohexane": "C1CCCCC1",
    "cyclopentadiene": "C1=CC=CC1",
    "glycine": "NCC(=O)O",
    "alanine": "C[C@H](N)C(=O)O",
}


def resolve_chemical_name_to_smiles(name: str, timeout: float = 4.0) -> Optional[str]:
    """Resolve a common chemical name, IUPAC name, or CAS number to an Isomeric SMILES string.

    Checks the built-in offline dictionary first, then queries the PubChem PUG REST API.
    """
    key = name.strip().lower()
    if not key:
        return None
    if key in _COMMON_CHEMICAL_NAMES_SMILES:
        return _COMMON_CHEMICAL_NAMES_SMILES[key]

    import urllib.parse
    import urllib.request

    encoded = urllib.parse.quote(name.strip())
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/IsomericSMILES,CanonicalSMILES/JSON"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AgyMolden/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                p0 = props[0]
                smi = p0.get("IsomericSMILES") or p0.get("CanonicalSMILES") or p0.get("ConnectivitySMILES")
                if smi:
                    return str(smi).strip()
    except Exception:
        logging.getLogger(__name__).debug("PubChem lookup failed for %r", name, exc_info=True)
    return None


def molecule_to_rdkit_mol(
    symbols: List[str],
    coords: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]] = None,
    bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
):
    """Convert molecular structure to an RDKit Mol object."""
    if not _HAS_RDKIT:
        raise RuntimeError(
            "RDKit is required to generate SMILES and InChI identifiers.\n"
            "Install it with:  pip install rdkit  (or: pip install .[all])"
        )

    xyz_block = format_xyz_string(symbols, coords)
    try:
        raw_mol = Chem.MolFromXYZBlock(xyz_block)
        if raw_mol is None:
            raise ValueError("Failed to create RDKit molecule from XYZ block.")
        mol = Chem.Mol(raw_mol)

        # If bonds and bond orders are explicitly known, set them
        if bonds and bond_orders:
            rw_mol = Chem.RWMol()
            conf = Chem.Conformer(len(symbols))
            for i, (sym, pos) in enumerate(zip(symbols, coords)):
                rw_mol.AddAtom(Chem.Atom(sym))
                conf.SetAtomPosition(i, [float(pos[0]), float(pos[1]), float(pos[2])])

            bo_map = {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
            }
            for u, v in bonds:
                b_key = (min(u, v), max(u, v))
                order = bond_orders.get(b_key, 1)
                rw_mol.AddBond(u, v, bo_map.get(order, Chem.BondType.SINGLE))

            m = rw_mol.GetMol()
            m.AddConformer(conf)
            Chem.SanitizeMol(m)
            return m
        else:
            rdDetermineBonds.DetermineConnectivity(mol)
            rdDetermineBonds.DetermineBondOrders(mol)
            return mol
    except Exception:
        # Fallback to direct connectivity determination
        raw_mol = Chem.MolFromXYZBlock(xyz_block)
        if raw_mol is None:
            return None
        mol = Chem.Mol(raw_mol)
        try:
            rdDetermineBonds.DetermineConnectivity(mol)
        except Exception:
            logging.getLogger(__name__).debug("Ignored exception", exc_info=True)
        return mol


def get_smiles(
    symbols: List[str],
    coords: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]] = None,
    bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
    isomeric: bool = True,
) -> str:
    """Generate SMILES string representation."""
    if not _HAS_RDKIT:
        raise RuntimeError("RDKit is required for SMILES generation. Please install rdkit.")
    mol = molecule_to_rdkit_mol(symbols, coords, bonds, bond_orders)
    if mol is None:
        raise ValueError("Could not parse molecular structure for SMILES conversion.")
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def get_inchi_and_key(
    symbols: List[str],
    coords: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]] = None,
    bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
) -> Tuple[str, str]:
    """Generate InChI string and InChIKey."""
    if not _HAS_RDKIT:
        raise RuntimeError("RDKit is required for InChI generation. Please install rdkit.")
    mol = molecule_to_rdkit_mol(symbols, coords, bonds, bond_orders)
    if mol is None:
        raise ValueError("Could not parse molecular structure for InChI conversion.")
    inchi = Chem.MolToInchi(mol)
    inchi_key = Chem.MolToInchiKey(mol)
    return inchi, inchi_key


def fetch_3d_from_cactus(query: str, timeout: float = 4.0) -> Optional[Tuple[List[str], np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]]:
    """Fetch 3D Cartesian coordinates from NIH Cactus Chemical Identifier Resolver."""
    import urllib.parse
    import urllib.request
    from .zmatrix import parse_xyz
    from .forcefield import detect_connectivity, detect_bond_orders

    enc = urllib.parse.quote(query.strip())
    url = f"https://cactus.nci.nih.gov/chemical/structure/{enc}/file?format=xyz"
    req = urllib.request.Request(url, headers={"User-Agent": "AgyMolden/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            symbols, coords = parse_xyz(content, is_path=False)
            if len(symbols) > 0 and len(coords) == len(symbols):
                bonds = detect_connectivity(symbols, coords)
                bo = detect_bond_orders(symbols, coords, bonds)
                return symbols, coords, bonds, bo
    except Exception as e:
        log.debug("NIH Cactus 3D fetch failed for '%s': %s", query, e)
    return None


def parse_sdf_string(content: str) -> Optional[Tuple[List[str], np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]]:
    """Parse standard MDL Molfile / SDF V2000 string into (symbols, coords, bonds, bond_orders)."""
    from .elements import normalize_symbol
    lines = content.splitlines()
    counts_line_idx = -1
    for idx, l in enumerate(lines[:12]):
        if "V2000" in l:
            counts_line_idx = idx
            break
    if counts_line_idx == -1:
        return None
    header = lines[counts_line_idx]
    try:
        n_atoms = int(header[:3].strip())
        n_bonds = int(header[3:6].strip())
    except ValueError:
        return None

    symbols = []
    coords = []
    for line in lines[counts_line_idx + 1 : counts_line_idx + 1 + n_atoms]:
        toks = line.split()
        if len(toks) >= 4:
            try:
                x, y, z = float(toks[0]), float(toks[1]), float(toks[2])
                sym = normalize_symbol(toks[3])
                symbols.append(sym)
                coords.append([x, y, z])
            except ValueError:
                pass

    if len(symbols) != n_atoms:
        return None

    bonds = []
    bo = {}
    b_start = counts_line_idx + 1 + n_atoms
    for line in lines[b_start : b_start + n_bonds]:
        try:
            u = int(line[:3].strip()) - 1
            v = int(line[3:6].strip()) - 1
            order = int(line[6:9].strip()) if len(line) >= 9 else 1
            if 0 <= u < n_atoms and 0 <= v < n_atoms:
                b_key = (min(u, v), max(u, v))
                bonds.append(b_key)
                bo[b_key] = order
        except ValueError:
            pass

    return symbols, np.array(coords, dtype=np.float64), bonds, bo


def fetch_3d_from_pubchem_sdf(query: str, timeout: float = 4.0) -> Optional[Tuple[List[str], np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]]:
    """Fetch 3D Cartesian coordinates from PubChem PUG REST 3D record."""
    import urllib.parse
    import urllib.request
    enc = urllib.parse.quote(query.strip())
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{enc}/SDF?record_type=3d"
    req = urllib.request.Request(url, headers={"User-Agent": "AgyMolden/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            res = parse_sdf_string(content)
            if res is not None:
                return res
    except Exception as e:
        log.debug("PubChem SDF fetch failed for '%s': %s", query, e)
    return None


_OFFLINE_TEMPLATES_3D: Dict[str, Tuple[List[str], np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]] = {
    "water": (
        ["O", "H", "H"],
        np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]], dtype=np.float64),
        [(0, 1), (0, 2)],
        {(0, 1): 1, (0, 2): 1},
    ),
    "methane": (
        ["C", "H", "H", "H", "H"],
        np.array([
            [0.0, 0.0, 0.0],
            [0.628, 0.628, 0.628],
            [-0.628, -0.628, 0.628],
            [-0.628, 0.628, -0.628],
            [0.628, -0.628, -0.628],
        ], dtype=np.float64),
        [(0, 1), (0, 2), (0, 3), (0, 4)],
        {(0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1},
    ),
    "ammonia": (
        ["N", "H", "H", "H"],
        np.array([
            [0.0, 0.0, 0.113],
            [0.0, 0.938, -0.264],
            [0.812, -0.469, -0.264],
            [-0.812, -0.469, -0.264],
        ], dtype=np.float64),
        [(0, 1), (0, 2), (0, 3)],
        {(0, 1): 1, (0, 2): 1, (0, 3): 1},
    ),
    "carbon dioxide": (
        ["C", "O", "O"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.162], [0.0, 0.0, -1.162]], dtype=np.float64),
        [(0, 1), (0, 2)],
        {(0, 1): 2, (0, 2): 2},
    ),
    "ethylene": (
        ["C", "C", "H", "H", "H", "H"],
        np.array([
            [0.0, 0.0, 0.665],
            [0.0, 0.0, -0.665],
            [0.0, 0.923, 1.232],
            [0.0, -0.923, 1.232],
            [0.0, 0.923, -1.232],
            [0.0, -0.923, -1.232],
        ], dtype=np.float64),
        [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)],
        {(0, 1): 2, (0, 2): 1, (0, 3): 1, (1, 4): 1, (1, 5): 1},
    ),
    "benzene": (
        ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
        np.array([
            [1.397, 0.0, 0.0],
            [0.6985, 1.2098, 0.0],
            [-0.6985, 1.2098, 0.0],
            [-1.397, 0.0, 0.0],
            [-0.6985, -1.2098, 0.0],
            [0.6985, -1.2098, 0.0],
            [2.481, 0.0, 0.0],
            [1.2405, 2.1486, 0.0],
            [-1.2405, 2.1486, 0.0],
            [-2.481, 0.0, 0.0],
            [-1.2405, -2.1486, 0.0],
            [1.2405, -2.1486, 0.0],
        ], dtype=np.float64),
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)],
        {(0, 1): 2, (1, 2): 1, (2, 3): 2, (3, 4): 1, (4, 5): 2, (5, 0): 1, (0, 6): 1, (1, 7): 1, (2, 8): 1, (3, 9): 1, (4, 10): 1, (5, 11): 1},
    ),
}
_OFFLINE_TEMPLATES_3D["h2o"] = _OFFLINE_TEMPLATES_3D["water"]
_OFFLINE_TEMPLATES_3D["ch4"] = _OFFLINE_TEMPLATES_3D["methane"]
_OFFLINE_TEMPLATES_3D["nh3"] = _OFFLINE_TEMPLATES_3D["ammonia"]
_OFFLINE_TEMPLATES_3D["co2"] = _OFFLINE_TEMPLATES_3D["carbon dioxide"]
_OFFLINE_TEMPLATES_3D["ethene"] = _OFFLINE_TEMPLATES_3D["ethylene"]


def _embed_rdkit_mol_to_3d(
    mol: Any,
    add_hydrogens: bool = True,
) -> Optional[Tuple[List[str], np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]]:
    """Helper to embed an RDKit Mol into 3D using ETKDG and UFF relaxation."""
    if not _HAS_RDKIT or mol is None:
        return None
    try:
        from rdkit.Chem import AllChem
        if add_hydrogens:
            mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res < 0:
            res2 = AllChem.EmbedMolecule(mol, useRandomCoords=True)
            if res2 < 0 or mol.GetNumConformers() == 0:
                AllChem.Compute2DCoords(mol)
        if mol.GetNumConformers() > 0:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception as e:
                log.debug("RDKit UFF optimization notice: %s", e)
            conf = mol.GetConformer()
            symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            coords = conf.GetPositions().astype(np.float64)
            bonds = []
            bond_orders = {}
            bo_map = {
                Chem.BondType.SINGLE: 1,
                Chem.BondType.DOUBLE: 2,
                Chem.BondType.TRIPLE: 3,
                Chem.BondType.AROMATIC: 1,
            }
            for b in mol.GetBonds():
                u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                norm_b = (min(u, v), max(u, v))
                bonds.append(norm_b)
                bond_orders[norm_b] = bo_map.get(b.GetBondType(), 1)
            return symbols, coords, bonds, bond_orders
    except Exception as e:
        log.debug("RDKit 3D embedding failed: %s", e)
    return None


def parse_smiles_or_inchi_to_3d(
    text: str,
    add_hydrogens: bool = True,
) -> Tuple[List[str], np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """
    Parse a chemical name, SMILES string, or InChI string and generate 3D Cartesian coordinates.

    Workflow:
      1. If offline template matches, returns immediate high-accuracy 3D coordinates.
      2. If RDKit is available, attempts offline ETKDG / UFF embedding directly for SMILES,
         InChI, or mapped common names (instant, zero network latency).
      3. Queries NIH Cactus and PubChem REST APIs for IUPAC names, CAS, or online SMILES.
      4. Falls back to PubChem name-to-SMILES resolution and local RDKit embedding.

    Returns:
        (symbols, coords, bonds, bond_orders)
    """
    s = text.strip()
    if not s:
        raise ValueError("Empty chemical identifier provided.")

    key_lower = s.lower()

    # 1. Check local offline template dictionary first
    if key_lower in _OFFLINE_TEMPLATES_3D:
        syms, coords, bonds, bo = _OFFLINE_TEMPLATES_3D[key_lower]
        return list(syms), np.copy(coords), list(bonds), dict(bo)

    # 2. Check if text is a known common chemical name with a mapped SMILES
    mapped_smi = _COMMON_CHEMICAL_NAMES_SMILES.get(key_lower)

    # 3. If RDKit is available, attempt direct local offline 3D embedding
    if _HAS_RDKIT:
        target_s = mapped_smi if mapped_smi else s
        mol = None
        if target_s.startswith("InChI="):
            mol = Chem.MolFromInchi(target_s)
        else:
            try:
                mol = Chem.MolFromSmiles(target_s)
            except Exception:
                mol = None
        if mol is not None:
            res_3d = _embed_rdkit_mol_to_3d(mol, add_hydrogens=add_hydrogens)
            if res_3d is not None:
                return res_3d

    # 4. Direct 3D resolution via NIH Cactus
    cactus_3d = fetch_3d_from_cactus(s)
    if cactus_3d is not None:
        return cactus_3d

    # 5. Direct 3D resolution via PubChem 3D SDF
    pubchem_3d = fetch_3d_from_pubchem_sdf(s)
    if pubchem_3d is not None:
        return pubchem_3d

    # 6. If mapped SMILES exists, try Cactus for mapped SMILES
    if mapped_smi:
        c_res = fetch_3d_from_cactus(mapped_smi)
        if c_res is not None:
            return c_res

    # 7. Online PubChem Name-to-SMILES resolution fallback
    resolved_smi = resolve_chemical_name_to_smiles(s)
    if resolved_smi:
        c_res = fetch_3d_from_cactus(resolved_smi)
        if c_res is not None:
            return c_res
        if _HAS_RDKIT:
            mol = Chem.MolFromSmiles(resolved_smi)
            if mol is not None:
                res_3d = _embed_rdkit_mol_to_3d(mol, add_hydrogens=add_hydrogens)
                if res_3d is not None:
                    return res_3d

    raise ValueError(
        f"Could not resolve 3D structure for chemical name or identifier:\n'{s}'\n\n"
        "Please check the name spelling (e.g. 'caffeine', 'aspirin', 'ferrocene', 'bipyridine')\n"
        "or provide a standard SMILES (e.g. 'c1ccccc1') or InChI string."
    )


def format_supporting_information_entry(
    symbols: List[str],
    coords: np.ndarray,
    title: str = "Structure",
    point_group: str = "C1",
    energy_ha: Optional[float] = None,
    zpe_ha: Optional[float] = None,
    enthalpy_ha: Optional[float] = None,
    gibbs_ha: Optional[float] = None,
    dipole_debye: Optional[float] = None,
    homo_ev: Optional[float] = None,
    lumo_ev: Optional[float] = None,
    n_imag: Optional[int] = None,
    lowest_freq_cm1: Optional[float] = None,
    style: str = "acs",
) -> str:
    """
    Format a publication-ready Supporting Information (SI) entry for a molecular structure
    in ACS/RSC plain text ('acs'), Markdown ('markdown'), or LaTeX ('latex') format.
    """
    counts: Dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    sorted_elems = sorted(
        counts.keys(),
        key=lambda e: (0, e) if e == "C" else ((1, e) if e == "H" else (2, e)),
    )
    formula = "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in sorted_elems) or "None"

    meta_items = [f"Formula: {formula}", f"Point Group: {point_group or 'C1'}", f"Atoms: {len(symbols)}"]
    if energy_ha is not None:
        meta_items.append(f"E(tot) = {energy_ha:.8f} Eh")
    if zpe_ha is not None:
        meta_items.append(f"ZPE = {zpe_ha:.6f} Eh")
    if enthalpy_ha is not None:
        meta_items.append(f"H(298K) = {enthalpy_ha:.8f} Eh")
    if gibbs_ha is not None:
        meta_items.append(f"G(298K) = {gibbs_ha:.8f} Eh")
    if n_imag is not None:
        freq_info = f"N_imag = {n_imag}"
        if lowest_freq_cm1 is not None:
            freq_info += f" (ν_1 = {lowest_freq_cm1:.1f} cm⁻¹)"
        meta_items.append(freq_info)
    if dipole_debye is not None:
        meta_items.append(f"Dipole = {dipole_debye:.3f} D")
    if homo_ev is not None and lumo_ev is not None:
        gap = lumo_ev - homo_ev
        meta_items.append(f"HOMO = {homo_ev:+.3f} eV, LUMO = {lumo_ev:+.3f} eV, Gap = {gap:.3f} eV")

    style_low = style.strip().lower()
    if style_low == "latex":
        lines = [
            f"% Supporting Information: {title} ({formula}, {point_group})",
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{Cartesian coordinates (\\AA) and energetic summary for \\textbf{{{title}}} ({formula}, {point_group}).}}",
            r"\begin{tabular}{lrrr}",
            r"\hline",
            r"Atom & $X$ (\AA) & $Y$ (\AA) & $Z$ (\AA) \\",
            r"\hline",
        ]
        for s, (x, y, z) in zip(symbols, coords):
            lines.append(f"{s:<4s} & {float(x):12.6f} & {float(y):12.6f} & {float(z):12.6f} \\\\")
        lines.append(r"\hline")
        for item in meta_items:
            safe_item = item.replace("_", r"\_").replace("⁻¹", "$^{-1}$")
            lines.append(f"\\multicolumn{{4}}{{l}}{{\\footnotesize {safe_item}}} \\\\")
        lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines) + "\n"

    if style_low == "markdown":
        lines = [
            f"### {title} ({formula}, Point Group: {point_group})",
            "",
            "- " + "\n- ".join(meta_items),
            "",
            "```xyz",
            f"{len(symbols)}",
            f"{title} | {' | '.join(meta_items[:4])}",
        ]
        for s, (x, y, z) in zip(symbols, coords):
            lines.append(f"{s:<4s}  {float(x):14.8f}  {float(y):14.8f}  {float(z):14.8f}")
        lines.append("```")
        return "\n".join(lines) + "\n"

    # Default: ACS / RSC Supporting Information plain text block
    sep = "=" * 72
    lines = [
        sep,
        f"{title}  ({formula}, Point Group: {point_group})",
        "-" * 72,
    ]
    for item in meta_items:
        lines.append(f"  {item}")
    lines.extend([
        "-" * 72,
        "Cartesian Coordinates (Angstroms):",
        f"{'Atom':<6s} {'X':>14s} {'Y':>14s} {'Z':>14s}",
    ])
    for s, (x, y, z) in zip(symbols, coords):
        lines.append(f"{s:<6s} {float(x):14.8f} {float(y):14.8f} {float(z):14.8f}")
    lines.append(sep)
    return "\n".join(lines) + "\n"

