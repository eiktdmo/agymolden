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
from typing import List, Tuple, Optional, Dict
import numpy as np

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


def parse_smiles_or_inchi_to_3d(
    text: str,
    add_hydrogens: bool = True,
) -> Tuple[List[str], np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """
    Parse a SMILES string or InChI string and generate 3D coordinates using ETKDG.
    
    Returns:
        (symbols, coords, bonds, bond_orders)
    """
    if not _HAS_RDKIT:
        raise RuntimeError(
            "RDKit is required to parse SMILES or InChI into 3D molecular structures.\n"
            "Install it with:  pip install rdkit  (or: pip install .[all])"
        )

    from rdkit.Chem import AllChem

    s = text.strip()
    mol = None
    if s.startswith("InChI="):
        mol = Chem.MolFromInchi(s)
    else:
        mol = Chem.MolFromSmiles(s)

    if mol is None:
        raise ValueError(f"Could not parse valid chemical structure from:\n{s}")

    if add_hydrogens:
        mol = Chem.AddHs(mol)

    # Embed 3D coordinates using ETKDG
    res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if res < 0:
        # Fallback to random coordinates + UFF optimization if ETKDG failed
        res2 = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res2 < 0 or mol.GetNumConformers() == 0:
            # Fallback 2: Compute 2D coordinates and map to 3D plane so it never crashes
            AllChem.Compute2DCoords(mol)

    if mol.GetNumConformers() == 0:
        raise ValueError(f"Could not generate 3D coordinates for structure:\n{s}")

    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = conf.GetPositions().astype(np.float64)

    bonds = []
    bond_orders = {}
    bo_map = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 1, # Kekule representation
    }

    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        norm_b = (min(u, v), max(u, v))
        bonds.append(norm_b)
        order = bo_map.get(b.GetBondType(), 1)
        if order > 1:
            bond_orders[norm_b] = order

    return symbols, coords, bonds, bond_orders
