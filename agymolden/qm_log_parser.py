"""
Universal Quantum Chemistry Output & Log File Parser via cclib.

Extracts:
- Molecular geometry and optimization trajectories (atomcoords)
- Vibrational frequencies, normal mode displacement vectors, and IR intensities
- Total energies along optimization paths
"""

import os
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

log = logging.getLogger(__name__)

try:
    import cclib
except ImportError:
    cclib = None

from .elements import ELEMENT_DATA
from .vibrations import VibrationalAnalysis


# Atomic number to element symbol mapping
ATOMIC_NUMBER_TO_SYMBOL: Dict[int, str] = {
    data["atomic_number"]: sym for sym, data in ELEMENT_DATA.items() if "atomic_number" in data
}


@dataclass
class QMLogParseResult:
    program_name: str
    symbols: List[str]
    trajectory_coords: List[np.ndarray]  # List of (n_atoms, 3) arrays
    trajectory_energies: List[float]    # Energies in eV or kcal/mol
    vibrational_analysis: Optional[VibrationalAnalysis]
    converged: bool = True
    comment: str = ""
    charges: Optional[np.ndarray] = None  # Atomic partial charges (e.g. CHELPG / Mulliken)
    et_energies: Optional[List[float]] = None  # Excited state transition energies in cm^-1 or nm
    et_oscs: Optional[List[float]] = None      # Oscillator strengths f
    nuclear_forces: Optional[np.ndarray] = None  # Nuclear Cartesian forces / gradients (n_atoms, 3) in Ha/Bohr or eV/A
    homo_ev: Optional[float] = None
    lumo_ev: Optional[float] = None
    dipole_debye: Optional[float] = None
    zpe_ha: Optional[float] = None
    enthalpy_ha: Optional[float] = None
    gibbs_ha: Optional[float] = None
    orbitals: Optional[List[Any]] = None  # List of MoldenMO objects if orbitals were present in the QM log
    homo_idx: Optional[int] = None
    is_irc: bool = False
    irc_path_points: Optional[List[float]] = None  # Reaction coordinate values along IRC (e.g. amu^(1/2)*Bohr)
    irc_pathway: Optional[Any] = None  # Assembled IRCPathway object with barriers and landmark indices
    trajectory_max_forces: Optional[List[float]] = None  # Per-step max atomic force norm (Ha/Bohr)
    trajectory_rms_forces: Optional[List[float]] = None  # Per-step RMS atomic force norm (Ha/Bohr)
    nmr_shieldings: Optional[List[Dict[str, Any]]] = None  # [{'atom_index': int, 'symbol': str, 'sigma_iso': float, 'anisotropy': float}]


def _extract_nmr_shieldings_from_text(text: str, symbols: Optional[List[str]] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Extract isotropic NMR chemical shielding tensors (sigma_iso in ppm) from ORCA, Gaussian, or Q-Chem output text.
    Returns a list of dicts: [{'atom_index': 0-based int, 'symbol': str, 'sigma_iso': float, 'anisotropy': float}].
    """
    if not text:
        return None

    results: Dict[int, Dict[str, Any]] = {}

    # 1. ORCA format: "CHEMICAL SHIELDING SUMMARY (ppm)"
    #   Nucleus  Element    Isotropic     Anisotropy
    #   -------  -------  ------------   ------------
    #      0       C          154.321        25.123
    sec_orca = text.rfind("CHEMICAL SHIELDING SUMMARY")
    if sec_orca != -1:
        blk = text[sec_orca:sec_orca + 15000]
        for m in re.finditer(r'^\s*(\d+)\s+([A-Z][a-z]?)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)', blk, re.MULTILINE):
            at_idx = int(m.group(1))
            sym = m.group(2)
            iso = float(m.group(3))
            aniso = float(m.group(4))
            results[at_idx] = {"atom_index": at_idx, "symbol": sym, "sigma_iso": iso, "anisotropy": aniso}

    # 2. Gaussian GIAO format:
    #   "  1  C    Isotropic =   187.4521   Anisotropy =    12.3456"
    if not results:
        for m in re.finditer(
            r'^\s*(\d+)\s+([A-Z][a-z]?)\s+Isotropic\s*=\s*([+-]?\d+\.\d+)\s+Anisotropy\s*=\s*([+-]?\d+\.\d+)',
            text,
            re.MULTILINE,
        ):
            at_idx = int(m.group(1)) - 1  # Gaussian is 1-indexed
            sym = m.group(2)
            iso = float(m.group(3))
            aniso = float(m.group(4))
            results[at_idx] = {"atom_index": at_idx, "symbol": sym, "sigma_iso": iso, "anisotropy": aniso}

    if not results:
        return None

    return [results[k] for k in sorted(results.keys())]


def _extract_thermo_and_frontier_from_text(text: str) -> Dict[str, Optional[float]]:
    """Lightweight regex extraction of ZPE, Enthalpy, Gibbs Free Energy (Ha), Dipole (D), and HOMO/LUMO (eV)."""
    info: Dict[str, Optional[float]] = {
        "zpe_ha": None,
        "enthalpy_ha": None,
        "gibbs_ha": None,
        "dipole_debye": None,
        "homo_ev": None,
        "lumo_ev": None,
    }
    # ORCA / PySCF / Gaussian ZPE (in Eh / Ha)
    m_zpe = re.findall(r'(?:Zero point energy|Zero-point correction|ZPE)\s*[:.=]*\s*([+-]?\d+\.\d+)\s*(?:Eh|Ha|Hartree)?', text, re.IGNORECASE)
    if m_zpe:
        try:
            info["zpe_ha"] = float(m_zpe[-1])
        except ValueError:
            pass
    # Total Enthalpy (ORCA: "Total enthalpy ... -76.123456 Eh", PySCF: "Enthalpy (H = E + ZPE + E_trans + ...) -76.123456 Eh")
    m_h = re.findall(r'(?:Total enthalpy|Sum of electronic and thermal Enthalpies|H_tot|\bEnthalpy\b[^\n]*?)\s*[:.=]*\s*([+-]?\d+\.\d+)\s*(?:Eh|Ha)', text, re.IGNORECASE)
    if m_h:
        try:
            info["enthalpy_ha"] = float(m_h[-1])
        except ValueError:
            pass
    # Final Gibbs free energy (ORCA: "Final Gibbs free energy ... -76.145678 Eh", PySCF: "Gibbs free energy ... -76.145678 Eh")
    m_g = re.findall(r'(?:Final Gibbs free energy|Sum of electronic and thermal Free Energies|G_tot|Gibbs free energy[^\n]*?)\s*[:.=]*\s*([+-]?\d+\.\d+)\s*(?:Eh|Ha)', text, re.IGNORECASE)
    if m_g:
        try:
            info["gibbs_ha"] = float(m_g[-1])
        except ValueError:
            pass
    # Dipole Magnitude in Debye (ORCA: "Magnitude (Debye) : 1.8543", PySCF: "Dipole moment(X, Y, Z, Debye): 0.00, 0.00, 1.85")
    m_dip = re.findall(r'Magnitude\s*\(Debye\)\s*:\s*([0-9]+\.[0-9]+)', text, re.IGNORECASE)
    if m_dip:
        try:
            info["dipole_debye"] = float(m_dip[-1])
        except ValueError:
            pass
    else:
        m_dip_pyscf = re.findall(r'Dipole moment\(X,\s*Y,\s*Z,\s*Debye\):\s*([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)', text)
        if m_dip_pyscf:
            try:
                dx, dy, dz = (float(x) for x in m_dip_pyscf[-1])
                info["dipole_debye"] = float(np.hypot(np.hypot(dx, dy), dz))
            except ValueError:
                pass
    # PySCF / ORCA HOMO & LUMO
    # PySCF: "HOMO (A) = -0.5042315 Ha  LUMO (A) = 0.123456 Ha" or "HOMO = -0.5042315  LUMO = 0.123456"
    m_hl = re.findall(r'HOMO[^\n=]*=\s*([+-]?\d+\.\d+)[^\n]*?LUMO[^\n=]*=\s*([+-]?\d+\.\d+)', text)
    if m_hl:
        try:
            h_val, l_val = float(m_hl[-1][0]), float(m_hl[-1][1])
            # Convert Ha to eV if magnitude looks like Hartree (< 3.0)
            scale = 27.211386245988 if abs(h_val) < 3.5 else 1.0
            info["homo_ev"] = h_val * scale
            info["lumo_ev"] = l_val * scale
        except ValueError:
            pass
    return info


def can_parse_with_cclib() -> bool:
    return cclib is not None


def parse_brorsen_logfile(path: str) -> Optional[QMLogParseResult]:
    """Dedicated parser for newBRORSEN and direct PySCF calculations (.out, .log, _summary.txt)."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().replace("\r\n", "\n").replace("\r", "\n")
    except Exception:
        return None

    is_brorsen = ("newBRORSEN" in text or "POSHF" in text or "newBRORSEN OPTIMIZATION SUMMARY" in text)
    is_pyscf = (
        "converged SCF energy =" in text
        or "SCF Total Energy:" in text
        or "PySCF" in text
        or "Exported orbitals.molden" in text
        or "Molden/Cube export" in text
    )
    if not is_brorsen and not is_pyscf:
        return None

    base_dir = os.path.dirname(os.path.abspath(path))
    base_name = os.path.splitext(os.path.basename(path))[0]

    # Find job names
    job_names = re.findall(r'\[STAGE\s+\d+\]\s+EXECUTING:\s*(\S+)', text, re.IGNORECASE)
    if not job_names:
        job_names = re.findall(r'###\s+\[Stage\s+\d+\]:\s*(\S+)', text, re.IGNORECASE)

    # Extract energies (Hartree to eV conversion: 1 Ha = 27.211386245988 eV)
    ha_to_ev = 27.211386245988
    # Search for all step-wise or cycle energies e.g.
    # "cycle 1: E = -40.1986689506" or "Step 0 : ... Energy = -40.1986689506"
    cycle_energy_matches = re.findall(
        r'cycle\s+\d+:\s*E\s*=\s*([+-]?\d+\.\d+)',
        text,
        re.IGNORECASE
    )
    if not cycle_energy_matches:
        cycle_energy_matches = re.findall(
            r'Step\s+\d+\s*:.*?Energy\s*=\s*([+-]?\d+\.\d+)',
            text,
            re.IGNORECASE
        )
    if not cycle_energy_matches:
        cycle_energy_matches = re.findall(
            r'(?:Converged\s+(?:\S+\s+)?Energy|Optimized Energy|Final Conventional Energy)\s*(?:\([^)]*\))?\s*:\s*([+-]?\d+\.\d+)\s*Ha',
            text,
            re.IGNORECASE
        )
    if not cycle_energy_matches:
        cycle_energy_matches = re.findall(
            r'(?:converged\s+SCF\s+energy\s*=|(?:SCF|MP2|CCSD(?:\(T\))?)\s+Total\s+Energy:)\s*([+-]?\d+\.\d+)',
            text,
            re.IGNORECASE
        )

    traj_energies = [float(e) * ha_to_ev for e in cycle_energy_matches]

    # Extract coordinates and trajectory
    final_coords = None
    symbols = None
    traj_coords: List[np.ndarray] = []

    # Priority A: Check for companion trajectory / checkpoint file <job>_checkpoint.xyz
    from .zmatrix import parse_multi_xyz, parse_xyz
    for jn in (job_names + [base_name]):
        chk_candidate = os.path.join(base_dir, f"{jn}_checkpoint.xyz")
        if os.path.isfile(chk_candidate):
            try:
                syms, frames = parse_multi_xyz(chk_candidate, is_path=True)
                if syms and frames:
                    symbols = syms
                    # If multiple frames exist in checkpoint.xyz, load whole optimization trajectory
                    if len(frames) > 1:
                        traj_coords = [f[1] for f in frames]
                        final_coords = traj_coords[-1]
                        break
                    else:
                        # Single frame in checkpoint
                        final_coords = frames[0][1]
            except Exception:
                logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

    # Check for companion artifact <job>_final.xyz
    if final_coords is None or len(traj_coords) <= 1:
        for jn in (job_names + [base_name]):
            xyz_candidate = os.path.join(base_dir, f"{jn}_final.xyz")
            if os.path.isfile(xyz_candidate):
                try:
                    syms, crds = parse_xyz(xyz_candidate, is_path=True)
                    if syms and len(syms) == len(crds):
                        symbols = syms
                        final_coords = crds
                        if not traj_coords:
                            traj_coords.append(crds)
                        break
                except Exception:
                    logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

    # Priority B: Check for "Final Cartesian Coordinates (Angstrom):" or "Cartesian Coordinates (Angstrom):" or PySCF "atoms / geometry" in text
    if final_coords is None:
        coord_blocks = re.findall(
            r'(?:Final\s+)?Cartesian\s+Coordinates\s*\((?:Angstrom|Angs|A)\):\s*\n((?:\s*[A-Za-z]{1,2}\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s*\n)+)',
            text
        )
        for blk in coord_blocks:
            syms = []
            crds = []
            for l in blk.strip().splitlines():
                parts = l.split()
                if len(parts) >= 4:
                    syms.append(parts[0])
                    try:
                        crds.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        pass
            if syms and len(syms) == len(crds):
                symbols = syms
                traj_coords.append(np.array(crds, dtype=np.float64))
        if traj_coords:
            final_coords = traj_coords[-1]

    # Priority B2: Check PySCF atom tables, Molden artifacts, companion scripts, or internal distances
    if final_coords is None:
        # Check PySCF verbose=4 atom coordinate table: "   1 O       0.000000    0.000000    0.117300"
        pyscf_at_blocks = re.findall(
            r'(?:^\s*\d+\s+[A-Z][a-z]?\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+[^\n]*\n)+',
            text,
            re.MULTILINE,
        )
        for blk in pyscf_at_blocks:
            syms = []
            crds = []
            for l in blk.strip().splitlines():
                parts = l.split()
                if len(parts) >= 5 and parts[0].isdigit() and parts[1].capitalize() in ELEMENT_DATA:
                    syms.append(parts[1].capitalize())
                    try:
                        crds.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    except ValueError:
                        pass
            if syms and len(syms) == len(crds):
                symbols = syms
                traj_coords.append(np.array(crds, dtype=np.float64))
        if traj_coords:
            final_coords = traj_coords[-1]

        # Also check companion .py script or .molden in the same directory
        if final_coords is None:
            py_cand = os.path.join(base_dir, f"{base_name}.py")
            if os.path.isfile(py_cand):
                try:
                    py_txt = open(py_cand, "r", encoding="utf-8", errors="ignore").read()
                    m_at = re.search(r'mol\.atom\s*=\s*"""(.*?)"""', py_txt, re.DOTALL)
                    if m_at:
                        syms, crds = [], []
                        for l in m_at.group(1).replace(";", "\n").replace("'", "").replace('"', "").splitlines():
                            p = l.split()
                            if len(p) >= 4 and p[0].capitalize() in ELEMENT_DATA:
                                syms.append(p[0].capitalize())
                                crds.append([float(p[1]), float(p[2]), float(p[3])])
                        if syms and len(syms) == len(crds):
                            symbols = syms
                            final_coords = np.array(crds, dtype=np.float64)
                            traj_coords = [final_coords]
                except Exception as e:
                    log.debug("Failed parsing py candidate %s: %s", py_cand, e)
        if final_coords is None:
            import glob
            # Check explicitly declared generated artifacts in log
            artifact_moldens = re.findall(r'\[Artifact Generated\]\s*(\S+\.molden)', text, re.IGNORECASE)
            molden_candidates = [os.path.join(base_dir, art) for art in artifact_moldens]

            # Also check job_names glob and base_name
            for jn in (job_names + [base_name]):
                molden_candidates.extend(glob.glob(os.path.join(base_dir, f"{jn}*.molden")))
            molden_candidates.extend([
                os.path.join(base_dir, f"{base_name}.molden"),
                os.path.join(base_dir, "orbitals.molden"),
            ])

            for m_cand in molden_candidates:
                if os.path.isfile(m_cand):
                    try:
                        from .volumetric import parse_molden_file
                        md = parse_molden_file(m_cand)
                        if md and md.symbols and len(md.coords) == len(md.symbols):
                            symbols = md.symbols
                            final_coords = md.coords
                            traj_coords = [final_coords]
                            break
                    except Exception as e:
                        log.debug("Failed parsing molden candidate %s: %s", m_cand, e)

            # Check for diatom distance in text: "Distance R(Li1 - H2) : 1.615306 A"
            if final_coords is None:
                m_diatom = re.search(
                    r'Distance\s+R\(([A-Za-z]{1,2})\d*\s*-\s*([A-Za-z]{1,2})\d*\)\s*:\s*([+-]?\d+\.\d+)\s*A',
                    text,
                    re.IGNORECASE,
                )
                if m_diatom:
                    s1 = m_diatom.group(1).capitalize()
                    s2 = m_diatom.group(2).capitalize()
                    d_val = float(m_diatom.group(3))
                    symbols = [s1, s2]
                    final_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, d_val]], dtype=np.float64)
                    traj_coords = [final_coords]

    # Priority C: Check for companion input file (.inp)
    # If we have optimization steps or only final coordinates, get initial structure as step 0 if available!
    init_coords = None
    inp_candidate = os.path.join(base_dir, f"{base_name}.inp")
    m_inp = re.search(r'input:\s*(\S+\.inp)', text)
    if m_inp:
        inp_ref = os.path.join(base_dir, m_inp.group(1))
        if os.path.isfile(inp_ref):
            inp_candidate = inp_ref

    if os.path.isfile(inp_candidate):
        from .brorsen_builder.generator import BrorsenInputGenerator
        gen = BrorsenInputGenerator()
        try:
            _, geom = gen.load_from_text(open(inp_candidate).read())
            if geom and geom.get("lines"):
                lines = geom["lines"]
                if geom.get("format") == "xyz":
                    syms_in, crds_in = parse_xyz(f"{len(lines)}\nfrom inp\n" + "\n".join(lines), is_path=False)
                    if syms_in and len(crds_in) == len(syms_in):
                        init_coords = crds_in
                        if not symbols:
                            symbols = syms_in
                elif geom.get("format") == "zmat":
                    from .zmatrix import parse_zmatrix_text, zmatrix_to_cartesian
                    zatoms_in, _ = parse_zmatrix_text("\n".join(lines))
                    if zatoms_in:
                        init_coords = zmatrix_to_cartesian(zatoms_in)
                        if not symbols:
                            symbols = [a.symbol for a in zatoms_in]
        except Exception:
            logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

    if final_coords is None and init_coords is not None:
        final_coords = init_coords

    # If we have initial coords and final coords and multiple cycle energies, construct trajectory!
    if init_coords is not None and final_coords is not None:
        if not traj_coords or len(traj_coords) == 1:
            # We have at least initial and final
            if len(cycle_energy_matches) > 1:
                # Interpolate intermediate steps if geometry output only saved initial & final
                n_c = len(cycle_energy_matches)
                interp_coords = []
                for s_i in range(n_c):
                    alpha = s_i / (n_c - 1)
                    c_step = (1.0 - alpha) * init_coords + alpha * final_coords
                    interp_coords.append(c_step)
                traj_coords = interp_coords
            else:
                traj_coords = [init_coords, final_coords]
        elif len(traj_coords) > 1 and not np.allclose(traj_coords[0], init_coords):
            # Prepend initial structure if not already at step 0
            traj_coords = [init_coords] + traj_coords

    if not symbols or final_coords is None:
        return None

    # Check for companion vibrational frequencies <job>_freq.molden or inline PySCF normal mode tables
    vib_analysis = None
    for jn in (job_names + [base_name]):
        for cand_m in (f"{jn}_freq.molden", f"{jn}.molden"):
            molden_freq = os.path.join(base_dir, cand_m)
            if os.path.isfile(molden_freq):
                try:
                    from .volumetric import parse_molden_file
                    m_data = parse_molden_file(molden_freq)
                    if m_data and m_data.vibrations:
                        vib_analysis = m_data.vibrations
                        break
                except Exception:
                    logging.getLogger(__name__).debug("Ignored exception", exc_info=True)
        if vib_analysis is not None:
            break

    # Fallback: Parse inline PySCF thermo.dump_normal_mode(mol, freq_info) output blocks
    if vib_analysis is None and ("Freq [cm^-1]" in text or "Frequencies (cm-1):" in text):
        try:
            n_atoms = len(symbols)
            mode_freqs = []
            mode_disps = []
            # Match PySCF dump_normal_mode blocks:
            # Mode              0                  1                  2
            # Freq [cm^-1]   1650.12            3750.45            3855.80
            lines = text.splitlines()
            idx_l = 0
            while idx_l < len(lines):
                l_str = lines[idx_l].strip()
                if l_str.startswith("Freq [cm^-1]"):
                    f_parts = l_str.split()[2:]
                    blk_freqs = []
                    for fp in f_parts:
                        try:
                            blk_freqs.append(float(fp))
                        except ValueError:
                            pass
                    if blk_freqs:
                        # Scan ahead for "Normal mode" header followed by atom x y z rows
                        k_l = idx_l + 1
                        while k_l < min(len(lines), idx_l + 15):
                            if "Normal mode" in lines[k_l] or (lines[k_l].strip().startswith("0 ") and len(lines[k_l].split()) >= 2 + 3 * len(blk_freqs)):
                                if "Normal mode" in lines[k_l]:
                                    k_l += 1
                                disp_Per_mode = [np.zeros((n_atoms, 3), dtype=np.float64) for _ in blk_freqs]
                                rows_read = 0
                                while k_l < len(lines) and rows_read < n_atoms:
                                    rtoks = lines[k_l].split()
                                    if len(rtoks) >= 2 + 3 * len(blk_freqs) and rtoks[0].isdigit():
                                        a_idx = int(rtoks[0])
                                        if 0 <= a_idx < n_atoms:
                                            for m_i in range(len(blk_freqs)):
                                                base_col = 2 + 3 * m_i
                                                try:
                                                    disp_Per_mode[m_i][a_idx, 0] = float(rtoks[base_col])
                                                    disp_Per_mode[m_i][a_idx, 1] = float(rtoks[base_col + 1])
                                                    disp_Per_mode[m_i][a_idx, 2] = float(rtoks[base_col + 2])
                                                except ValueError:
                                                    pass
                                            rows_read += 1
                                        k_l += 1
                                    else:
                                        break
                                if rows_read == n_atoms:
                                    for f_v, d_m in zip(blk_freqs, disp_Per_mode):
                                        if abs(f_v) > 5.0:
                                            mode_freqs.append(f_v)
                                            mode_disps.append(d_m)
                                break
                            k_l += 1
                idx_l += 1
            if mode_freqs and mode_disps:
                vib_analysis = VibrationalAnalysis(
                    frequencies=mode_freqs,
                    displacements=np.array(mode_disps, dtype=np.float64),
                    symbols=symbols,
                    base_coords=final_coords,
                    source="PySCF / newBRORSEN Output",
                )
        except Exception:
            logging.getLogger(__name__).debug("Ignored exception parsing inline PySCF normal modes", exc_info=True)

    # Extract CHELPG atomic charges from log if present
    parsed_charges = None
    chelpg_matches = re.findall(
        r'\[CHELPG\]\s+\w+\s+charges[^\n]*\n\s*#\s+Atom\s+q\s*\n((?:\s*\d+\s+[A-Za-z0-9]+\s+[-+]?\d+\.\d+\s*\n)+)',
        text
    )
    if chelpg_matches:
        q_list = []
        for line in chelpg_matches[-1].strip().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                try:
                    q_list.append(float(parts[2]))
                except ValueError:
                    pass
        if len(q_list) == len(symbols):
            parsed_charges = np.array(q_list, dtype=np.float64)

    # Extract TD-DFT UV-Vis transitions if printed by PySCF / newBRORSEN
    et_energies = None
    et_oscs = None
    td_matches = re.findall(
        r'Excited\s+State\s+\d+:\s*E\s*=\s*([+-]?\d+\.\d+)\s*eV\s+lambda\s*=\s*([+-]?\d+\.\d+)\s*nm\s+f\s*=\s*([+-]?\d+\.\d+)',
        text,
        re.IGNORECASE,
    )
    if td_matches:
        et_energies = [float(m[1]) for m in td_matches]
        et_oscs = [float(m[2]) for m in td_matches]

    nmr_sh = _extract_nmr_shieldings_from_text(text, symbols=symbols)
    prog_label = "newBRORSEN" if is_brorsen else "PySCF"
    comment = f"{prog_label} output: {len(symbols)} atoms, {len(cycle_energy_matches)} energy evaluation(s)"
    if nmr_sh:
        comment += f", {len(nmr_sh)} NMR shielding(s)"
    extra = _extract_thermo_and_frontier_from_text(text)
    return QMLogParseResult(
        program_name=prog_label,
        symbols=symbols,
        trajectory_coords=traj_coords if traj_coords else [final_coords],
        trajectory_energies=traj_energies,
        vibrational_analysis=vib_analysis,
        converged="converged" in text.lower(),
        comment=comment,
        charges=parsed_charges,
        et_energies=et_energies,
        et_oscs=et_oscs,
        nmr_shieldings=nmr_sh,
        **extra,
    )


def _parse_openlowdin_logfile(path: str) -> Optional[QMLogParseResult]:
    """Dedicated parser for openLOWDIN APMO output files (.out, .log)."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().replace("\r\n", "\n").replace("\r", "\n")
    except Exception:
        return None

    if "LOWDIN" not in text.upper() and "APMO" not in text.upper() and "SYSTEM_DESCRIPTION" not in text:
        return None

    base_dir = os.path.dirname(os.path.abspath(path))
    base_name = os.path.splitext(os.path.basename(path))[0]
    ha_to_ev = 27.211386245988

    # Extract total energies (Hartree)
    e_matches = re.findall(
        r'(?:TOTAL\s+ENERGY\s*=|Total\s+Energy\s*[:=]|E\(TOTAL\)\s*=)\s*([+-]?\d+\.\d+)',
        text,
        re.IGNORECASE,
    )
    energies = [float(e) * ha_to_ev for e in e_matches]

    symbols: List[str] = []
    steps: List[np.ndarray] = []

    # 1. Try parsing classical nuclei / geometry blocks in output or companion .lowdin input
    lowdin_inp = os.path.join(base_dir, f"{base_name}.lowdin")
    for src_txt in ([text] + ([open(lowdin_inp, encoding="utf-8", errors="ignore").read()] if os.path.isfile(lowdin_inp) else [])):
        # Match classical dirac lines: "O  dirac  0.000  0.000  0.117" or e-(O) lines
        dirac_matches = re.findall(
            r'^\s*([A-Z][a-z]?)\s+dirac\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)',
            src_txt,
            re.MULTILINE,
        )
        if dirac_matches:
            syms = [m[0].capitalize() for m in dirac_matches]
            crds = np.array([[float(m[1]), float(m[2]), float(m[3])] for m in dirac_matches], dtype=np.float64)
            symbols = syms
            steps.append(crds)
            break
        e_matches_geom = re.findall(
            r'^\s*e-\(([A-Za-z]+)\)\s+\S+\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)',
            src_txt,
            re.MULTILINE,
        )
        if e_matches_geom:
            syms = [m[0].capitalize() for m in e_matches_geom]
            crds = np.array([[float(m[1]), float(m[2]), float(m[3])] for m in e_matches_geom], dtype=np.float64)
            symbols = syms
            steps.append(crds)
            break

    # 2. Check companion Molden files exported by openLOWDIN
    if not symbols or not steps:
        for ext in (".molden", ".e-.molden", ".E-.molden", ".E-ALPHA.molden", ".molf"):
            cand_m = os.path.join(base_dir, f"{base_name}{ext}")
            if os.path.isfile(cand_m):
                try:
                    from .volumetric import parse_molden_file
                    md = parse_molden_file(cand_m)
                    if md and md.symbols and len(md.coords) == len(md.symbols):
                        symbols = md.symbols
                        steps = [md.coords]
                        break
                except Exception as e:
                    log.debug("Failed parsing openLOWDIN molden candidate %s: %s", cand_m, e)

    if not symbols or not steps:
        return None

    extra = _extract_thermo_and_frontier_from_text(text)
    comment = f"openLOWDIN APMO calculation: {len(symbols)} atoms, {len(energies)} energy evaluation(s)"
    return QMLogParseResult(
        program_name="openLOWDIN",
        symbols=symbols,
        trajectory_coords=steps,
        trajectory_energies=energies,
        vibrational_analysis=None,
        converged=("END OF LOWDIN" in text.upper() or "CONVERGED" in text.upper() or len(energies) > 0),
        comment=comment,
        **extra,
    )


def _parse_psi4_logfile_native(path: str) -> Optional[QMLogParseResult]:
    """Dedicated native parser for Psi4 output logs (.out, .dat, .log)."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().replace("\r\n", "\n").replace("\r", "\n")
    except Exception:
        return None

    if "PSI4" not in text.upper() and "BUY A DEVELOPER A BEER" not in text.upper():
        return None

    ha_to_ev = 27.211386245988
    lines = text.splitlines()
    i = 0
    symbols: List[str] = []
    geom_blocks: List[np.ndarray] = []

    # Parse geometry tables
    while i < len(lines):
        line = lines[i]
        if "Center" in line and ("X" in line or "x" in line) and ("Y" in line or "y" in line):
            if i + 1 < len(lines) and ("----" in lines[i + 1] or "====" in lines[i + 1]):
                i += 2
                coords_step = []
                step_syms = []
                while i < len(lines):
                    row = lines[i].strip()
                    if not row or "----" in row or "====" in row or "Total Mass" in row:
                        break
                    toks = row.split()
                    if len(toks) >= 4:
                        s_cand = toks[0]
                        try:
                            x, y, z = float(toks[1]), float(toks[2]), float(toks[3])
                            from .elements import normalize_symbol
                            norm_s = normalize_symbol(s_cand)
                            if norm_s:
                                step_syms.append(norm_s)
                                coords_step.append([x, y, z])
                            else:
                                break
                        except ValueError:
                            break
                    else:
                        break
                    i += 1
                if coords_step and (not symbols or len(step_syms) == len(symbols)):
                    if not symbols:
                        symbols = step_syms
                    geom_blocks.append(np.array(coords_step, dtype=np.float64))
        i += 1

    if not symbols or not geom_blocks:
        return None

    # Parse total electronic energies (excluding thermochemical summaries)
    traj_energies = []
    for l in lines:
        if any(w in l for w in ("ZPE", "Enthalpy", "Gibbs", "thermal", "Correction", "Zero-point")):
            continue
        m_e = re.search(
            r'(?:(?:Final\s+Energy|Total\s+Energy|Current\s+energy)\s*[:=]|@.*Final\s+Energy:\s*)\s*([+-]?\d+\.\d+)',
            l,
            re.IGNORECASE,
        )
        if m_e:
            traj_energies.append(float(m_e.group(1)) * ha_to_ev)

    # Parse vibrational frequencies & IR intensities
    vib_analysis = None
    if "Harmonic Vibrational Frequencies" in text or "Freq [cm^-1]" in text:
        vib_freqs: List[float] = []
        vib_irs: List[float] = []
        n_atoms = len(symbols)

        for match_m in re.finditer(r'Freq\s*\[cm\^-1\]\s+((?:[+-]?\d+\.?\d*\s*)+)', text):
            freq_tokens = [float(x) for x in match_m.group(1).split()]
            vib_freqs.extend(freq_tokens)

        for match_ir in re.finditer(r'IR\s+Intens\s*\[km/mol\]\s+((?:[+-]?\d+\.?\d*\s*)+)', text):
            ir_tokens = [float(x) for x in match_ir.group(1).split()]
            vib_irs.extend(ir_tokens)

        if vib_freqs:
            disps = np.zeros((len(vib_freqs), n_atoms, 3), dtype=np.float64)
            vib_analysis = VibrationalAnalysis(
                frequencies=vib_freqs,
                displacements=disps,
                symbols=symbols,
                base_coords=geom_blocks[-1],
                source="Psi4 QM Output",
                ir_intensities=vib_irs if len(vib_irs) == len(vib_freqs) else None,
            )

    converged = "exit" in text.lower() or "success" in text.lower() or "optimization complete" in text.lower()

    comment = f"Psi4 calculation: {len(symbols)} atoms, {len(geom_blocks)} geometry step(s)"
    if vib_analysis:
        comment += f", {vib_analysis.n_modes} vibrational mode(s)"

    extra = _extract_thermo_and_frontier_from_text(text)

    return QMLogParseResult(
        program_name="Psi4",
        symbols=symbols,
        trajectory_coords=geom_blocks,
        trajectory_energies=traj_energies,
        vibrational_analysis=vib_analysis,
        converged=converged,
        comment=comment,
        **extra,
    )


def _parse_orca_logfile_fallback(path: str) -> Optional[QMLogParseResult]:
    """Robust native fallback parser for ORCA log/out files when cclib fails."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().replace("\r\n", "\n").replace("\r", "\n")
    except Exception:
        return None

    if "O   R   C   A" not in text and "* O R C A *" not in text and "ORCA" not in text[:5000]:
        return None

    # 1. Parse Cartesian coordinate steps
    coord_blocks = re.findall(
        r'CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n-+\s*\n((?:\s*[A-Za-z-]{1,2}\s+[-+]?\d+\.\d+(?:[EeDd][+-]?\d+)?\s+[-+]?\d+\.\d+(?:[EeDd][+-]?\d+)?\s+[-+]?\d+\.\d+(?:[EeDd][+-]?\d+)?\s*\n)+)',
        text
    )
    if not coord_blocks:
        return None

    steps = []
    symbols = []
    for blk in coord_blocks:
        block_syms = []
        block_coords = []
        for l in blk.strip().splitlines():
            parts = l.split()
            # Ignore dummy atoms / points denoted by '-'
            if len(parts) >= 4 and parts[0] != '-':
                sym = parts[0].capitalize()
                block_syms.append(sym)
                try:
                    block_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    pass
        if block_coords:
            symbols = block_syms
            steps.append(np.array(block_coords, dtype=np.float64))

    if not symbols or not steps:
        return None

    # 2. Extract final single point energies
    e_matches = re.findall(r'FINAL SINGLE POINT ENERGY\s+([+-]?\d+\.\d+)', text, re.IGNORECASE)
    ha_to_ev = 27.211386245988
    energies = [float(e) * ha_to_ev for e in e_matches]

    # 3. Extract vibrational frequencies & normal mode displacements
    vib_analysis = None
    freq_matches = re.findall(r'^\s*(\d+):\s*([0-9]+\.[0-9]+)\s*cm\*\*-1', text, re.MULTILINE)
    if freq_matches:
        all_freqs = [(int(idx), float(f)) for idx, f in freq_matches]
        # Keep non-zero / real vibrational frequencies
        vib_freq_list = [f for idx, f in all_freqs if f > 10.0]
        vib_idx_set = {idx for idx, f in all_freqs if f > 10.0}

        # Parse normal mode displacement vectors
        n_atoms = len(symbols)
        start_modes = text.find("NORMAL MODES\n------------")
        mode_cols: Dict[int, List[float]] = {}
        if start_modes != -1:
            end_modes = text.find("IR SPECTRUM", start_modes)
            block = text[start_modes:end_modes] if end_modes != -1 else text[start_modes:start_modes + 15000]
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            i = 0
            while i < len(lines):
                toks = lines[i].split()
                if toks and all(t.isdigit() for t in toks):
                    col_modes = [int(t) for t in toks]
                    i += 1
                    for r in range(3 * n_atoms):
                        if i >= len(lines):
                            break
                        row_toks = lines[i].split()
                        if len(row_toks) >= len(col_modes) + 1 and row_toks[0].isdigit():
                            coord_idx = int(row_toks[0])
                            for c_idx, m_idx in enumerate(col_modes):
                                if m_idx not in mode_cols:
                                    mode_cols[m_idx] = [0.0] * (3 * n_atoms)
                                if coord_idx < 3 * n_atoms:
                                    try:
                                        mode_cols[m_idx][coord_idx] = float(row_toks[c_idx + 1])
                                    except ValueError:
                                        pass
                        i += 1
                else:
                    i += 1

        # Extract IR intensities if available
        ir_intensities = None
        ir_sec = text.find("IR SPECTRUM\n-----------")
        if ir_sec != -1:
            ir_end = text.find("RAMAN SPECTRUM", ir_sec)
            ir_blk = text[ir_sec:ir_end] if ir_end != -1 else text[ir_sec:ir_sec + 8000]
            ir_matches = re.findall(r'^\s*(\d+):\s+([0-9.]+)\s+[0-9.]+\s+([0-9.]+)', ir_blk, re.MULTILINE)
            if ir_matches:
                ir_dict = {int(m[0]): float(m[2]) for m in ir_matches}
                ir_intensities = [ir_dict.get(idx, 0.0) for idx, f in all_freqs if f > 10.0]

        # Extract Raman activities if available
        raman_activities = None
        raman_sec = text.find("RAMAN SPECTRUM\n--------------")
        if raman_sec != -1:
            raman_end = text.find("ABSORPTION SPECTRUM", raman_sec)
            raman_blk = text[raman_sec:raman_end] if raman_end != -1 else text[raman_sec:raman_sec + 8000]
            raman_matches = re.findall(r'^\s*(\d+):\s+([0-9.]+)\s+([0-9.]+)', raman_blk, re.MULTILINE)
            if raman_matches:
                raman_dict = {int(m[0]): float(m[2]) for m in raman_matches}
                raman_activities = [raman_dict.get(idx, 0.0) for idx, f in all_freqs if f > 10.0]

        # Assemble displacements array: shape (n_modes, n_atoms, 3)
        disps_list = []
        final_freqs = []
        for idx, f in all_freqs:
            if f > 10.0 and idx in mode_cols:
                disp_mat = np.array(mode_cols[idx], dtype=np.float64).reshape((n_atoms, 3))
                disps_list.append(disp_mat)
                final_freqs.append(f)

        if disps_list and len(disps_list) == len(final_freqs):
            vib_analysis = VibrationalAnalysis(
                frequencies=final_freqs,
                displacements=np.array(disps_list, dtype=np.float64),
                symbols=symbols,
                base_coords=steps[-1],
                source="ORCA QM Output",
                ir_intensities=ir_intensities[:len(final_freqs)] if ir_intensities else None,
                raman_activities=raman_activities[:len(final_freqs)] if raman_activities else None,
            )

    # 4. Extract TD-DFT / CIS UV-Vis electronic transitions if available
    et_energies = None
    et_oscs = None
    uv_sec = text.find("ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS")
    if uv_sec != -1:
        uv_blk = text[uv_sec:uv_sec + 12000]
        # Format in ORCA 5 & 6:
        # State  Energy(cm-1)  Wavelength(nm)   fosc ...
        # or:  0-1A  ->  1-1A    3.4567   27880.1   358.7   0.0123456
        wls = []
        oscs = []
        for line in uv_blk.splitlines():
            parts = line.split()
            # Match rows containing numeric transition data
            floats = []
            for tok in parts:
                try:
                    floats.append(float(tok))
                except ValueError:
                    pass
            # In ORCA 5: [State#, Energy(cm-1), Wavelength(nm), fosc, T2, TX, TY, TZ]
            # In ORCA 6: [Energy(eV), Energy(cm-1), Wavelength(nm), fosc, T2, TX, TY, TZ]
            if len(floats) >= 4:
                # Find pair (cm-1, nm) where 1e7 / cm-1 ~= nm
                for k in range(len(floats) - 2):
                    w_cm = floats[k]
                    wl_nm = floats[k + 1]
                    f_osc = floats[k + 2]
                    if w_cm > 1000.0 and wl_nm > 50.0 and abs((1e7 / w_cm) - wl_nm) < 5.0 and 0.0 <= f_osc <= 50.0:
                        wls.append(wl_nm)
                        oscs.append(f_osc)
                        break
        if wls:
            et_energies = wls
            et_oscs = oscs

    comment = f"ORCA calculation: {len(steps)} geometry step(s)"
    if vib_analysis:
        comment += f", {vib_analysis.n_modes} vibrational mode(s)"
    if et_energies:
        comment += f", {len(et_energies)} electronic transition(s)"

    extra = _extract_thermo_and_frontier_from_text(text)
    nmr_sh = _extract_nmr_shieldings_from_text(text, symbols=symbols)
    if nmr_sh:
        comment += f", {len(nmr_sh)} NMR shielding(s)"
    return QMLogParseResult(
        program_name="Orca",
        symbols=symbols,
        trajectory_coords=steps,
        trajectory_energies=energies,
        vibrational_analysis=vib_analysis,
        converged="THE OPTIMIZATION HAS CONVERGED" in text or "ORCA TERMINATED NORMALLY" in text,
        comment=comment,
        et_energies=et_energies,
        et_oscs=et_oscs,
        nmr_shieldings=nmr_sh,
        **extra,
    )


def _parse_fchk_logfile_native(path: str) -> Optional[QMLogParseResult]:
    """Native parser for Gaussian Formatted Checkpoint (.fchk / .fch) files."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return None

    if len(lines) < 2:
        return None

    # Verify fchk signature in first few lines
    header_found = False
    for line in lines[:5]:
        if any(kw in line for kw in ("SP ", "Opt ", "Freq ", "RHF", "UHF", "ROHF", "RB3LYP", "UB3LYP", "Number of atoms")):
            header_found = True
            break
    if not header_found and not path.lower().endswith((".fchk", ".fch")):
        return None

    BOHR_TO_ANGSTROM = 0.529177210903
    HARTREE_TO_EV = 27.211386245988

    atomic_numbers: List[int] = []
    coords_bohr: List[float] = []
    total_energy_hartree: Optional[float] = None
    charge = 0
    mult = 1
    forces_list: List[float] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Atomic numbers"):
            m = re.search(r"N=\s*(\d+)", line)
            n_vals = int(m.group(1)) if m else 0
            i += 1
            vals = []
            while i < len(lines) and len(vals) < n_vals:
                vals.extend([int(x) for x in lines[i].split()])
                i += 1
            atomic_numbers = vals[:n_vals]
            continue
        elif line.startswith("Current cartesian coordinates"):
            m = re.search(r"N=\s*(\d+)", line)
            n_vals = int(m.group(1)) if m else 0
            i += 1
            vals = []
            while i < len(lines) and len(vals) < n_vals:
                cleaned = lines[i].replace("D", "E").replace("d", "e").split()
                vals.extend([float(x) for x in cleaned])
                i += 1
            coords_bohr = vals[:n_vals]
            continue
        elif line.startswith("Cartesian Gradient") or line.startswith("Cartesian Force"):
            m = re.search(r"N=\s*(\d+)", line)
            n_vals = int(m.group(1)) if m else 0
            i += 1
            vals = []
            while i < len(lines) and len(vals) < n_vals:
                cleaned = lines[i].replace("D", "E").replace("d", "e").split()
                vals.extend([float(x) for x in cleaned])
                i += 1
            forces_list = vals[:n_vals]
            continue
        elif line.startswith("Total Energy"):
            parts = line.replace("D", "E").replace("d", "e").split()
            if len(parts) >= 3:
                try:
                    total_energy_hartree = float(parts[-1])
                except ValueError:
                    pass
        elif line.startswith("Charge"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    charge = int(parts[-1])
                except ValueError:
                    pass
        elif line.startswith("Multiplicity"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    mult = int(parts[-1])
                except ValueError:
                    pass
        i += 1

    if not atomic_numbers or not coords_bohr:
        return None

    symbols = [ATOMIC_NUMBER_TO_SYMBOL.get(num, f"X{num}") for num in atomic_numbers]
    n_atoms = len(symbols)
    if len(coords_bohr) < n_atoms * 3:
        return None

    coords_angstrom = np.array(coords_bohr[:n_atoms * 3], dtype=np.float64).reshape(-1, 3) * BOHR_TO_ANGSTROM

    energies_ev = []
    if total_energy_hartree is not None:
        energies_ev = [total_energy_hartree * HARTREE_TO_EV]

    nuclear_forces = None
    if len(forces_list) >= n_atoms * 3:
        # Negative gradient is force; in fchk Cartesian Gradient is dE/dx (Hartree/Bohr)
        # Convert to eV/Angstrom or keep normalized vector
        grad_arr = np.array(forces_list[:n_atoms * 3], dtype=np.float64).reshape(-1, 3)
        nuclear_forces = -grad_arr

    comment = f"Gaussian Formatted Checkpoint (FCHK): {n_atoms} atom(s)"
    if total_energy_hartree is not None:
        comment += f", Total Energy = {total_energy_hartree:.8f} Eh"

    return QMLogParseResult(
        program_name="Gaussian",
        symbols=symbols,
        trajectory_coords=[coords_angstrom],
        trajectory_energies=energies_ev,
        vibrational_analysis=None,
        converged=True,
        comment=comment,
        nuclear_forces=nuclear_forces,
    )


def parse_qm_logfile(path: str) -> QMLogParseResult:
    """Parse quantum chemistry output/log file into geometries, trajectories and vibrational data."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    # 1. Try dedicated newBRORSEN / PySCF, openLOWDIN, Psi4, and Gaussian FCHK parsers first
    if path.lower().endswith((".fchk", ".fch")):
        fchk_res = _parse_fchk_logfile_native(path)
        if fchk_res is not None:
            return fchk_res

    brorsen_res = parse_brorsen_logfile(path)
    if brorsen_res is not None:
        return brorsen_res
    lowdin_res = _parse_openlowdin_logfile(path)
    if lowdin_res is not None:
        return lowdin_res
    psi4_res = _parse_psi4_logfile_native(path)
    if psi4_res is not None:
        return psi4_res
    fchk_res = _parse_fchk_logfile_native(path)
    if fchk_res is not None:
        return fchk_res

    # 2. Use cclib to open and parse
    if not cclib:
        # If cclib is missing, try native ORCA fallback parser
        orca_fallback = _parse_orca_logfile_fallback(path)
        if orca_fallback is not None:
            return orca_fallback
        raise RuntimeError(
            "cclib is required to parse third-party QM log files (ORCA, Gaussian, Q-Chem, etc.).\n"
            "Install it with:  pip install cclib  (or: pip install .[qm])"
        )

    try:
        parser = cclib.io.ccopen(path)
        data = parser.parse() if parser is not None else None
    except Exception as e_cclib:
        log.warning(f"cclib parsing error for {path}: {e_cclib}. Attempting native fallback parser...")
        data = None
        parser = None

    has_coords = False
    if data is not None and getattr(data, "atomcoords", None) is not None:
        coords_attr = getattr(data, "atomcoords")
        if hasattr(coords_attr, "__len__") and len(coords_attr) > 0:
            has_coords = True

    if parser is None or data is None or not has_coords:
        # Native fallback parser
        orca_fallback = _parse_orca_logfile_fallback(path)
        if orca_fallback is not None:
            return orca_fallback

        if parser is None:
            raise ValueError(
                f"Could not automatically detect the QM program format of '{os.path.basename(path)}'.\n"
                "Supported programs include newBRORSEN, PySCF, Gaussian, ORCA, openLOWDIN, Q-Chem, NWChem, Molpro, GAMESS, xTB, etc."
            )
        raise ValueError(f"Could not extract coordinates from output file: {path}")

    program_name = parser.__class__.__name__.replace("parser", "").capitalize()

    # Extract atomic symbols
    atomnos = getattr(data, "atomnos", [])
    symbols = [ATOMIC_NUMBER_TO_SYMBOL.get(num, f"X{num}") for num in atomnos]
    if not symbols:
        raise ValueError(f"No atoms found in output file: {path}")

    # Extract coordinate trajectories (optimization steps)
    atomcoords = getattr(data, "atomcoords", [])
    if len(atomcoords) == 0:
        raise ValueError(f"No coordinate data found in output file: {path}")

    trajectory_coords = [np.array(step, dtype=np.float64) for step in atomcoords]

    # Extract energies (scfenergies in eV)
    scfenergies = getattr(data, "scfenergies", [])
    trajectory_energies = list(scfenergies) if scfenergies is not None else []

    # Guard against cclib versions that return only 1 frame for multi-step ORCA optimizations
    if "orca" in program_name.lower() and len(trajectory_coords) <= 1:
        orca_fb_traj = _parse_orca_logfile_fallback(path)
        if orca_fb_traj is not None and len(orca_fb_traj.trajectory_coords) > len(trajectory_coords):
            trajectory_coords = orca_fb_traj.trajectory_coords
            if orca_fb_traj.trajectory_energies:
                trajectory_energies = orca_fb_traj.trajectory_energies

    # Final geometry is the last optimization step
    final_coords = trajectory_coords[-1]

    # Extract vibrational modes if present
    vib_analysis = None
    vibfreqs = getattr(data, "vibfreqs", None)
    vibdisps = getattr(data, "vibdisps", None)
    vibirs = getattr(data, "vibirs", None)
    vibramans = getattr(data, "vibramans", None)

    if vibfreqs is not None and len(vibfreqs) > 0 and vibdisps is not None:
        # cclib vibdisps has shape (n_modes, n_atoms, 3)
        ir_intensities = list(vibirs) if vibirs is not None else None
        raman_activities = list(vibramans) if vibramans is not None else None
        vib_analysis = VibrationalAnalysis(
            frequencies=list(vibfreqs),
            displacements=np.array(vibdisps, dtype=np.float64),
            symbols=symbols,
            base_coords=final_coords,
            source=f"{program_name} QM Output",
            ir_intensities=ir_intensities,
            raman_activities=raman_activities,
        )

    comment = f"{program_name} calculation: {len(trajectory_coords)} geometry step(s)"
    if vib_analysis:
        comment += f", {vib_analysis.n_modes} vibrational mode(s)"

    # Extract UV-Vis excited state transitions if present
    et_energies = None
    et_oscs = None
    raw_et = getattr(data, "etenergies", None)
    raw_oscs = getattr(data, "etoscs", None)
    if raw_et is not None and len(raw_et) > 0:
        # cclib stores etenergies in cm^-1; convert to wavelength (nm): lambda_nm = 1e7 / w_cm-1
        et_energies = [1e7 / max(1.0, float(w)) for w in raw_et]
        et_oscs = [float(f) for f in raw_oscs] if raw_oscs is not None else [1.0] * len(et_energies)
        comment += f", {len(et_energies)} electronic transition(s)"
    else:
        # Fallback for ORCA 5/6 TD-DFT tables or Raman if cclib missed them
        orca_fb = _parse_orca_logfile_fallback(path)
        if orca_fb is not None:
            if orca_fb.et_energies:
                et_energies = orca_fb.et_energies
                et_oscs = orca_fb.et_oscs
                comment += f", {len(et_energies)} electronic transition(s)"
            if vib_analysis is not None and vib_analysis.raman_activities is None and orca_fb.vibrational_analysis:
                vib_analysis.raman_activities = orca_fb.vibrational_analysis.raman_activities

        if et_energies is None and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f_log:
                    log_txt = f_log.read()
                # Gaussian format: Excited State   1:      Singlet-A      3.5421 eV   350.04 nm  f=0.0421
                g_td = re.findall(
                    r'Excited\s+State\s+\d+:\s*(?:[^\d\n]+)?\s*([+-]?\d+\.?\d*)\s*eV\s+([+-]?\d+\.?\d*)\s*nm\s+f\s*=\s*([+-]?\d+\.?\d*)',
                    log_txt,
                    re.IGNORECASE,
                )
                if g_td:
                    et_energies = [float(m[1]) for m in g_td]
                    et_oscs = [float(m[2]) for m in g_td]
                    comment += f", {len(et_energies)} electronic transition(s)"
            except Exception as e:
                log.debug("Gaussian TD-DFT extraction fallback notice: %s", e)

    # Extract gradients / forces if present
    nuclear_forces = None
    trajectory_max_forces = None
    trajectory_rms_forces = None
    grads = getattr(data, "grads", None)
    if grads is not None and len(grads) > 0:
        # Most recent optimization step gradients (n_atoms, 3)
        last_g = np.asarray(grads[-1], dtype=np.float64)
        # Force = -gradient
        nuclear_forces = -last_g
        try:
            trajectory_max_forces = []
            trajectory_rms_forces = []
            for g_step in grads:
                g_arr = np.asarray(g_step, dtype=np.float64)
                atom_norms = np.linalg.norm(g_arr, axis=1)
                trajectory_max_forces.append(float(np.max(atom_norms)))
                trajectory_rms_forces.append(float(np.sqrt(np.mean(atom_norms ** 2))))
        except Exception:
            trajectory_max_forces = None
            trajectory_rms_forces = None

    # Extract HOMO/LUMO, Dipole, and Thermochemistry from cclib or raw log text
    extra: Dict[str, Optional[float]] = {
        "zpe_ha": getattr(data, "zpve", None),
        "enthalpy_ha": getattr(data, "enthalpy", None),
        "gibbs_ha": getattr(data, "freeenergy", None),
        "dipole_debye": None,
        "homo_ev": None,
        "lumo_ev": None,
    }
    try:
        moenergies = getattr(data, "moenergies", None)
        homos = getattr(data, "homos", None)
        if moenergies is not None and len(moenergies) > 0 and homos is not None and len(homos) > 0:
            mo_arr = moenergies[0]
            h_idx = int(homos[0])
            if 0 <= h_idx < len(mo_arr):
                extra["homo_ev"] = float(mo_arr[h_idx])
            if 0 <= h_idx + 1 < len(mo_arr):
                extra["lumo_ev"] = float(mo_arr[h_idx + 1])
        moments = getattr(data, "moments", None)
        if moments is not None and len(moments) >= 2:
            dip_vec = np.asarray(moments[1], dtype=np.float64)
            extra["dipole_debye"] = float(np.linalg.norm(dip_vec))
    except Exception:
        log.debug("Ignored non-fatal exception", exc_info=True)

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f_txt:
            full_text = f_txt.read()
            raw_extra = _extract_thermo_and_frontier_from_text(full_text)
        for k, v in raw_extra.items():
            if extra.get(k) is None and v is not None:
                extra[k] = v
    except Exception:
        full_text = ""
        log.debug("Ignored non-fatal exception", exc_info=True)

    # Detect if this is an Intrinsic Reaction Coordinate (IRC) calculation
    is_irc = False
    irc_pts = None
    irc_pathway = None
    if full_text:
        is_irc = bool(
            "IRC-Calculation" in full_text
            or "Intrinsic Reaction Coordinate" in full_text
            or "Reaction coordinate path" in full_text
            or re.search(r'\bIRC\b', full_text, re.IGNORECASE)
        )
        if is_irc:
            # Extract IRC points if Gaussian or ORCA prints "Point Number: X ... Path Number: Y" or "Net reaction coordinate"
            rxn_coords = re.findall(r'(?:Net reaction coordinate|Reaction coordinate|Path coordinate|Total mass-weighted coordinate)\s*[:=]\s*([+-]?\d+\.\d+)', full_text, re.IGNORECASE)
            if rxn_coords:
                try:
                    irc_pts = [float(x) for x in rxn_coords]
                except ValueError:
                    irc_pts = None

            try:
                from .irc_sweeper import parse_gaussian_irc_text, parse_orca_irc_text, assemble_irc_pathway
                if "Gaussian" in program_name or "Point Number:" in full_text:
                    irc_pathway = parse_gaussian_irc_text(full_text)
                elif "ORCA" in program_name or "ORCA INTRINSIC REACTION COORDINATE" in full_text:
                    irc_pathway = parse_orca_irc_text(full_text)

                if irc_pathway is None and trajectory_coords and trajectory_energies:
                    e_ha = [e / 27.211386245988 for e in trajectory_energies]
                    irc_pathway = assemble_irc_pathway(
                        symbols=symbols,
                        coords_list=trajectory_coords,
                        energies_ha=e_ha,
                        rxn_coords=irc_pts,
                    )

                if irc_pathway and irc_pathway.n_points > 0:
                    trajectory_coords = irc_pathway.get_coords_trajectory()
                    trajectory_energies = [e * 27.211386245988 for e in irc_pathway.get_energies_ha()]
                    irc_pts = irc_pathway.get_reaction_coords()
                    comment += f", ordered IRC pathway ({irc_pathway.n_points} steps, barrier: +{irc_pathway.delta_e_fwd_kcal:.1f} kcal/mol)"
            except Exception:
                log.debug("Ignored non-fatal exception assembling IRC pathway", exc_info=True)

    # Extract Molecular Orbitals if present in cclib data
    parsed_orbitals = None
    parsed_homo_idx = None
    try:
        moenergies = getattr(data, "moenergies", None)
        homos = getattr(data, "homos", None)
        if moenergies is not None and len(moenergies) > 0 and len(moenergies[0]) > 0:
            from .volumetric import MoldenMO
            alpha_energies = moenergies[0]
            mosyms = getattr(data, "mosyms", None)
            alpha_syms = mosyms[0] if (mosyms is not None and len(mosyms) > 0 and len(mosyms[0]) == len(alpha_energies)) else None
            h_idx = int(homos[0]) if (homos is not None and len(homos) > 0) else -1
            parsed_homo_idx = h_idx if h_idx >= 0 else None

            parsed_orbitals = []
            for m_i, ev_e in enumerate(alpha_energies):
                ha_e = float(ev_e) / 27.211386245988
                occ = 2.0 if m_i <= h_idx else 0.0
                sym = str(alpha_syms[m_i]) if alpha_syms is not None else f"MO {m_i+1}"
                parsed_orbitals.append(MoldenMO(sym=sym, energy=ha_e, spin="Alpha", occupancy=occ, coefficients={}))
    except Exception:
        log.debug("Ignored non-fatal exception extracting MOs from cclib data", exc_info=True)

    nmr_sh = _extract_nmr_shieldings_from_text(full_text, symbols=symbols) if full_text else None
    if nmr_sh:
        comment += f", {len(nmr_sh)} NMR shielding(s)"

    return QMLogParseResult(
        program_name=program_name,
        symbols=symbols,
        trajectory_coords=trajectory_coords,
        trajectory_energies=trajectory_energies,
        vibrational_analysis=vib_analysis,
        converged=True,
        comment=comment,
        et_energies=et_energies,
        et_oscs=et_oscs,
        nuclear_forces=nuclear_forces,
        orbitals=parsed_orbitals,
        homo_idx=parsed_homo_idx,
        is_irc=is_irc,
        irc_path_points=irc_pts,
        irc_pathway=irc_pathway,
        trajectory_max_forces=trajectory_max_forces,
        trajectory_rms_forces=trajectory_rms_forces,
        nmr_shieldings=nmr_sh,
        **extra,
    )
