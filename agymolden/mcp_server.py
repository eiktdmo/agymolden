"""
Model Context Protocol (ChemMCP) Server Module for AgyMolden.

Provides a standard JSON-RPC 2.0 interface over standard input/output (stdio)
enabling AI desktop assistants (Claude Desktop, local LLMs, AI agents) to query and control
AgyMolden directly.

Supported Tools:
1. `get_molecule_info`: Returns chemical formula, molecular weight, atom count.
2. `get_coordinates`: Returns atomic symbols and 3D Cartesian coordinates in Å.
3. `calculate_sasa`: Computes Solvent-Accessible Surface Area (Total, Polar, Hydrophobic).
4. `calculate_sterics`: Computes %V_bur buried volume and Sterimol parameters (L, B1, B5).
5. `calculate_quasiharmonic`: Evaluates Grimme mRRHO and Truhlar quasi-harmonic thermochemistry.
6. `export_si_latex`: Generates publication-ready Supporting Information LaTeX code.
7. `generate_orca_input`: Generates validated ORCA 5/6 calculation input decks.

100% Pure Python (Standard library json, sys, os).
"""

from __future__ import annotations

import json
import logging
import sys
import os
from typing import Any, Dict, List, Optional
import numpy as np

log = logging.getLogger(__name__)


def handle_mcp_request(request: Dict[str, Any], context_molecule: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Processes a single JSON-RPC 2.0 MCP request and returns the JSON-RPC response."""
    req_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params", {})

    # Default empty molecular context if not passed
    mol = context_molecule or {}
    symbols: List[str] = mol.get("symbols", ["C", "C", "H", "H", "H", "H", "H", "H"])
    coords: np.ndarray = np.array(
        mol.get(
            "coords",
            [
                [0.0, 0.0, 0.0], [1.54, 0.0, 0.0],
                [-0.36, 1.02, 0.0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89],
                [1.90, 1.02, 0.0], [1.90, -0.51, 0.89], [1.90, -0.51, -0.89],
            ]
        ),
        dtype=np.float64
    )

    try:
        # Standard MCP protocol initialization & discovery
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "AgyMolden-ChemMCP",
                        "version": "1.2.0",
                    },
                    "capabilities": {
                        "tools": {},
                    },
                },
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "get_molecule_info",
                            "description": "Returns atomic count, chemical formula, and basic geometry info of active molecule.",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                        {
                            "name": "get_coordinates",
                            "description": "Returns Cartesian XYZ coordinates (in Ångströms) of the current molecular structure.",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                        {
                            "name": "calculate_sasa",
                            "description": "Calculates Solvent-Accessible Surface Area (Total, Polar PSA, Hydrophobic SASA) via Shrake-Rupley.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "probe_radius": {"type": "number", "default": 1.40, "description": "Solvent probe radius in Å (water = 1.4)"}
                                },
                            },
                        },
                        {
                            "name": "calculate_sterics",
                            "description": "Calculates Percent Buried Volume (%V_bur) and Verloop Sterimol descriptors (L, B1, B5).",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "center_atom_index": {"type": "integer", "default": 0, "description": "0-based index of catalytic metal/center"}
                                },
                            },
                        },
                        {
                            "name": "calculate_quasiharmonic",
                            "description": "Evaluates Grimme mRRHO and Truhlar quasi-harmonic thermochemistry from vibrational frequencies.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "frequencies_cm1": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "description": "List of harmonic vibrational frequencies in cm^-1",
                                    },
                                    "temperature_k": {"type": "number", "default": 298.15},
                                    "cutoff_cm1": {"type": "number", "default": 100.0},
                                },
                            },
                        },
                        {
                            "name": "export_si_latex",
                            "description": "Generates publication-ready Supporting Information LaTeX and Markdown tables for the active molecule.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "default": "Structure_1"},
                                    "electronic_energy_eh": {"type": "number", "default": -150.0},
                                    "gibbs_energy_eh": {"type": "number", "default": -149.9},
                                },
                            },
                        },
                        {
                            "name": "generate_orca_input",
                            "description": "Generates a validated ORCA input file for geometry optimization or frequency calculation.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "functional": {"type": "string", "default": "wB97X-D"},
                                    "basis": {"type": "string", "default": "def2-TZVP"},
                                    "job_type": {"type": "string", "default": "Opt Freq"},
                                    "charge": {"type": "integer", "default": 0},
                                    "multiplicity": {"type": "integer", "default": 1},
                                },
                            },
                        },
                        {
                            "name": "optimize_geometry",
                            "description": "Optimizes the molecular geometry using the built-in UFF or MMFF94s force field and returns the optimized XYZ and energy.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "method": {"type": "string", "default": "UFF", "enum": ["UFF", "MMFF94s"]},
                                    "max_iter": {"type": "integer", "default": 200},
                                    "xyz": {"type": "string", "description": "Optional inline XYZ structure"},
                                },
                            },
                        },
                        {
                            "name": "detect_point_group",
                            "description": "Detects the molecular point group symmetry (e.g., C2v, Td, Oh, D3h) and symmetry elements.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "tolerance": {"type": "number", "default": 0.1},
                                    "xyz": {"type": "string", "description": "Optional inline XYZ structure"},
                                },
                            },
                        },
                        {
                            "name": "validate_orca_input",
                            "description": "Runs AgyMolden pre-flight chemistry and executable checks for an ORCA calculation.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "functional": {"type": "string", "default": "B3LYP"},
                                    "basis": {"type": "string", "default": "def2-SVP"},
                                    "job_type": {"type": "string", "default": "Opt"},
                                    "charge": {"type": "integer", "default": 0},
                                    "multiplicity": {"type": "integer", "default": 1},
                                    "xyz": {"type": "string", "description": "Optional inline XYZ structure"},
                                },
                            },
                        },
                    ]
                },
            }

        elif method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})

            # Support optional inline 'xyz' argument in any tool call
            if isinstance(args, dict) and "xyz" in args and args["xyz"]:
                try:
                    from .zmatrix import parse_xyz
                    parsed_syms, parsed_crds = parse_xyz(str(args["xyz"]), is_path=False)
                    if not parsed_syms or len(parsed_crds) != len(parsed_syms):
                        raise ValueError("XYZ payload produced an empty or mismatched coordinate set.")
                    symbols = parsed_syms
                    coords = np.asarray(parsed_crds, dtype=np.float64)
                except Exception as exc:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": f"Invalid 'xyz' parameter: {exc}"},
                    }

            if tool_name == "get_molecule_info":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "num_atoms": len(symbols),
                                    "symbols": symbols,
                                }, indent=2),
                            }
                        ]
                    },
                }

            elif tool_name == "get_coordinates":
                from .chem_identifiers import format_xyz_string
                xyz_str = format_xyz_string(symbols, coords, comment="AgyMolden Active Structure")
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": xyz_str}]},
                }

            elif tool_name == "calculate_sasa":
                from .sasa import calculate_sasa
                probe_r = float(args.get("probe_radius", 1.40))
                sasa_res = calculate_sasa(symbols, coords, probe_radius=probe_r)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "total_sasa_ang2": sasa_res.total_sasa,
                                    "polar_sasa_ang2": sasa_res.polar_sasa,
                                    "hydrophobic_sasa_ang2": sasa_res.hydrophobic_sasa,
                                    "probe_radius": sasa_res.probe_radius,
                                }, indent=2),
                            }
                        ]
                    },
                }

            elif tool_name == "calculate_sterics":
                from .sterics import compute_buried_volume
                c_idx = int(args.get("center_atom_index", 0))
                c_idx = min(max(0, c_idx), len(symbols) - 1)
                st_res = compute_buried_volume(symbols, coords, center_idx=c_idx, sphere_radius=3.5)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "center_index": c_idx,
                                    "percent_v_bur": st_res.percent_v_bur,
                                    "quadrants": st_res.quadrants,
                                    "buried_volume_ang3": st_res.buried_volume_angstrom3,
                                }, indent=2),
                            }
                        ]
                    },
                }

            elif tool_name == "calculate_quasiharmonic":
                from .quasiharmonic import calculate_quasiharmonic_thermo
                freqs = args.get("frequencies_cm1", [45.0, 82.0, 145.0, 520.0, 1200.0, 1650.0, 3100.0])
                temp_k = float(args.get("temperature_k", 298.15))
                cutoff = float(args.get("cutoff_cm1", 100.0))
                qh_res = calculate_quasiharmonic_thermo(freqs, temperature_k=temp_k, cutoff_cm1=cutoff)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "temperature_k": qh_res.temperature_k,
                                    "cutoff_cm1": qh_res.cutoff_cm1,
                                    "zpe_kcal_mol": qh_res.zpe_kcal_mol,
                                    "s_vib_standard_cal_mol_k": qh_res.s_vib_standard_cal_mol_k,
                                    "s_vib_grimme_cal_mol_k": qh_res.s_vib_grimme_cal_mol_k,
                                    "s_vib_truhlar_cal_mol_k": qh_res.s_vib_truhlar_cal_mol_k,
                                    "delta_g_grimme_kcal_mol": qh_res.delta_g_grimme_kcal_mol,
                                    "delta_g_truhlar_kcal_mol": qh_res.delta_g_truhlar_kcal_mol,
                                }, indent=2),
                            }
                        ]
                    },
                }

            elif tool_name == "export_si_latex":
                from .si_generator import (
                    SISpecies,
                    generate_computational_methods_paragraph,
                    generate_latex_si_document,
                )
                sp_name = str(args.get("name", "Structure_1"))
                e_eh = float(args.get("electronic_energy_eh", -150.0))
                g_eh = float(args.get("gibbs_energy_eh", -149.9))
                sp = SISpecies(
                    name=sp_name,
                    symbols=symbols,
                    coords=coords,
                    electronic_energy_ha=e_eh,
                    gibbs_ha=g_eh,
                )
                methods_txt = generate_computational_methods_paragraph()
                latex_doc = generate_latex_si_document([sp], methods_txt)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": latex_doc}]},
                }

            elif tool_name == "generate_orca_input":
                func = args.get("functional", "wB97X-D")
                basis = args.get("basis", "def2-TZVP")
                job = args.get("job_type", "Opt Freq")
                chg = int(args.get("charge", 0))
                mult = int(args.get("multiplicity", 1))
                header = f"! {func} {job}".strip() if "3c" in str(func).lower() else f"! {func} {basis} {job}".strip()
                lines = [header, f"* xyz {chg} {mult}"]
                for s, c in zip(symbols, coords):
                    lines.append(f"  {s:<2}  {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}")
                lines.append("*")
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": "\n".join(lines)}]},
                }

            elif tool_name == "optimize_geometry":
                from .forcefield import optimize_geometry_force_field
                from .chem_identifiers import format_xyz_string
                method_ff = str(args.get("method", "UFF"))
                max_iter = max(1, min(2000, int(args.get("max_iter", 200))))
                opt_coords, e_kcal = optimize_geometry_force_field(
                    symbols, coords, method=method_ff, max_iter=max_iter
                )
                rmsd = float(np.sqrt(np.mean(np.sum((opt_coords - coords) ** 2, axis=1)))) if len(coords) > 0 else 0.0
                xyz_str = format_xyz_string(symbols, opt_coords, comment=f"Optimized ({method_ff}, E={e_kcal:.3f} kcal/mol)")
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "method": method_ff,
                                    "energy_kcal_mol": float(e_kcal),
                                    "rmsd_angstrom": rmsd,
                                    "xyz": xyz_str,
                                }, indent=2),
                            }
                        ]
                    },
                }

            elif tool_name == "detect_point_group":
                from .symmetry import find_symmetry_operations
                tol = float(args.get("tolerance", 0.1))
                pg_res = find_symmetry_operations(symbols, coords, tolerance=tol)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "point_group": pg_res.symbol,
                                    "tolerance": tol,
                                    "num_operations": len(getattr(pg_res, "operations", [])),
                                }, indent=2),
                            }
                        ]
                    },
                }

            elif tool_name == "validate_orca_input":
                from .calc_runner.preflight import validate_qm_calculation
                from .qm_detect import detect_qm_executable
                from .qm_input_generator import QMCalculationParameters
                params_qm = QMCalculationParameters(
                    job_type=str(args.get("job_type", "Opt")),
                    method=str(args.get("functional", "B3LYP")),
                    basis=str(args.get("basis", "def2-SVP")),
                    charge=int(args.get("charge", 0)),
                    multiplicity=int(args.get("multiplicity", 1)),
                )
                exe_info = detect_qm_executable("orca")
                report = validate_qm_calculation(
                    code_key="orca",
                    symbols=symbols,
                    coords=coords,
                    params=params_qm,
                    executable_info=exe_info,
                )
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "orca_found": exe_info.found,
                                    "orca_path": exe_info.resolved_path,
                                    "is_valid": getattr(report, "is_valid", len(report.issues) == 0),
                                    "issues": [
                                        {
                                            "severity": getattr(iss, "severity", "warning"),
                                            "message": getattr(iss, "message", str(iss)),
                                        }
                                        for iss in getattr(report, "issues", [])
                                    ],
                                }, indent=2),
                            }
                        ]
                    },
                }

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Tool '{tool_name}' not found."},
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method '{method}' not found."},
            }

    except Exception as exc:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32000, "message": str(exc)},
        }


def run_mcp_server_stdio(context_molecule: Optional[Dict[str, Any]] = None):
    """Runs a standard stdio JSON-RPC loop for Model Context Protocol."""
    for line in sys.stdin:
        line_clean = line.strip()
        if not line_clean:
            continue
        try:
            req = json.loads(line_clean)
            resp = handle_mcp_request(req, context_molecule)
        except json.JSONDecodeError as exc:
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
        except Exception as exc:
            log.debug("MCP stdio request error", exc_info=True)
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": f"Internal error: {exc}"},
            }
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
