"""
Aromatic HOMA & Delocalization Index Analyzer Plugin for AgyMolden.

Computes the Harmonic Oscillator Model of Aromaticity (HOMA) and ring planarity metrics
for detected conjugated carbo- and heterocyclic rings in the active molecular structure.
"""

from typing import Dict, Any, List
import numpy as np

from agymolden.plugin_api import PluginBase, PluginContext, register_plugin
from agymolden.aromatic import compute_ring_homa_and_planarity, find_aromatic_rings
from agymolden.forcefield import detect_connectivity


@register_plugin(
    name="Aromatic Ring HOMA & Planarity Descriptor",
    version="1.1.0",
    author="AgyMolden Computational Chemistry Team",
    description="Identifies conjugated carbocycles and calculates HOMA indices and RMSD planarity deviations.",
    menu_label="🧪 Aromatic HOMA & Ring Planarity",
    category="Analysis",
)
class AromaticHomaPlugin(PluginBase):

    def run(self, context: PluginContext) -> Dict[str, Any]:
        symbols = context.symbols
        coords = context.coords
        n = len(symbols)

        if n < 3 or len(coords) != n:
            context.show_message("Aromatic Analysis", "At least 3 atoms required for ring analysis.")
            return {"rings": [], "homa": []}

        bonds = context.get_bonds()
        if not bonds:
            bonds = detect_connectivity(symbols, coords)

        rings = find_aromatic_rings(symbols, coords, bonds)
        results = []
        for r_indices in rings:
            res = compute_ring_homa_and_planarity(symbols, coords, r_indices)
            results.append({
                "atoms": r_indices,
                "size": len(r_indices),
                "homa": res.get("homa", 0.0),
                "planarity_rmsd": res.get("planarity_rmsd", 0.0),
            })

        summary = f"Detected {len(rings)} aromatic ring(s)."
        if results:
            summary += f" Top HOMA: {results[0]['homa']:.3f} (RMSD: {results[0]['planarity_rmsd']:.3f} Å)"
        context.status_message(summary, 5000)

        return {"rings": rings, "results": results}
