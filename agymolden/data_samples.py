"""Curated Sample and Tutorial Chemistry Systems for AgyMolden.

Provides instant 1-click loading of benchmark systems ranging from organometallics,
nanostructures, and transition states to periodic crystals and hydrogen bonding networks.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple


class ChemistrySample(NamedTuple):
    title: str
    category: str
    formula: str
    description: str
    xyz_data: str


SAMPLE_CATALOG: List[ChemistrySample] = [
    ChemistrySample(
        title="Ferrocene (D5d staggered)",
        category="Organometallics & Sandwich Complexes",
        formula="Fe(C5H5)2",
        description="Archetypal metallocene featuring an Fe(II) center sandwiched between two cyclopentadienyl aromatic rings.",
        xyz_data="""21
Ferrocene (Fe(C5H5)2) staggered D5d
Fe   0.000000   0.000000   0.000000
C    1.215000   0.000000   1.650000
C    0.375456   1.155554   1.650000
C   -0.982956   0.714170   1.650000
C   -0.982956  -0.714170   1.650000
C    0.375456  -1.155554   1.650000
H    2.164500   0.000000   1.650000
H    0.668869   2.058607   1.650000
H   -1.751119   1.272304   1.650000
H   -1.751119  -1.272304   1.650000
H    0.668869  -2.058607   1.650000
C   -1.215000   0.000000  -1.650000
C   -0.375456  -1.155554  -1.650000
C    0.982956  -0.714170  -1.650000
C    0.982956   0.714170  -1.650000
C   -0.375456   1.155554  -1.650000
H   -2.164500   0.000000  -1.650000
H   -0.668869  -2.058607  -1.650000
H    1.751119  -1.272304  -1.650000
H    1.751119   1.272304  -1.650000
H   -0.668869   2.058607  -1.650000
"""
    ),
    ChemistrySample(
        title="Diels-Alder Transition State (Cyclopentadiene + Ethylene)",
        category="Reaction Mechanisms & Transition States",
        formula="C7H10 TS",
        description="Synchronous pericyclic [4+2] cycloaddition saddle point with forming partial C...C bonds at ~2.1 Å.",
        xyz_data="""17
Diels-Alder [4+2] Transition State
C   -1.196324   0.742301  -0.456012
C   -1.196324  -0.742301  -0.456012
C    0.043512  -1.298412  -0.092104
C    0.854120   0.000000   0.210512
C    0.043512   1.298412  -0.092104
C    0.892301   0.681204  -1.890124
C    0.892301  -0.681204  -1.890124
H   -2.112045   1.304512  -0.561204
H   -2.112045  -1.304512  -0.561204
H    0.165201  -2.341205   0.142012
H    1.421500   0.000000   1.135201
H    1.612045   0.000000  -0.568102
H    0.165201   2.341205   0.142012
H    1.452104   1.295412  -2.581023
H    0.412056   1.210452  -1.072045
H    1.452104  -1.295412  -2.581023
H    0.412056  -1.210452  -1.072045
"""
    ),
    ChemistrySample(
        title="Buckminsterfullerene C60",
        category="Nanostructures & Carbon Allotropes",
        formula="C60",
        description="Truncated icosahedron cage consisting of 12 pentagons and 20 hexagons with Ih point group symmetry.",
        xyz_data="""60
Buckminsterfullerene C60 Ih
C   0.000000   0.714500   3.435700
C   0.000000  -0.714500   3.435700
C   1.171800   1.411600   3.027400
C  -1.171800   1.411600   3.027400
C   1.171800  -1.411600   3.027400
C  -1.171800  -1.411600   3.027400
C   1.896300   0.714500   2.368300
C  -1.896300   0.714500   2.368300
C   1.896300  -0.714500   2.368300
C  -1.896300  -0.714500   2.368300
C   0.000000   2.548400   2.368300
C   0.000000  -2.548400   2.368300
C   0.724500   3.074700   1.300900
C  -0.724500   3.074700   1.300900
C   0.724500  -3.074700   1.300900
C  -0.724500  -3.074700   1.300900
C   2.620800   1.411600   1.300900
C  -2.620800   1.411600   1.300900
C   2.620800  -1.411600   1.300900
C  -2.620800  -1.411600   1.300900
C   1.171800   3.266100   0.000000
C  -1.171800   3.266100   0.000000
C   1.171800  -3.266100   0.000000
C  -1.171800  -3.266100   0.000000
C   2.343600   2.235600   0.000000
C  -2.343600   2.235600   0.000000
C   2.343600  -2.235600   0.000000
C  -2.343600  -2.235600   0.000000
C   3.068100   0.714500   0.000000
C  -3.068100   0.714500   0.000000
C   3.068100  -0.714500   0.000000
C  -3.068100  -0.714500   0.000000
C   0.724500   3.074700  -1.300900
C  -0.724500   3.074700  -1.300900
C   0.724500  -3.074700  -1.300900
C  -0.724500  -3.074700  -1.300900
C   2.620800   1.411600  -1.300900
C  -2.620800   1.411600  -1.300900
C   2.620800  -1.411600  -1.300900
C  -2.620800  -1.411600  -1.300900
C   0.000000   2.548400  -2.368300
C   0.000000  -2.548400  -2.368300
C   1.896300   0.714500  -2.368300
C  -1.896300   0.714500  -2.368300
C   1.896300  -0.714500  -2.368300
C  -1.896300  -0.714500  -2.368300
C   1.171800   1.411600  -3.027400
C  -1.171800   1.411600  -3.027400
C   1.171800  -1.411600  -3.027400
C  -1.171800  -1.411600  -3.027400
C   0.000000   0.714500  -3.435700
C   0.000000  -0.714500  -3.435700
C   2.620800   0.000000   1.300900
C  -2.620800   0.000000   1.300900
C   2.620800   0.000000  -1.300900
C  -2.620800   0.000000  -1.300900
C   0.000000   0.000000   3.541000
C   0.000000   0.000000  -3.541000
C   3.541000   0.000000   0.000000
C  -3.541000   0.000000   0.000000
"""
    ),
    ChemistrySample(
        title="Water Hexamer Prism (H2O)6",
        category="Clusters & Non-Covalent Interactions",
        formula="(H2O)6",
        description="Global lowest-energy prism isomer displaying 9 cooperative directional hydrogen bonds.",
        xyz_data="""18
Water Hexamer Prism Cluster
O   -1.1012   -0.9631   -0.8421
H   -1.4215   -0.0821   -0.6120
H   -0.1251   -0.9324   -0.8124
O    1.4921   -0.8312   -0.7612
H    1.8210    0.0451   -0.5421
H    1.5214   -1.3912    0.0124
O   -0.1245    1.4921   -0.6512
H   -0.7312    1.6214    0.0821
H    0.7321    1.2514   -0.2814
O   -1.4821   -0.0214    1.5214
H   -1.2914   -0.8921    1.0912
H   -0.7124    0.5124    1.4215
O    1.0912   -0.9821    1.4912
H    0.1512   -0.8124    1.6124
H    1.4512   -0.1612    1.1821
O    0.1214    1.4124    1.4512
H    0.8512    1.0912    0.9124
H   -0.1821    0.6912    1.9821
"""
    ),
]
