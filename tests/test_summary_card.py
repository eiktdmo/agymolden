import pytest
import numpy as np
from agymolden.qm_log_parser import QMLogParseResult
from agymolden.vibrations import VibrationalAnalysis
from agymolden.dialog_summary_card import QMSummaryCardDialog


def test_qm_summary_card_data():
    res = QMLogParseResult(
        program_name="ORCA 6.0",
        symbols=["O", "H", "H"],
        trajectory_coords=[np.zeros((3, 3)), np.zeros((3, 3))],
        trajectory_energies=[-76.4321, -76.4398],
        vibrational_analysis=VibrationalAnalysis(
            frequencies=[1595.0, 3657.0, 3756.0],
            displacements=np.zeros((3, 3, 3)),
            symbols=["O", "H", "H"],
            base_coords=np.zeros((3, 3)),
            source="ORCA",
            ir_intensities=[25.0, 10.0, 45.0],
        ),
        dipole_debye=1.854,
        homo_ev=-12.61,
        lumo_ev=1.25,
        zpe_ha=0.021,
    )
    assert res.program_name == "ORCA 6.0"
    assert len(res.symbols) == 3
    assert len(res.trajectory_energies) == 2
    assert res.dipole_debye == 1.854
    assert res.homo_ev == -12.61
    assert res.lumo_ev == 1.25
