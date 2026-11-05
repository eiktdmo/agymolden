"""Test Cross-Engine Benchmark & Calculation Comparison Dialog."""

import os
import numpy as np
import pytest

from agymolden.dialog_benchmark import BenchmarkComparisonDialog, BenchmarkEntry


@pytest.fixture
def example_log():
    p = os.path.join(os.path.dirname(__file__), "..", "examples", "AgyMolden_generated_calculation.log")
    if not os.path.isfile(p):
        pytest.skip("example calculation log not found")
    return p


def test_benchmark_dialog_load_and_energetics(qapp, example_log):
    dlg = BenchmarkComparisonDialog(None, initial_files=[example_log])
    assert len(dlg.entries) == 1
    assert dlg.table_energetics.rowCount() == 1

    entry = dlg.entries[0]
    assert entry.program == "Gaussian"
    assert entry.final_energy_ha is not None
    assert abs(entry.delta_e_kcal) < 1e-4

    # Add a mock second calculation to test delta E computation & RMSD
    c2 = entry.coords.copy()
    c2[0, 0] += 0.15  # Internal coordinate deformation
    entry2 = BenchmarkEntry(
        filepath="/tmp/mock_orca.out",
        label="mock_orca.out",
        program="ORCA",
        symbols=list(entry.symbols),
        coords=c2,
        final_energy_ha=entry.final_energy_ha + 0.005,  # 0.005 Ha ~ 3.14 kcal/mol higher
        converged=True,
        n_opt_steps=3,
        homo_ev=-10.40,
        lumo_ev=2.00,
        gap_ev=12.40,
    )
    dlg.entries.append(entry2)
    dlg._recompute_and_refresh()

    assert len(dlg.entries) == 2
    assert dlg.table_energetics.rowCount() == 2
    assert dlg.table_rmsd.rowCount() == 2
    assert dlg.table_rmsd.columnCount() == 2

    # Verify delta E computed properly
    assert dlg.entries[0].delta_e_kcal == 0.0
    assert dlg.entries[1].delta_e_kcal > 2.0

    # Verify RMSD computed
    rmsd_cell = dlg.table_rmsd.item(0, 1).text()
    assert float(rmsd_cell) > 0.0

    # Verify exports
    csv = dlg.get_csv_string()
    assert "File,Program,Final_Energy_Ha" in csv
    assert "Gaussian" in csv and "ORCA" in csv

    md = dlg.get_markdown_table()
    assert "| Calculation | Program |" in md
    assert "mock_orca.out" in md

    # Test signal emission
    emitted = []
    dlg.structure_selected.connect(lambda syms, coords, lbl: emitted.append(lbl))
    dlg.table_energetics.selectRow(1)
    dlg._on_load_viewport_clicked()
    assert len(emitted) == 1
    assert emitted[0] == "mock_orca.out"

    dlg.close()
