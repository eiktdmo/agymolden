"""Test NBO Second-Order Perturbation Analyzer and Dialog."""

import pytest
from agymolden.nbo_analyzer import parse_nbo_perturbation_table, NBOPerturbation
from agymolden.dialog_nbo import NBODonorAcceptorDialog


SAMPLE_NBO_OUTPUT = """
 Second Order Perturbation Theory Analysis of Fock Matrix in NBO Basis

 Threshold for printing:   0.50 kcal/mol
                                                                  E(2)  E(j)-E(i) F(i,j)
      Donor NBO (i)                       Acceptor NBO (j)      kcal/mol   a.u.    a.u.
 =======================================================================================
  within unit  1
   1. BD ( 1) C   1 - H   2       /111. RY*( 1) C   1             0.63    2.44   0.035
   2. BD ( 1) C   1 - H   3       / 74. BD*( 1) C   1 - C   4     3.12    1.15   0.054
  13. LP ( 1) N   5               / 74. BD*( 1) C   1 - C   4    14.85    0.42   0.075
  14. LP ( 1) O   6               / 75. BD*( 2) C   1 - O   2    22.30    0.35   0.088
  15. BD ( 2) C   1 - C   4       / 76. BD*( 2) C   2 - C   3    18.40    0.28   0.065
 Natural Bond Orbitals
"""


def test_parse_nbo_perturbation_table():
    interactions = parse_nbo_perturbation_table(SAMPLE_NBO_OUTPUT)
    assert len(interactions) == 5

    # Check highest energy first (sorted descending)
    top = interactions[0]
    assert top.donor_idx == 14
    assert "LP" in top.donor_label
    assert top.energy_kcal == 22.30
    assert top.energy_diff_au == 0.35
    assert top.fock_element_au == 0.088

    # Verify classification
    assert top.interaction_type == "n -> π*"

    # Verify atom index extraction
    assert 5 in top.donor_atoms  # O 6 is 0-based index 5
    assert 0 in top.acceptor_atoms and 1 in top.acceptor_atoms  # C 1 - O 2 is 0 and 1

    # Verify hyperconjugation
    hyp = [i for i in interactions if "Hyperconjugation" in i.interaction_type]
    assert len(hyp) >= 1
    assert hyp[0].donor_idx == 2


def test_nbo_dialog_filtering_and_export(qapp):
    dlg = NBODonorAcceptorDialog(None, log_filepath_or_text=SAMPLE_NBO_OUTPUT, is_path=False)
    assert len(dlg.all_interactions) == 5

    # Default min_e is 2.0 kcal/mol -> excludes 0.63
    assert len(dlg.filtered_interactions) == 4
    assert dlg.table.rowCount() == 4

    # Filter to min_e = 15.0 kcal/mol -> only 22.30 and 18.40
    dlg.spin_min_e.setValue(15.0)
    assert len(dlg.filtered_interactions) == 2

    # Filter by type: n -> π*
    dlg.spin_min_e.setValue(0.5)
    dlg.combo_type.setCurrentText("n -> π* (Resonance / Conjugation)")
    assert len(dlg.filtered_interactions) == 1
    assert dlg.filtered_interactions[0].energy_kcal == 22.30

    # Exports
    csv = dlg.get_csv_string()
    assert "Donor_NBO,Acceptor_NBO" in csv
    assert "22.30" in csv

    md = dlg.get_markdown_string()
    assert "| Donor NBO (i) | Acceptor NBO (j) |" in md
    assert "22.30" in md

    # Signal emission
    emitted = []
    dlg.interaction_highlighted.connect(lambda d, a, e, l: emitted.append((d, a, e)))
    dlg.table.selectRow(0)
    assert len(emitted) == 1
    assert emitted[0][2] == 22.30

    dlg.close()
