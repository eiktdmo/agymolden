"""Tests for NBO Second-Order Perturbation E(2) Network & Chord diagram."""

import pytest
from PyQt6 import QtWidgets

from agymolden.nbo_analyzer import NBOPerturbation
from agymolden.dialog_nbo_network import NBONetworkDialog


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_nbo_network_dialog(qapp):
    """Verify NBO Network dialog initializes, filters by E(2), and renders bipartite/chord layouts."""
    interactions = [
        NBOPerturbation(
            donor_idx=1,
            donor_label="LP ( 1) O 3",
            donor_type="LP",
            donor_atoms=[3],
            acceptor_idx=42,
            acceptor_label="BD*( 1) C 1 - H 5",
            acceptor_type="BD*",
            acceptor_atoms=[1, 5],
            interaction_type="n -> σ*",
            energy_kcal=12.45,
            energy_diff_au=0.85,
            fock_element_au=0.092,
        ),
        NBOPerturbation(
            donor_idx=2,
            donor_label="BD ( 1) C 1 - C 2",
            donor_type="BD",
            donor_atoms=[1, 2],
            acceptor_idx=43,
            acceptor_label="BD*( 1) C 2 - H 6",
            acceptor_type="BD*",
            acceptor_atoms=[2, 6],
            interaction_type="σ -> σ* (Hyperconjugation)",
            energy_kcal=3.80,
            energy_diff_au=0.92,
            fock_element_au=0.052,
        ),
        NBOPerturbation(
            donor_idx=3,
            donor_label="LP ( 2) O 3",
            donor_type="LP",
            donor_atoms=[3],
            acceptor_idx=44,
            acceptor_label="BD*( 1) C 2 - H 7",
            acceptor_type="BD*",
            acceptor_atoms=[2, 7],
            interaction_type="n -> σ*",
            energy_kcal=0.95,
            energy_diff_au=1.10,
            fock_element_au=0.028,
        ),
    ]

    dlg = NBONetworkDialog(interactions=interactions)
    assert len(dlg.interactions) == 3

    # Default min_e is 2.0 -> top 2 interactions shown
    assert "Showing top 2 of 2" in dlg.lbl_stats.text()

    # Switch to Circular / Chord layout
    dlg.combo_layout.setCurrentIndex(1)
    dlg.update_network_plot()
    assert "NBO E(2) Chord Graph" in dlg.fig.axes[0].get_title()

    # Switch to Bipartite layout
    dlg.combo_layout.setCurrentIndex(0)
    dlg.update_network_plot()
    assert "NBO Second-Order Perturbation Network" in dlg.fig.axes[0].get_title()
