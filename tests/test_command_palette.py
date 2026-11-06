"""
Tests for modern Command Palette (Ctrl+K) in AgyMolden.
Verifies command filtering, execution, and UI activation.
"""

import pytest
from PyQt6 import QtCore, QtWidgets
from agymolden.dialog_command_palette import CommandPaletteDialog, CommandItem


def test_command_item_matching():
    executed = []
    item = CommandItem(
        title="Run ORCA Calculation",
        category="Calculate",
        shortcut="Ctrl+R",
        callback=lambda: executed.append("orca"),
        keywords="dft opt freq",
    )

    assert item.matches("orca")
    assert item.matches("ORCA")
    assert item.matches("dft")
    assert item.matches("calculate")
    assert item.matches("run orca")
    assert not item.matches("gaussian")


def test_command_palette_filter_and_execute(qapp):
    executed = []

    commands = [
        CommandItem("Run ORCA Calculation", "Calculate", "Ctrl+R", lambda: executed.append("orca"), "dft"),
        CommandItem("Volumetric Orbitals", "Surfaces", "Ctrl+Alt+O", lambda: executed.append("orbitals"), "homo lumo"),
        CommandItem("Export Scalable Vector Graphics (SVG)", "Export", "", lambda: executed.append("svg"), "vector"),
    ]

    dlg = CommandPaletteDialog(commands=commands)
    assert dlg.list_widget.count() == 3

    # Filter for 'svg'
    dlg.search_edit.setText("svg")
    assert dlg.list_widget.count() == 1
    assert "SVG" in dlg.list_widget.item(0).text()

    # Execute selected
    dlg._execute_selected()
    assert executed == ["svg"]
