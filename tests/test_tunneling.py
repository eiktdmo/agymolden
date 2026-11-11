"""Tests for quantum tunneling corrections in Reaction Thermochemistry."""

import pytest
from agymolden.thermochemistry import SpeciesThermo, calculate_reaction_thermochemistry


def test_tunneling_corrections():
    """Verify Wigner and Eckart quantum tunneling factors κ(T) increase at low T."""
    r = SpeciesThermo(name="Reactant", role="reactant", coefficient=1.0, gibbs_ha=-100.0, enthalpy_ha=-100.0)
    # TS with imaginary frequency of 1200 cm^-1 (typical hydrogen transfer / proton transfer)
    ts = SpeciesThermo(
        name="TS", role="ts", coefficient=1.0, gibbs_ha=-99.98, enthalpy_ha=-99.98, imaginary_freq_cm1=1200.0
    )
    p = SpeciesThermo(name="Product", role="product", coefficient=1.0, gibbs_ha=-100.02, enthalpy_ha=-100.02)

    # At 298.15 K:
    res_298 = calculate_reaction_thermochemistry([r, ts, p], temperature_k=298.15)
    assert res_298.tunneling_kappa_wigner is not None
    assert res_298.tunneling_kappa_wigner > 1.5  # Significant tunneling for 1200 cm^-1 at 298K
    assert res_298.rate_constant_k_fwd_tunneled > res_298.rate_constant_k_fwd

    # At lower temperature (200 K), tunneling transmission κ should be even higher
    res_200 = calculate_reaction_thermochemistry([r, ts, p], temperature_k=200.0)
    assert res_200.tunneling_kappa_wigner > res_298.tunneling_kappa_wigner
