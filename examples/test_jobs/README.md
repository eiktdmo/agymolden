# Quick Verification Jobs for AgyMolden

This directory contains lightweight sample inputs to test local and remote quantum chemistry engines.

## 1. newBRORSEN Test Job
- **`CH4_SP_HF_cc-pvdz.inp`**: Methane single point calculation with Hartree-Fock / cc-pVDZ.
  - Runtime: ~5-15 seconds.
  - Can be launched directly from AgyMolden:
    - Open AgyMolden -> **File -> Open newBRORSEN Input File (*.inp)**.
    - Click **▶ Launch Calculation** in the Job Monitor.
    - Select your Python environment and `newBRORSEN.v04.09.py`.

- **`C4H4O_SP_HF_cc-pvdz.inp`**: Furan single point calculation with cc-pVDZ.

## 2. ORCA Test Job
- **`h2o_orca.inp`**: Water geometry optimization + vibrational frequencies.
  - Runtime: ~10 seconds.
  - Command line test: `orca h2o_orca.inp > h2o_orca.out`

## 3. Z-Matrix Test Structures
- **`bp.zmat`**: Biphenyl Z-matrix with dihedral angle parameters.
- **`oh.zmat`**: Hydroxyl radical Z-matrix.
