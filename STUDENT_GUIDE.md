# AgyMolden Student Quickstart Guide: Running ORCA

Welcome to the Computational Chemistry laboratory exercises! This guide provides a 3-step walkthrough to build molecules, execute ORCA calculations, and analyze results using **AgyMolden**.

---

## 🚀 Step 1: Build or Open Your Molecule

### Option A: Open an Existing Molecule
- Click **File → Open File... (Ctrl+O)** and select any `.xyz` or `.zmat` file (or try an example from `examples/`).

### Option B: Build from Internal Coordinates (Z-Matrix)
1. In the **Z-Matrix** tab on the left:
   - Click **Add Line** to add atoms.
   - Set bond lengths (Å), angles (°), and dihedrals (°).
2. Or use the **Fragments** menu to add functional groups (Methyl, Phenyl, Hydroxyl, etc.).

---

## ⚡ Step 2: Run ORCA (1-Click)

1. Click the blue **"🚀 Run ORCA"** button on the top toolbar (or press **Ctrl+R**).
2. Choose your lab exercise preset:
   - **📐 Geometry Optimization (`! B3LYP def2-SVP Opt`):** Relaxes the molecule to its lowest energy geometry.
   - **🎵 Vibrations & IR Spectrum (`! B3LYP def2-SVP Opt Freq`):** Calculates harmonic vibrational frequencies and IR intensities.
   - **🌐 Molecular Orbitals (`! B3LYP def2-SVP SP`):** Computes single point energy and orbital eigenvalues.
3. Check **Charge** (0 for neutral) and **Multiplicity** (1 for closed-shell singlet).
4. Click **🚀 Run ORCA Now**.

---

## 📊 Step 3: Inspect Real-Time Progress & Results

- **Live 3D Optimization:** As ORCA computes each optimization step, the 3D viewport on your screen updates automatically with the new coordinates and energy.
- **Log Stream:** The **Job Monitor** displays live output.
- **When Complete:**
  - Click **Load Result into Viewer** to inspect final bond lengths and angles.
  - To view IR vibrations: Click the **🎵 Vibrations** toolbar button.
  - To view Molecular Orbitals: Click the **🌐 Orbitals/Grid** toolbar button.

---

## ❓ Troubleshooting & Error Messages

If a calculation finishes with an error notice, AgyMolden will highlight the issue with suggested remedies:
- **Parallel / MPI error:** In the Quick ORCA dialog, set **Cores (%pal)** to `1`.
- **SCF Not Converged:** Add `! SlowConv` to the ORCA route line.
- **ORCA Executable Not Found:** Open **File → Configure QM Engines & Routes** and set the path to your ORCA executable folder.
