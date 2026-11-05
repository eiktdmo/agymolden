# AgyMolden One-Click Portable Package

This bundle runs **AgyMolden** natively on **Windows**, **macOS**, and **Linux** without compiling binaries or using GitHub.

---

## How to Run:

### 🪟 On Windows:
- Double-click **`Run_Windows.bat`**.
- *That's it!* On the first run, it sets up an isolated environment (~20 seconds) and automatically launches the GUI. Subsequent double-clicks launch immediately.

### 🍎 On macOS:
- Double-click or run in Terminal:
  ```bash
  bash Run_Mac_or_Linux.sh
  ```

### 🐧 On Linux:
- Double-click or run:
  ```bash
  ./Run_Mac_or_Linux.sh
  ```

---

### What makes this foolproof:
1. **Zero interference**: It creates its own isolated `agymolden_env` subfolder so it never messes with the user's system Python or other software.
2. **Self-updating**: If you drop a newer `.whl` into this folder, double-clicking the launcher script updates the installation automatically.
3. **Completely offline**: All application logic is inside the `.whl` package.
