# Using UNAL VPN with AgyMolden for Remote Calculations

This guide explains how to connect **AgyMolden** to UNAL remote clusters (e.g., Bourbaki or campus servers) through the GlobalProtect VPN (`vpnpa.unal.edu.co`) without ever opening a terminal or manually typing credentials.

---

## 1. Prerequisites (Done Once)

Ensure you ran the setup script on the computer:
```bash
./configure_unal_vpn_connection.sh
```
This generates the non-interactive VPN script at:
`~/.local/bin/unal-vpn-toggle.sh`

---

## 2. Setting Up the Remote Calculation in AgyMolden

### Step 1: Open Your Input in AgyMolden
1. Launch AgyMolden:
   ```bash
   ./run.py
   ```
2. Build your molecule or open an existing `.inp` file:
   - **File → Open newBRORSEN Input File (*.inp)** (e.g. from `examples/test_jobs/CH4_SP_HF_cc-pvdz.inp`).

### Step 2: Open the Launch Configuration
1. Open the **Job Monitor** (or click **▶ Launch Calculation**).
2. The **Launch newBRORSEN Calculation** modal dialog will appear.
3. Click on the second tab: **🌐 Remote Server (SSH / VPN)**.

---

## 3. Configure the Remote & VPN Fields

Fill out the fields in the **Remote Server (SSH / VPN)** tab:

| Field | What to Put | Explanation |
| :--- | :--- | :--- |
| **VPN Command** | `~/.local/bin/unal-vpn-toggle.sh` | AgyMolden will execute this script in the background to bring up the UNAL VPN automatically before testing or running. |
| **Remote Host (SSH)** | `your_user@remote_host_or_ip` | e.g. `areyesv@bourbaki.unal.edu.co` or your cluster's IP address. |
| **SSH Private Key** | `~/.ssh/id_rsa` *(Optional)* | Path to your SSH key (leave blank if using password). |
| **SSH Password** | *Your cluster password* | Required only if you don't use SSH key authentication. |
| **Remote Workdir** | `~/agymolden_scratch` | Directory on the remote cluster where calculations will run. |
| **Remote Python** | `python3` | Python binary on the cluster (or path to cluster conda environment). |
| **Remote Wrapper** | `newBRORSEN.v04.09.py` | Name or path of the wrapper script on the cluster. |

---

## 4. One-Click Verification & Saving Profile

1. **Test the Connection:**
   - Click the blue button: **🔌 Test Connection (Verify Reachability)**.
   - AgyMolden will:
     1. Automatically call `~/.local/bin/unal-vpn-toggle.sh` to ensure the VPN is up.
     2. Ping and SSH into the remote server.
     3. Show a green confirmation: **✓ Connected successfully**.
2. **Save the Profile:**
   - Click **💾 Save Profile**. You only have to type this configuration once; next time it will be pre-filled!
3. *(Optional)* If the remote server does not have the latest wrapper script yet:
   - Click **📤 Upload Local Wrapper to Remote** to upload `newBRORSEN.v04.09.py` directly to the cluster with one click.

---

## 5. Launching Remote Calculations

1. Click **▶ Launch Calculation** (green button).
2. AgyMolden will:
   - Verify the UNAL VPN tunnel is active.
   - Securely transfer your input file (`.inp`) to the remote server.
   - Launch `newBRORSEN` or `ORCA` on the remote cluster.
   - **Stream live output directly into the AgyMolden Job Monitor on your local screen.**
   - Update the **3D OpenGL viewport in real-time** as optimization steps converge on the remote server.
3. When finished, click **Load Result into Viewer** to inspect the final geometry, orbitals, or spectra.
