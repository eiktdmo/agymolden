# Complete Guide: Building Windows Standalone Binaries via GitHub Actions

This guide explains how to compile the standalone Windows executable (`agymolden.exe` inside a `.zip`) directly from your Linux machine using GitHub's free cloud infrastructure.

---

## 🛠️ Step 1: Workflow Configuration Created

A complete GitHub Actions workflow has been configured at:
[`.github/workflows/build_standalone.yml`](file:///home/andres/Desktop/newagy/.github/workflows/build_standalone.yml)

### What it does automatically:
1. Spins up a native **Windows Server 2022** runner and an **Ubuntu 22.04** runner in the cloud simultaneously.
2. Installs Python 3.11, PyQt6, OpenGL, NumPy, SciPy, Matplotlib, Scikit-Image, and PyInstaller.
3. Compiles the directory bundle (`--onedir`) using [`agymolden.spec`](file:///home/andres/Desktop/newagy/agymolden.spec).
4. Compresses the output into:
   - **Windows:** `agymolden-windows-x64.zip` (containing `agymolden.exe` + all DLLs)
   - **Linux:** `agymolden-linux-x64.tar.gz`
5. Uploads the ready-to-download `.zip` files as workflow artifacts.
6. If you push a release tag (e.g. `v1.0.0`), it automatically publishes them to a **GitHub Release** page for students to download directly.

---

## 🚀 Step 2: Initialize Git & Push to GitHub

In your project directory (`/home/andres/Desktop/newagy`), run the following commands:

### 1. Initialize Git (if not already done):
```bash
git init
git add .
git commit -m "Initial commit of AgyMolden with GitHub Actions build workflow"
```

### 2. Connect to your GitHub Repository:
1. Create a new repository on [GitHub](https://github.com/new) (e.g., `agymolden`).
2. Link your local project and push:
```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/agymolden.git
git push -u origin main
```

---

## 📥 Step 3: Download the Windows `.zip` for Your Students

1. Open your repository in your web browser:
   `https://github.com/YOUR_USERNAME/agymolden`
2. Click on the **Actions** tab at the top.
3. You will see the **"Build Multi-Platform Standalone Binaries"** workflow running.
4. Once completed (~3 to 4 minutes), click on the run.
5. Under the **Artifacts** section at the bottom of the page, click:
   - **`agymolden-windows-x64`** $\to$ Downloads the complete, self-contained Windows package!

---

## 🏷️ Optional: Creating Official Student Releases (Permanent Direct Link)

To generate a permanent, public download link (e.g., for a class syllabus or LMS):

```bash
git tag v1.0.0
git push origin v1.0.0
```

GitHub Actions will automatically create a **Release** under `https://github.com/YOUR_USERNAME/agymolden/releases`, attaching both:
- `agymolden-windows-x64.zip`
- `agymolden-linux-x64.tar.gz`

Students can simply click the link, unzip, and double-click `agymolden.exe` with zero installation required!
