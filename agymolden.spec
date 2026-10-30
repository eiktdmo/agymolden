# -*- mode: python ; coding: utf-8 -*-
# Build with:  pyinstaller --noconfirm agymolden.spec   (see build.sh / build.bat)
# PyInstaller's built-in PyQt6 hook collects only the Qt modules that are imported.
import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = (
    ['agymolden']
    + collect_submodules('agymolden')
    + [
        'PyQt6.QtOpenGLWidgets',
        'scipy.special',
        'scipy.spatial.transform',
        'skimage.measure',
        'cclib',
    ]
)

datas = [('USER_MANUAL.md', '.'), ('README.md', '.')]
if os.path.isdir('agymolden/resources'):
    datas.append(('agymolden/resources', 'agymolden/resources'))
if os.path.isdir('agymolden/data'):
    datas.append(('agymolden/data', 'agymolden/data'))
if os.path.isdir('examples'):
    datas.append(('examples', 'examples'))

# Exclude heavy optional/unused packages so the standalone binary stays lean (~65-80 MB)
# External QM engines (PySCF, ORCA, Gaussian, openLOWDIN, xTB) run out-of-process via QProcess.
excludes = [
    'tkinter',
    'pandas',
    'IPython',
    'pytest',
    'pyscf',
    'rdkit',
    'basis_set_exchange',
    'PyQt6.QtWebEngineCore',
    'PyQt6.QtWebEngineWidgets',
    'PyQt6.QtQml',
    'PyQt6.QtQuick',
    'PyQt6.QtMultimedia',
    'PyQt6.QtNetwork',
    'PyQt6.QtSql',
    'PyQt6.QtPdf',
    'PyQt6.QtDBus',
    'PyQt6.QtBluetooth',
    'PyQt6.QtPositioning',
    'PyQt6.QtSensors',
    'PyQt6.QtSerialPort',
    'PyQt6.QtTest',
    'PyQt6.QtXml',
    'scipy.spatial.distance.tests',
    'scipy.stats.tests',
    'numpy.tests',
]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# Strip unnecessary heavy non-code assets (e.g. ICU data tables, duplicate BLAS libs, test suites)
# Preserves 100% of calculation code, scientific solvers, and mathematical precision
filtered_binaries = []
seen_lib_names = set()
for b in a.binaries:
    dest_name = os.path.basename(b[0]).lower()
    # Strip heavy foreign locale database (saves ~31 MB without affecting standard text/formulas)
    if 'libicudata' in dest_name:
        continue
    # De-duplicate redundant OpenBLAS shared library copies if both NumPy and SciPy vendor them
    if 'openblas' in dest_name:
        if dest_name in seen_lib_names:
            continue
        seen_lib_names.add(dest_name)
    filtered_binaries.append(b)
a.binaries = filtered_binaries

# Strip test suites and non-essential translation files
filtered_datas = []
for d in a.datas:
    dest = d[0].lower()
    if 'qt6/translations' in dest and not dest.endswith('_en.qm'):
        continue
    if any(t in dest for t in ['/tests/', '/test/', 'scipy/tests', 'numpy/tests', 'skimage/tests']):
        continue
    filtered_datas.append(d)
a.datas = filtered_datas

pyz = PYZ(a.pure)

if os.environ.get('AGYMOLDEN_ONEDIR', '0') == '1':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='agymolden',
        debug=False,
        bootloader_ignore_signals=False,
        strip=(sys.platform != 'win32'),
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=(sys.platform == 'darwin'),
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=sys.platform.startswith('linux'),
        upx=True,
        upx_exclude=[],
        name='agymolden',
    )
    if sys.platform == 'darwin':
        app = BUNDLE(
            coll,
            name='AgyMolden.app',
            icon=None,
            bundle_identifier='org.agymolden.app',
            info_plist={
                'CFBundleShortVersionString': '1.0.0',
                'CFBundleVersion': '1.0.0',
                'NSHighResolutionCapable': 'True',
            },
        )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='agymolden',
        debug=False,
        bootloader_ignore_signals=False,
        strip=sys.platform.startswith('linux'),
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=(sys.platform == 'darwin'),
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    if sys.platform == 'darwin':
        app = BUNDLE(
            exe,
            name='AgyMolden.app',
            icon=None,
            bundle_identifier='org.agymolden.app',
            info_plist={
                'CFBundleShortVersionString': '1.0.0',
                'CFBundleVersion': '1.0.0',
                'NSHighResolutionCapable': 'True',
            },
        )

