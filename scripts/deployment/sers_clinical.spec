# PyInstaller spec for SERS Clinical Webapp
# Build with: pyinstaller scripts/deployment/sers_clinical.spec
#
# Output: dist/SERS_Clinical/SERS_Clinical.exe

import sys
from pathlib import Path

# When running PyInstaller, __file__ may not be defined.
# Use SPECPATH which is the directory of this .spec file.
PROJECT_ROOT = Path(SPECPATH).resolve().parents[1]

# Files to bundle (templates, static files, model artifacts)
datas = [
    # Templates and static
    (str(PROJECT_ROOT / 'scripts' / 'deployment' / 'templates'), 'scripts/deployment/templates'),
    (str(PROJECT_ROOT / 'scripts' / 'deployment' / 'static'), 'scripts/deployment/static'),

    # Model artifacts (Stacking V2)
    (str(PROJECT_ROOT / 'models' / 'production_stacking'), 'models/production_stacking'),

    # Fallback LR model
    (str(PROJECT_ROOT / 'models' / 'production'), 'models/production'),

    # Source code
    (str(PROJECT_ROOT / 'src' / 'sers'), 'src/sers'),
    (str(PROJECT_ROOT / 'scripts' / 'analysis' / 'stacking_ensemble.py'),
     'scripts/analysis'),
    (str(PROJECT_ROOT / 'scripts' / 'analysis' / 'weekend_experiments' / 'stacking_optimization.py'),
     'scripts/analysis/weekend_experiments'),
]

hiddenimports = [
    # Web framework
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'fastapi',
    'starlette',
    'starlette.middleware',
    'jinja2',
    'multipart',

    # ML / data
    'sklearn',
    'sklearn.linear_model',
    'sklearn.ensemble',
    'sklearn.pipeline',
    'sklearn.preprocessing',
    'sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel',
    'sklearn.neighbors._partition_nodes',
    'scipy.signal',
    'scipy.special',
    'scipy.optimize',
    'scipy.integrate',
    'xgboost',
    'joblib',

    # Project modules
    'scripts.deployment.sers_clinical_webapp',
    'scripts.deployment.sers_predict',
    'scripts.deployment.clinical_db',
    'scripts.deployment.clinical_auth',
    'scripts.deployment.clinical_i18n',
    'scripts.deployment.clinical_audit',
    'scripts.deployment.clinical_report',
    'src.sers.io',
    'src.sers.preprocessing',
]

block_cipher = None

a = Analysis(
    [str(PROJECT_ROOT / 'scripts' / 'deployment' / 'launcher.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'PIL',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'mlflow',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SERS_Clinical',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False to hide console window (use False for production)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / 'scripts' / 'deployment' / 'static' / 'icon.ico')
        if (PROJECT_ROOT / 'scripts' / 'deployment' / 'static' / 'icon.ico').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SERS_Clinical',
)
