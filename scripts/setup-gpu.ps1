# FrameSentry Windows NVIDIA GPU fresh install
# No full CUDA Toolkit required; onnxruntime-gpu[cuda,cudnn] bundles needed libs.
# Do NOT install both onnxruntime and onnxruntime-gpu.
# No secrets required.

$ErrorActionPreference = "Stop"

Write-Host "=== FrameSentry GPU setup ===" -ForegroundColor Cyan

# 1. Confirm Python 3.11
$pyVer = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($pyVer -ne "3.11") {
    Write-Error "Python 3.11 required (found $pyVer). Install Python 3.11 and retry."
    exit 1
}
Write-Host "[ok] Python $pyVer"

# 2. Uninstall onnxruntime and onnxruntime-gpu if present
Write-Host "Uninstalling onnxruntime / onnxruntime-gpu if present..."
python -m pip uninstall -y onnxruntime onnxruntime-gpu 2>$null
# pip uninstall returns non-zero when packages are absent; ignore that.
$ErrorActionPreference = "Stop"

# 3. Install core GUI / decode deps
Write-Host "Installing PySide6, opencv-python-headless, numpy..."
python -m pip install --upgrade pip
python -m pip install "PySide6>=6.5" "opencv-python-headless>=4.8" "numpy>=1.24"

# 4. NudeNet without pulling its onnxruntime CPU dep
Write-Host "Installing nudenet==3.4.2 (--no-deps)..."
python -m pip install --no-deps "nudenet==3.4.2"

# 5. GPU ORT (bundles cuda/cudnn wheels — no full Toolkit required)
Write-Host "Installing onnxruntime-gpu[cuda,cudnn]==1.29.0..."
python -m pip install "onnxruntime-gpu[cuda,cudnn]==1.29.0"

# 6. Editable project without re-resolving deps
Write-Host "Installing FrameSentry editable (--no-deps)..."
python -m pip install -e . --no-deps

# 7. Verify ORT version / providers; ensure not both distributions present
Write-Host "Verifying onnxruntime..."
python -c @"
import importlib.metadata as md
import sys

dists = {d.metadata['Name'].lower() for d in md.distributions()
         if d.metadata['Name'].lower() in ('onnxruntime', 'onnxruntime-gpu')}
print('distributions:', sorted(dists))
if 'onnxruntime' in dists and 'onnxruntime-gpu' in dists:
    print('ERROR: both onnxruntime and onnxruntime-gpu are installed', file=sys.stderr)
    sys.exit(2)
if 'onnxruntime-gpu' not in dists and 'onnxruntime' not in dists:
    print('ERROR: no onnxruntime distribution found', file=sys.stderr)
    sys.exit(3)

import onnxruntime as ort
print('onnxruntime.__version__ =', ort.__version__)
providers = ort.get_available_providers()
print('get_available_providers() =', providers)
if 'CUDAExecutionProvider' not in providers:
    print('ERROR: CUDAExecutionProvider not listed in get_available_providers(). '
          'Check NVIDIA driver / onnxruntime-gpu install.', file=sys.stderr)
    sys.exit(4)
print('[ok] CUDAExecutionProvider listed')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Error "ORT verification failed (exit $LASTEXITCODE). Setup NOT complete."
    exit $LASTEXITCODE
}

# 8. Project-level smoke: actually construct NudeNetBackend(device="gpu")
#    and verify the live session activated CUDAExecutionProvider.
Write-Host "Smoke-testing FrameSentry NudeNet CUDA InferenceSession..."
python -c @"
import sys

from framesentry.detectors.nudenet_backend import NudeNetBackend

try:
    backend = NudeNetBackend(device='gpu')
except Exception as exc:
    print(f'ERROR: NudeNetBackend(device="gpu") failed: {exc}', file=sys.stderr)
    sys.exit(5)

active = list(backend.session.get_providers())
print('active providers:', active)
if not active or active[0] != 'CUDAExecutionProvider':
    if 'CUDAExecutionProvider' not in active:
        print(
            'ERROR: CUDAExecutionProvider not active on NudeNet session. '
            f'Active: {active}',
            file=sys.stderr,
        )
        sys.exit(6)
    print(
        'ERROR: CUDAExecutionProvider is not the first/active primary provider. '
        f'Active: {active}',
        file=sys.stderr,
    )
    sys.exit(6)

print('FrameSentry NudeNet CUDA session: OK')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Error "NudeNet CUDA session smoke failed (exit $LASTEXITCODE). Setup NOT complete."
    exit $LASTEXITCODE
}

Write-Host "=== FrameSentry GPU setup complete ===" -ForegroundColor Green
Write-Host "Launch: python -m framesentry   (or framesentry)"
Write-Host "GUI should show 检测到 CUDA Provider：是 ; real CUDA session is verified above."
