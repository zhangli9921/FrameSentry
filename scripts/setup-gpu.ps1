# FrameSentry Windows NVIDIA GPU fresh install
# No full CUDA Toolkit required; onnxruntime-gpu[cuda,cudnn] bundles needed libs.
# Do NOT install both onnxruntime and onnxruntime-gpu.
# No secrets required.
# Compatible with Windows PowerShell 5.1 and PowerShell 7+.

[CmdletBinding()]
param(
    # Safe CI / local regression: exercise native invoker only (no pip/GPU side effects).
    [switch]$NativeInvokerSelfTest
)

$ErrorActionPreference = "Stop"

function Invoke-NativeChecked {
    <#
    .SYNOPSIS
      Run a native executable and judge success by $LASTEXITCODE only.

    Windows PowerShell 5.1 turns native stderr into NativeCommandError when
    $ErrorActionPreference is Stop, even if exit code is 0 (e.g. pip uninstall
    WARNING for a missing package). PowerShell 7+ can also fail native commands
    via $PSNativeCommandUseErrorActionPreference.

    This helper temporarily suppresses that termination for the call, still
    streams stdout/stderr to the host (unless -CaptureStdout), then fails only
    when exit code != 0.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $false)]
        [string[]]$ArgumentList = @(),

        [Parameter(Mandatory = $false)]
        [string]$FailureMessage = "",

        [Parameter(Mandatory = $false)]
        [switch]$CaptureStdout
    )

    $prevEap = $ErrorActionPreference
    $hadNativePref = $false
    $prevNativePref = $null
    if (Test-Path variable:global:PSNativeCommandUseErrorActionPreference) {
        $hadNativePref = $true
        $prevNativePref = $global:PSNativeCommandUseErrorActionPreference
        $global:PSNativeCommandUseErrorActionPreference = $false
    }

    # Local Continue so PS 5.1 does not terminate on stderr ErrorRecords.
    $ErrorActionPreference = "Continue"
    $stdoutText = $null
    try {
        $global:LASTEXITCODE = 0
        if ($CaptureStdout) {
            # Capture stdout objects; stderr ErrorRecords still surface to host via Write-Error stream
            # when they become error records — redirect merge then split by type for display.
            $streams = & $FilePath @ArgumentList 2>&1
            $exitCode = $LASTEXITCODE
            $outLines = New-Object System.Collections.Generic.List[string]
            foreach ($item in @($streams)) {
                if ($item -is [System.Management.Automation.ErrorRecord]) {
                    # Re-emit native stderr text so users still see pip warnings.
                    [Console]::Error.WriteLine([string]$item)
                }
                else {
                    $line = [string]$item
                    [void]$outLines.Add($line)
                    Write-Host $line
                }
            }
            $stdoutText = ($outLines -join "`n").Trim()
        }
        else {
            & $FilePath @ArgumentList
            $exitCode = $LASTEXITCODE
        }
        if ($null -eq $exitCode) {
            $exitCode = 0
        }
    }
    finally {
        $ErrorActionPreference = $prevEap
        if ($hadNativePref) {
            $global:PSNativeCommandUseErrorActionPreference = $prevNativePref
        }
    }

    if ($exitCode -ne 0) {
        $detail = $FailureMessage
        if ([string]::IsNullOrWhiteSpace($detail)) {
            $detail = "Native command failed: $FilePath $($ArgumentList -join ' ')"
        }
        throw "$detail (exit code $exitCode)"
    }

    if ($CaptureStdout) {
        return $stdoutText
    }
}

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList,

        [Parameter(Mandatory = $false)]
        [string]$FailureMessage = "Python command failed",

        [Parameter(Mandatory = $false)]
        [switch]$CaptureStdout
    )
    Invoke-NativeChecked -FilePath "python" -ArgumentList $ArgumentList `
        -FailureMessage $FailureMessage -CaptureStdout:$CaptureStdout
}

function Invoke-NativeInvokerSelfTest {
    Write-Host "=== NativeInvokerSelfTest (no install / no GPU) ===" -ForegroundColor Cyan

    # Scenario A: stderr warning + exit 0 → must succeed under PS 5.1 Stop EAP.
    Write-Host "[self-test A] stderr + exit 0 (expect success)..."
    Invoke-PythonChecked -ArgumentList @(
        "-c",
        "import sys; print('WARN: missing package simulation', file=sys.stderr); sys.exit(0)"
    ) -FailureMessage "Self-test A unexpectedly failed"
    Write-Host "[ok] self-test A: stderr + exit 0 treated as success"

    # Scenario B: stderr + exit 7 → must fail; catch and verify.
    Write-Host "[self-test B] stderr + exit 7 (expect failure detection)..."
    $failedAsExpected = $false
    $caught = $null
    try {
        Invoke-PythonChecked -ArgumentList @(
            "-c",
            "import sys; print('ERROR: simulated failure', file=sys.stderr); sys.exit(7)"
        ) -FailureMessage "Self-test B expected failure"
    }
    catch {
        $failedAsExpected = $true
        $caught = $_
    }
    if (-not $failedAsExpected) {
        throw "Self-test B: wrapper did not fail on non-zero exit code"
    }
    $msg = [string]$caught
    if ($msg -notmatch "exit code 7") {
        throw "Self-test B: failure message missing exit code 7: $msg"
    }
    Write-Host "[ok] self-test B: stderr + exit 7 detected as failure"

    Write-Host "=== NativeInvokerSelfTest passed ===" -ForegroundColor Green
    exit 0
}

if ($NativeInvokerSelfTest) {
    Invoke-NativeInvokerSelfTest
}

Write-Host "=== FrameSentry GPU setup ===" -ForegroundColor Cyan

# 1. Confirm Python 3.11
$pyVer = Invoke-PythonChecked -CaptureStdout -ArgumentList @(
    "-c",
    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
) -FailureMessage "Failed to query Python version"
if ($pyVer -ne "3.11") {
    Write-Error "Python 3.11 required (found $pyVer). Install Python 3.11 and retry."
    exit 1
}
Write-Host "[ok] Python $pyVer"

# 2. Uninstall onnxruntime and onnxruntime-gpu if present
#    Harmless pip WARNING on stderr when a package is absent must NOT abort under PS 5.1.
Write-Host "Uninstalling onnxruntime / onnxruntime-gpu if present..."
Invoke-PythonChecked -ArgumentList @(
    "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"
) -FailureMessage "pip uninstall onnxruntime / onnxruntime-gpu failed"

# 3. Install core GUI / decode deps
Write-Host "Installing PySide6, opencv-python-headless, numpy..."
Invoke-PythonChecked -ArgumentList @(
    "-m", "pip", "install", "--upgrade", "pip"
) -FailureMessage "pip upgrade failed"
Invoke-PythonChecked -ArgumentList @(
    "-m", "pip", "install", "PySide6>=6.5", "opencv-python-headless>=4.8", "numpy>=1.24"
) -FailureMessage "Failed installing PySide6 / OpenCV / numpy"

# 4. NudeNet without pulling its onnxruntime CPU dep
Write-Host "Installing nudenet==3.4.2 (--no-deps)..."
Invoke-PythonChecked -ArgumentList @(
    "-m", "pip", "install", "--no-deps", "nudenet==3.4.2"
) -FailureMessage "Failed installing nudenet==3.4.2 --no-deps"

# 5. GPU ORT (bundles cuda/cudnn wheels — no full Toolkit required)
Write-Host "Installing onnxruntime-gpu[cuda,cudnn]==1.29.0..."
Invoke-PythonChecked -ArgumentList @(
    "-m", "pip", "install", "onnxruntime-gpu[cuda,cudnn]==1.29.0"
) -FailureMessage "Failed installing onnxruntime-gpu[cuda,cudnn]==1.29.0"

# 6. Editable project without re-resolving deps
Write-Host "Installing FrameSentry editable (--no-deps)..."
Invoke-PythonChecked -ArgumentList @(
    "-m", "pip", "install", "-e", ".", "--no-deps"
) -FailureMessage "Failed installing FrameSentry editable --no-deps"

# 7. Verify ORT version / providers; ensure not both distributions present
Write-Host "Verifying onnxruntime..."
Invoke-PythonChecked -ArgumentList @(
    "-c",
    @"
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
) -FailureMessage "ORT verification failed. Setup NOT complete."

# 8. Project-level smoke: actually construct NudeNetBackend(device="gpu")
Write-Host "Smoke-testing FrameSentry NudeNet CUDA InferenceSession..."
Invoke-PythonChecked -ArgumentList @(
    "-c",
    @"
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
) -FailureMessage "NudeNet CUDA session smoke failed. Setup NOT complete."

Write-Host "=== FrameSentry GPU setup complete ===" -ForegroundColor Green
Write-Host "Launch: python -m framesentry   (or framesentry)"
Write-Host "GUI should show 检测到 CUDA Provider：是 ; real CUDA session is verified above."
