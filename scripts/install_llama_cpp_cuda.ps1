# llama-cpp-python AVEC CUDA sur Windows — sans CPU-only.
#
# Problemes connus:
#   - MSBuild + CUDA 13.2.targets => CudaToolkitDir vide (incontournable en build source VS).
#   - --no-build-isolation => ModuleNotFoundError scikit_build_core si PYTHONPATH pas fixe.
#
# Strategie:
#   A) Wheel precompile (cu121/cu124) — driver NVIDIA accepte souvent runtime CUDA plus ancien.
#   B) Build source avec Ninja + PYTHONPATH + deps explicites.
#
# Usage: .\scripts\install_llama_cpp_cuda.ps1
#        .\scripts\install_llama_cpp_cuda.ps1 -ForceSource   # ignore wheels, compile

param(
    [switch]$CpuOnly,
    [switch]$ForceSource
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$venvPip = ".\.venv\Scripts\pip.exe"
$venvPy  = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPip)) { Write-Host "Venv introuvable." -ForegroundColor Red; exit 1 }

cmd /c "$venvPip uninstall llama-cpp-python -y 2>nul"

if ($CpuOnly) {
    & $venvPip install llama-cpp-python --no-cache-dir
    exit $LASTEXITCODE
}

# --- CUDA sur PATH ---
$cudaRoots = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
)
foreach ($root in $cudaRoots) {
    if (Test-Path (Join-Path $root "bin\nvcc.exe")) {
        $env:CUDAToolkit_ROOT = $root
        $env:CUDA_PATH = $root
        $env:CUDA_HOME = $root
        $env:PATH = "$root\bin;$env:PATH"
        Write-Host "[CUDA] $root" -ForegroundColor Green
        break
    }
}

# ========== A) WHEELS PRECOMPILES (recommande) ==========
if (-not $ForceSource) {
    Write-Host "[A] Tentative wheels CUDA (cu121 / cu124)..." -ForegroundColor Cyan
    $indexes = @(
        "https://abetlen.github.io/llama-cpp-python/whl/cu124",
        "https://abetlen.github.io/llama-cpp-python/whl/cu121",
        "https://abetlen.github.io/llama-cpp-python/whl/cu118"
    )
    foreach ($url in $indexes) {
        Write-Host "  -> $url" -ForegroundColor Gray
        & $venvPip install llama-cpp-python --extra-index-url $url --no-cache-dir 2>$null
        if ($LASTEXITCODE -eq 0) {
            & $venvPy -c "from llama_cpp import Llama; print('llama_cpp OK')"
            Write-Host "[OK] Wheel installe depuis $url" -ForegroundColor Green
            exit 0
        }
    }
    Write-Host "[A] Aucun wheel compatible — passage build source." -ForegroundColor Yellow
}

# ========== B) BUILD SOURCE — Ninja + PYTHONPATH ==========
Write-Host "[B] Build source : deps + Ninja + CMAKE_GENERATOR=Ninja..." -ForegroundColor Cyan
& $venvPip install --quiet scikit-build-core cmake ninja packaging pathspec
$sitePkgs = (Resolve-Path ".\.venv\Lib\site-packages").Path
$env:PYTHONPATH = $sitePkgs
$env:PYTHONNOUSERSITE = "1"

& $venvPip install ninja --quiet
$ninjaExe = Join-Path (Get-Location) ".venv\Scripts\ninja.exe"
if (Test-Path $ninjaExe) {
    $env:PATH = "$(Split-Path $ninjaExe -Parent);$env:PATH"
    $ninjaForward = ($ninjaExe -replace '\\','/')
} else {
    $ninjaForward = "ninja"
}

$buildRoot = "C:\llm_cpp_ninja"
if (-not (Test-Path $buildRoot)) { New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null }
$env:TMP = $buildRoot
$env:TEMP = $buildRoot
# Force Ninja partout (CMake + scikit-build)
$env:CMAKE_GENERATOR = "Ninja"
$env:SKBUILD_CMAKE_ARGS = "-DGGML_CUDA=on"
$env:CMAKE_ARGS = "-G Ninja -DGGML_CUDA=on -DCMAKE_MAKE_PROGRAM=$ninjaForward"

Write-Host "[B] pip install --no-build-isolation (PYTHONPATH=site-packages)..." -ForegroundColor Cyan
& $venvPip install llama-cpp-python --no-cache-dir --no-build-isolation --verbose

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Build source reussi." -ForegroundColor Green
    exit 0
}

# ========== C) Wheel communautaire HF (cp311) ==========
Write-Host "[C] Si tu as Python 3.11, wheel HF possible (cu128) — a telecharger a la main :" -ForegroundColor Yellow
Write-Host "  https://huggingface.co/boneylizardwizard/llama_cpp_python-0.3.16-cp312-cp312-win_amd64 (cp312)"
Write-Host "  Cherche cp311-cp311-win_amd64 + cu sur HF puis : pip install chemin\vers\fichier.whl"
Write-Host ""
Write-Host "Sinon : installer CUDA 12.4 EN PLUS de 13.2, puis relancer ce script (Ninja + nvcc 12.4 parfois plus stable avec ggml)." -ForegroundColor Gray
exit 1
