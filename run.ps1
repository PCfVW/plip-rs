# PLIP-rs environment setup and runner
# Usage: .\run.ps1 [command]
# Examples:
#   .\run.ps1 build
#   .\run.ps1 load_model
#   .\run.ps1 inference
#   .\run.ps1 tokenize

# Set CUDA environment (using CUDA 12.6 for candle 0.9 compatibility)
$env:CUDA_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_ROOT\bin;$env:PATH"
$env:NVCC_CCBIN = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"

$command = $args[0]

switch ($command) {
    "build" {
        cargo build --release --all-targets
    }
    "rebuild" {
        Write-Host "Forcing clean rebuild..." -ForegroundColor Cyan
        # Remove the binary to force rebuild
        if (Test-Path .\target\release\plip-rs.exe) {
            Remove-Item .\target\release\plip-rs.exe
        }
        cargo build --release --examples
    }
    "clippy" {
        Write-Host "Running clippy..." -ForegroundColor Cyan
        cargo clippy --release -- -W clippy::all
    }
    "clippy_pedantic" {
        Write-Host "Running pedantic clippy..." -ForegroundColor Cyan
        cargo clippy --features cuda -- -W clippy::all -W clippy::pedantic -A clippy::module_name_repetitions -A clippy::must_use_candidate -A clippy::missing_errors_doc -A clippy::missing_panics_doc
    }
    "clippy_fix" {
        Write-Host "Auto-fixing clippy warnings..." -ForegroundColor Cyan
        cargo clippy --features cuda --fix --allow-dirty --lib -p plip-rs
    }
    "load_model" {
        Write-Host "Starting load_model example (CUDA)..." -ForegroundColor Cyan
        & .\target\release\examples\load_model.exe
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "load_model_cpu" {
        Write-Host "Starting load_model example (CPU mode)..." -ForegroundColor Cyan
        & .\target\release\examples\load_model.exe --cpu
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "inference" {
        Write-Host "Starting inference example..." -ForegroundColor Cyan
        & .\target\release\examples\inference.exe
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "tokenize" {
        Write-Host "Starting tokenize example..." -ForegroundColor Cyan
        & .\target\release\examples\tokenize.exe
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "logit_lens" {
        Write-Host "Running Logit Lens on Rust code..." -ForegroundColor Cyan
        & .\target\release\examples\logit_lens.exe
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "logit_lens_detailed" {
        Write-Host "Running Logit Lens (detailed)..." -ForegroundColor Cyan
        & .\target\release\examples\logit_lens.exe --detailed --top-k 10
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "logit_lens_python" {
        Write-Host "Running Logit Lens on Python code..." -ForegroundColor Cyan
        & .\target\release\examples\logit_lens.exe --code "def add(a, b):`n    return a + b`n"
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "list_tensors" {
        Write-Host "Listing tensor names from safetensors..." -ForegroundColor Cyan
        & .\target\release\examples\list_tensors.exe
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "test_emergence" {
        Write-Host "Running Test-Awareness Logit Lens Experiment..." -ForegroundColor Cyan
        & .\target\release\examples\test_emergence.exe
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "attention" {
        Write-Host "Running Attention Pattern Analysis..." -ForegroundColor Cyan
        & .\target\release\examples\attention_patterns.exe
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "run" {
        Write-Host "Running full PLIP experiment..." -ForegroundColor Cyan
        & .\target\release\plip-rs.exe --corpus corpus/samples.json --output outputs/ --verbose
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "run_cpu" {
        Write-Host "Running PLIP experiment on CPU..." -ForegroundColor Cyan
        & .\target\release\plip-rs.exe --corpus corpus/samples.json --output outputs/ --verbose --cpu
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "test" {
        Write-Host "Running tests with CUDA..." -ForegroundColor Cyan
        cargo test --features cuda
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "test_corpus" {
        Write-Host "Running corpus tests with CUDA..." -ForegroundColor Cyan
        cargo test corpus --features cuda
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "test_gpu" {
        Write-Host "Running GPU-dependent tests (ignored tests)..." -ForegroundColor Cyan
        cargo test --features cuda -- --ignored
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    }
    "check" {
        Write-Host "=== Environment Check ===" -ForegroundColor Green
        Write-Host "CUDA_ROOT: $env:CUDA_ROOT"
        Write-Host ""

        # Check nvcc
        Write-Host "Checking nvcc..." -ForegroundColor Cyan
        if (Get-Command nvcc -ErrorAction SilentlyContinue) {
            nvcc --version | Select-Object -First 4
        } else {
            Write-Host "ERROR: nvcc not found in PATH" -ForegroundColor Red
        }
        Write-Host ""

        # Check CUDA DLLs (CUDA 12.6 uses version 12 DLLs)
        Write-Host "Checking CUDA DLLs..." -ForegroundColor Cyan
        $cudaDlls = @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")
        foreach ($dll in $cudaDlls) {
            $path = "$env:CUDA_ROOT\bin\$dll"
            if (Test-Path $path) {
                Write-Host "  OK: $dll" -ForegroundColor Green
            } else {
                Write-Host "  MISSING: $dll" -ForegroundColor Red
            }
        }
        Write-Host ""

        # List all DLLs that load_model.exe needs
        Write-Host "Checking executable dependencies..." -ForegroundColor Cyan
        if (Test-Path ".\target\release\examples\load_model.exe") {
            Write-Host "  load_model.exe exists" -ForegroundColor Green
        } else {
            Write-Host "  load_model.exe NOT FOUND - run '.\run.ps1 build' first" -ForegroundColor Red
        }
    }
    default {
        Write-Host "PLIP-rs runner" -ForegroundColor Green
        Write-Host "Usage: .\run.ps1 [command]"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  build          - Build with CUDA support"
        Write-Host "  clippy         - Run clippy linter"
        Write-Host "  check          - Check CUDA environment and dependencies"
        Write-Host "  test           - Run all tests with CUDA"
        Write-Host "  test_corpus    - Run corpus tests with CUDA"
        Write-Host "  test_gpu       - Run GPU-dependent tests (model loading, activations)"
        Write-Host "  load_model     - Download and load StarCoder2-3B (CUDA)"
        Write-Host "  load_model_cpu - Download and load StarCoder2-3B (CPU mode)"
        Write-Host "  inference      - Run inference example"
        Write-Host "  tokenize       - Run tokenizer example"
        Write-Host "  logit_lens     - Run Logit Lens on Rust code"
        Write-Host "  logit_lens_detailed - Run Logit Lens with detailed output"
        Write-Host "  logit_lens_python   - Run Logit Lens on Python code"
        Write-Host "  run            - Run main CLI (PLIP experiment)"
    }
}
