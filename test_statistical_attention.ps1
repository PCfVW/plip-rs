# Quick Test Script for Statistical Attention Analysis
# Tests the implementation before running the full multi-hour experiment

Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  PLIP-rs: Statistical Attention - Quick Test" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if corpus file exists
Write-Host "Step 1: Checking test corpus..." -ForegroundColor Yellow
$corpusPath = "corpus/test_attention_samples.json"
if (Test-Path $corpusPath) {
    Write-Host "  ✓ Test corpus found: $corpusPath" -ForegroundColor Green
} else {
    Write-Host "  ✗ Test corpus not found: $corpusPath" -ForegroundColor Red
    exit 1
}

# Step 2: Validate JSON structure
Write-Host "`nStep 2: Validating JSON structure..." -ForegroundColor Yellow
try {
    $corpus = Get-Content $corpusPath | ConvertFrom-Json
    Write-Host "  ✓ JSON is valid" -ForegroundColor Green
    Write-Host "    - Python doctest: $($corpus.python_doctest.Count) samples" -ForegroundColor Gray
    Write-Host "    - Rust test: $($corpus.rust_test.Count) samples" -ForegroundColor Gray
    Write-Host "    - Python baseline: $($corpus.python_baseline.Count) samples" -ForegroundColor Gray
    Write-Host "    - Rust baseline: $($corpus.rust_baseline.Count) samples" -ForegroundColor Gray
} catch {
    Write-Host "  ✗ JSON validation failed: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Build the project
Write-Host "`nStep 3: Building statistical_attention example..." -ForegroundColor Yellow
Write-Host "  (This may take a few minutes on first build)" -ForegroundColor Gray
$buildOutput = cargo build --example statistical_attention 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Build successful" -ForegroundColor Green
} else {
    Write-Host "  ✗ Build failed" -ForegroundColor Red
    Write-Host $buildOutput -ForegroundColor Red
    exit 1
}

# Step 4: Run quick test (CPU mode, smaller model if available)
Write-Host "`nStep 4: Running quick test..." -ForegroundColor Yellow
Write-Host "  Model: Qwen/Qwen2.5-Coder-7B-Instruct (may download ~14GB)" -ForegroundColor Gray
Write-Host "  Mode: CPU (safer for testing)" -ForegroundColor Gray
Write-Host "  Corpus: Test corpus (5 samples total)" -ForegroundColor Gray
Write-Host ""
Write-Host "  NOTE: First run will download the model. This may take 10-30 minutes." -ForegroundColor Yellow
Write-Host "  Press Ctrl+C to cancel if you want to test with a smaller model first." -ForegroundColor Yellow
Write-Host ""

$confirmation = Read-Host "Continue with test run? (y/N)"
if ($confirmation -ne 'y') {
    Write-Host "  Test cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host "`nStarting test run..." -ForegroundColor Cyan
$testStart = Get-Date

cargo run --release --example statistical_attention -- `
    --corpus corpus/test_attention_samples.json `
    --output outputs/test_statistical_attention.json `
    --cpu

$testEnd = Get-Date
$duration = $testEnd - $testStart

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "  ✓ TEST SUCCESSFUL" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "Test duration: $($duration.TotalSeconds) seconds" -ForegroundColor Cyan
    Write-Host "Results saved to: outputs/test_statistical_attention.json" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Review the test output above" -ForegroundColor Gray
    Write-Host "  2. Check outputs/test_statistical_attention.json" -ForegroundColor Gray
    Write-Host "  3. If results look good, run full experiment:" -ForegroundColor Gray
    Write-Host "     cargo run --release --example statistical_attention" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Estimated full experiment runtime:" -ForegroundColor Yellow
    Write-Host "  - With test corpus (5 samples): ~$([math]::Round($duration.TotalMinutes, 1)) minutes" -ForegroundColor Gray
    Write-Host "  - With full corpus (35 samples): ~$([math]::Round($duration.TotalMinutes * 7, 1)) minutes" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Red
    Write-Host "  ✗ TEST FAILED" -ForegroundColor Red
    Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check the error messages above for details." -ForegroundColor Yellow
    exit 1
}
