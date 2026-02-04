# Statistical Attention Analysis - Test Checklist

## Pre-Test Setup (5 minutes)

- [ ] **Corpus files exist**
  - [ ] `corpus/test_attention_samples.json` (test corpus, 5 samples)
  - [ ] `corpus/attention_samples.json` (full corpus, 35 samples)

- [ ] **Directory structure**
  - [ ] `outputs/` directory exists (or will be created)
  - [ ] `examples/statistical_attention.rs` exists

- [ ] **Dependencies**
  - [ ] Rust toolchain installed (`cargo --version`)
  - [ ] CUDA available (optional, can use `--cpu`)
  - [ ] ~20GB free disk space (for model download)

## Quick Test Run (10-40 minutes)

### Option 1: Automated Test Script (Recommended)

```powershell
.\test_statistical_attention.ps1
```

### Option 2: Manual Test

```powershell
# Build
cargo build --release --example statistical_attention

# Run test
cargo run --release --example statistical_attention -- `
    --corpus corpus/test_attention_samples.json `
    --output outputs/test_statistical_attention.json `
    --cpu
```

## Validation Checklist

### ✓ Build Phase
- [ ] Code compiles without errors
- [ ] No warnings about missing dependencies
- [ ] Binary created successfully

### ✓ Runtime Phase
- [ ] Model downloads (first run only, ~14GB)
- [ ] Corpus JSON loads successfully
- [ ] Token counts displayed for each sample
- [ ] No "marker not found" errors
- [ ] No "out of bounds" errors
- [ ] All samples analyzed (counts match corpus)

### ✓ Statistics Phase
- [ ] Mean values computed (not zero, not NaN)
- [ ] Standard deviations reasonable (< mean)
- [ ] T-test p-values computed (0.0 to 1.0)
- [ ] Hypothesis validation runs

### ✓ Output Phase
- [ ] Console output shows formatted tables
- [ ] JSON file created at specified path
- [ ] JSON is valid and contains all sections:
  - [ ] `python_doctest` statistics
  - [ ] `rust_test` statistics
  - [ ] `python_baseline` statistics
  - [ ] `rust_baseline` statistics
  - [ ] `python_vs_rust` t-test
  - [ ] `python_test_vs_baseline` t-test
  - [ ] `rust_test_vs_baseline` t-test

## Expected Test Results (Sanity Check)

**If using test corpus:**

| Metric | Expected Range | Red Flag |
|--------|---------------|----------|
| Python mean attention | 5-30% | < 1% or > 50% |
| Rust mean attention | 1-15% | > Python mean |
| Python std dev | 2-10% | > 20% |
| Rust std dev | 1-8% | > 15% |
| Python vs Rust p-value | 0.001-0.5 | = 1.0 (no difference) |

**Red flags that indicate problems:**

- ❌ All attention values are 0.0
- ❌ All p-values are 1.0 (no significance)
- ❌ Means are identical across groups
- ❌ Standard deviations are 0.0
- ❌ "NaN" or "Infinity" in output

## Debugging Failed Tests

### Problem: "Marker not found"
**Diagnosis:** Token positions don't match actual tokenizer output

**Fix:**
1. Run token verification script (Appendix A in RIGOR_EXPERIMENT.md)
2. Print actual tokens for one sample
3. Manually count position of `>>>` or `#[`
4. Update JSON positions

### Problem: Zero attention values
**Diagnosis:** Positions point to wrong tokens or layer has no attention

**Fix:**
1. Verify layer index (try final layer: `--layer 31` for 32-layer model)
2. Check if positions are within sequence length
3. Try different sample to isolate issue

### Problem: CUDA out of memory
**Diagnosis:** Model too large for GPU VRAM

**Fix:**
```powershell
# Use CPU mode
cargo run --release --example statistical_attention -- --cpu

# Or use smaller model (if available)
cargo run --release --example statistical_attention -- `
    --model bigcode/starcoder2-3b
```

### Problem: Very slow on CPU
**Diagnosis:** Normal behavior - transformer inference is slow on CPU

**Fix:**
- Accept slower runtime (~2-5x slower than GPU)
- Use smaller model if available
- Run overnight if needed

## Post-Test Actions

### ✅ If test passes:

1. Review `outputs/test_statistical_attention.json`
2. Check that hypothesis validation makes sense
3. Note the test runtime
4. Extrapolate to full corpus: `test_time × 7`
5. Plan full experiment run time
6. Proceed to full corpus:
   ```powershell
   cargo run --release --example statistical_attention -- `
       --corpus corpus/attention_samples.json `
       --output outputs/stats_qwen_7b.json
   ```

### ❌ If test fails:

1. Check error messages above
2. Review debugging section
3. Fix identified issues
4. Re-run test
5. **Do not proceed to full corpus until test passes**

## Full Experiment Estimate

Based on test results:

- **Test runtime:** _______ minutes
- **Samples in test:** 5
- **Samples in full corpus:** 35
- **Estimated full runtime:** _______ minutes (test_time × 7)

**Recommended:**
- Schedule full run for overnight or dedicated time block
- Monitor first few samples to ensure no errors
- Check disk space for output files

## Support

If issues persist after debugging:
1. Check plip-rs issues: [GitHub](https://github.com/your-repo/plip-rs)
2. Review RIGOR_EXPERIMENT.md Appendix A
3. Verify tokenizer matches model
4. Try with minimal 1-sample corpus to isolate issue
