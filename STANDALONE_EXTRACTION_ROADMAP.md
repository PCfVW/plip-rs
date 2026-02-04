# PLIP-rs Standalone Repository Extraction Roadmap

**Purpose**: Extract PLIP-rs into a standalone GitHub repository for AIware 2026 supplementary materials
**Created**: February 4, 2026
**Status**: Planning Phase

---

## Overview

This document provides a step-by-step roadmap for extracting PLIP-rs from its current location (`experiment/plip-rs/` within the Priority Queues repository) into a standalone GitHub repository suitable for:

1. **Supplementary materials** for AIware 2026 submission
2. **Reproducibility** - reviewers and readers can clone and run experiments
3. **Citation target** - persistent DOI via Zenodo integration
4. **Independent visibility** - separate repo can gain recognition independently

### Current State

| Aspect | Status |
|--------|--------|
| Location | `experiment/plip-rs/` in [PCfVW/d-Heap-priority-queue](https://github.com/PCfVW/d-Heap-priority-queue) |
| Version | 0.1.0 (recommend bumping to 1.0.0) |
| License | Apache-2.0 |
| README | Comprehensive (240 lines) - exists, needs minor updates |
| Dependencies on parent | 6 path references (2 Docker + 3 example fallbacks + 1 doc link) |
| Code dependencies | None (fully self-contained) |

---

## Phase 1: Pre-Extraction Cleanup (In Priority Queues)

**Goal**: Fix all external path references while still in the parent repository, allowing testing before extraction.

### 1.1 Fix Docker Compose Volume Mounts

**File**: `docker/docker-compose.yml`

| Line | Current | Change To |
|------|---------|-----------|
| 31 | `- ../corpus:/data/corpus:ro` | `- ./corpus:/data/corpus:ro` |
| 60 | `- ../corpus:/data/corpus:ro` | `- ./corpus:/data/corpus:ro` |

**Rationale**: The `../corpus` path assumes plip-rs is a subdirectory. After extraction, corpus will be at `./corpus` relative to the repo root.

### 1.2 Remove Example Corpus Path Fallbacks

The example files already have `corpus/attention_samples_universal.json` as their **primary path** (which works after extraction). The `../corpus/` paths are fallbacks that will become invalid. **Remove them entirely** for cleaner code.

**File**: `examples/ablation_experiment.rs` (lines 208-219)

```rust
// BEFORE: Three candidates with parent directory fallback
let candidates = [
    PathBuf::from("corpus/attention_samples_universal.json"),
    PathBuf::from("experiment/plip-rs/corpus/attention_samples_universal.json"),
    PathBuf::from("../corpus/attention_samples_universal.json"),  // REMOVE
];

// AFTER: Two candidates (remove line 212)
let candidates = [
    PathBuf::from("corpus/attention_samples_universal.json"),
    PathBuf::from("experiment/plip-rs/corpus/attention_samples_universal.json"),
];
```

**File**: `examples/steering_calibrate.rs` (lines 72-83)
- Remove line 76: `PathBuf::from("../corpus/attention_samples_universal.json")`

**File**: `examples/steering_experiment.rs` (lines 107-118)
- Remove line 111: `PathBuf::from("../corpus/attention_samples_universal.json")`

**Rationale**: The primary path `corpus/...` already works from the repo root. The `experiment/plip-rs/corpus/...` fallback supports running from the parent Priority Queues directory during testing. The `../corpus/` fallback is only needed while embedded and should be removed.

### 1.3 Update Documentation Cross-References

**File**: `COMMANDS.md`

| Line | Current | Change To |
|------|---------|-----------|
| 1343 | `- [SEGA COMMANDS.md](../experiment-runner/COMMANDS.md) - Related experiment runner commands` | `- [SEGA experiment-runner](https://github.com/PCfVW/d-Heap-priority-queue/tree/master/experiment/experiment-runner) - Related experiment runner (in parent repository)` |

**Rationale**: After extraction, the relative path won't exist. Link to the parent repo instead.

### 1.4 Update Version Number

**File**: `Cargo.toml`

| Line | Current | Change To |
|------|---------|-----------|
| 3 | `version = "0.1.0"` | `version = "1.0.0"` |

**Rationale**: PLIP-rs has successfully generated all AIware 2026 data, has a stable API, and is feature-complete for its research purpose.

### 1.5 Fix RIKEN Instructions Placeholders

**File**: `docs/RIKEN_INSTRUCTIONS.md`

Replace all `YOUR_USERNAME` placeholders with `PCfVW`:

| Lines | Current | Change To |
|-------|---------|-----------|
| 32, 52, 63, 72 | `ghcr.io/YOUR_USERNAME/plip-rs:h100` | `ghcr.io/PCfVW/plip-rs:h100` |
| 193 | `https://github.com/YOUR_USERNAME/plip-rs/issues` | `https://github.com/PCfVW/plip-rs/issues` |

**Rationale**: Documentation should have working URLs after extraction.

### Phase 1 Checklist

- [ ] Update `docker/docker-compose.yml` line 31
- [ ] Update `docker/docker-compose.yml` line 60
- [ ] Remove `examples/ablation_experiment.rs` line 212 (parent path fallback)
- [ ] Remove `examples/steering_calibrate.rs` line 76 (parent path fallback)
- [ ] Remove `examples/steering_experiment.rs` line 111 (parent path fallback)
- [ ] Update `COMMANDS.md` line 1343
- [ ] Update `Cargo.toml` version to 1.0.0
- [ ] Update `docs/RIKEN_INSTRUCTIONS.md` - replace `YOUR_USERNAME` with `PCfVW` (5 occurrences)
- [ ] Commit changes with message: "Prepare PLIP-rs for standalone extraction"

---

## Phase 2: Local Testing

**Goal**: Verify all examples and builds work with the new paths before extraction.

### 2.1 Build Verification

```powershell
cd "C:\Users\Eric JACOPIN\Documents\Code\Source\Priority Queues\experiment\plip-rs"

# Clean build
cargo clean
cargo build --release

# Run tests
cargo test
```

### 2.2 Example Verification

Test the examples that had path changes:

```powershell
# Test ablation experiment (uses corpus path)
cargo run --release --example ablation_experiment -- --help

# Test steering calibrate (uses corpus path)
cargo run --release --example steering_calibrate -- --help

# Test steering experiment (uses corpus path)
cargo run --release --example steering_experiment -- --help
```

### 2.3 Docker Verification

Docker is available. Test the container build:

```powershell
cd docker
docker-compose build
docker-compose run --rm plip-rs --help
```

> **Note**: After Phase 1 changes, the volume mount `./corpus:/data/corpus:ro` should work correctly.

### Phase 2 Checklist

- [ ] `cargo build --release` succeeds
- [ ] `cargo test` passes
- [ ] ablation_experiment example runs (--help)
- [ ] steering_calibrate example runs (--help)
- [ ] steering_experiment example runs (--help)
- [ ] Docker build succeeds

---

## Phase 3: Repository Extraction

**Goal**: Create a new standalone Git repository with clean history.

### 3.1 Create New Repository Directory

```powershell
# Create standalone directory
mkdir "C:\Users\Eric JACOPIN\Documents\Code\Source\plip-rs"
cd "C:\Users\Eric JACOPIN\Documents\Code\Source\plip-rs"

# Initialize Git
git init
```

### 3.2 Copy Files (Excluding Build Artifacts)

```powershell
# Using robocopy to exclude target/ and .git/
robocopy "C:\Users\Eric JACOPIN\Documents\Code\Source\Priority Queues\experiment\plip-rs" . /E /XD target .git

# Alternatively, manual copy excluding:
# - target/           (build artifacts, ~GB)
# - .git/             (parent repo history)
# - Cargo.lock        (will regenerate, optional to include)
```

### 3.3 Verify File Structure

Expected structure after copy:

```
plip-rs/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── container.yml
├── corpus/
│   ├── README.md
│   ├── samples.json
│   ├── attention_samples.json
│   ├── attention_samples_universal.json
│   └── ... (other JSON files)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   └── RIKEN_INSTRUCTIONS.md
├── examples/
│   └── ... (24 example files)
├── outputs/
│   └── ... (JSON result files)
├── src/
│   ├── lib.rs
│   ├── main.rs
│   └── ... (module files)
├── tests/
│   └── integration.rs
├── .gitignore
├── ABLATION_RESULTS.md
├── ABLATION_AMPLIFICATION_ROADMAP.md
├── Cargo.toml
├── COMMANDS.md
├── INTERVENTION_ROADMAP.md
├── LICENSE
├── README.md                         # Comprehensive, already exists
├── RIGOR_EXPERIMENT.md
├── run.ps1
├── STANDALONE_EXTRACTION_ROADMAP.md  (this file - remove after extraction)
├── STEERING_RESULTS.md
├── TEST_CHECKLIST.md
└── test_statistical_attention.ps1
```

### 3.4 Initial Commit

```powershell
git add .
git commit -m "$(cat <<'EOF'
Initial commit: PLIP-rs v1.0.0

Programming Language Internal Probing - Mechanistic interpretability
toolkit for code language models, implemented entirely in Rust.

Features:
- Attention pattern analysis across transformer layers
- Attention knockout (ablation) experiments
- Attention steering with dose-response calibration
- End-to-end generation with KV-cache
- Multi-model support (StarCoder2, Qwen2, CodeGemma)
- Statistical analysis (Welch's t-test, p-values)

Extracted from: https://github.com/PCfVW/d-Heap-priority-queue
For: AIware 2026 supplementary materials
EOF
)"
```

### Phase 3 Checklist

- [ ] Create new directory
- [ ] Initialize Git repository
- [ ] Copy files (excluding target/, .git/)
- [ ] Verify file structure matches expected
- [ ] Create initial commit
- [ ] Verify `cargo build --release` works in new location
- [ ] Verify `cargo test` passes in new location

---

## Phase 4: Enhance Existing README

**Goal**: Update the existing README.md for standalone repository context.

> **Note**: PLIP-rs already has a comprehensive README.md (240 lines) with Quick Start, Hardware Requirements, Citation, and "MI for the Rest of Us" sections. Only minor enhancements are needed.

### 4.1 README Enhancements

The existing README is well-structured. Add/update these elements:

**1. Add CI/DOI badges** (after title):

```markdown
[![CI](https://github.com/PCfVW/plip-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/PCfVW/plip-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

**2. Update the "Part of" reference** (line 3 currently references Amphigraphic):

```markdown
# BEFORE
Part of the [Amphigraphic](https://github.com/PCfVW/Amphigraphic) research project.

# AFTER
Developed as part of the [d-Heap Priority Queue](https://github.com/PCfVW/d-Heap-priority-queue) research project. Supplementary material for AIware 2026.
```

**3. Update Citation section** (add DOI after Zenodo integration):

```bibtex
@software{plip_rs_2026,
  author = {Jacopin, Eric and Claude},
  title = {PLIP-rs: Programming Language Internal Probing in Rust},
  year = {2026},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/PCfVW/plip-rs},
  note = {Mechanistic interpretability toolkit for AIware 2026}
}
```

**4. Verify URL in existing citation** (line 237):
- Current: `url = {https://github.com/PCfVW/plip-rs}` - Correct (matches target repo)

**5. Update Amphigraphic reference** to correct URL: `https://github.com/PCfVW/Amphigraphic-Strict`

### 4.2 Additional Documentation Review

Ensure these files are present and accurate:

| File | Purpose | Action |
|------|---------|--------|
| `LICENSE` | Apache-2.0 license text | Verify present |
| `COMMANDS.md` | CLI reference | Already updated in Phase 1 |
| `corpus/README.md` | Corpus format docs | Verify accurate |
| `.gitignore` | Git ignore patterns | Verify includes target/, outputs/ optionally |

### Phase 4 Checklist

- [ ] Add CI/License/DOI badges to existing README.md
- [ ] Update "Part of" reference (Amphigraphic → d-Heap Priority Queue)
- [ ] Update Citation section with DOI (after Phase 6)
- [ ] Verify Quick Start section works with standalone paths
- [ ] Verify LICENSE file present
- [ ] Verify .gitignore appropriate for standalone repo
- [ ] Consider removing this STANDALONE_EXTRACTION_ROADMAP.md after extraction

---

## Phase 5: Update Parent Repository Documentation

**Goal**: Fix broken links in Priority Queues after PLIP-rs extraction.

### 5.1 Experiment-Runner Documentation

**File**: `experiment/experiment-runner/COMMANDS.md`

| Line | Current | Change To |
|------|---------|-----------|
| 577 | `- [PLIP-rs COMMANDS.md](../plip-rs/COMMANDS.md) - Related model probing commands` | `- [PLIP-rs](https://github.com/PCfVW/plip-rs) - Related model probing commands (standalone repository)` |

### 5.2 Results Analysis Documentation

**File**: `experiment/results/analysis/plip_rust_roadmap.md`

Update any references to `experiment/plip-rs/` paths to point to the standalone repository URL.

**File**: `experiment/reports/grit-llm-proposal.md`

| Line | Current | Change To |
|------|---------|-----------|
| ~933 | `../plip-rs/INTERVENTION_ROADMAP.md` | `https://github.com/PCfVW/plip-rs/blob/main/INTERVENTION_ROADMAP.md` |

### 5.3 Experiment README (Optional)

**File**: `experiment/README.md`

Consider adding a note about PLIP-rs being extracted:

```markdown
## Related Tools

- **[PLIP-rs](https://github.com/PCfVW/plip-rs)** - Mechanistic interpretability toolkit (standalone repository)
- **experiment-runner/** - SEGA experiment automation
```

### Phase 5 Checklist

- [ ] Update `experiment/experiment-runner/COMMANDS.md` line 577
- [ ] Update `experiment/results/analysis/plip_rust_roadmap.md` path references
- [ ] Update `experiment/reports/grit-llm-proposal.md` line ~933
- [ ] Optionally update `experiment/README.md` with PLIP-rs link
- [ ] Commit changes with message: "Update docs: PLIP-rs extracted to standalone repo"

---

## Phase 6: GitHub Repository Setup & Zenodo Integration

**Goal**: Publish the standalone repository and create a DOI for citation.

### 6.1 Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `plip-rs`
3. Description: "Programming Language Internal Probing - Mechanistic interpretability toolkit for code models"
4. Visibility: **Public** (required for supplementary materials)
5. Do NOT initialize with README (we already have one)

### 6.2 Push to GitHub

```powershell
cd "C:\Users\Eric JACOPIN\Documents\Code\Source\plip-rs"
git remote add origin https://github.com/PCfVW/plip-rs.git
git branch -M main
git push -u origin main
```

### 6.3 Configure Repository Settings

1. **About section**: Add description and topics
   - Topics: `mechanistic-interpretability`, `transformers`, `rust`, `code-models`, `attention-analysis`
2. **Releases**: Create v1.0.0 release
3. **Actions**: Verify CI workflow runs successfully

### 6.4 Zenodo Integration (DOI)

1. Go to https://zenodo.org
2. Log in with GitHub
3. Enable the `plip-rs` repository for Zenodo
4. Create a GitHub release (v1.0.0)
5. Zenodo automatically creates DOI
6. Add DOI badge to README.md:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

### 6.5 Update Citation

After DOI is created, update README.md citation:

```bibtex
@software{plip_rs_2026,
  author = {Jacopin, Eric},
  title = {PLIP-rs: Programming Language Internal Probing},
  year = {2026},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/PCfVW/plip-rs}
}
```

### Phase 6 Checklist

- [ ] Create GitHub repository (public)
- [ ] Push code to GitHub
- [ ] Add repository description and topics
- [ ] Create v1.0.0 release with release notes
- [ ] Verify CI Actions run successfully
- [ ] Enable Zenodo integration
- [ ] Obtain DOI from Zenodo
- [ ] Update README.md with DOI badge
- [ ] Update citation with DOI

---

## Summary: Complete Checklist

### Pre-Extraction (Phase 1-2)
- [ ] Fix 2 Docker volume mounts (docker-compose.yml lines 31, 60)
- [ ] Remove 3 parent-path fallbacks from examples (ablation_experiment.rs, steering_calibrate.rs, steering_experiment.rs)
- [ ] Update 1 documentation link (COMMANDS.md line 1343)
- [ ] Update version to 1.0.0 (Cargo.toml line 3)
- [ ] Fix 5 `YOUR_USERNAME` placeholders (docs/RIKEN_INSTRUCTIONS.md)
- [ ] Test build, examples, and Docker locally
- [ ] Commit preparation changes

### Extraction (Phase 3-4)
- [ ] Create new directory and Git repo
- [ ] Copy files (exclude target/, .git/)
- [ ] Verify build in new location
- [ ] Enhance existing README.md (add badges, update references)
- [ ] Initial commit

### Post-Extraction (Phase 5-6)
- [ ] Update 3 documentation links in Priority Queues
- [ ] Push to GitHub
- [ ] Configure repository settings
- [ ] Create v1.0.0 release
- [ ] Integrate with Zenodo for DOI
- [ ] Update citation with DOI

---

## Timeline Estimate

| Phase | Tasks | Dependencies |
|-------|-------|--------------|
| **Phase 1** | Fix paths, update version | None |
| **Phase 2** | Local testing | Phase 1 |
| **Phase 3** | Extract to new repo | Phase 2 |
| **Phase 4** | Create README | Phase 3 |
| **Phase 5** | Update parent docs | Phase 3 |
| **Phase 6** | GitHub + Zenodo | Phase 4 |

---

## Rollback Plan

If issues are discovered after extraction:

1. **PLIP-rs changes are reversible**: Git history in Priority Queues preserves original state
2. **Standalone repo can be deleted**: No impact on parent repo
3. **Documentation links can point back**: If standalone repo is abandoned, revert doc changes

---

## Notes

- This roadmap assumes the target repository URL is `https://github.com/PCfVW/plip-rs`
- Adjust URLs if using a different GitHub organization or repository name
- The `outputs/` directory contains experiment results; decide whether to include in repo or add to `.gitignore`
- Consider adding a `CHANGELOG.md` for future versions

---

## Appendix: Consistency Validation

This roadmap was validated on February 4, 2026 by verifying:

### Line Numbers Verified

| File | Lines Referenced | Verified |
|------|------------------|----------|
| `docker/docker-compose.yml` | 31, 60 | Yes - `../corpus:/data/corpus:ro` |
| `examples/ablation_experiment.rs` | 208-219 (candidates array) | Yes - line 212 has `../corpus` |
| `examples/steering_calibrate.rs` | 72-83 (candidates array) | Yes - line 76 has `../corpus` |
| `examples/steering_experiment.rs` | 107-118 (candidates array) | Yes - line 111 has `../corpus` |
| `COMMANDS.md` | 1343 | Yes - SEGA cross-reference |
| `Cargo.toml` | 3 | Yes - `version = "0.1.0"` |
| `docs/RIKEN_INSTRUCTIONS.md` | 32, 52, 63, 72, 193 | Yes - `YOUR_USERNAME` placeholders |

### Key Findings During Review

1. **Example files already have working primary paths**: The `corpus/attention_samples_universal.json` path is the FIRST candidate in all three examples, so they will work after extraction without any changes. The `../corpus/` paths are fallbacks that should be removed for cleanliness.

2. **README.md already exists**: A comprehensive 240-line README with Quick Start, Hardware Requirements, Citation, and "MI for the Rest of Us" sections already exists. It needs enhancement (badges, updated references) rather than replacement.

3. **Root CI workflow is unrelated**: The Priority Queues `.github/workflows/deploy-demo.yml` only handles the demo folder and has no PLIP-rs references.

4. **No shared corpus data**: PLIP-rs uses its own `plip-rs/corpus/` directory; experiment-runner uses `test-corpus/`. No shared data files.

### Potential Issues Identified

1. **Amphigraphic reference**: README line 3 references "Amphigraphic" project.
   - **Resolution**: Update to correct URL: `https://github.com/PCfVW/Amphigraphic-Strict`

2. **outputs/ directory decision**: Contains 26 files (25 JSON + 1 TXT) - all **PLIP-rs experiment outputs**:
   - `ablation_*.json` - Knockout experiment results (5 files)
   - `layer_scan_*.json` - Layer-by-layer attention analysis (8 files)
   - `stats_*.json` - Statistical analysis per model (3 files)
   - `verify_*.json` - Position verification outputs (3 files)
   - `test_statistical_attention*.json` - Development test outputs (3 files)
   - `full_results_*.json` - Complete results (2 files)
   - `plip_results.json` - Summary results (1 file)
   - `ablation_qwen7b_layer_scan.txt` - Text log (1 file)

   **Recommendation**: Include for reproducibility - these are the actual AIware 2026 data.

3. **Author in citation**: Current citation includes "Claude" as co-author.
   - **Resolution**: Keep as-is. AI coding assistants (Augment Code, Kiro, Claude Code) have been the primary coding tools since July 2025 and are legitimate co-authors.

4. **RIKEN_INSTRUCTIONS.md**: Contains `YOUR_USERNAME` placeholders (lines 32, 52, 63, 72, 193).
   - **Action needed**: Replace with actual GitHub username `PCfVW` before extraction.

---

*Document created: February 4, 2026*
*Last validated: February 4, 2026*
*For: PLIP-rs standalone extraction / AIware 2026 supplementary materials*
