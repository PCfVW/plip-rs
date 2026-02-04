//! Attention Steering Calibration for PLIP-rs
//!
//! Utilities for measuring baseline attention levels and calibrating
//! steering targets for dose-response experiments.
//!
//! ## Usage
//!
//! ```ignore
//! use plip_rs::{PlipModel, calibrate_from_samples};
//!
//! let model = PlipModel::from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")?;
//!
//! // Measure baseline attention for Python and Rust samples
//! let calibration = calibrate_from_samples(&model, &python_samples, &rust_samples, 16)?;
//!
//! println!("Python baseline: {:.2}%", calibration.python_baseline * 100.0);
//! println!("Rust baseline: {:.2}%", calibration.rust_baseline * 100.0);
//! println!("Recommended target: {:.2}%", calibration.recommended_target * 100.0);
//! ```

use anyhow::Result;

use crate::intervention::measure_attention_to_targets;
use crate::PlipModel;

/// Standard dose levels for dose-response experiments
///
/// These are multipliers for the baseline attention level:
/// - 0.5x: Reduce attention by half
/// - 1.0x: Baseline (no change)
/// - 2.0x: Double the attention
/// - 3.0x: Triple the attention
/// - 4.0x: Quadruple (approximately Python level for Rust)
/// - 6.0x: Six times baseline
pub const DOSE_LEVELS: [f32; 6] = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0];

/// Calibration data for steering experiments
#[derive(Debug, Clone)]
pub struct SteeringCalibration {
    /// Mean attention for Python doctest samples (marker → function)
    pub python_baseline: f32,

    /// Mean attention for Rust test samples (marker → function)
    pub rust_baseline: f32,

    /// Recommended target attention level (typically Python level)
    pub recommended_target: f32,

    /// Ratio of Python to Rust attention (python_baseline / rust_baseline)
    pub attention_ratio: f32,

    /// Layer index used for calibration
    pub layer: usize,

    /// Number of Python samples used
    pub n_python_samples: usize,

    /// Number of Rust samples used
    pub n_rust_samples: usize,
}

impl SteeringCalibration {
    /// Create a new calibration result
    pub fn new(
        python_baseline: f32,
        rust_baseline: f32,
        layer: usize,
        n_python_samples: usize,
        n_rust_samples: usize,
    ) -> Self {
        let attention_ratio = if rust_baseline > 1e-10 {
            python_baseline / rust_baseline
        } else {
            0.0
        };

        Self {
            python_baseline,
            rust_baseline,
            recommended_target: python_baseline, // Default to Python level
            attention_ratio,
            layer,
            n_python_samples,
            n_rust_samples,
        }
    }

    /// Set a custom recommended target
    pub fn with_target(mut self, target: f32) -> Self {
        self.recommended_target = target;
        self
    }

    /// Calculate the scale factor needed to boost Rust attention to target
    pub fn scale_factor_for_target(&self, target: f32) -> f32 {
        if self.rust_baseline > 1e-10 {
            target / self.rust_baseline
        } else {
            1.0
        }
    }

    /// Calculate the scale factor to boost Rust to Python level
    pub fn scale_factor_to_python(&self) -> f32 {
        self.scale_factor_for_target(self.python_baseline)
    }

    /// Get dose levels as absolute attention values (based on Rust baseline)
    pub fn dose_levels_absolute(&self) -> Vec<(f32, f32)> {
        DOSE_LEVELS
            .iter()
            .map(|&scale| (scale, self.rust_baseline * scale))
            .collect()
    }
}

/// Sample data for calibration
#[derive(Debug, Clone)]
pub struct CalibrationSample {
    /// Sample identifier
    pub id: String,

    /// The code text
    pub code: String,

    /// Character position of the test marker (e.g., `>>>` or `#[test]`)
    pub marker_char_pos: usize,

    /// Character positions of function tokens to measure attention to
    pub target_char_positions: Vec<usize>,
}

impl CalibrationSample {
    /// Create a new calibration sample
    pub fn new(
        id: impl Into<String>,
        code: impl Into<String>,
        marker_char_pos: usize,
        target_char_positions: Vec<usize>,
    ) -> Self {
        Self {
            id: id.into(),
            code: code.into(),
            marker_char_pos,
            target_char_positions,
        }
    }
}

/// Calibrate steering targets from sample data
///
/// Measures the mean attention from test markers to function tokens
/// for both Python and Rust samples at the specified layer.
///
/// # Arguments
/// * `model` - The loaded PLIP model
/// * `python_samples` - Python doctest samples with marker and target positions
/// * `rust_samples` - Rust test samples with marker and target positions
/// * `layer` - Layer index to measure attention at
///
/// # Returns
/// `SteeringCalibration` with measured baselines and recommended target
pub fn calibrate_from_samples(
    model: &PlipModel,
    python_samples: &[CalibrationSample],
    rust_samples: &[CalibrationSample],
    layer: usize,
) -> Result<SteeringCalibration> {
    // Measure Python attention
    let python_attentions = measure_samples(model, python_samples, layer)?;
    let python_baseline = if python_attentions.is_empty() {
        0.0
    } else {
        python_attentions.iter().sum::<f32>() / python_attentions.len() as f32
    };

    // Measure Rust attention
    let rust_attentions = measure_samples(model, rust_samples, layer)?;
    let rust_baseline = if rust_attentions.is_empty() {
        0.0
    } else {
        rust_attentions.iter().sum::<f32>() / rust_attentions.len() as f32
    };

    Ok(SteeringCalibration::new(
        python_baseline,
        rust_baseline,
        layer,
        python_samples.len(),
        rust_samples.len(),
    ))
}

/// Measure attention for a set of samples
fn measure_samples(
    model: &PlipModel,
    samples: &[CalibrationSample],
    layer: usize,
) -> Result<Vec<f32>> {
    let mut attentions = Vec::with_capacity(samples.len());

    for sample in samples {
        match measure_sample_attention(model, sample, layer) {
            Ok(attn) => attentions.push(attn),
            Err(e) => {
                tracing::warn!("Failed to measure sample {}: {}", sample.id, e);
            }
        }
    }

    Ok(attentions)
}

/// Measure attention for a single sample
fn measure_sample_attention(
    model: &PlipModel,
    sample: &CalibrationSample,
    layer: usize,
) -> Result<f32> {
    // Convert character positions to token positions
    let encoding = model.tokenize_with_offsets(&sample.code)?;

    let marker_token_pos = encoding
        .char_to_token(sample.marker_char_pos)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Could not find marker token for char pos {}",
                sample.marker_char_pos
            )
        })?;

    let target_token_positions: Vec<usize> = sample
        .target_char_positions
        .iter()
        .filter_map(|&char_pos| encoding.char_to_token(char_pos))
        .collect();

    if target_token_positions.is_empty() {
        anyhow::bail!("No target tokens found for sample {}", sample.id);
    }

    // Get attention and measure
    let attn_cache = model.get_attention(&sample.code)?;
    measure_attention_to_targets(
        &attn_cache,
        marker_token_pos,
        &target_token_positions,
        layer,
    )
}

/// Result of a dose-response measurement
#[derive(Debug, Clone)]
pub struct DoseResponsePoint {
    /// Scale factor applied
    pub scale_factor: f32,

    /// Resulting attention level (after steering)
    pub attention_level: f32,

    /// KL divergence from baseline
    pub kl_divergence: f32,
}

/// Full dose-response curve
#[derive(Debug, Clone)]
pub struct DoseResponseCurve {
    /// Sample identifier
    pub sample_id: String,

    /// Language (python or rust)
    pub language: String,

    /// Layer used for steering
    pub layer: usize,

    /// Baseline attention (no steering)
    pub baseline_attention: f32,

    /// Data points on the curve
    pub points: Vec<DoseResponsePoint>,
}

impl DoseResponseCurve {
    /// Create a new dose-response curve
    pub fn new(sample_id: String, language: String, layer: usize, baseline_attention: f32) -> Self {
        Self {
            sample_id,
            language,
            layer,
            baseline_attention,
            points: Vec::new(),
        }
    }

    /// Add a data point
    pub fn add_point(&mut self, scale_factor: f32, attention_level: f32, kl_divergence: f32) {
        self.points.push(DoseResponsePoint {
            scale_factor,
            attention_level,
            kl_divergence,
        });
    }

    /// Find the scale factor that achieves target attention level
    pub fn scale_for_target(&self, target: f32) -> Option<f32> {
        // Linear interpolation between points
        for window in self.points.windows(2) {
            let (p1, p2) = (&window[0], &window[1]);
            if (p1.attention_level <= target && target <= p2.attention_level)
                || (p2.attention_level <= target && target <= p1.attention_level)
            {
                // Interpolate
                let t = (target - p1.attention_level) / (p2.attention_level - p1.attention_level);
                return Some(p1.scale_factor + t * (p2.scale_factor - p1.scale_factor));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_new() {
        let cal = SteeringCalibration::new(0.09, 0.025, 16, 10, 10);

        assert!((cal.python_baseline - 0.09).abs() < 1e-6);
        assert!((cal.rust_baseline - 0.025).abs() < 1e-6);
        assert!((cal.attention_ratio - 3.6).abs() < 1e-6);
        assert_eq!(cal.layer, 16);
    }

    #[test]
    fn test_scale_factor_calculation() {
        let cal = SteeringCalibration::new(0.09, 0.025, 16, 10, 10);

        // Scale factor to reach Python level (0.09) from Rust (0.025)
        let scale = cal.scale_factor_to_python();
        assert!((scale - 3.6).abs() < 1e-6);

        // Scale factor to reach 0.05
        let scale = cal.scale_factor_for_target(0.05);
        assert!((scale - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dose_levels() {
        assert_eq!(DOSE_LEVELS.len(), 6);
        assert!((DOSE_LEVELS[0] - 0.5).abs() < 1e-6);
        assert!((DOSE_LEVELS[1] - 1.0).abs() < 1e-6);
        assert!((DOSE_LEVELS[5] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_dose_response_curve() {
        let mut curve =
            DoseResponseCurve::new("test_sample".to_string(), "rust".to_string(), 16, 0.025);

        curve.add_point(1.0, 0.025, 0.0);
        curve.add_point(2.0, 0.050, 0.01);
        curve.add_point(4.0, 0.100, 0.05);

        // Find scale for 0.075 (between 2x and 4x)
        let scale = curve.scale_for_target(0.075);
        assert!(scale.is_some());
        let scale = scale.unwrap();
        assert!(scale > 2.0 && scale < 4.0);
    }
}
