use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Resample a single channel from `from_rate` to `to_rate`.
///
/// Returns the input unchanged if rates already match.
/// Uses sinc interpolation for high-quality audio resampling.
pub fn resample_channel(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, String> {
    if from_rate == to_rate || samples.is_empty() {
        return Ok(samples.to_vec());
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let chunk_size = 1024;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1)
        .map_err(|e| format!("failed to create resampler: {}", e))?;

    let expected_out = (samples.len() as f64 * ratio).ceil() as usize + chunk_size;
    let mut output = Vec::with_capacity(expected_out);

    // Process full chunks
    let mut pos = 0;
    while pos + chunk_size <= samples.len() {
        let chunk = &samples[pos..pos + chunk_size];
        let result = resampler
            .process(&[chunk], None)
            .map_err(|e| format!("resample error: {}", e))?;
        output.extend_from_slice(&result[0]);
        pos += chunk_size;
    }

    // Process remaining samples + flush
    if pos < samples.len() {
        let remaining = &samples[pos..];
        let result = resampler
            .process_partial(Some(&[remaining]), None)
            .map_err(|e| format!("resample error: {}", e))?;
        output.extend_from_slice(&result[0]);
    } else {
        let result = resampler
            .process_partial(None::<&[&[f32]]>, None)
            .map_err(|e| format!("resample error: {}", e))?;
        output.extend_from_slice(&result[0]);
    }

    Ok(output)
}
