use burn::{prelude::Backend, tensor::TensorData, Tensor};
use realfft::num_complex::Complex;

use crate::{DemucsError, Result};

/// Convert a spectrogram from STFT format (flat `[frame × bin]` layout with complex values)
/// to "Complex as Channels" (CaC) format — a 3D tensor of shape `[2, n_fft/2, num_frames]`
/// with separate real and imaginary channels.
///
/// * `spectrogram` is a flat slice of complex values in `[frame × bin]` order, where each frame
///   has `n_fft / 2` bins (Nyquist already dropped by the forward STFT).
/// * `n_fft` is the FFT size used to compute the STFT, which determines the number of bins.
/// * `device` is the target device for the output tensor (e.g., CPU or GPU).
pub fn stft_to_cac<B: Backend>(
    spectrogram: &[Complex<f32>],
    n_fft: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let freq_bins = n_fft / 2; // 2048
    let num_frames = spectrogram.len() / freq_bins;

    // Allocate [2, freq_bins, num_frames] row-major
    let mut data = vec![0.0f32; 2 * freq_bins * num_frames];

    for bin in 0..freq_bins {
        for frame in 0..num_frames {
            let c = spectrogram[frame * freq_bins + bin];
            data[bin * num_frames + frame] = c.re;
            data[freq_bins * num_frames + bin * num_frames + frame] = c.im;
        }
    }

    Tensor::from_data(TensorData::new(data, [2, freq_bins, num_frames]), device)
}

/// Convert a spectrogram from CaC format back to STFT format (flat `[frame × bin]` layout
/// with complex values).
///
/// * `tensor` is a 3D tensor of shape `[2, freq_bins, num_frames]` where the first dimension
///   separates real and imaginary parts, the second dimension is frequency bins (`n_fft/2`)
///   and the third dimension is time frames.
///
/// Returns a flat vector of `num_frames × (freq_bins + 1)` complex values, with zeroed Nyquist
/// bins appended to each frame.
pub async fn cac_to_stft<B: Backend>(tensor: &Tensor<B, 3>) -> Result<Vec<Complex<f32>>> {
    let [_, freq_bins, num_frames] = tensor.dims();
    let bins = freq_bins + 1; // Add Nyquist bin

    let mut spectrogram = vec![Complex::new(0.0, 0.0); num_frames * bins];

    let data: Vec<f32> = tensor
        .to_data_async()
        .await
        .map_err(|e| DemucsError::Tensor(format!("cac_to_stft async read failed: {}", e)))?
        .to_vec()
        .map_err(|e| DemucsError::Tensor(format!("cac_to_stft conversion failed: {}", e)))?;
    let (reals, imags) = data.split_at(freq_bins * num_frames);

    reals
        .iter()
        .zip(imags)
        .map(|(&re, &im)| Complex::new(re, im))
        .enumerate()
        .for_each(|(i, c)| {
            let frame = i % num_frames;
            let bin = i / num_frames;
            spectrogram[frame * bins + bin] = c;
        });

    Ok(spectrogram)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn output_shape() {
        let n_fft = 8;
        let freq_bins = n_fft / 2; // 4
        let num_frames = 3;
        let spec = vec![Complex::new(0.0, 0.0); num_frames * freq_bins];

        let tensor = stft_to_cac::<B>(&spec, n_fft, &Default::default());

        assert_eq!(tensor.dims(), [2, 4, 3]); // [2, n_fft/2, num_frames]
    }

    #[test]
    fn real_imag_split() {
        let n_fft = 4;

        // One frame, 2 bins (no Nyquist in input)
        let spec = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
        ];

        let tensor = stft_to_cac::<B>(&spec, n_fft, &Default::default());
        let data: Vec<f32> = tensor.to_data().to_vec().unwrap();

        // [2, 2, 1] row-major: [real_bin0, real_bin1, imag_bin0, imag_bin1]
        assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn frame_ordering() {
        let n_fft = 4;

        // 2 frames, 2 bins each (no Nyquist)
        let spec = vec![
            // Frame 0
            Complex::new(1.0, 10.0),
            Complex::new(2.0, 20.0),
            // Frame 1
            Complex::new(3.0, 30.0),
            Complex::new(4.0, 40.0),
        ];

        let tensor = stft_to_cac::<B>(&spec, n_fft, &Default::default());
        let data: Vec<f32> = tensor.to_data().to_vec().unwrap();

        assert_eq!(
            data,
            vec![
                1.0, 3.0, // real bin0: frame0, frame1
                2.0, 4.0, // real bin1: frame0, frame1
                10.0, 30.0, // imag bin0: frame0, frame1
                20.0, 40.0, // imag bin1: frame0, frame1
            ]
        );
    }

    #[test]
    fn round_trip_preserves_values() {
        let n_fft = 8;
        let freq_bins = n_fft / 2; // 4
        let num_frames = 4;

        // Fill with known values — n_fft/2 bins per frame (no Nyquist)
        let mut spec: Vec<Complex<f32>> = Vec::with_capacity(num_frames * freq_bins);
        for frame in 0..num_frames {
            for bin in 0..freq_bins {
                spec.push(Complex::new(
                    (frame * 100 + bin) as f32,
                    (frame * 100 + bin + 50) as f32,
                ));
            }
        }

        let device = Default::default();
        let tensor = stft_to_cac::<B>(&spec, n_fft, &device);
        let roundtrip = pollster::block_on(cac_to_stft::<B>(&tensor)).unwrap();

        // cac_to_stft returns n_fft/2+1 bins per frame (adds zero Nyquist)
        let bins_out = freq_bins + 1;
        assert_eq!(roundtrip.len(), num_frames * bins_out);

        for frame in 0..num_frames {
            for bin in 0..freq_bins {
                let orig = spec[frame * freq_bins + bin];
                let rt = roundtrip[frame * bins_out + bin];
                assert!(
                    (orig.re - rt.re).abs() < 1e-6,
                    "real mismatch at frame={frame} bin={bin}: {} vs {}",
                    orig.re,
                    rt.re,
                );
                assert!(
                    (orig.im - rt.im).abs() < 1e-6,
                    "imag mismatch at frame={frame} bin={bin}: {} vs {}",
                    orig.im,
                    rt.im,
                );
            }
            // Nyquist bin should be zero
            let nyquist = roundtrip[frame * bins_out + freq_bins];
            assert_eq!(nyquist.re, 0.0);
            assert_eq!(nyquist.im, 0.0);
        }
    }
}
