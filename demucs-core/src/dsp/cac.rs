use burn::{prelude::Backend, tensor::TensorData, Tensor};
use realfft::num_complex::Complex;

/// Convert a spectrogram from STFT format (flat `[frame × bin]` layout with complex values)
/// to "Complex as Channels" (CaC) format — a 3D tensor of shape `[2, n_fft/2, num_frames]`
/// with separate real and imaginary channels. The Nyquist bin is dropped.
///
/// * `spectrogram` is a flat slice of complex values in `[frame × bin]` order, where each frame has
///   `n_fft / 2 + 1` bins, like [frame₀bin₀, frame₀bin₁, ..., frame₁bin₀, ...]
/// * `n_fft` is the FFT size used to compute the STFT, which determines the number of bins.
/// * `device` is the target device for the output tensor (e.g., CPU or GPU).
pub fn stft_to_cac<B: Backend>(
    spectrogram: &[Complex<f32>],
    n_fft: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let freq_bins = n_fft / 2;
    let bins = freq_bins + 1; // Add Nyquist bin
    let num_frames = spectrogram.len() / bins;

    // Allocate [2, freq_bins, num_frames] row-major
    let mut data = vec![0.0f32; 2 * freq_bins * num_frames];

    for bin in 0..freq_bins {
        for frame in 0..num_frames {
            let c = spectrogram[frame * bins + bin];
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
pub fn cac_to_stft<B: Backend>(tensor: &Tensor<B, 3>) -> Vec<Complex<f32>> {
    let [_, freq_bins, num_frames] = tensor.dims();
    let bins = freq_bins + 1; // Add Nyquist bin

    let mut spectrogram = vec![Complex::new(0.0, 0.0); num_frames * bins];

    let data: Vec<f32> = tensor.to_data().to_vec().unwrap();
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

    spectrogram
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn output_shape() {
        let n_fft = 8;
        let bins = n_fft / 2 + 1; // 5
        let num_frames = 3;
        let spec = vec![Complex::new(0.0, 0.0); num_frames * bins];

        let tensor = stft_to_cac::<B>(&spec, n_fft, &Default::default());

        assert_eq!(tensor.dims(), [2, 4, 3]); // [2, n_fft/2, num_frames]
    }

    #[test]
    fn nyquist_bin_dropped() {
        let n_fft = 8;
        let bins = n_fft / 2 + 1; // 5

        // Put a distinctive value in the Nyquist bin (index 4)
        let mut spec = vec![Complex::new(0.0, 0.0); bins];
        spec[4] = Complex::new(99.0, 99.0);

        let tensor = stft_to_cac::<B>(&spec, n_fft, &Default::default());
        let data: Vec<f32> = tensor.to_data().to_vec().unwrap();

        // 99.0 should not appear anywhere in the tensor
        assert!(data.iter().all(|&v| v != 99.0));
    }

    #[test]
    fn real_imag_split() {
        let n_fft = 4;

        // One frame: bin0=(1,2), bin1=(3,4), bin2=Nyquist(ignored)
        let spec = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(99.0, 99.0), // Nyquist
        ];

        let tensor = stft_to_cac::<B>(&spec, n_fft, &Default::default());
        let data: Vec<f32> = tensor.to_data().to_vec().unwrap();

        // [2, 2, 1] row-major: [real_bin0, real_bin1, imag_bin0, imag_bin1]
        assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn frame_ordering() {
        let n_fft = 4;

        // 2 frames, 3 bins each (last is Nyquist)
        let spec = vec![
            // Frame 0
            Complex::new(1.0, 10.0),
            Complex::new(2.0, 20.0),
            Complex::new(0.0, 0.0), // Nyquist
            // Frame 1
            Complex::new(3.0, 30.0),
            Complex::new(4.0, 40.0),
            Complex::new(0.0, 0.0), // Nyquist
        ];

        let tensor = stft_to_cac::<B>(&spec, n_fft, &Default::default());
        let data: Vec<f32> = tensor.to_data().to_vec().unwrap();

        // [2, 2, 2] row-major: bins-major within each channel
        // Channel 0 (real): bin0=[frame0,frame1], bin1=[frame0,frame1]
        // Channel 1 (imag): bin0=[frame0,frame1], bin1=[frame0,frame1]
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
    fn round_trip_preserves_non_nyquist() {
        let n_fft = 8;
        let bins = n_fft / 2 + 1; // 5
        let num_frames = 4;

        // Fill with known values, Nyquist bins set to zero
        let mut spec: Vec<Complex<f32>> = Vec::with_capacity(num_frames * bins);
        for frame in 0..num_frames {
            for bin in 0..bins {
                if bin < n_fft / 2 {
                    spec.push(Complex::new(
                        (frame * 100 + bin) as f32,
                        (frame * 100 + bin + 50) as f32,
                    ));
                } else {
                    spec.push(Complex::new(0.0, 0.0)); // Nyquist
                }
            }
        }

        let device = Default::default();
        let tensor = stft_to_cac::<B>(&spec, n_fft, &device);
        let roundtrip = cac_to_stft::<B>(&tensor);

        assert_eq!(roundtrip.len(), spec.len());
        for (a, b) in spec.iter().zip(roundtrip.iter()) {
            assert!(
                (a.re - b.re).abs() < 1e-6,
                "real mismatch: {} vs {}",
                a.re,
                b.re
            );
            assert!(
                (a.im - b.im).abs() < 1e-6,
                "imag mismatch: {} vs {}",
                a.im,
                b.im
            );
        }
    }

    #[test]
    fn round_trip_zeroes_nyquist() {
        let n_fft = 8;
        let bins = n_fft / 2 + 1; // 5
        let num_frames = 2;

        // Put nonzero values in Nyquist bins
        let spec = vec![Complex::new(1.0, 1.0); num_frames * bins];

        let device = Default::default();
        let tensor = stft_to_cac::<B>(&spec, n_fft, &device);
        let roundtrip = cac_to_stft::<B>(&tensor);

        // Nyquist bins (index 4 in each frame) should be zeroed
        for frame in 0..num_frames {
            let nyquist = roundtrip[frame * bins + n_fft / 2];
            assert_eq!(nyquist.re, 0.0);
            assert_eq!(nyquist.im, 0.0);
        }
    }
}
