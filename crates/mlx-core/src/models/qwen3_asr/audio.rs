pub(crate) use crate::audio::dsp::resample_mono;
use crate::audio::dsp::{mel_filter_bank, reflect_index};
use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use super::config::ProcessorConfig;

/// CPU-side features ready for the MLX audio tower.
pub(crate) struct AudioFeatures {
    /// Mel-major `[num_mels, padded_frames]` buffer.
    pub values: Vec<f32>,
    pub num_mels: usize,
    pub padded_frames: usize,
    pub valid_frames: usize,
}

/// Numerically matches Transformers' `Qwen3ASRFeatureExtractor` for a mono
/// signal: centered `torch.stft`, periodic Hann, squared magnitude, Slaney
/// mel filters, log10/dynamic-range normalization, then mel-time padding.
pub(crate) struct FeatureExtractor {
    config: ProcessorConfig,
    num_mels: usize,
    n_window: usize,
    window: Vec<f32>,
    mel_filters: Vec<f32>, // frequency-major [n_freqs, num_mels]
    fft: Arc<dyn Fft<f32>>,
}

impl FeatureExtractor {
    pub(crate) fn new(
        config: ProcessorConfig,
        num_mels: usize,
        n_window: usize,
    ) -> Result<Self, String> {
        if config.n_fft == 0 || config.hop_length == 0 || !config.n_fft.is_multiple_of(2) {
            return Err(
                "n_fft must be a non-zero even number and hop_length must be non-zero".into(),
            );
        }
        if config.sampling_rate != 16_000 {
            return Err(format!(
                "Qwen3-ASR checkpoint expects a 16 kHz processor, got {} Hz",
                config.sampling_rate
            ));
        }
        if n_window == 0 {
            return Err("n_window must be non-zero".into());
        }
        if config.feature_size != num_mels || config.n_window != n_window {
            return Err(format!(
                "processor feature_size/n_window ({}/{}) must match audio config ({num_mels}/{n_window})",
                config.feature_size, config.n_window
            ));
        }
        if config.dither != 0.0 || config.padding_value != 0.0 {
            return Err(
                "Only zero dither and zero waveform padding are supported for Qwen3-ASR".into(),
            );
        }

        // `torch.hann_window(n_fft)` is periodic by default.
        let window = (0..config.n_fft)
            .map(|i| {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / config.n_fft as f32).cos()
            })
            .collect();
        let mel_filters = mel_filter_bank(
            config.n_fft / 2 + 1,
            num_mels,
            0.0,
            config.sampling_rate as f32 / 2.0,
            config.sampling_rate as f32,
        );
        let fft = FftPlanner::<f32>::new().plan_fft_forward(config.n_fft);
        Ok(Self {
            config,
            num_mels,
            n_window,
            window,
            mel_filters,
            fft,
        })
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.config.sampling_rate
    }

    pub(crate) fn hop_length(&self) -> usize {
        self.config.hop_length
    }

    pub(crate) fn extract(&self, audio: &[f32]) -> Result<AudioFeatures, String> {
        if audio.is_empty() {
            return Err("audio must contain at least one sample".into());
        }
        if audio.iter().any(|sample| !sample.is_finite()) {
            return Err("audio contains NaN or infinity".into());
        }

        let signal_len = audio.len().max(self.config.min_length);
        let mut signal = vec![0.0f32; signal_len];
        signal[..audio.len()].copy_from_slice(audio);

        let valid_frames = signal_len / self.config.hop_length;
        if valid_frames == 0 {
            return Err("audio is too short to produce a feature frame".into());
        }
        let padded_frames = valid_frames.div_ceil(self.n_window * 2) * self.n_window * 2;
        let n_freqs = self.config.n_fft / 2 + 1;
        let center = self.config.n_fft / 2;
        let mut fft_buf = vec![Complex32::new(0.0, 0.0); self.config.n_fft];
        let mut power = vec![0.0f32; n_freqs];
        let mut mel = vec![0.0f32; self.num_mels * padded_frames];

        // Torch returns one more centered frame and the HF extractor drops
        // its final frame. The remaining count is floor(samples / hop).
        for frame in 0..valid_frames {
            let start = frame as isize * self.config.hop_length as isize - center as isize;
            for (i, slot) in fft_buf.iter_mut().enumerate() {
                let source = reflect_index(start + i as isize, signal_len);
                *slot = Complex32::new(signal[source] * self.window[i], 0.0);
            }
            self.fft.process(&mut fft_buf);
            for freq in 0..n_freqs {
                power[freq] = fft_buf[freq].norm_sqr();
            }
            for mel_idx in 0..self.num_mels {
                let mut value = 0.0f32;
                for (freq, power_value) in power.iter().enumerate().take(n_freqs) {
                    value += self.mel_filters[freq * self.num_mels + mel_idx] * *power_value;
                }
                mel[mel_idx * padded_frames + frame] = value.max(1e-10).log10();
            }
        }

        let max_log = mel
            .iter()
            .enumerate()
            .filter(|(i, _)| i % padded_frames < valid_frames)
            .map(|(_, value)| *value)
            .fold(f32::NEG_INFINITY, f32::max);
        let floor = max_log - 8.0;
        for mel_idx in 0..self.num_mels {
            for frame in 0..valid_frames {
                let value = &mut mel[mel_idx * padded_frames + frame];
                *value = (value.max(floor) + 4.0) / 4.0;
            }
        }

        Ok(AudioFeatures {
            values: mel,
            num_mels: self.num_mels,
            padded_frames,
            valid_frames,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_padding_excludes_endpoints() {
        let got: Vec<_> = (-4..9).map(|i| reflect_index(i, 5)).collect();
        assert_eq!(got, vec![4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn feature_shapes_follow_qwen3_asr_padding_contract() {
        let extractor = FeatureExtractor::new(ProcessorConfig::default(), 128, 50).unwrap();
        let short = extractor.extract(&vec![0.0; 1_600]).unwrap();
        assert_eq!(
            (short.num_mels, short.valid_frames, short.padded_frames),
            (128, 50, 100)
        );
        let uneven = extractor.extract(&vec![0.0; 16_159]).unwrap();
        assert_eq!((uneven.valid_frames, uneven.padded_frames), (100, 100));
    }

    #[test]
    fn feature_values_match_refreshed_transformers_fixture() {
        let extractor = FeatureExtractor::new(ProcessorConfig::default(), 128, 50).unwrap();
        let audio: Vec<_> = (0..8_000)
            .map(|sample| {
                0.25 * (2.0 * std::f32::consts::PI * 440.0 * sample as f32 / 16_000.0).sin()
            })
            .collect();
        let features = extractor.extract(&audio).unwrap();
        // Generated from the updated local Transformers checkout's
        // Qwen3ASRFeatureExtractor._torch_extract_fbank_features.
        let expected = [
            (0, 0, 0.757_004_56),
            (1, 1, 0.342_560_77),
            (10, 2, -0.665_160_4),
            (20, 10, 1.136_878_1),
            (40, 20, -0.665_160_4),
            (64, 25, -0.665_160_4),
            (80, 30, -0.665_160_4),
            (100, 40, -0.665_160_4),
            (127, 49, -0.665_160_4),
        ];
        for (mel, frame, want) in expected {
            let got = features.values[mel * features.padded_frames + frame];
            assert!(
                (got - want).abs() < 5e-4,
                "mel={mel} frame={frame}: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn resampler_preserves_constant_signal() {
        let source = vec![0.25; 4_800];
        let output = resample_mono(&source, 48_000, 16_000);
        assert_eq!(output.len(), 1_600);
        assert!(
            output[64..output.len() - 64]
                .iter()
                .all(|value| (*value - 0.25).abs() < 1e-5)
        );
    }
}
