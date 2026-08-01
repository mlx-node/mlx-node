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

/// PyTorch's centered STFT defaults to `pad_mode="reflect"`. This is the
/// one-dimensional equivalent of its reflection padding (endpoints excluded).
fn reflect_index(index: isize, len: usize) -> usize {
    debug_assert!(len > 0);
    if len == 1 {
        return 0;
    }
    let period = (2 * (len - 1)) as isize;
    let folded = index.rem_euclid(period);
    if folded < len as isize {
        folded as usize
    } else {
        (period - folded) as usize
    }
}

fn hertz_to_mel(freq: f32) -> f32 {
    const MIN_LOG_HERTZ: f32 = 1_000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    const LOGSTEP: f32 = 0.068_751_78; // ln(6.4) / 27
    let linear = 3.0 * freq / 200.0;
    if freq >= MIN_LOG_HERTZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HERTZ).ln() / LOGSTEP
    } else {
        linear
    }
}

fn mel_to_hertz(mel: f32) -> f32 {
    const MIN_LOG_HERTZ: f32 = 1_000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    const LOGSTEP: f32 = 0.068_751_78;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HERTZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// Transformers/librosa-style Slaney filter bank, returned frequency-major.
fn mel_filter_bank(
    n_freqs: usize,
    n_mels: usize,
    min_frequency: f32,
    max_frequency: f32,
    sample_rate: f32,
) -> Vec<f32> {
    let min_mel = hertz_to_mel(min_frequency);
    let max_mel = hertz_to_mel(max_frequency);
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32)
        .map(mel_to_hertz)
        .collect();
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate / (2 * (n_freqs - 1)) as f32)
        .collect();
    let mut filters = vec![0.0f32; n_freqs * n_mels];
    for mel in 0..n_mels {
        let left = mel_points[mel];
        let center = mel_points[mel + 1];
        let right = mel_points[mel + 2];
        let enorm = 2.0 / (right - left);
        for (freq_idx, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - left) / (center - left);
            let upper = (right - freq) / (right - center);
            filters[freq_idx * n_mels + mel] = lower.min(upper).max(0.0) * enorm;
        }
    }
    filters
}

/// Streaming/capture inputs commonly arrive at 44.1 or 48 kHz. A windowed-
/// sinc resampler keeps that conversion deterministic and avoids aliasing from
/// a cheap linear interpolation path. The cutoff follows the lower Nyquist
/// rate; 32 taps on either side is a practical audio-quality/perf trade-off.
pub(crate) fn resample_mono(input: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if input.is_empty() || source_rate == target_rate {
        return input.to_vec();
    }
    let output_len = ((input.len() as u64 * target_rate as u64) / source_rate as u64) as usize;
    let ratio = source_rate as f64 / target_rate as f64;
    let cutoff = (target_rate as f64 / source_rate as f64).min(1.0) * 0.94;
    const RADIUS: isize = 32;
    let mut output = Vec::with_capacity(output_len);
    for out_idx in 0..output_len {
        let source_pos = out_idx as f64 * ratio;
        let center = source_pos.floor() as isize;
        let mut sum = 0.0f64;
        let mut norm = 0.0f64;
        for tap in -RADIUS + 1..=RADIUS {
            let sample_idx = center + tap;
            if sample_idx < 0 || sample_idx >= input.len() as isize {
                continue;
            }
            let distance = source_pos - sample_idx as f64;
            let x = std::f64::consts::PI * distance * cutoff;
            let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
            let window_pos = distance / RADIUS as f64;
            let window = if window_pos.abs() <= 1.0 {
                0.5 + 0.5 * (std::f64::consts::PI * window_pos).cos()
            } else {
                0.0
            };
            let weight = cutoff * sinc * window;
            sum += input[sample_idx as usize] as f64 * weight;
            norm += weight;
        }
        output.push(if norm.abs() > 1e-12 {
            (sum / norm) as f32
        } else {
            0.0
        });
    }
    output
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
