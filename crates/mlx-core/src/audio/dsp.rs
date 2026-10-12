/// PyTorch's centered STFT defaults to `pad_mode="reflect"`. This is the
/// one-dimensional equivalent of its reflection padding (endpoints excluded).
pub(crate) fn reflect_index(index: isize, len: usize) -> usize {
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
pub(crate) fn mel_filter_bank(
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
