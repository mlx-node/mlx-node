/// PyTorch's centered STFT defaults to `pad_mode="reflect"`. This is the
/// one-dimensional equivalent of its reflection padding (endpoints excluded).
pub fn reflect_index(index: isize, len: usize) -> usize {
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
pub fn mel_filter_bank(
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
pub fn resample_mono(input: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if input.is_empty() || source_rate == target_rate {
        return input.to_vec();
    }
    let resampler = SincResampler::new(input.len(), source_rate, target_rate);
    match exact_phase_period(source_rate, target_rate, resampler.output_len) {
        Some(period) => resampler.phase_table(input, period, normalize),
        None => resampler.per_tap(input, normalize),
    }
}

const RESAMPLE_RADIUS: isize = 32;
const RESAMPLE_TAPS: usize = 2 * RESAMPLE_RADIUS as usize;
const MAX_RESAMPLE_PHASES: u64 = 256;

/// `out_idx * ratio` is exact in f64 only when source/target reduces to p/q
/// with q a power of two and every `out_idx * p` stays within 2^53. Then the
/// tap distances of output `out_idx` repeat bit for bit every q outputs, so
/// cached weights reproduce the per-tap path byte for byte. Any other pair
/// (44.1 kHz) must stay on the per-tap path.
fn exact_phase_period(source_rate: u32, target_rate: u32, output_len: usize) -> Option<usize> {
    let (mut gcd, mut rest) = (source_rate, target_rate);
    while rest != 0 {
        (gcd, rest) = (rest, gcd % rest);
    }
    if gcd == 0 {
        return None;
    }
    let step = u64::from(source_rate / gcd);
    let period = u64::from(target_rate / gcd);
    (period.is_power_of_two()
        && period <= MAX_RESAMPLE_PHASES
        && (output_len as u64).checked_mul(step)? <= 1 << 53)
        .then_some(period as usize)
}

fn normalize(sum: f64, norm: f64) -> f32 {
    if norm.abs() > 1e-12 {
        (sum / norm) as f32
    } else {
        0.0
    }
}

struct SincResampler {
    output_len: usize,
    ratio: f64,
    cutoff: f64,
}

impl SincResampler {
    fn new(input_len: usize, source_rate: u32, target_rate: u32) -> Self {
        Self {
            output_len: ((input_len as u64 * target_rate as u64) / source_rate as u64) as usize,
            ratio: source_rate as f64 / target_rate as f64,
            cutoff: (target_rate as f64 / source_rate as f64).min(1.0) * 0.94,
        }
    }

    fn per_tap<T>(&self, input: &[f32], finish: impl Fn(f64, f64) -> T) -> Vec<T> {
        (0..self.output_len)
            .map(|out_idx| self.sample_at(input, out_idx as f64 * self.ratio, &finish))
            .collect()
    }

    /// Interior outputs take every tap in the per-tap order, so the cached
    /// weights and norm are summed in that same order. Outputs whose window
    /// is clipped by an input edge skip taps and must use `sample_at`.
    fn phase_table<T>(
        &self,
        input: &[f32],
        period: usize,
        finish: impl Fn(f64, f64) -> T,
    ) -> Vec<T> {
        let phases: Vec<([f64; RESAMPLE_TAPS], f64)> = (0..period)
            .map(|phase| {
                let source_pos = phase as f64 * self.ratio;
                let first = source_pos.floor() as isize - RESAMPLE_RADIUS + 1;
                let mut weights = [0.0f64; RESAMPLE_TAPS];
                let mut norm = 0.0f64;
                for (offset, weight) in weights.iter_mut().enumerate() {
                    *weight = self.tap_weight(source_pos - (first + offset as isize) as f64);
                    norm += *weight;
                }
                (weights, norm)
            })
            .collect();
        (0..self.output_len)
            .map(|out_idx| {
                let source_pos = out_idx as f64 * self.ratio;
                let first = source_pos.floor() as isize - RESAMPLE_RADIUS + 1;
                if first < 0 || first as usize + RESAMPLE_TAPS > input.len() {
                    return self.sample_at(input, source_pos, &finish);
                }
                let (weights, norm) = &phases[out_idx % period];
                let sum = input[first as usize..first as usize + RESAMPLE_TAPS]
                    .iter()
                    .zip(weights)
                    .fold(0.0f64, |sum, (&sample, &weight)| {
                        sum + sample as f64 * weight
                    });
                finish(sum, *norm)
            })
            .collect()
    }

    fn sample_at<T>(&self, input: &[f32], source_pos: f64, finish: impl Fn(f64, f64) -> T) -> T {
        let center = source_pos.floor() as isize;
        let mut sum = 0.0f64;
        let mut norm = 0.0f64;
        for tap in -RESAMPLE_RADIUS + 1..=RESAMPLE_RADIUS {
            let sample_idx = center + tap;
            if sample_idx < 0 || sample_idx >= input.len() as isize {
                continue;
            }
            let weight = self.tap_weight(source_pos - sample_idx as f64);
            sum += input[sample_idx as usize] as f64 * weight;
            norm += weight;
        }
        finish(sum, norm)
    }

    fn tap_weight(&self, distance: f64) -> f64 {
        let x = std::f64::consts::PI * distance * self.cutoff;
        let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
        let window_pos = distance / RESAMPLE_RADIUS as f64;
        let window = if window_pos.abs() <= 1.0 {
            0.5 + 0.5 * (std::f64::consts::PI * window_pos).cos()
        } else {
            0.0
        };
        self.cutoff * sinc * window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn assert_same<T: PartialEq + std::fmt::Debug>(got: &[T], want: &[T], case: &str) {
        assert_eq!(got.len(), want.len(), "{case}: length");
        if let Some(index) = got.iter().zip(want).position(|(g, w)| g != w) {
            panic!(
                "{case}, output {index}: {:?} != {:?}",
                got[index], want[index]
            );
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    // The f32 cast hides most f64 rounding changes, so the f64 sum and norm
    // are compared too: they must match the per-tap path bit for bit.
    #[test]
    fn phase_table_matches_per_tap_bytes() {
        let raw = |sum: f64, norm: f64| (sum.to_bits(), norm.to_bits());
        for source_rate in [8_000, 16_000, 24_000, 32_000, 44_100, 48_000] {
            for len in [0, 1, 31, 63, 64, 65, 1_000, 48_007] {
                let case = format!("{source_rate} Hz, {len} samples");
                let input = noise(len, u64::from(source_rate) * 100_003 + len as u64);
                let output = resample_mono(&input, source_rate, 16_000);
                if len == 0 || source_rate == 16_000 {
                    assert_same(&bits(&output), &bits(&input), &case);
                    continue;
                }
                let resampler = SincResampler::new(len, source_rate, 16_000);
                assert_same(
                    &bits(&output),
                    &bits(&resampler.per_tap(&input, normalize)),
                    &case,
                );
                let period = exact_phase_period(source_rate, 16_000, resampler.output_len);
                assert_eq!(period.is_some(), source_rate != 44_100, "{case}");
                if let Some(period) = period {
                    assert_same(
                        &resampler.phase_table(&input, period, raw),
                        &resampler.per_tap(&input, raw),
                        &case,
                    );
                }
            }
        }
    }
}
