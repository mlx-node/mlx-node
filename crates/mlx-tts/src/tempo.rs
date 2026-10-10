//! MIT-licensed, independent waveform-similarity time scaling for speech.
//! Fixed grains follow a global time map. Similarity search aligns overlapping
//! waveforms without estimating pitch or inserting/deleting pitch periods.

#[derive(Debug, thiserror::Error)]
pub enum TempoError {
    #[error("invalid PCM format or time-scale configuration")]
    Format,
    #[error("speech speed must be finite and within 0.25..4")]
    Speed,
    #[error("PCM must contain complete frames of finite samples")]
    Samples,
    #[error("speech tempo processor is closed")]
    Closed,
    #[error("speech tempo sample counter overflow")]
    Overflow,
}

/// Durations are audio policy, independent of the model, voice and sample rate.
#[derive(Clone, Copy, Debug)]
pub struct TempoConfig {
    pub grain_ms: f64,
    pub search_ms: f64,
    pub coarse_rate: u32,
    /// Maximum local rate deviation used only to align the finite endpoint.
    pub endpoint_rate_deviation: f64,
}
impl Default for TempoConfig {
    fn default() -> Self {
        Self {
            grain_ms: 40.,
            search_ms: 8.,
            coarse_rate: 6000,
            endpoint_rate_deviation: 0.25,
        }
    }
}

pub struct SpeechTempo {
    channels: usize,
    speed: f64,
    hop: usize,
    search: usize,
    stride: usize,
    endpoint: usize,
    endpoint_rate_deviation: f64,
    input: Vec<f32>,
    base: u64,
    received: u64,
    generated: u64,
    delivered: u64,
    tail: Vec<f32>,
    previous_start: u64,
    pending: Vec<f32>,
    prefix: Vec<f64>,
    reference_prefix: Vec<f64>,
    window: Vec<f64>,
    closed: bool,
    /// Grain picks recorded for tests: (desired, excluded verbatim start, low,
    /// high, chosen). The second field is None when no position was excluded.
    #[cfg(test)]
    selections: Vec<(u64, Option<u64>, u64, u64, u64)>,
}
impl SpeechTempo {
    pub fn new(sample_rate: u32, channels: u32, speed: f64) -> Result<Self, TempoError> {
        Self::with_config(sample_rate, channels, speed, TempoConfig::default())
    }
    pub fn with_config(
        sample_rate: u32,
        channels: u32,
        speed: f64,
        config: TempoConfig,
    ) -> Result<Self, TempoError> {
        if !speed.is_finite() || !(0.25..=4.).contains(&speed) {
            return Err(TempoError::Speed);
        }
        if !(1000..=500_000).contains(&sample_rate)
            || !(1..=32).contains(&channels)
            || !config.grain_ms.is_finite()
            || !(8. ..=100.).contains(&config.grain_ms)
            || !config.search_ms.is_finite()
            || config.search_ms < 0.
            || config.search_ms > config.grain_ms
            || config.coarse_rate == 0
            || !config.endpoint_rate_deviation.is_finite()
            || !(0.01..=0.5).contains(&config.endpoint_rate_deviation)
        {
            return Err(TempoError::Format);
        }
        let hop = (sample_rate as f64 * config.grain_ms / 2000.).round() as usize;
        let search = (sample_rate as f64 * config.search_ms / 1000.).round() as usize;
        let channels = channels as usize;
        let stride = (sample_rate / config.coarse_rate).max(1) as usize;
        // Cubic smoothstep has peak derivative 1.5. Reserve enough correction
        // time for the configured displacement and local rate deviation.
        let endpoint =
            hop + (1.5 * search as f64 / config.endpoint_rate_deviation).ceil() as usize + 1;
        if stride > hop {
            return Err(TempoError::Format);
        }
        Ok(Self {
            channels,
            speed,
            hop,
            search,
            stride,
            endpoint,
            endpoint_rate_deviation: config.endpoint_rate_deviation,
            input: Vec::new(),
            base: 0,
            received: 0,
            generated: 0,
            delivered: 0,
            tail: Vec::new(),
            previous_start: 0,
            pending: Vec::new(),
            prefix: Vec::new(),
            reference_prefix: Vec::new(),
            window: (0..hop).map(|i| fade(i, hop)).collect(),
            closed: false,
            #[cfg(test)]
            selections: Vec::new(),
        })
    }
    /// Input frames required for the first nonempty write, including the
    /// uncommitted endpoint. Subsequent grains follow the global speed ratio.
    pub fn lookahead_frames(&self) -> usize {
        if self.speed == 1. {
            0
        } else {
            ((self.endpoint_frames() / self.hop * self.hop) as f64 * self.speed).round() as usize
                + 2 * self.hop
                + self.search
        }
    }
    /// Bound on retained source frames for all supported input chunkings.
    pub fn history_limit_frames(&self) -> usize {
        (self.speed.ceil() as usize + 6) * self.hop + 3 * self.search + self.endpoint
    }
    pub fn buffered_frames(&self) -> usize {
        self.input.len() / self.channels
    }
    pub fn buffered_output_frames(&self) -> usize {
        self.pending.len() / self.channels
    }
    fn endpoint_frames(&self) -> usize {
        self.endpoint
    }
    pub fn write(&mut self, samples: &[f32]) -> Result<Vec<f32>, TempoError> {
        if self.closed {
            return Err(TempoError::Closed);
        }
        if !samples.len().is_multiple_of(self.channels) || samples.iter().any(|x| !x.is_finite()) {
            return Err(TempoError::Samples);
        }
        let total = self
            .received
            .checked_add((samples.len() / self.channels) as u64)
            .filter(|n| *n <= 1u64 << 52)
            .ok_or(TempoError::Overflow)?;
        if self.speed == 1. {
            self.received = total;
            self.delivered = total;
            self.generated = total;
            return Ok(samples.to_vec());
        }
        let mut output = std::mem::take(&mut self.pending);
        for block in samples.chunks(self.hop * self.channels) {
            self.input.extend_from_slice(block);
            self.received += (block.len() / self.channels) as u64;
            while self.ready() {
                self.render(false, &mut output);
                self.prune();
            }
        }
        let budget = self.target() - self.delivered;
        // Keep an endpoint editable until EOF is known. Otherwise a final
        // transient can have been discarded before finish can align the tail.
        let count = output
            .len()
            .saturating_sub(self.endpoint_frames() * self.channels)
            .min(budget as usize * self.channels);
        self.pending = output.split_off(count);
        self.delivered += (output.len() / self.channels) as u64;
        Ok(output)
    }
    pub fn finish(&mut self) -> Result<Vec<f32>, TempoError> {
        if self.closed {
            return Ok(Vec::new());
        }
        self.closed = true;
        let target = self.target();
        let remaining = target
            .checked_sub(self.delivered)
            .ok_or(TempoError::Overflow)? as usize;
        let mut output = std::mem::take(&mut self.pending);
        if self.generated == 0 && self.received < 2 {
            // A single frame has no waveform to align; preserve its value.
            if self.received == 1 {
                for _ in 0..remaining {
                    output.extend_from_slice(&self.input[..self.channels]);
                }
            }
        } else {
            if self.generated == 0 && self.received < (2 * self.hop) as u64 {
                self.hop = (self.received as usize / 2).max(1);
                self.search = self.search.min(self.hop);
                self.stride = self.stride.min(self.hop);
                self.window = (0..self.hop).map(|i| fade(i, self.hop)).collect();
            }
            while self.generated < target {
                self.render(true, &mut output);
            }
        }
        output.truncate(remaining * self.channels);
        self.align_endpoint(&mut output);
        self.delivered = target;
        self.input.clear();
        self.tail.clear();
        Ok(output)
    }
    fn target(&self) -> u64 {
        (self.received as f64 / self.speed).round() as u64
    }
    fn nominal(&self) -> u64 {
        (self.generated as f64 * self.speed).round() as u64
    }
    fn align_endpoint(&mut self, output: &mut [f32]) {
        let remaining = output.len() / self.channels;
        let count = self
            .endpoint_frames()
            .min(remaining)
            .min(self.received as usize);
        if count == 0 {
            return;
        }
        let offset = (remaining - count) * self.channels;
        if count == 1 {
            for channel in 0..self.channels {
                output[offset + channel] = self.sample(self.received - 1, channel);
            }
            return;
        }
        let overlap = (count / 3).max(1).min(self.hop);
        self.tail.clear();
        self.tail
            .extend_from_slice(&output[offset..offset + overlap * self.channels]);
        let desired = self.received - count as u64;
        // Limit displacement so the smooth source map remains strictly
        // increasing, even when a caller configures an unusually wide search.
        let radius = self.search.min(
            ((count - overlap - 1) as f64 * self.endpoint_rate_deviation / 1.5).floor() as usize,
        ) as u64;
        let low = desired.saturating_sub(radius).max(self.base);
        let high = (desired + radius).min(self.received - overlap as u64);
        let start = self.select(desired, low, high, false);
        let correction = desired as f64 - start as f64;
        for frame in 0..count {
            let position = if frame < overlap {
                start as f64 + frame as f64
            } else {
                let t = (frame - overlap) as f64 / (count - overlap - 1).max(1) as f64;
                start as f64 + frame as f64 + correction * t * t * (3. - 2. * t)
            };
            let position = position.clamp(self.base as f64, (self.received - 1) as f64);
            let left = position.floor() as u64;
            let fraction = position - left as f64;
            // Ramp length is overlap+1 so frame 0 keeps weight 0 even at
            // overlap==1 (short tails); the frame at `overlap` crosses to 1.
            let weight = if frame < overlap {
                fade(frame, overlap + 1)
            } else {
                1.
            };
            for channel in 0..self.channels {
                let a = self.sample(left, channel) as f64;
                let b = self.sample((left + 1).min(self.received - 1), channel) as f64;
                let source = a * (1. - fraction) + b * fraction;
                let index = offset + frame * self.channels + channel;
                output[index] = (output[index] as f64 * (1. - weight) + source * weight) as f32;
            }
        }
    }
    fn ready(&self) -> bool {
        if self.tail.is_empty() {
            self.received >= (2 * self.hop) as u64
        } else {
            self.received >= self.nominal() + (self.search + 2 * self.hop) as u64
        }
    }
    fn prune(&mut self) {
        // Retain endpoint history too: EOF may arrive before another full grain.
        let keep = self.nominal().saturating_sub(self.search as u64).min(
            self.received
                .saturating_sub((self.endpoint_frames().max(2 * self.hop) + self.search) as u64),
        );
        if keep > self.base {
            let frames = (keep - self.base).min(self.buffered_frames() as u64) as usize;
            self.input.drain(..frames * self.channels);
            self.base += frames as u64;
        }
    }
    fn sample(&self, frame: u64, channel: usize) -> f32 {
        // Only real retained PCM may participate in similarity or overlap.
        debug_assert!(frame >= self.base && frame < self.received);
        self.input[(frame - self.base) as usize * self.channels + channel]
    }
    fn prepare_search(&mut self, end: u64) {
        let frames = (end - self.base) as usize;
        self.prefix.clear();
        self.prefix.resize((frames + 1) * self.channels, 0.);
        for i in 0..frames {
            for c in 0..self.channels {
                self.prefix[(i + 1) * self.channels + c] = self.prefix[i * self.channels + c]
                    + self.sample(self.base + i as u64, c) as f64;
            }
        }
        self.reference_prefix.clear();
        let reference_frames = self.tail.len() / self.channels;
        self.reference_prefix
            .resize((reference_frames + 1) * self.channels, 0.);
        for i in 0..reference_frames {
            for c in 0..self.channels {
                self.reference_prefix[(i + 1) * self.channels + c] = self.reference_prefix
                    [i * self.channels + c]
                    + self.tail[i * self.channels + c] as f64;
            }
        }
    }
    fn similarity(&self, start: u64, stride: usize) -> f64 {
        let offset = (start - self.base) as usize;
        let mut cross = 0.;
        let mut energy = 0.;
        let reference_frames = self.tail.len() / self.channels;
        for i in (0..reference_frames).step_by(stride) {
            let end = (i + stride).min(reference_frames);
            for c in 0..self.channels {
                let a = self.reference_prefix[end * self.channels + c]
                    - self.reference_prefix[i * self.channels + c];
                let b = self.prefix[(offset + end) * self.channels + c]
                    - self.prefix[(offset + i) * self.channels + c];
                cross += a * b;
                energy += a * a + b * b;
            }
        }
        // Energy-aware normalized correlation. Channels contribute separately,
        // so opposite-phase stereo cannot cancel during the search.
        if energy <= f64::MIN_POSITIVE {
            1.
        } else {
            2. * cross / energy
        }
    }
    fn select(&mut self, desired: u64, low: u64, high: u64, verbatim_tail: bool) -> u64 {
        self.prepare_search(high + (self.tail.len() / self.channels) as u64);
        let mut best = desired.clamp(low, high);
        // On the render path the tail is copied verbatim from input at
        // previous_start + hop, so that position always scores a perfect
        // self-match and would lock grains into rate-1 playback. Exclude it
        // unless it is the nominal continuation (natural == desired), where
        // verbatim is correct. The endpoint tail is synthesized output rather
        // than verbatim input, so no position is excluded there.
        let natural = self.previous_start + self.hop as u64;
        let verbatim = |candidate: u64| verbatim_tail && candidate == natural && natural != desired;
        // The clamped seed itself can land on the excluded position at the
        // finishing boundary; reseed to a neighbor so it cannot win by default.
        if verbatim(best) {
            best = if best > low { best - 1 } else { high };
        }
        let mut score = self.similarity(best, self.stride);
        for candidate in (low..=high).step_by(self.stride) {
            if verbatim(candidate) {
                continue;
            }
            let value = self.similarity(candidate, self.stride);
            if preferable(value, candidate, score, best, desired) {
                best = candidate;
                score = value;
            }
        }
        let coarse = best;
        score = self.similarity(best, 1);
        for candidate in coarse.saturating_sub(self.stride as u64).max(low)
            ..=(coarse + self.stride as u64).min(high)
        {
            if verbatim(candidate) {
                continue;
            }
            let value = self.similarity(candidate, 1);
            if preferable(value, candidate, score, best, desired) {
                best = candidate;
                score = value;
            }
        }
        #[cfg(test)]
        self.selections
            .push((desired, verbatim_tail.then_some(natural), low, high, best));
        best
    }
    fn render(&mut self, finishing: bool, output: &mut Vec<f32>) {
        let start = if self.tail.is_empty() {
            0
        } else {
            let desired = self.nominal();
            let limit = if finishing {
                self.received - (2 * self.hop) as u64
            } else {
                desired + self.search as u64
            };
            let high = (desired + self.search as u64).min(limit).max(self.base);
            let low = desired
                .saturating_sub(self.search as u64)
                .max(self.base)
                .min(high);
            self.select(desired, low, high, true)
        };
        for frame in 0..self.hop {
            for channel in 0..self.channels {
                let new = self.sample(start + frame as u64, channel) as f64;
                let value = if self.tail.is_empty() {
                    new
                } else {
                    let old = self.tail[frame * self.channels + channel] as f64;
                    let weight = self.window[frame];
                    old * (1. - weight) + new * weight
                };
                output.push(value as f32);
            }
        }
        self.tail.clear();
        for frame in 0..self.hop {
            for channel in 0..self.channels {
                self.tail
                    .push(self.sample(start + (self.hop + frame) as u64, channel));
            }
        }
        self.previous_start = start;
        self.generated += self.hop as u64;
    }
}
fn fade(frame: usize, length: usize) -> f64 {
    if length <= 1 {
        1.
    } else {
        0.5 - 0.5 * (std::f64::consts::PI * frame as f64 / (length - 1) as f64).cos()
    }
}
fn preferable(
    score: f64,
    position: u64,
    best_score: f64,
    best_position: u64,
    desired: u64,
) -> bool {
    score > best_score + 1e-9
        || ((score - best_score).abs() <= 1e-9
            && position.abs_diff(desired) < best_position.abs_diff(desired))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn wave(frames: usize, channels: usize) -> Vec<f32> {
        (0..frames)
            .flat_map(|i| {
                (0..channels).map(move |c| {
                    (std::f32::consts::TAU * (200. + c as f32 * 100.) * i as f32 / 24000.).sin()
                        * 0.5
                })
            })
            .collect()
    }
    fn run(samples: &[f32], channels: usize, speed: f64, chunk: usize) -> Vec<f32> {
        let mut tempo = SpeechTempo::new(24000, channels as u32, speed).unwrap();
        let mut output = Vec::new();
        for block in samples.chunks(chunk * channels) {
            output.extend(tempo.write(block).unwrap());
            assert!(tempo.buffered_frames() <= tempo.history_limit_frames());
            assert!(tempo.buffered_output_frames() <= tempo.history_limit_frames());
        }
        output.extend(tempo.finish().unwrap());
        assert!(tempo.finish().unwrap().is_empty());
        assert!(tempo.write(&[]).is_err());
        output
    }
    #[test]
    fn chunking_preserves_pcm_and_stereo_alignment() {
        let samples = wave(24000, 2);
        for speed in [0.25, 0.5, 0.8, 1., 1.25, 2., 4.] {
            let whole = run(&samples, 2, speed, 24000);
            for chunk in [1, 137, 3840] {
                assert_eq!(run(&samples, 2, speed, chunk), whole);
            }
            assert_eq!(whole.len(), (24000f64 / speed).round() as usize * 2);
            if speed == 1. {
                assert_eq!(whole, samples);
            }
        }
    }
    #[test]
    fn preserves_pitch_in_each_channel() {
        let samples = wave(48000, 2);
        for speed in [0.8, 1.25, 2.] {
            let output = run(&samples, 2, speed, 997);
            for channel in 0..2 {
                let center: Vec<f32> = output
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|x| x[channel])
                    .skip(2400)
                    .take(12000)
                    .collect();
                let crossings = center
                    .windows(2)
                    .filter(|x| x[0] <= 0. && x[1] > 0.)
                    .count();
                assert!(
                    (crossings as f64 * 2. - (200. + channel as f64 * 100.)).abs() < 5.,
                    "{speed} {channel} {crossings}"
                );
            }
        }
    }
    #[test]
    fn silence_and_short_tails_have_exact_length() {
        for count in [0, 1, 2, 31, 735, 736, 737, 5000] {
            for speed in [0.25, 0.8, 1., 1.25, 4.] {
                let output = run(&vec![0.; count], 1, speed, 7);
                assert_eq!(output.len(), (count as f64 / speed).round() as usize);
                assert!(output.iter().all(|x| *x == 0.));
            }
        }
    }
    #[test]
    fn nonperiodic_audio_is_chunk_invariant() {
        let mut state = 123u32;
        let input: Vec<f32> = (0..20000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as f32 / u32::MAX as f32 - 0.5
            })
            .collect();
        for speed in [0.8, 1.001, 1.25, 3.9] {
            assert_eq!(
                run(&input, 1, speed, 109),
                run(&input, 1, speed, input.len())
            );
        }
    }
    #[test]
    fn slow_near_unity_retains_excess_until_source_arrives() {
        for speed in [0.95, 0.99, 0.999, 0.999999, 1.000001] {
            for count in [737, 798, 800, 1472, 2208, 2999, 10_003] {
                let input = vec![0.25; count];
                let whole = run(&input, 1, speed, count);
                let small = run(&input, 1, speed, 17);
                assert_eq!(whole.len(), (count as f64 / speed).round() as usize);
                assert_eq!(whole, small);
            }
        }
    }

    #[test]
    fn opposite_phase_stereo_does_not_cancel_pitch_analysis() {
        let input: Vec<f32> = wave(48_000, 1).into_iter().flat_map(|x| [x, -x]).collect();
        for speed in [0.8, 1.25, 2.] {
            let output = run(&input, 2, speed, 997);
            assert!(output.as_chunks::<2>().0.iter().all(|x| x[0] == -x[1]));
            let center: Vec<f32> = output
                .as_chunks::<2>()
                .0
                .iter()
                .skip(2400)
                .take(12000)
                .map(|x| x[0])
                .collect();
            let frequency = center
                .windows(2)
                .filter(|x| x[0] <= 0. && x[1] > 0.)
                .count() as f64
                * 2.;
            assert!((frequency - 200.).abs() <= 2.);
            let rms = (center.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()
                / center.len() as f64)
                .sqrt();
            assert!(rms > 0.3);
        }
    }

    #[test]
    fn eof_preserves_nonzero_short_and_slow_input() {
        for count in [1, 2, 3, 100, 959, 960, 2400, 10_003] {
            for speed in [0.25, 0.8, 0.95, 1., 1.15, 2., 4.] {
                let input = vec![0.75; count];
                let output = run(&input, 1, speed, 17);
                assert_eq!(output.len(), (count as f64 / speed).round() as usize);
                assert!(
                    output.iter().all(|x| (*x - 0.75).abs() < 1e-6),
                    "{count} {speed}"
                );
                assert_eq!(output, run(&input, 1, speed, count));
            }
        }
    }
    #[test]
    fn eof_preserves_last_transient() {
        for (count, speed) in [(960, 2.), (1200, 4.), (100, 1.25), (10_003, 1.15)] {
            let mut input = vec![0.; count];
            input[count - 1] = 1.;
            let output = run(&input, 1, speed, 7);
            assert_eq!(output.last(), Some(&1.), "{count} {speed}");
            assert_eq!(output, run(&input, 1, speed, count));
        }
    }
    #[test]
    fn endpoint_alignment_avoids_periodic_phase_cancellation() {
        for speed in [0.8, 1.15, 1.25, 2., 4.] {
            for count in (24_000..24_480).step_by(24) {
                let output = run(&wave(count, 1), 1, speed, 137);
                for period in output[output.len() - 1440..].windows(120).step_by(12) {
                    let rms =
                        (period.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / 120.).sqrt();
                    assert!(rms > 0.27, "speed={speed} count={count} rms={rms}");
                }
            }
        }
    }
    #[test]
    fn reported_lookahead_includes_uncommitted_endpoint() {
        for speed in [0.25, 0.8, 1.15, 2., 4.] {
            let mut tempo = SpeechTempo::new(24000, 1, speed).unwrap();
            let required = tempo.lookahead_frames();
            assert!(tempo.write(&vec![0.5; required - 1]).unwrap().is_empty());
            assert!(!tempo.write(&[0.5]).unwrap().is_empty());
        }
    }
    #[test]
    fn finite_eof_retains_a_complete_grain_with_narrow_search() {
        for search_ms in [0., 0.05, 1.] {
            for speed in [0.25, 0.8, 1.15, 4.] {
                let mut tempo = SpeechTempo::with_config(
                    24000,
                    1,
                    speed,
                    TempoConfig {
                        search_ms,
                        ..TempoConfig::default()
                    },
                )
                .unwrap();
                let mut output = Vec::new();
                for _ in 0..24 {
                    output.extend(tempo.write(&[1.; 100]).unwrap());
                }
                output.extend(tempo.finish().unwrap());
                assert_eq!(output.len(), (2400. / speed).round() as usize);
                assert!(
                    output.iter().all(|x| (*x - 1.).abs() < 1e-6),
                    "{search_ms} {speed}"
                );
            }
        }
    }
    #[test]
    fn rejects_invalid_parameters_and_input_without_advancing() {
        for speed in [f64::NAN, f64::INFINITY, 0., -1., 10.] {
            assert!(SpeechTempo::new(24000, 1, speed).is_err());
        }
        assert!(SpeechTempo::new(0, 1, 1.).is_err());
        assert!(SpeechTempo::new(24000, 0, 1.).is_err());
        let mut tempo = SpeechTempo::new(24000, 2, 1.25).unwrap();
        assert!(tempo.write(&[0.]).is_err());
        assert!(tempo.write(&[f32::NAN, 0.]).is_err());
        assert_eq!(tempo.received, 0);
        assert!(tempo.finish().unwrap().is_empty());
    }
    #[test]
    fn selection_never_locks_onto_the_verbatim_continuation() {
        // The tail mirrors input at previous_start + hop, so that candidate
        // always correlates to exactly 1. Unless it is also the nominal time
        // map position, choosing it stalls the map and later forces a skip.
        let mut state = 0x9e37_79b9u32;
        let input: Vec<f32> = (0..48000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as f32 / u32::MAX as f32 - 0.5
            })
            .collect();
        for speed in [0.7, 0.8, 1.1, 1.25, 1.35] {
            let mut tempo = SpeechTempo::new(24000, 1, speed).unwrap();
            for block in input.chunks(1531) {
                tempo.write(block).unwrap();
            }
            tempo.finish().unwrap();
            assert!(!tempo.selections.is_empty(), "speed {speed}");
            assert!(
                tempo.selections.iter().any(|s| s.1.is_some()),
                "speed {speed}: no render-path selections recorded"
            );
            for &(desired, natural, low, high, chosen) in &tempo.selections {
                if let Some(natural) = natural {
                    assert!(
                        chosen != natural || chosen == desired.clamp(low, high),
                        "speed {speed}: grain at {chosen} locked onto the verbatim \
                         continuation {natural} (desired {desired} in {low}..={high})"
                    );
                }
            }
        }
    }
    #[test]
    fn endpoint_replacement_crossfades_the_first_frame() {
        // With a short final tail the overlap resolves to a single frame; the
        // replaced region must still crossfade instead of hard-overwriting the
        // boundary sample with a verbatim input frame.
        let input: Vec<f32> = (0..5)
            .map(|i| (std::f32::consts::TAU * 800. * i as f32 / 24000.).sin())
            .collect();
        let output = run(&input, 1, 0.25, 5);
        let boundary = output.len() - 5;
        assert_ne!(output[boundary], input[0]);
        let step = (output[boundary] - output[boundary - 1]).abs();
        assert!(step < 0.4, "boundary discontinuity {step}");
    }
}
