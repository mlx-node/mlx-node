//! CoreAudio PCM sink. The realtime callback touches only a preallocated SPSC
//! ring and atomics; producer waits and device lifecycle run outside it.
use coreaudio::audio_unit::{
    AudioUnit, Element, IOType, SampleFormat, Scope, StreamFormat,
    audio_format::LinearPcmFlags,
    render_callback::{self, data},
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use objc2_core_audio::{AudioGetCurrentHostTime, AudioGetHostClockFrequency};
use objc2_core_audio_types::AudioTimeStampFlags;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

// A stalled CoreAudio render callback (device unplugged or suspended) makes
// the ring stop draining; bound every producer wait so callers can error out.
const STALL: Duration = Duration::from_secs(10);

struct Ring {
    samples: Box<[AtomicU32]>,
    head: AtomicUsize,
    tail: AtomicUsize,
    enabled: AtomicBool,
    closed: AtomicBool,
    cancelled: AtomicBool,
    // Set before cancelling on a stall so finish() reports the stall rather
    // than racing cancel() to a clean partial drain.
    stalled: AtomicBool,
    underruns: AtomicU64,
    played: AtomicU64,
    first_ns: AtomicU64,
    drain_ns: AtomicU64,
    started_host: u64,
    host_ns_per_tick: f64,
    samples_per_second: f64,
}
impl Ring {
    fn host_elapsed_ns(&self, host: u64) -> u64 {
        (host.saturating_sub(self.started_host) as f64 * self.host_ns_per_tick) as u64
    }
    fn now_ns(&self) -> u64 {
        self.host_elapsed_ns(unsafe { AudioGetCurrentHostTime() })
    }
    fn queued(&self) -> usize {
        self.head
            .load(Ordering::Acquire)
            .wrapping_sub(self.tail.load(Ordering::Acquire))
    }
    fn push(&self, input: &[f32]) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let free = self.samples.len() - head.wrapping_sub(self.tail.load(Ordering::Acquire));
        let count = free.min(input.len());
        for (i, &value) in input[..count].iter().enumerate() {
            self.samples[(head + i) % self.samples.len()].store(value.to_bits(), Ordering::Relaxed);
        }
        self.head.store(head.wrapping_add(count), Ordering::Release);
        count
    }
    fn render(&self, output: &mut [f32], presentation_ns: u64) {
        output.fill(0.);
        if !self.enabled.load(Ordering::Acquire) || self.cancelled.load(Ordering::Acquire) {
            return;
        }
        let tail = self.tail.load(Ordering::Relaxed);
        let count = self
            .head
            .load(Ordering::Acquire)
            .wrapping_sub(tail)
            .min(output.len());
        for (i, value) in output[..count].iter_mut().enumerate() {
            *value = f32::from_bits(
                self.samples[(tail + i) % self.samples.len()].load(Ordering::Relaxed),
            );
        }
        self.played.fetch_add(count as u64, Ordering::Relaxed);
        if count > 0 {
            self.drain_ns.store(
                presentation_ns
                    .saturating_add((count as f64 / self.samples_per_second * 1e9).ceil() as u64),
                Ordering::Release,
            );
            let _ = self.first_ns.compare_exchange(
                0,
                presentation_ns.max(1),
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
        // Publish the deadline before the consumer observes an empty ring.
        self.tail.store(tail.wrapping_add(count), Ordering::Release);
        if count < output.len() && !self.closed.load(Ordering::Acquire) {
            self.underruns.fetch_add(1, Ordering::Relaxed);
        }
    }
}
struct Playback {
    ring: Arc<Ring>,
    unit: Mutex<AudioUnit>,
    writer: tokio::sync::Mutex<()>,
    prebuffer: usize,
    sample_rate: u32,
    channels: u32,
}
impl Playback {
    fn stop(&self) {
        self.ring.cancelled.store(true, Ordering::Release);
        if let Ok(mut unit) = self.unit.lock() {
            let _ = unit.stop();
        }
    }
    fn stall(&self) {
        self.ring.stalled.store(true, Ordering::Release);
        self.stop();
    }
}
#[napi(object)]
pub struct PcmPlaybackStats {
    pub played_seconds: f64,
    pub underruns: f64,
    pub first_playback_ms: Option<f64>,
}
#[napi]
pub struct PcmPlayer {
    inner: Arc<Playback>,
}
#[napi]
impl PcmPlayer {
    #[napi]
    pub fn open(
        sample_rate: u32,
        channels: u32,
        buffer_seconds: Option<f64>,
        prebuffer_seconds: Option<f64>,
    ) -> Result<Self> {
        let seconds = buffer_seconds.unwrap_or(1.);
        let prebuffer = prebuffer_seconds.unwrap_or(0.32);
        if sample_rate == 0
            || !(1..=2).contains(&channels)
            || !seconds.is_finite()
            || !(0.01..=60.).contains(&seconds)
            || !prebuffer.is_finite()
            || prebuffer < 0.
            || prebuffer > seconds
        {
            return Err(Error::from_reason(
                "Invalid PCM playback format or buffer duration",
            ));
        }
        let capacity = ((seconds * sample_rate as f64).ceil() as usize) * channels as usize;
        let host_frequency = unsafe { AudioGetHostClockFrequency() };
        if !host_frequency.is_finite() || host_frequency <= 0. {
            return Err(Error::from_reason("CoreAudio host clock unavailable"));
        }
        let ring = Arc::new(Ring {
            samples: (0..capacity).map(|_| AtomicU32::new(0)).collect(),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            enabled: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            stalled: AtomicBool::new(false),
            underruns: AtomicU64::new(0),
            played: AtomicU64::new(0),
            first_ns: AtomicU64::new(0),
            drain_ns: AtomicU64::new(0),
            started_host: unsafe { AudioGetCurrentHostTime() },
            host_ns_per_tick: 1e9 / host_frequency,
            samples_per_second: sample_rate as f64 * channels as f64,
        });
        let error = |e: coreaudio::Error| Error::from_reason(format!("CoreAudio playback: {e}"));
        let mut unit = AudioUnit::new(IOType::DefaultOutput).map_err(error)?;
        unit.uninitialize().map_err(error)?;
        // The AudioUnit converts this source format to the hardware format.
        unit.set_stream_format(
            StreamFormat {
                sample_rate: sample_rate as f64,
                sample_format: SampleFormat::F32,
                flags: LinearPcmFlags::IS_FLOAT | LinearPcmFlags::IS_PACKED,
                channels,
            },
            Scope::Input,
            Element::Output,
        )
        .map_err(error)?;
        let callback_ring = ring.clone();
        unit.set_render_callback(move |args: render_callback::Args<data::Interleaved<f32>>| {
            let presentation = if args
                .time_stamp
                .mFlags
                .contains(AudioTimeStampFlags::HostTimeValid)
            {
                callback_ring.host_elapsed_ns(args.time_stamp.mHostTime)
            } else {
                callback_ring.now_ns()
            };
            callback_ring.render(args.data.buffer, presentation);
            Ok(())
        })
        .map_err(error)?;
        unit.initialize().map_err(error)?;
        unit.start().map_err(error)?;
        Ok(Self {
            inner: Arc::new(Playback {
                ring,
                unit: Mutex::new(unit),
                writer: tokio::sync::Mutex::new(()),
                prebuffer: (prebuffer * sample_rate as f64).ceil() as usize * channels as usize,
                sample_rate,
                channels,
            }),
        })
    }
    #[napi]
    pub fn write<'env>(
        &self,
        env: &'env Env,
        samples: Float32Array,
    ) -> Result<PromiseRaw<'env, ()>> {
        if !samples.len().is_multiple_of(self.inner.channels as usize)
            || samples.iter().any(|x| !x.is_finite())
        {
            return Err(Error::from_reason("Invalid PCM sample buffer"));
        }
        let samples = samples.to_vec();
        let inner = self.inner.clone();
        env.spawn_future(async move {
            let _writer = inner.writer.lock().await;
            let mut at = 0;
            let mut last_progress = Instant::now();
            while at < samples.len() {
                if inner.ring.stalled.load(Ordering::Acquire) {
                    return Err(Error::from_reason("PCM playback stalled"));
                }
                if inner.ring.cancelled.load(Ordering::Acquire)
                    || inner.ring.closed.load(Ordering::Acquire)
                {
                    return Err(Error::from_reason("PCM player closed"));
                }
                let pushed = inner.ring.push(&samples[at..]);
                at += pushed;
                if inner.ring.queued() >= inner.prebuffer {
                    inner.ring.enabled.store(true, Ordering::Release);
                }
                if at < samples.len() {
                    if pushed > 0 {
                        last_progress = Instant::now();
                    } else if last_progress.elapsed() >= STALL {
                        inner.stall();
                        return Err(Error::from_reason("PCM playback stalled"));
                    }
                    tokio::time::sleep(Duration::from_millis(4)).await;
                }
            }
            Ok(())
        })
    }
    #[napi]
    pub fn finish<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, PcmPlaybackStats>> {
        let inner = self.inner.clone();
        env.spawn_future(async move {
            let _writer = inner.writer.lock().await;
            inner.ring.closed.store(true, Ordering::Release);
            inner.ring.enabled.store(true, Ordering::Release);
            let mut last_progress = Instant::now();
            let mut last_queued = inner.ring.queued();
            while inner.ring.queued() > 0 && !inner.ring.cancelled.load(Ordering::Acquire) {
                let queued = inner.ring.queued();
                if queued != last_queued {
                    last_queued = queued;
                    last_progress = Instant::now();
                } else if last_progress.elapsed() >= STALL {
                    inner.stall();
                    return Err(Error::from_reason("PCM playback stalled"));
                }
                tokio::time::sleep(Duration::from_millis(4)).await;
            }
            // An empty software ring only means CoreAudio accepted the last
            // block. Wait until its scheduled presentation ends before stop.
            let deadline = Instant::now() + STALL;
            while !inner.ring.cancelled.load(Ordering::Acquire) {
                let remaining = inner
                    .ring
                    .drain_ns
                    .load(Ordering::Acquire)
                    .saturating_sub(inner.ring.now_ns());
                if remaining == 0 || Instant::now() >= deadline {
                    break;
                }
                tokio::time::sleep(Duration::from_nanos(remaining).min(Duration::from_millis(4)))
                    .await;
            }
            // A stall on either side sets `stalled`; a concurrent finish must
            // surface it instead of resolving a partial drain as success.
            if inner.ring.stalled.load(Ordering::Acquire) {
                return Err(Error::from_reason("PCM playback stalled"));
            }
            let stats = stats(&inner);
            inner.stop();
            Ok(stats)
        })
    }
    #[napi]
    pub fn cancel(&self) {
        self.inner.stop();
    }
}
fn stats(inner: &Playback) -> PcmPlaybackStats {
    let first = inner.ring.first_ns.load(Ordering::Acquire);
    PcmPlaybackStats {
        played_seconds: inner.ring.played.load(Ordering::Acquire) as f64
            / (inner.sample_rate as f64 * inner.channels as f64),
        underruns: inner.ring.underruns.load(Ordering::Acquire) as f64,
        first_playback_ms: if first == 0 {
            None
        } else {
            Some(first as f64 / 1e6)
        },
    }
}
impl Drop for PcmPlayer {
    fn drop(&mut self) {
        self.inner.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(capacity: usize, samples_per_second: f64) -> Ring {
        Ring {
            samples: (0..capacity).map(|_| AtomicU32::new(0)).collect(),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            enabled: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            stalled: AtomicBool::new(false),
            underruns: AtomicU64::new(0),
            played: AtomicU64::new(0),
            first_ns: AtomicU64::new(0),
            drain_ns: AtomicU64::new(0),
            started_host: 0,
            host_ns_per_tick: 1.,
            samples_per_second,
        }
    }

    #[test]
    fn interleaved_pcm_stays_in_order_across_wrap_and_full_ring() {
        let ring = ring(6, 48_000. * 2.);
        ring.enabled.store(true, Ordering::Release);
        assert_eq!(ring.push(&[1., -1., 2., -2., 3., -3., 4., -4.]), 6);
        assert_eq!(ring.push(&[4., -4.]), 0);
        let mut first = [0.; 4];
        ring.render(&mut first, 1);
        assert_eq!(first, [1., -1., 2., -2.]);
        assert_eq!(ring.push(&[4., -4., 5., -5.]), 4);
        let mut rest = [0.; 6];
        ring.render(&mut rest, 2);
        assert_eq!(rest, [3., -3., 4., -4., 5., -5.]);
        assert_eq!(ring.queued(), 0);
        assert_eq!(ring.played.load(Ordering::Acquire), 10);
        assert_eq!(ring.underruns.load(Ordering::Acquire), 0);
    }

    #[test]
    fn prebuffer_and_cancellation_do_not_consume_or_report_starvation() {
        let ring = ring(4, 24_000.);
        ring.push(&[1., 2.]);
        let mut output = [9.; 4];
        ring.render(&mut output, 100);
        assert_eq!(output, [0.; 4]);
        assert_eq!(ring.queued(), 2);
        ring.enabled.store(true, Ordering::Release);
        ring.cancelled.store(true, Ordering::Release);
        output.fill(9.);
        ring.render(&mut output, 200);
        assert_eq!(output, [0.; 4]);
        assert_eq!(ring.queued(), 2);
        assert_eq!(ring.played.load(Ordering::Acquire), 0);
        assert_eq!(ring.first_ns.load(Ordering::Acquire), 0);
        assert_eq!(ring.underruns.load(Ordering::Acquire), 0);
    }

    #[test]
    fn starvation_and_normal_tail_have_distinct_statistics_and_deadlines() {
        let ring = ring(4, 2_000.);
        ring.enabled.store(true, Ordering::Release);
        ring.push(&[1., 2.]);
        let mut output = [0.; 4];
        ring.render(&mut output, 10_000_000);
        assert_eq!(output, [1., 2., 0., 0.]);
        assert_eq!(ring.underruns.load(Ordering::Acquire), 1);
        assert_eq!(ring.drain_ns.load(Ordering::Acquire), 11_000_000);
        ring.push(&[3.]);
        ring.closed.store(true, Ordering::Release);
        ring.render(&mut output, 20_000_000);
        assert_eq!(output, [3., 0., 0., 0.]);
        assert_eq!(ring.underruns.load(Ordering::Acquire), 1);
        assert_eq!(ring.first_ns.load(Ordering::Acquire), 10_000_000);
        assert_eq!(ring.drain_ns.load(Ordering::Acquire), 20_500_000);
        ring.render(&mut output, 30_000_000);
        assert_eq!(output, [0.; 4]);
        assert_eq!(ring.played.load(Ordering::Acquire), 3);
        assert_eq!(ring.drain_ns.load(Ordering::Acquire), 20_500_000);
        assert_eq!(ring.underruns.load(Ordering::Acquire), 1);
    }
}
