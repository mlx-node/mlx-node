//! OS audio device discovery and real-time capture primitives.

use std::cell::{Cell, UnsafeCell};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::error::AudioError;

#[cfg(target_os = "macos")]
mod core_audio;
#[cfg(windows)]
mod wasapi;

/// Audio capture source kind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CaptureSource {
    #[default]
    Microphone,
    SystemAudio,
}

impl CaptureSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Microphone => "microphone",
            Self::SystemAudio => "system-audio",
        }
    }
}

/// Options controlling native audio device selection, buffering, and feed cadence.
#[derive(Debug, Clone, Default)]
pub struct CaptureOptions {
    /// Audio source. Omit to capture the microphone.
    pub source: Option<CaptureSource>,
    /// Stable native device ID returned by [`audio_devices`] or
    /// [`input_devices`] (Core Audio UID on macOS, WASAPI endpoint ID on Windows).
    pub device_id: Option<String>,
    /// Device name. Omit to use the default input or output device for the
    /// selected source.
    pub device_name: Option<String>,
    /// For system audio, optionally capture only processes with these bundle
    /// identifiers. Empty or omitted captures all audio sent to the device.
    pub application_bundle_ids: Option<Vec<String>>,
    /// Lock-free callback ring capacity in seconds (default 10).
    pub ring_seconds: Option<f64>,
    /// Amount drained from the ring into each consumer feed (default 100 ms).
    pub feed_milliseconds: Option<u32>,
}

/// Discovered audio input endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputDevice {
    pub id: String,
    pub name: String,
    pub is_default: bool,
    pub sample_rate: u32,
    pub channels: u32,
    pub sample_format: String,
}

/// Discovered audio capture endpoint (microphone or system output).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioDevice {
    pub id: String,
    pub name: String,
    pub source: CaptureSource,
    pub is_default: bool,
    pub sample_rate: u32,
    pub channels: u32,
}

/// Fixed-capacity single-producer/single-consumer ring storage. The native
/// audio callback performs no allocation, lock acquisition, inference, or
/// resampling. Acquire/release ordering makes each slot visible before the
/// producer publishes its write cursor.
struct AudioRingInner {
    data: Box<[UnsafeCell<f32>]>,
    capacity: usize,
    read: AtomicUsize,
    write: AtomicUsize,
    captured: AtomicU64,
    dropped: AtomicU64,
    stopped: AtomicBool,
    wait_lock: Mutex<()>,
    ready: Condvar,
}

// SAFETY: `AudioRingInner` is private and only handed out as a unique
// (`!Clone`, `!Sync`) `AudioRingProducer`/`AudioRingConsumer` pair plus
// atomic/condvar-only `AudioRingControl` handles. Only the producer writes
// `data` slots (`&mut self`) and only the consumer reads them (`&mut self`);
// acquire/release cursors publish slot ownership transitions.
unsafe impl Sync for AudioRingInner {}

impl AudioRingInner {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capture ring capacity must be positive");
        let mut data = Vec::with_capacity(capacity);
        data.resize_with(capacity, || UnsafeCell::new(0.0));
        Self {
            data: data.into_boxed_slice(),
            capacity,
            read: AtomicUsize::new(0),
            write: AtomicUsize::new(0),
            captured: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            stopped: AtomicBool::new(false),
            wait_lock: Mutex::new(()),
            ready: Condvar::new(),
        }
    }

    #[inline]
    fn push(&self, sample: f32) {
        let write = self.write.load(Ordering::Relaxed);
        let read = self.read.load(Ordering::Acquire);
        if write.wrapping_sub(read) >= self.capacity {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        unsafe { *self.data[write % self.capacity].get() = sample };
        self.write.store(write.wrapping_add(1), Ordering::Release);
        self.captured.fetch_add(1, Ordering::Relaxed);
    }

    fn push_mono(&self, samples: &[f32]) {
        for &sample in samples {
            self.push(sample);
        }
        self.wake();
    }

    fn available(&self) -> usize {
        self.write
            .load(Ordering::Acquire)
            .wrapping_sub(self.read.load(Ordering::Relaxed))
    }

    fn drain(&self, max_samples: usize) -> Vec<f32> {
        let read = self.read.load(Ordering::Relaxed);
        let count = max_samples.min(self.available());
        let mut output = Vec::with_capacity(count);
        for offset in 0..count {
            output.push(unsafe { *self.data[(read + offset) % self.capacity].get() });
        }
        self.read.store(read.wrapping_add(count), Ordering::Release);
        output
    }

    fn wake(&self) {
        self.ready.notify_one();
    }

    fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
        self.ready.notify_all();
    }

    fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    fn captured_frames(&self) -> u64 {
        self.captured.load(Ordering::Relaxed)
    }

    fn dropped_frames(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    fn wait_timeout_while(&self, timeout: Duration, should_wait: impl FnOnce() -> bool) {
        let guard = self
            .wait_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if should_wait() {
            let _ = self.ready.wait_timeout(guard, timeout);
        }
    }

    fn wait_timeout(&self, timeout: Duration) {
        self.wait_timeout_while(timeout, || true);
    }
}

/// Unique producer half of an SPSC audio capture ring (`Send`, `!Sync`, `!Clone`).
pub struct AudioRingProducer {
    inner: Arc<AudioRingInner>,
    _not_sync: PhantomData<Cell<()>>,
}

impl AudioRingProducer {
    #[inline]
    pub fn push(&mut self, sample: f32) {
        self.inner.push(sample);
    }

    pub fn push_mono(&mut self, samples: &[f32]) {
        self.inner.push_mono(samples);
    }

    pub fn wake(&self) {
        self.inner.wake();
    }
}

/// Unique consumer half of an SPSC audio capture ring (`Send`, `!Sync`, `!Clone`).
pub struct AudioRingConsumer {
    inner: Arc<AudioRingInner>,
    _not_sync: PhantomData<Cell<()>>,
}

impl AudioRingConsumer {
    pub fn available(&self) -> usize {
        self.inner.available()
    }

    pub fn drain(&mut self, max_samples: usize) -> Vec<f32> {
        self.inner.drain(max_samples)
    }

    pub fn is_stopped(&self) -> bool {
        self.inner.is_stopped()
    }

    /// Waits up to `timeout` for a wake or stop notification while `should_wait`
    /// remains true under the internal wait lock.
    pub fn wait_timeout_while(&self, timeout: Duration, should_wait: impl FnOnce() -> bool) {
        self.inner.wait_timeout_while(timeout, should_wait);
    }

    /// Waits up to `timeout` for a wake or stop notification.
    pub fn wait_timeout(&self, timeout: Duration) {
        self.inner.wait_timeout(timeout);
    }

    pub fn stop(&self) {
        self.inner.stop();
    }

    pub fn wake(&self) {
        self.inner.wake();
    }

    pub fn captured_frames(&self) -> u64 {
        self.inner.captured_frames()
    }

    pub fn dropped_frames(&self) -> u64 {
        self.inner.dropped_frames()
    }

    pub fn control(&self) -> AudioRingControl {
        AudioRingControl {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Shared, thread-safe control and metrics handle for an SPSC audio ring
/// (`Clone`, `Send`, `Sync`). Exposes only atomic/condvar operations.
#[derive(Clone)]
pub struct AudioRingControl {
    inner: Arc<AudioRingInner>,
}

impl AudioRingControl {
    pub fn stop(&self) {
        self.inner.stop();
    }

    pub fn wake(&self) {
        self.inner.wake();
    }

    pub fn is_stopped(&self) -> bool {
        self.inner.is_stopped()
    }

    pub fn captured_frames(&self) -> u64 {
        self.inner.captured_frames()
    }

    pub fn dropped_frames(&self) -> u64 {
        self.inner.dropped_frames()
    }
}

/// Create a fixed-capacity single-producer/single-consumer audio capture ring.
pub fn audio_ring(capacity: usize) -> (AudioRingProducer, AudioRingConsumer) {
    let inner = Arc::new(AudioRingInner::new(capacity));
    (
        AudioRingProducer {
            inner: Arc::clone(&inner),
            _not_sync: PhantomData,
        },
        AudioRingConsumer {
            inner,
            _not_sync: PhantomData,
        },
    )
}

/// Abstract start/stop handle for a native capture stream.
pub trait CaptureStream: Send {
    fn start(&mut self) -> Result<(), AudioError>;
    fn stop(&mut self) -> Result<(), AudioError>;
}

/// Native capture stream paired with its SPSC ring buffer and stream metadata,
/// ready for a consumer worker to attach and start.
pub struct PreparedCapture {
    pub stream: Box<dyn CaptureStream>,
    pub ring: AudioRingConsumer,
    pub source: CaptureSource,
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u32,
}

/// Validate and normalize capture ring duration (seconds) and consumer feed
/// cadence (milliseconds).
pub fn capture_timing(options: &CaptureOptions) -> Result<(f64, u32), AudioError> {
    let ring_seconds = options.ring_seconds.unwrap_or(10.0);
    if !ring_seconds.is_finite() || !(1.0..=120.0).contains(&ring_seconds) {
        return Err(AudioError::InvalidCaptureConfig(
            "ring_seconds must be between 1 and 120".into(),
        ));
    }
    Ok((
        ring_seconds,
        options.feed_milliseconds.unwrap_or(100).clamp(10, 1_000),
    ))
}

#[cfg(target_os = "macos")]
mod platform {
    use super::core_audio::{self, NativeCapture};
    use super::*;

    impl CaptureStream for NativeCapture {
        fn start(&mut self) -> Result<(), AudioError> {
            NativeCapture::start(self)
        }

        fn stop(&mut self) -> Result<(), AudioError> {
            NativeCapture::stop(self)
        }
    }

    pub(super) fn audio_devices() -> Result<Vec<AudioDevice>, AudioError> {
        let mut devices = Vec::new();
        for source in [CaptureSource::Microphone, CaptureSource::SystemAudio] {
            let default = core_audio::default_device_id(source);
            devices.extend(
                core_audio::audio_devices(source)?
                    .into_iter()
                    .map(|device| AudioDevice {
                        is_default: default == Some(device.object_id),
                        id: device.id,
                        name: device.name,
                        source,
                        sample_rate: device.sample_rate,
                        channels: device.channels,
                    }),
            );
        }
        Ok(devices)
    }

    pub(super) fn input_devices() -> Result<Vec<InputDevice>, AudioError> {
        let source = CaptureSource::Microphone;
        let default = core_audio::default_device_id(source);
        Ok(core_audio::audio_devices(source)?
            .into_iter()
            .map(|device| InputDevice {
                id: device.id,
                name: device.name,
                is_default: default == Some(device.object_id),
                sample_rate: device.sample_rate,
                channels: device.channels,
                sample_format: "f32".into(),
            })
            .collect())
    }

    pub(super) fn prepare_capture(options: &CaptureOptions) -> Result<PreparedCapture, AudioError> {
        let (ring_seconds, _) = capture_timing(options)?;
        let (_, selected_device) = core_audio::selected_device(options)?;
        let (mut producer, ring) =
            audio_ring((ring_seconds * selected_device.sample_rate as f64).ceil() as usize);
        let (source, stream) = core_audio::build(options, move |samples| {
            producer.push_mono(samples);
        })?;
        let sample_rate = stream.device.sample_rate;
        let channels = stream.device.channels;
        let device_name = stream.device.name.clone();
        if sample_rate == 0 || channels == 0 {
            return Err(AudioError::InvalidCaptureConfig(
                "Core Audio returned an invalid capture configuration".into(),
            ));
        }
        Ok(PreparedCapture {
            stream: Box::new(stream),
            ring,
            source,
            device_name,
            sample_rate,
            channels,
        })
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::*;

    pub(super) fn input_devices() -> Result<Vec<InputDevice>, AudioError> {
        #[cfg(windows)]
        {
            super::wasapi::input_devices()
        }
        #[cfg(not(windows))]
        {
            Err(AudioError::Unsupported(
                "Qwen3-ASR audio device discovery is currently built only for macOS and Windows",
            ))
        }
    }

    pub(super) fn audio_devices() -> Result<Vec<AudioDevice>, AudioError> {
        #[cfg(windows)]
        {
            super::wasapi::audio_devices()
        }
        #[cfg(not(windows))]
        {
            Err(AudioError::Unsupported(
                "Qwen3-ASR audio device discovery is currently built only for macOS and Windows",
            ))
        }
    }

    pub(super) fn prepare_capture(
        _options: &CaptureOptions,
    ) -> Result<PreparedCapture, AudioError> {
        Err(AudioError::Unsupported(
            "Qwen3-ASR Core Audio capture is currently built only for macOS",
        ))
    }
}

/// List available microphone input devices.
pub fn input_devices() -> Result<Vec<InputDevice>, AudioError> {
    platform::input_devices()
}

/// List available microphone and system-output audio devices.
pub fn audio_devices() -> Result<Vec<AudioDevice>, AudioError> {
    platform::audio_devices()
}

/// Prepare a native capture stream and ring buffer according to `options`.
pub fn prepare_capture(options: &CaptureOptions) -> Result<PreparedCapture, AudioError> {
    platform::prepare_capture(options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_ring_is_bounded_and_preserves_order() {
        let (mut producer, mut consumer) = audio_ring(3);
        producer.push_mono(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(consumer.captured_frames(), 3);
        assert_eq!(consumer.dropped_frames(), 1);
        assert_eq!(consumer.drain(2), vec![1.0, 2.0]);
        producer.push_mono(&[5.0]);
        assert_eq!(consumer.drain(3), vec![3.0, 5.0]);
    }

    #[test]
    fn audio_ring_accounts_for_all_frames_with_concurrent_producer_and_consumer() {
        const FRAME_COUNT: usize = 2_000;
        let (mut producer_ring, mut consumer_ring) = audio_ring(64);
        let control = consumer_ring.control();
        let producer_done = Arc::new(AtomicBool::new(false));
        let producer = {
            let producer_done = producer_done.clone();
            std::thread::spawn(move || {
                for frame in 0..FRAME_COUNT {
                    producer_ring.push(frame as f32);
                }
                producer_done.store(true, Ordering::Release);
                producer_ring.wake();
            })
        };
        let consumer = {
            let producer_done = producer_done.clone();
            std::thread::spawn(move || {
                let mut frames = Vec::new();
                while !producer_done.load(Ordering::Acquire) || consumer_ring.available() > 0 {
                    let drained = consumer_ring.drain(7);
                    if drained.is_empty() {
                        std::thread::yield_now();
                    } else {
                        frames.extend(drained);
                    }
                }
                frames
            })
        };
        producer.join().unwrap();
        let frames = consumer.join().unwrap();
        let captured = control.captured_frames();
        let dropped = control.dropped_frames();
        assert_eq!(captured as usize, frames.len());
        assert_eq!(captured + dropped, FRAME_COUNT as u64);
        assert!(!frames.is_empty());
        assert!(frames.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
