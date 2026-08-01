use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio::sync::mpsc;

use super::config::{Qwen3AsrCaptureOptions, Qwen3AsrResult};
use super::model::Qwen3AsrCmd;

#[napi(object)]
pub struct Qwen3AsrInputDevice {
    pub id: String,
    pub name: String,
    pub is_default: bool,
    pub sample_rate: u32,
    pub channels: u32,
    pub sample_format: String,
}

#[napi(object)]
pub struct Qwen3AsrCaptureStats {
    pub captured_frames: i64,
    pub dropped_frames: i64,
}

#[cfg(target_os = "macos")]
mod platform {
    use std::cell::UnsafeCell;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread::JoinHandle;
    use std::time::Duration;

    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::{FromSample, Sample, SampleFormat};

    use super::*;

    /// Fixed-capacity single-producer/single-consumer ring. The Core Audio
    /// callback performs no allocation, lock acquisition, inference, or
    /// resampling. Acquire/release ordering makes each slot visible before the
    /// producer publishes its write cursor.
    struct AudioRing {
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

    // SPSC discipline: only the CPAL callback writes slots and only the feeder
    // worker reads them; cursors publish ownership transitions.
    unsafe impl Sync for AudioRing {}

    impl AudioRing {
        fn new(capacity: usize) -> Self {
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

        fn available(&self) -> usize {
            self.write
                .load(Ordering::Acquire)
                .wrapping_sub(self.read.load(Ordering::Relaxed))
        }

        fn drain(&self, count: usize) -> Vec<f32> {
            let read = self.read.load(Ordering::Relaxed);
            let count = count.min(self.available());
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
    }

    fn push_interleaved<T>(ring: &AudioRing, data: &[T], channels: usize)
    where
        T: Sample,
        f32: FromSample<T>,
    {
        for frame in data.chunks_exact(channels) {
            let mut mono = 0.0f32;
            for &sample in frame {
                mono += f32::from_sample(sample);
            }
            ring.push(mono / channels as f32);
        }
        ring.wake();
    }

    fn select_device(host: &cpal::Host, options: &Qwen3AsrCaptureOptions) -> Result<cpal::Device> {
        if let Some(id) = options.device_id.as_deref() {
            let id = id
                .parse()
                .map_err(|error| Error::from_reason(format!("Invalid CPAL device id: {error}")))?;
            return host.device_by_id(&id).ok_or_else(|| {
                Error::from_reason(format!("CPAL input device is unavailable: {id}"))
            });
        }
        if let Some(name) = options.device_name.as_deref() {
            return host
                .input_devices()
                .map_err(|error| {
                    Error::from_reason(format!("Failed to list input devices: {error}"))
                })?
                .find(|device| {
                    device
                        .description()
                        .is_ok_and(|description| description.name() == name)
                })
                .ok_or_else(|| Error::from_reason(format!("CPAL input device not found: {name}")));
        }
        host.default_input_device()
            .ok_or_else(|| Error::from_reason("No default CPAL input device is available"))
    }

    #[napi]
    pub struct Qwen3AsrCapture {
        stream: Option<cpal::Stream>,
        ring: Arc<AudioRing>,
        worker: Option<JoinHandle<()>>,
        device_name: String,
        sample_rate: u32,
        channels: u32,
    }

    #[napi]
    impl Qwen3AsrCapture {
        #[napi(getter)]
        pub fn device_name(&self) -> String {
            self.device_name.clone()
        }

        #[napi(getter)]
        pub fn sample_rate(&self) -> u32 {
            self.sample_rate
        }

        #[napi(getter)]
        pub fn channels(&self) -> u32 {
            self.channels
        }

        #[napi]
        pub fn pause(&self) -> Result<()> {
            self.stream
                .as_ref()
                .ok_or_else(|| Error::from_reason("Capture is stopped"))?
                .pause()
                .map_err(|error| Error::from_reason(format!("Failed to pause capture: {error}")))
        }

        #[napi]
        pub fn resume(&self) -> Result<()> {
            self.stream
                .as_ref()
                .ok_or_else(|| Error::from_reason("Capture is stopped"))?
                .play()
                .map_err(|error| Error::from_reason(format!("Failed to resume capture: {error}")))
        }

        #[napi]
        pub fn stop<'env>(
            &mut self,
            env: &'env Env,
        ) -> Result<PromiseRaw<'env, Qwen3AsrCaptureStats>> {
            if let Some(stream) = self.stream.take() {
                let _ = stream.pause();
                drop(stream);
            }
            self.ring.stop();
            let worker = self.worker.take();
            let ring = self.ring.clone();
            env.spawn_future(async move {
                if let Some(worker) = worker {
                    napi::bindgen_prelude::spawn_blocking(move || worker.join())
                        .await
                        .map_err(|error| {
                            Error::from_reason(format!("Capture join failed: {error}"))
                        })?
                        .map_err(|_| Error::from_reason("Capture worker panicked"))?;
                }
                Ok(Qwen3AsrCaptureStats {
                    captured_frames: ring.captured.load(Ordering::Relaxed) as i64,
                    dropped_frames: ring.dropped.load(Ordering::Relaxed) as i64,
                })
            })
        }
    }

    pub(super) fn input_devices() -> Result<Vec<Qwen3AsrInputDevice>> {
        let host = cpal::default_host();
        let default_id = host
            .default_input_device()
            .and_then(|device| device.id().ok())
            .map(|id| id.to_string());
        host.input_devices()
            .map_err(|error| {
                Error::from_reason(format!("Failed to list CPAL input devices: {error}"))
            })?
            .map(|device| {
                let id = device.id().map_err(|error| {
                    Error::from_reason(format!("Failed to identify CPAL device: {error}"))
                })?;
                let description = device.description().map_err(|error| {
                    Error::from_reason(format!("Failed to describe CPAL device: {error}"))
                })?;
                let config = device.default_input_config().map_err(|error| {
                    Error::from_reason(format!(
                        "Failed to query default input config for {}: {error}",
                        description.name()
                    ))
                })?;
                let id_string = id.to_string();
                Ok(Qwen3AsrInputDevice {
                    is_default: default_id.as_deref() == Some(id_string.as_str()),
                    id: id_string,
                    name: description.name().to_string(),
                    sample_rate: config.sample_rate(),
                    channels: config.channels() as u32,
                    sample_format: config.sample_format().to_string(),
                })
            })
            .collect()
    }

    impl Drop for Qwen3AsrCapture {
        fn drop(&mut self) {
            self.ring.stop();
            self.stream.take();
            // Joining can block behind a rolling decode. Detach on GC; callers
            // that need deterministic teardown use `await capture.stop()`.
            self.worker.take();
        }
    }

    pub(super) fn start_capture(
        sender: mpsc::UnboundedSender<Qwen3AsrCmd>,
        stream_id: String,
        options: Qwen3AsrCaptureOptions,
        callback: ThreadsafeFunction<Qwen3AsrResult, ()>,
    ) -> Result<Qwen3AsrCapture> {
        let host = cpal::default_host();
        let device = select_device(&host, &options)?;
        let description = device.description().map_err(|error| {
            Error::from_reason(format!("Failed to describe CPAL input device: {error}"))
        })?;
        let supported = device.default_input_config().map_err(|error| {
            Error::from_reason(format!("Failed to get CPAL input configuration: {error}"))
        })?;
        let sample_rate = supported.sample_rate();
        let channels = supported.channels() as usize;
        if channels == 0 || sample_rate == 0 {
            return Err(Error::from_reason(
                "CPAL returned an invalid input configuration",
            ));
        }
        let (prepare_reply, prepare_rx) = tokio::sync::oneshot::channel();
        sender
            .send(Qwen3AsrCmd::PrepareCapture {
                id: stream_id.clone(),
                sample_rate,
                reply: prepare_reply,
            })
            .map_err(|_| Error::from_reason("Qwen3-ASR model thread has exited"))?;
        prepare_rx.blocking_recv().map_err(|_| {
            Error::from_reason("Qwen3-ASR model thread exited during capture setup")
        })??;
        let ring_seconds = options.ring_seconds.unwrap_or(10.0);
        if !ring_seconds.is_finite() || ring_seconds < 1.0 || ring_seconds > 120.0 {
            return Err(Error::from_reason("ring_seconds must be between 1 and 120"));
        }
        let feed_ms = options.feed_milliseconds.unwrap_or(100).clamp(10, 1_000);
        let ring = Arc::new(AudioRing::new(
            (ring_seconds * sample_rate as f64).ceil() as usize
        ));
        let callback = Arc::new(callback);
        let error_callback = callback.clone();
        let error_fn = move |error: cpal::Error| {
            error_callback.call(
                Err(Error::from_reason(format!("CPAL input error: {error}"))),
                ThreadsafeFunctionCallMode::NonBlocking,
            );
        };
        let callback_ring = ring.clone();
        let sample_format = supported.sample_format();
        let stream = device
            .build_input_stream_raw(
                supported.config(),
                sample_format,
                move |data, _| {
                    macro_rules! push_as {
                        ($ty:ty) => {
                            if let Some(samples) = data.as_slice::<$ty>() {
                                push_interleaved(&callback_ring, samples, channels);
                            }
                        };
                    }
                    match sample_format {
                        SampleFormat::I8 => push_as!(i8),
                        SampleFormat::I16 => push_as!(i16),
                        SampleFormat::I24 => push_as!(cpal::I24),
                        SampleFormat::I32 => push_as!(i32),
                        SampleFormat::I64 => push_as!(i64),
                        SampleFormat::U8 => push_as!(u8),
                        SampleFormat::U16 => push_as!(u16),
                        SampleFormat::U24 => push_as!(cpal::U24),
                        SampleFormat::U32 => push_as!(u32),
                        SampleFormat::U64 => push_as!(u64),
                        SampleFormat::F32 => push_as!(f32),
                        SampleFormat::F64 => push_as!(f64),
                        _ => {}
                    }
                },
                error_fn,
                None,
            )
            .map_err(|error| {
                Error::from_reason(format!("Failed to build CPAL input stream: {error}"))
            })?;

        let worker_ring = ring.clone();
        let worker_callback = callback.clone();
        let feed_frames = ((sample_rate as u64 * feed_ms as u64) / 1_000).max(1) as usize;
        let worker = std::thread::Builder::new()
            .name("mlx-asr-capture".into())
            .spawn(move || {
                loop {
                    let available = worker_ring.available();
                    if available < feed_frames && !worker_ring.stopped.load(Ordering::Acquire) {
                        let guard = worker_ring
                            .wait_lock
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        let _ = worker_ring
                            .ready
                            .wait_timeout(guard, Duration::from_millis(20));
                        continue;
                    }
                    if available == 0 && worker_ring.stopped.load(Ordering::Acquire) {
                        break;
                    }
                    let samples =
                        worker_ring.drain(if worker_ring.stopped.load(Ordering::Acquire) {
                            available
                        } else {
                            feed_frames
                        });
                    if samples.is_empty() {
                        continue;
                    }
                    let (reply, rx) = tokio::sync::oneshot::channel();
                    if sender
                        .send(Qwen3AsrCmd::FeedStream {
                            id: stream_id.clone(),
                            samples,
                            reply,
                        })
                        .is_err()
                    {
                        worker_callback.call(
                            Err(Error::from_reason("Qwen3-ASR model thread exited")),
                            ThreadsafeFunctionCallMode::NonBlocking,
                        );
                        break;
                    }
                    match rx.blocking_recv() {
                        Ok(Ok(Some(result))) => {
                            worker_callback
                                .call(Ok(result), ThreadsafeFunctionCallMode::NonBlocking);
                        }
                        Ok(Ok(None)) => {}
                        Ok(Err(error)) => {
                            worker_callback
                                .call(Err(error), ThreadsafeFunctionCallMode::NonBlocking);
                        }
                        Err(_) => {
                            worker_callback.call(
                                Err(Error::from_reason("Qwen3-ASR model thread exited")),
                                ThreadsafeFunctionCallMode::NonBlocking,
                            );
                            break;
                        }
                    }
                }
            })
            .map_err(|error| {
                Error::from_reason(format!("Failed to start capture worker: {error}"))
            })?;

        if let Err(error) = stream.play() {
            ring.stop();
            drop(stream);
            let _ = worker.join();
            return Err(Error::from_reason(format!(
                "Failed to start CPAL input stream: {error}"
            )));
        }
        Ok(Qwen3AsrCapture {
            stream: Some(stream),
            ring,
            worker: Some(worker),
            device_name: description.name().to_string(),
            sample_rate,
            channels: channels as u32,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn audio_ring_is_bounded_and_preserves_order() {
            let ring = AudioRing::new(3);
            ring.push(1.0);
            ring.push(2.0);
            ring.push(3.0);
            ring.push(4.0);
            assert_eq!(ring.captured.load(Ordering::Relaxed), 3);
            assert_eq!(ring.dropped.load(Ordering::Relaxed), 1);
            assert_eq!(ring.drain(2), vec![1.0, 2.0]);
            ring.push(5.0);
            assert_eq!(ring.drain(3), vec![3.0, 5.0]);
        }

        #[test]
        fn interleaved_capture_downmixes_without_changing_frame_count() {
            let ring = AudioRing::new(4);
            push_interleaved(&ring, &[1.0f32, -1.0, 0.5, 0.25], 2);
            assert_eq!(ring.drain(4), vec![0.0, 0.375]);
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::*;

    #[napi]
    pub struct Qwen3AsrCapture;

    #[napi]
    impl Qwen3AsrCapture {}

    pub(super) fn input_devices() -> Result<Vec<Qwen3AsrInputDevice>> {
        Err(Error::from_reason(
            "Qwen3-ASR CPAL capture is currently built only for macOS",
        ))
    }

    pub(super) fn start_capture(
        _sender: mpsc::UnboundedSender<Qwen3AsrCmd>,
        _stream_id: String,
        _options: Qwen3AsrCaptureOptions,
        _callback: ThreadsafeFunction<Qwen3AsrResult, ()>,
    ) -> Result<Qwen3AsrCapture> {
        Err(Error::from_reason(
            "Qwen3-ASR CPAL capture is currently built only for macOS",
        ))
    }
}

pub use platform::Qwen3AsrCapture;

pub(super) fn start_capture(
    sender: mpsc::UnboundedSender<Qwen3AsrCmd>,
    stream_id: String,
    options: Qwen3AsrCaptureOptions,
    callback: ThreadsafeFunction<Qwen3AsrResult, ()>,
) -> Result<Qwen3AsrCapture> {
    platform::start_capture(sender, stream_id, options, callback)
}

#[napi]
pub fn qwen3_asr_input_devices() -> Result<Vec<Qwen3AsrInputDevice>> {
    platform::input_devices()
}
