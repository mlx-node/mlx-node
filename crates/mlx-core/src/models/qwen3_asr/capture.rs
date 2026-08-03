use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio::sync::mpsc;

use super::config::{Qwen3AsrCaptureOptions, Qwen3AsrCaptureSource, Qwen3AsrResult};
use super::model::{Qwen3AsrCmd, StreamFeedSource};

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
pub struct Qwen3AsrAudioDevice {
    pub id: String,
    pub name: String,
    pub source: Qwen3AsrCaptureSource,
    pub is_default: bool,
    pub sample_rate: u32,
    pub channels: u32,
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

    use super::*;
    use crate::models::qwen3_asr::core_audio_capture::{self, NativeCapture};

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

    // SPSC discipline: only the Core Audio callback writes slots and only the
    // feeder worker reads them; cursors publish ownership transitions.
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

    #[napi]
    pub struct Qwen3AsrCapture {
        stream: Option<NativeCapture>,
        ring: Arc<AudioRing>,
        worker: Option<JoinHandle<()>>,
        source: Qwen3AsrCaptureSource,
        device_name: String,
        sample_rate: u32,
        channels: u32,
    }

    #[napi]
    impl Qwen3AsrCapture {
        #[napi(getter)]
        pub fn source(&self) -> Qwen3AsrCaptureSource {
            self.source
        }

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
        pub fn pause(&mut self) -> Result<()> {
            self.stream
                .as_mut()
                .ok_or_else(|| Error::from_reason("Capture is stopped"))?
                .stop()
        }

        #[napi]
        pub fn resume(&mut self) -> Result<()> {
            self.stream
                .as_mut()
                .ok_or_else(|| Error::from_reason("Capture is stopped"))?
                .start()
        }

        #[napi]
        pub fn stop<'env>(
            &mut self,
            env: &'env Env,
        ) -> Result<PromiseRaw<'env, Qwen3AsrCaptureStats>> {
            if let Some(mut stream) = self.stream.take() {
                let _ = stream.stop();
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

    impl Drop for Qwen3AsrCapture {
        fn drop(&mut self) {
            self.ring.stop();
            self.stream.take();
            // Joining can block behind a rolling decode. Detach on GC; callers
            // that need deterministic teardown use `await capture.stop()`.
            self.worker.take();
        }
    }

    pub(super) fn audio_devices() -> Result<Vec<Qwen3AsrAudioDevice>> {
        let mut devices = Vec::new();
        for source in [
            Qwen3AsrCaptureSource::Microphone,
            Qwen3AsrCaptureSource::SystemAudio,
        ] {
            let default = core_audio_capture::default_device_id(source);
            devices.extend(
                core_audio_capture::audio_devices(source)?
                    .into_iter()
                    .map(|device| Qwen3AsrAudioDevice {
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

    pub(super) fn input_devices() -> Result<Vec<Qwen3AsrInputDevice>> {
        let source = Qwen3AsrCaptureSource::Microphone;
        let default = core_audio_capture::default_device_id(source);
        Ok(core_audio_capture::audio_devices(source)?
            .into_iter()
            .map(|device| Qwen3AsrInputDevice {
                id: device.id,
                name: device.name,
                is_default: default == Some(device.object_id),
                sample_rate: device.sample_rate,
                channels: device.channels,
                sample_format: "f32".into(),
            })
            .collect())
    }

    pub(super) fn start_capture(
        sender: mpsc::UnboundedSender<Qwen3AsrCmd>,
        stream_id: String,
        options: Qwen3AsrCaptureOptions,
        callback: ThreadsafeFunction<Qwen3AsrResult, ()>,
    ) -> Result<Qwen3AsrCapture> {
        let ring_seconds = options.ring_seconds.unwrap_or(10.0);
        if !ring_seconds.is_finite() || !(1.0..=120.0).contains(&ring_seconds) {
            return Err(Error::from_reason("ring_seconds must be between 1 and 120"));
        }
        let feed_ms = options.feed_milliseconds.unwrap_or(100).clamp(10, 1_000);

        let (_, selected_device) = core_audio_capture::selected_device(&options)?;
        let ring = Arc::new(AudioRing::new(
            (ring_seconds * selected_device.sample_rate as f64).ceil() as usize,
        ));
        let callback_ring = ring.clone();
        let (source, mut stream) = core_audio_capture::build(&options, move |samples| {
            callback_ring.push_mono(samples);
        })?;
        let sample_rate = stream.device.sample_rate;
        let channels = stream.device.channels;
        let device_name = stream.device.name.clone();
        if sample_rate == 0 || channels == 0 {
            return Err(Error::from_reason(
                "Core Audio returned an invalid capture configuration",
            ));
        }
        let callback = Arc::new(callback);
        let worker_ring = ring.clone();
        let worker_callback = callback.clone();
        let worker_ready = Arc::new(AtomicBool::new(false));
        let capture_ready = worker_ready.clone();
        let worker_sender = sender.clone();
        let worker_stream_id = stream_id.clone();
        let feed_frames = ((sample_rate as u64 * feed_ms as u64) / 1_000).max(1) as usize;
        let worker = std::thread::Builder::new()
            .name(format!("mlx-asr-{}-capture", source_name(source)))
            .spawn(move || {
                // Native setup, worker creation, and model preparation are all
                // fallible. Keep the worker parked until every step succeeds.
                while !worker_ready.load(Ordering::Acquire) {
                    if worker_ring.stopped.load(Ordering::Acquire) {
                        return;
                    }
                    let guard = worker_ring
                        .wait_lock
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if worker_ready.load(Ordering::Acquire) {
                        break;
                    }
                    let _ = worker_ring
                        .ready
                        .wait_timeout(guard, Duration::from_millis(20));
                }
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
                    if worker_sender
                        .send(Qwen3AsrCmd::FeedStream {
                            id: worker_stream_id.clone(),
                            samples,
                            source: StreamFeedSource::Capture,
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
                let _ = worker_sender.send(Qwen3AsrCmd::ReleaseCapture {
                    id: worker_stream_id,
                });
            })
            .map_err(|error| {
                Error::from_reason(format!("Failed to start capture worker: {error}"))
            })?;

        if let Err(error) = stream.start() {
            ring.stop();
            drop(stream);
            let _ = worker.join();
            return Err(error);
        }

        let prepare_result = (|| {
            let (prepare_reply, prepare_rx) = tokio::sync::oneshot::channel();
            sender
                .send(Qwen3AsrCmd::PrepareCapture {
                    id: stream_id,
                    sample_rate,
                    reply: prepare_reply,
                })
                .map_err(|_| Error::from_reason("Qwen3-ASR model thread has exited"))?;
            prepare_rx.blocking_recv().map_err(|_| {
                Error::from_reason("Qwen3-ASR model thread exited during capture setup")
            })??;
            Ok(())
        })();
        if let Err(error) = prepare_result {
            ring.stop();
            drop(stream);
            let _ = worker.join();
            return Err(error);
        }
        capture_ready.store(true, Ordering::Release);
        ring.wake();

        Ok(Qwen3AsrCapture {
            stream: Some(stream),
            ring,
            worker: Some(worker),
            source,
            device_name,
            sample_rate,
            channels,
        })
    }

    fn source_name(source: Qwen3AsrCaptureSource) -> &'static str {
        match source {
            Qwen3AsrCaptureSource::Microphone => "microphone",
            Qwen3AsrCaptureSource::SystemAudio => "system-audio",
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn audio_ring_is_bounded_and_preserves_order() {
            let ring = AudioRing::new(3);
            ring.push_mono(&[1.0, 2.0, 3.0, 4.0]);
            assert_eq!(ring.captured.load(Ordering::Relaxed), 3);
            assert_eq!(ring.dropped.load(Ordering::Relaxed), 1);
            assert_eq!(ring.drain(2), vec![1.0, 2.0]);
            ring.push_mono(&[5.0]);
            assert_eq!(ring.drain(3), vec![3.0, 5.0]);
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
            "Qwen3-ASR Core Audio capture is currently built only for macOS",
        ))
    }

    pub(super) fn audio_devices() -> Result<Vec<Qwen3AsrAudioDevice>> {
        Err(Error::from_reason(
            "Qwen3-ASR Core Audio capture is currently built only for macOS",
        ))
    }

    pub(super) fn start_capture(
        _sender: mpsc::UnboundedSender<Qwen3AsrCmd>,
        _stream_id: String,
        _options: Qwen3AsrCaptureOptions,
        _callback: ThreadsafeFunction<Qwen3AsrResult, ()>,
    ) -> Result<Qwen3AsrCapture> {
        Err(Error::from_reason(
            "Qwen3-ASR Core Audio capture is currently built only for macOS",
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

#[napi]
pub fn qwen3_asr_audio_devices() -> Result<Vec<Qwen3AsrAudioDevice>> {
    platform::audio_devices()
}
