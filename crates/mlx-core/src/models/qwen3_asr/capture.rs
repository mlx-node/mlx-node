use std::cell::UnsafeCell;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

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

/// Fixed-capacity single-producer/single-consumer ring. The native audio
/// callback performs no allocation, lock acquisition, inference, or resampling.
/// Acquire/release ordering makes each slot visible before the producer
/// publishes its write cursor.
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

// SPSC discipline: only the native audio callback writes slots and only the
// feeder worker reads them; cursors publish ownership transitions.
unsafe impl Sync for AudioRing {}

impl AudioRing {
    #[cfg(any(target_os = "macos", test))]
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

    #[cfg(any(target_os = "macos", test))]
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

    #[cfg(any(target_os = "macos", test))]
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

trait CaptureStream: Send {
    fn start(&mut self) -> Result<()>;
    fn stop(&mut self) -> Result<()>;
}

struct PreparedCapture {
    stream: Box<dyn CaptureStream>,
    ring: Arc<AudioRing>,
    source: Qwen3AsrCaptureSource,
    device_name: String,
    sample_rate: u32,
    channels: u32,
}

#[napi]
pub struct Qwen3AsrCapture {
    stream: Option<Box<dyn CaptureStream>>,
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
    pub fn stop<'env>(&mut self, env: &'env Env) -> Result<PromiseRaw<'env, Qwen3AsrCaptureStats>> {
        env.spawn_future(stop_capture(self))
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

fn capture_timing(options: &Qwen3AsrCaptureOptions) -> Result<(f64, u32)> {
    let ring_seconds = options.ring_seconds.unwrap_or(10.0);
    if !ring_seconds.is_finite() || !(1.0..=120.0).contains(&ring_seconds) {
        return Err(Error::from_reason("ring_seconds must be between 1 and 120"));
    }
    Ok((
        ring_seconds,
        options.feed_milliseconds.unwrap_or(100).clamp(10, 1_000),
    ))
}

fn start_prepared_capture(
    sender: mpsc::UnboundedSender<Qwen3AsrCmd>,
    stream_id: String,
    options: Qwen3AsrCaptureOptions,
    callback: ThreadsafeFunction<Qwen3AsrResult, ()>,
    prepared: Result<PreparedCapture>,
) -> Result<Qwen3AsrCapture> {
    let callback = Arc::new(callback);
    start_prepared_capture_with_callback(sender, stream_id, options, prepared, move |result| {
        callback.call(result, ThreadsafeFunctionCallMode::NonBlocking);
    })
}

fn start_prepared_capture_with_callback(
    sender: mpsc::UnboundedSender<Qwen3AsrCmd>,
    stream_id: String,
    options: Qwen3AsrCaptureOptions,
    prepared: Result<PreparedCapture>,
    callback: impl Fn(Result<Qwen3AsrResult>) + Send + Sync + 'static,
) -> Result<Qwen3AsrCapture> {
    let PreparedCapture {
        stream,
        ring,
        source,
        device_name,
        sample_rate,
        channels,
    } = prepared?;
    let mut stream = stream;
    let (_, feed_ms) = capture_timing(&options)?;
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
                let samples = worker_ring.drain(if worker_ring.stopped.load(Ordering::Acquire) {
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
                    worker_callback(Err(Error::from_reason("Qwen3-ASR model thread exited")));
                    break;
                }
                match rx.blocking_recv() {
                    Ok(Ok(Some(result))) => {
                        worker_callback(Ok(result));
                    }
                    Ok(Ok(None)) => {}
                    Ok(Err(error)) => {
                        worker_callback(Err(error));
                    }
                    Err(_) => {
                        worker_callback(Err(Error::from_reason("Qwen3-ASR model thread exited")));
                        break;
                    }
                }
            }
            let _ = worker_sender.send(Qwen3AsrCmd::ReleaseCapture {
                id: worker_stream_id,
            });
        })
        .map_err(|error| Error::from_reason(format!("Failed to start capture worker: {error}")))?;

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

fn stop_capture(
    capture: &mut Qwen3AsrCapture,
) -> impl Future<Output = Result<Qwen3AsrCaptureStats>> + Send + 'static {
    if let Some(mut stream) = capture.stream.take() {
        let _ = stream.stop();
        drop(stream);
    }
    capture.ring.stop();
    let worker = capture.worker.take();
    let ring = capture.ring.clone();
    async move {
        if let Some(worker) = worker {
            napi::bindgen_prelude::spawn_blocking(move || worker.join())
                .await
                .map_err(|error| Error::from_reason(format!("Capture join failed: {error}")))?
                .map_err(|_| Error::from_reason("Capture worker panicked"))?;
        }
        Ok(Qwen3AsrCaptureStats {
            captured_frames: ring.captured.load(Ordering::Relaxed) as i64,
            dropped_frames: ring.dropped.load(Ordering::Relaxed) as i64,
        })
    }
}

fn source_name(source: Qwen3AsrCaptureSource) -> &'static str {
    match source {
        Qwen3AsrCaptureSource::Microphone => "microphone",
        Qwen3AsrCaptureSource::SystemAudio => "system-audio",
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use crate::models::qwen3_asr::core_audio_capture::{self, NativeCapture};

    impl CaptureStream for NativeCapture {
        fn start(&mut self) -> Result<()> {
            NativeCapture::start(self)
        }

        fn stop(&mut self) -> Result<()> {
            NativeCapture::stop(self)
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
        let (ring_seconds, _) = capture_timing(&options)?;
        let (_, selected_device) = core_audio_capture::selected_device(&options)?;
        let ring = Arc::new(AudioRing::new(
            (ring_seconds * selected_device.sample_rate as f64).ceil() as usize,
        ));
        let callback_ring = ring.clone();
        let (source, stream) = core_audio_capture::build(&options, move |samples| {
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
        start_prepared_capture(
            sender,
            stream_id,
            options,
            callback,
            Ok(PreparedCapture {
                stream: Box::new(stream),
                ring,
                source,
                device_name,
                sample_rate,
                channels,
            }),
        )
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::*;

    pub(super) fn input_devices() -> Result<Vec<Qwen3AsrInputDevice>> {
        #[cfg(windows)]
        {
            super::super::wasapi_devices::input_devices()
        }
        #[cfg(not(windows))]
        {
            Err(Error::from_reason(
                "Qwen3-ASR audio device discovery is currently built only for macOS and Windows",
            ))
        }
    }

    pub(super) fn audio_devices() -> Result<Vec<Qwen3AsrAudioDevice>> {
        #[cfg(windows)]
        {
            super::super::wasapi_devices::audio_devices()
        }
        #[cfg(not(windows))]
        {
            Err(Error::from_reason(
                "Qwen3-ASR audio device discovery is currently built only for macOS and Windows",
            ))
        }
    }

    pub(super) fn start_capture(
        sender: mpsc::UnboundedSender<Qwen3AsrCmd>,
        stream_id: String,
        options: Qwen3AsrCaptureOptions,
        callback: ThreadsafeFunction<Qwen3AsrResult, ()>,
    ) -> Result<Qwen3AsrCapture> {
        start_prepared_capture(
            sender,
            stream_id,
            options,
            callback,
            Err(Error::from_reason(
                "Qwen3-ASR Core Audio capture is currently built only for macOS",
            )),
        )
    }
}

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

#[cfg(test)]
mod tests {
    use std::sync::mpsc as std_mpsc;
    use std::time::Instant;

    use super::*;

    #[derive(Default)]
    struct TestStreamState {
        started: AtomicBool,
        stopped: AtomicBool,
        dropped: AtomicBool,
    }

    struct TestStream {
        state: Arc<TestStreamState>,
        fail_start: bool,
    }

    impl CaptureStream for TestStream {
        fn start(&mut self) -> Result<()> {
            self.state.started.store(true, Ordering::Release);
            if self.fail_start {
                Err(Error::from_reason("test stream start failed"))
            } else {
                Ok(())
            }
        }

        fn stop(&mut self) -> Result<()> {
            self.state.stopped.store(true, Ordering::Release);
            Ok(())
        }
    }

    impl Drop for TestStream {
        fn drop(&mut self) {
            self.state.dropped.store(true, Ordering::Release);
        }
    }

    fn test_options() -> Qwen3AsrCaptureOptions {
        Qwen3AsrCaptureOptions {
            ring_seconds: Some(1.0),
            feed_milliseconds: Some(100),
            ..Default::default()
        }
    }

    fn test_prepared(
        ring: Arc<AudioRing>,
        state: Arc<TestStreamState>,
        fail_start: bool,
    ) -> PreparedCapture {
        PreparedCapture {
            stream: Box::new(TestStream { state, fail_start }),
            ring,
            source: Qwen3AsrCaptureSource::Microphone,
            device_name: "test input".into(),
            sample_rate: 10,
            channels: 1,
        }
    }

    fn test_result(text: &str) -> Qwen3AsrResult {
        Qwen3AsrResult {
            text: text.into(),
            stable_text: text.into(),
            provisional_text: String::new(),
            language: None,
            token_ids: Vec::new(),
            reached_max_tokens: false,
            audio_seconds: 0.1,
            segment_audio_seconds: 0.1,
            feature_ms: 0.0,
            prefill_ms: 0.0,
            decode_ms: 0.0,
            total_ms: 0.0,
            tokens_per_second: 0.0,
            real_time_factor: 0.0,
            revision: 0,
            is_final: false,
        }
    }

    fn callback_events() -> (
        impl Fn(Result<Qwen3AsrResult>) + Send + Sync + 'static,
        std_mpsc::Receiver<String>,
    ) {
        let (sender, receiver) = std_mpsc::channel();
        (
            move |result| {
                let event = match result {
                    Ok(result) => format!("ok:{}", result.text),
                    Err(error) => format!("error:{error}"),
                };
                let _ = sender.send(event);
            },
            receiver,
        )
    }

    fn recv_command(commands: &mut mpsc::UnboundedReceiver<Qwen3AsrCmd>) -> Qwen3AsrCmd {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match commands.try_recv() {
                Ok(command) => return command,
                Err(mpsc::error::TryRecvError::Empty) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    panic!("timed out waiting for a capture command");
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    panic!("capture command sender disconnected");
                }
            }
        }
    }

    fn assert_no_command(commands: &mut mpsc::UnboundedReceiver<Qwen3AsrCmd>) {
        if let Ok(Qwen3AsrCmd::ReleaseCapture { .. }) = commands.try_recv() {
            // The worker releases its capture on exit; that notification is expected.
        }
        assert!(matches!(
            commands.try_recv(),
            Err(mpsc::error::TryRecvError::Empty) | Err(mpsc::error::TryRecvError::Disconnected)
        ));
    }

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

    #[test]
    fn audio_ring_accounts_for_all_frames_with_concurrent_producer_and_consumer() {
        const FRAME_COUNT: usize = 2_000;
        let ring = Arc::new(AudioRing::new(64));
        let producer_done = Arc::new(AtomicBool::new(false));
        let producer = {
            let ring = ring.clone();
            let producer_done = producer_done.clone();
            std::thread::spawn(move || {
                for frame in 0..FRAME_COUNT {
                    ring.push(frame as f32);
                }
                producer_done.store(true, Ordering::Release);
                ring.wake();
            })
        };
        let consumer = {
            let ring = ring.clone();
            let producer_done = producer_done.clone();
            std::thread::spawn(move || {
                let mut frames = Vec::new();
                while !producer_done.load(Ordering::Acquire) || ring.available() > 0 {
                    let drained = ring.drain(7);
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
        let captured = ring.captured.load(Ordering::Relaxed);
        let dropped = ring.dropped.load(Ordering::Relaxed);
        assert_eq!(captured as usize, frames.len());
        assert_eq!(captured + dropped, FRAME_COUNT as u64);
        assert!(!frames.is_empty());
        assert!(frames.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn native_start_failure_stops_worker_without_preparing_model() {
        let (sender, mut commands) = mpsc::unbounded_channel();
        let ring = Arc::new(AudioRing::new(10));
        let state = Arc::new(TestStreamState::default());
        let (callback, _events) = callback_events();
        let result = start_prepared_capture_with_callback(
            sender,
            "stream".into(),
            test_options(),
            Ok(test_prepared(ring, state.clone(), true)),
            callback,
        );
        let error = result.err().expect("native start should fail");
        assert!(error.to_string().contains("test stream start failed"));
        assert!(state.started.load(Ordering::Acquire));
        assert!(state.dropped.load(Ordering::Acquire));
        assert_no_command(&mut commands);
    }

    #[test]
    fn prepare_failure_propagates_and_tears_down_native_stream() {
        let (sender, mut commands) = mpsc::unbounded_channel();
        let ring = Arc::new(AudioRing::new(10));
        let state = Arc::new(TestStreamState::default());
        let (callback, _events) = callback_events();
        let startup = {
            let ring = ring.clone();
            let state = state.clone();
            std::thread::spawn(move || {
                start_prepared_capture_with_callback(
                    sender,
                    "stream".into(),
                    test_options(),
                    Ok(test_prepared(ring, state, false)),
                    callback,
                )
            })
        };
        let prepare_reply = match recv_command(&mut commands) {
            Qwen3AsrCmd::PrepareCapture { reply, .. } => reply,
            _ => panic!("expected PrepareCapture"),
        };
        ring.push_mono(&[1.0]);
        std::thread::sleep(Duration::from_millis(40));
        assert_no_command(&mut commands);
        prepare_reply
            .send(Err(Error::from_reason("prepare rejected")))
            .unwrap();
        let error = startup.join().unwrap().err().expect("prepare should fail");
        assert!(error.to_string().contains("prepare rejected"));
        assert!(state.started.load(Ordering::Acquire));
        assert!(state.dropped.load(Ordering::Acquire));
        assert_no_command(&mut commands);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn feeder_gates_on_prepare_serializes_replies_and_delivers_results() {
        let (sender, mut commands) = mpsc::unbounded_channel();
        let ring = Arc::new(AudioRing::new(10));
        let state = Arc::new(TestStreamState::default());
        let (callback, events) = callback_events();
        let startup = {
            let ring = ring.clone();
            let state = state.clone();
            std::thread::spawn(move || {
                start_prepared_capture_with_callback(
                    sender,
                    "stream".into(),
                    test_options(),
                    Ok(test_prepared(ring, state, false)),
                    callback,
                )
            })
        };
        let prepare_reply = match recv_command(&mut commands) {
            Qwen3AsrCmd::PrepareCapture {
                id,
                sample_rate,
                reply,
            } => {
                assert_eq!(id, "stream");
                assert_eq!(sample_rate, 10);
                reply
            }
            _ => panic!("expected PrepareCapture"),
        };
        ring.push_mono(&[1.0]);
        std::thread::sleep(Duration::from_millis(40));
        assert_no_command(&mut commands);
        prepare_reply.send(Ok(())).unwrap();
        let mut capture = startup.join().unwrap().unwrap();
        assert!(state.started.load(Ordering::Acquire));

        let first_reply = match recv_command(&mut commands) {
            Qwen3AsrCmd::FeedStream {
                id,
                samples,
                source: StreamFeedSource::Capture,
                reply,
            } => {
                assert_eq!(id, "stream");
                assert_eq!(samples, vec![1.0]);
                reply
            }
            _ => panic!("expected captured FeedStream"),
        };
        ring.push_mono(&[2.0]);
        std::thread::sleep(Duration::from_millis(40));
        assert_no_command(&mut commands);
        first_reply.send(Ok(Some(test_result("first")))).unwrap();
        assert_eq!(
            events.recv_timeout(Duration::from_secs(2)).unwrap(),
            "ok:first"
        );

        let second_reply = match recv_command(&mut commands) {
            Qwen3AsrCmd::FeedStream { samples, reply, .. } => {
                assert_eq!(samples, vec![2.0]);
                reply
            }
            _ => panic!("expected second FeedStream"),
        };
        second_reply.send(Ok(None)).unwrap();
        let stats = stop_capture(&mut capture).await.unwrap();
        assert_eq!(stats.captured_frames, 2);
        assert_eq!(stats.dropped_frames, 0);
        assert!(state.stopped.load(Ordering::Acquire));
        assert!(state.dropped.load(Ordering::Acquire));
        assert!(matches!(
            recv_command(&mut commands),
            Qwen3AsrCmd::ReleaseCapture { id } if id == "stream"
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn model_exit_is_delivered_and_stop_still_joins_worker() {
        let (sender, mut commands) = mpsc::unbounded_channel();
        let ring = Arc::new(AudioRing::new(10));
        ring.push_mono(&[1.0]);
        let state = Arc::new(TestStreamState::default());
        let (callback, events) = callback_events();
        let startup = {
            let ring = ring.clone();
            let state = state.clone();
            std::thread::spawn(move || {
                start_prepared_capture_with_callback(
                    sender,
                    "stream".into(),
                    test_options(),
                    Ok(test_prepared(ring, state, false)),
                    callback,
                )
            })
        };
        let prepare_reply = match recv_command(&mut commands) {
            Qwen3AsrCmd::PrepareCapture { reply, .. } => reply,
            _ => panic!("expected PrepareCapture"),
        };
        prepare_reply.send(Ok(())).unwrap();
        let mut capture = startup.join().unwrap().unwrap();
        drop(commands);
        let event = events.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(event.contains("Qwen3-ASR model thread exited"));
        let stats = stop_capture(&mut capture).await.unwrap();
        assert_eq!(stats.captured_frames, 1);
        assert_eq!(stats.dropped_frames, 0);
        assert!(state.stopped.load(Ordering::Acquire));
        assert!(state.dropped.load(Ordering::Acquire));
    }
}
