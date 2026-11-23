//! N-API capture adapter and Qwen3-ASR stream feeder worker.
//!
//! Native OS audio discovery, ring buffering, and stream capture live in `mlx-audio`.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use mlx_audio::capture::{
    AudioDevice, AudioRingControl, CaptureOptions, CaptureSource, CaptureStream, InputDevice,
    PreparedCapture, capture_timing,
};
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

impl From<Qwen3AsrCaptureSource> for CaptureSource {
    fn from(source: Qwen3AsrCaptureSource) -> Self {
        match source {
            Qwen3AsrCaptureSource::Microphone => Self::Microphone,
            Qwen3AsrCaptureSource::SystemAudio => Self::SystemAudio,
        }
    }
}

impl From<CaptureSource> for Qwen3AsrCaptureSource {
    fn from(source: CaptureSource) -> Self {
        match source {
            CaptureSource::Microphone => Self::Microphone,
            CaptureSource::SystemAudio => Self::SystemAudio,
        }
    }
}

impl From<&Qwen3AsrCaptureOptions> for CaptureOptions {
    fn from(options: &Qwen3AsrCaptureOptions) -> Self {
        Self {
            source: options.source.map(Into::into),
            device_id: options.device_id.clone(),
            device_name: options.device_name.clone(),
            application_bundle_ids: options.application_bundle_ids.clone(),
            ring_seconds: options.ring_seconds,
            feed_milliseconds: options.feed_milliseconds,
        }
    }
}

impl From<InputDevice> for Qwen3AsrInputDevice {
    fn from(device: InputDevice) -> Self {
        Self {
            id: device.id,
            name: device.name,
            is_default: device.is_default,
            sample_rate: device.sample_rate,
            channels: device.channels,
            sample_format: device.sample_format,
        }
    }
}

impl From<AudioDevice> for Qwen3AsrAudioDevice {
    fn from(device: AudioDevice) -> Self {
        Self {
            id: device.id,
            name: device.name,
            source: device.source.into(),
            is_default: device.is_default,
            sample_rate: device.sample_rate,
            channels: device.channels,
        }
    }
}

#[napi]
pub struct Qwen3AsrCapture {
    stream: Option<Box<dyn CaptureStream>>,
    ring: AudioRingControl,
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
            .map_err(|error| Error::from_reason(error.to_string()))
    }

    #[napi]
    pub fn resume(&mut self) -> Result<()> {
        self.stream
            .as_mut()
            .ok_or_else(|| Error::from_reason("Capture is stopped"))?
            .start()
            .map_err(|error| Error::from_reason(error.to_string()))
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
        mut stream,
        ring,
        source,
        device_name,
        sample_rate,
        channels,
    } = prepared?;
    let audio_options = CaptureOptions::from(&options);
    let (_, feed_ms) =
        capture_timing(&audio_options).map_err(|error| Error::from_reason(error.to_string()))?;
    let callback = Arc::new(callback);
    let control = ring.control();
    let mut worker_ring = ring;
    let worker_callback = callback.clone();
    let worker_ready = Arc::new(AtomicBool::new(false));
    let capture_ready = worker_ready.clone();
    let worker_sender = sender.clone();
    let worker_stream_id = stream_id.clone();
    let feed_frames = ((sample_rate as u64 * feed_ms as u64) / 1_000).max(1) as usize;
    let worker = std::thread::Builder::new()
        .name(format!("mlx-asr-{}-capture", source.as_str()))
        .spawn(move || {
            // Native setup, worker creation, and model preparation are all
            // fallible. Keep the worker parked until every step succeeds.
            while !worker_ready.load(Ordering::Acquire) {
                if worker_ring.is_stopped() {
                    return;
                }
                worker_ring.wait_timeout_while(Duration::from_millis(20), || {
                    !worker_ready.load(Ordering::Acquire)
                });
            }
            loop {
                let available = worker_ring.available();
                if available < feed_frames && !worker_ring.is_stopped() {
                    worker_ring.wait_timeout(Duration::from_millis(20));
                    continue;
                }
                if available == 0 && worker_ring.is_stopped() {
                    break;
                }
                let samples = worker_ring.drain(if worker_ring.is_stopped() {
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
        control.stop();
        drop(stream);
        let _ = worker.join();
        return Err(Error::from_reason(error.to_string()));
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
        control.stop();
        drop(stream);
        let _ = worker.join();
        return Err(error);
    }
    capture_ready.store(true, Ordering::Release);
    control.wake();

    Ok(Qwen3AsrCapture {
        stream: Some(stream),
        ring: control,
        worker: Some(worker),
        source: source.into(),
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
            captured_frames: ring.captured_frames() as i64,
            dropped_frames: ring.dropped_frames() as i64,
        })
    }
}

pub(super) fn start_capture(
    sender: mpsc::UnboundedSender<Qwen3AsrCmd>,
    stream_id: String,
    options: Qwen3AsrCaptureOptions,
    callback: ThreadsafeFunction<Qwen3AsrResult, ()>,
) -> Result<Qwen3AsrCapture> {
    let audio_options = CaptureOptions::from(&options);
    let prepared = mlx_audio::capture::prepare_capture(&audio_options)
        .map_err(|error| Error::from_reason(error.to_string()));
    start_prepared_capture(sender, stream_id, options, callback, prepared)
}

#[napi]
pub fn qwen3_asr_input_devices() -> Result<Vec<Qwen3AsrInputDevice>> {
    mlx_audio::capture::input_devices()
        .map(|devices| devices.into_iter().map(Into::into).collect())
        .map_err(|error| Error::from_reason(error.to_string()))
}

#[napi]
pub fn qwen3_asr_audio_devices() -> Result<Vec<Qwen3AsrAudioDevice>> {
    mlx_audio::capture::audio_devices()
        .map(|devices| devices.into_iter().map(Into::into).collect())
        .map_err(|error| Error::from_reason(error.to_string()))
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc as std_mpsc;
    use std::time::Instant;

    use mlx_audio::capture::{AudioRingConsumer, audio_ring};

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
        fn start(&mut self) -> std::result::Result<(), mlx_audio::AudioError> {
            self.state.started.store(true, Ordering::Release);
            if self.fail_start {
                Err(mlx_audio::AudioError::Capture(
                    "test stream start failed".into(),
                ))
            } else {
                Ok(())
            }
        }

        fn stop(&mut self) -> std::result::Result<(), mlx_audio::AudioError> {
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
        ring: AudioRingConsumer,
        state: Arc<TestStreamState>,
        fail_start: bool,
    ) -> PreparedCapture {
        PreparedCapture {
            stream: Box::new(TestStream { state, fail_start }),
            ring,
            source: CaptureSource::Microphone,
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
    fn native_start_failure_stops_worker_without_preparing_model() {
        let (sender, mut commands) = mpsc::unbounded_channel();
        let (_producer, ring) = audio_ring(10);
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
        let (mut producer, ring) = audio_ring(10);
        let state = Arc::new(TestStreamState::default());
        let (callback, _events) = callback_events();
        let startup = {
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
        producer.push_mono(&[1.0]);
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
        let (mut producer, ring) = audio_ring(10);
        let state = Arc::new(TestStreamState::default());
        let (callback, events) = callback_events();
        let startup = {
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
        producer.push_mono(&[1.0]);
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
        producer.push_mono(&[2.0]);
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
        let (mut producer, ring) = audio_ring(10);
        producer.push_mono(&[1.0]);
        let state = Arc::new(TestStreamState::default());
        let (callback, events) = callback_events();
        let startup = {
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
