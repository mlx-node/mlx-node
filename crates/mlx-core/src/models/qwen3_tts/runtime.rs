//! Pull-based N-API transport. Only owned PCM crosses the model thread boundary.
use super::{
    model::{NativeModel, Options},
    prefix::{LoadOptions, PrefixCache},
};
use crate::model_thread::{ModelThread, ResponseTx, stream_channel};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::{Mutex as AsyncMutex, Notify, mpsc};

// Registered ceilings feed the shared process-wide MLX free pool, so this
// limit applies to every loaded model in the process, not just TTS.
const CACHE_LIMIT_ENV: &str = "TTS_MLX_CACHE_LIMIT";

fn parse_cache_limit(raw: &str) -> Result<Option<usize>> {
    let invalid = || {
        Error::from_reason(format!(
            "{CACHE_LIMIT_ENV} must be a finite non-negative GiB amount (0 = automatic policy); got {raw:?}"
        ))
    };
    let gib = raw.trim().parse::<f64>().map_err(|_| invalid())?;
    if !gib.is_finite() || gib < 0.0 {
        return Err(invalid());
    }
    if gib == 0.0 {
        return Ok(None);
    }
    let bytes = (gib * (1u64 << 30) as f64).round();
    if bytes < 1.0 || bytes >= usize::MAX as f64 {
        return Err(invalid());
    }
    Ok(Some(bytes as usize))
}

fn cache_limit_from_env() -> Result<Option<usize>> {
    match std::env::var(CACHE_LIMIT_ENV) {
        Ok(raw) => parse_cache_limit(&raw),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(Error::from_reason(format!(
            "{CACHE_LIMIT_ENV} must be valid UTF-8"
        ))),
    }
}

#[napi(object)]
pub struct TtsNativeChunk {
    pub samples: Float32Array,
    pub finished: bool,
    pub finish_reason: Option<String>,
    pub synthesis_ms: Option<f64>,
    pub first_pcm_ms: Option<f64>,
}
struct Packet {
    samples: Vec<f32>,
    reason: Option<String>,
    synthesis_ms: Option<f64>,
    first_pcm_ms: Option<f64>,
}
struct Control {
    cancelled: AtomicBool,
    notify: Notify,
    receiver: AsyncMutex<mpsc::Receiver<Result<Packet>>>,
    pulling: AtomicBool,
    finished: AtomicBool,
    ended: Notify,
}
impl Control {
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        if let Ok(mut receiver) = self.receiver.try_lock() {
            receiver.close();
        }
        self.notify.notify_waiters();
    }
}
enum ActiveOperation {
    Generate(Arc<Control>),
    Prepare(Arc<AtomicBool>),
}
impl ActiveOperation {
    fn cancel(&self) {
        match self {
            Self::Generate(control) => control.cancel(),
            Self::Prepare(cancelled) => cancelled.store(true, Ordering::Release),
        }
    }
}
/// Complete the lifecycle even if a model operation unwinds. The transport
/// then closes and reports a worker error instead of leaving disposal waiting.
struct RunGuard {
    busy: Arc<AtomicBool>,
    control: Option<Arc<Control>>,
}
impl Drop for RunGuard {
    fn drop(&mut self) {
        self.busy.store(false, Ordering::Release);
        if let Some(control) = &self.control {
            control.finished.store(true, Ordering::Release);
            control.ended.notify_waiters();
        }
    }
}
enum Command {
    Generate {
        text: String,
        options: Box<Options>,
        control: Arc<Control>,
        tx: crate::model_thread::StreamTx<Packet>,
        busy: Arc<AtomicBool>,
    },
    Prepare {
        audio: Vec<f32>,
        sample_rate: u32,
        transcript: String,
        cancelled: Arc<AtomicBool>,
        reply: ResponseTx<String>,
        busy: Arc<AtomicBool>,
    },
    // Releases stay valid while a generation or preparation runs: the command
    // channel serializes them behind the active operation.
    Release {
        id: String,
        reply: ResponseTx<()>,
    },
    Dispose(ResponseTx<()>),
}
#[napi]
pub struct TtsNativeModel {
    thread: ModelThread<Command>,
    metadata: String,
    busy: Arc<AtomicBool>,
    disposed: AtomicBool,
    active: Mutex<Option<ActiveOperation>>,
}
#[napi]
impl TtsNativeModel {
    #[napi]
    pub fn load<'env>(
        env: &'env Env,
        path: String,
        options_json: Option<String>,
    ) -> Result<PromiseRaw<'env, TtsNativeModel>> {
        env.spawn_future(async move {
            let cache_limit = cache_limit_from_env()?;
            let options: LoadOptions =
                serde_json::from_str(options_json.as_deref().unwrap_or("{}"))
                    .map_err(|e| Error::from_reason(format!("Invalid TTS load options: {e}")))?;
            let (thread, rx) = ModelThread::spawn_with_init(
                move || {
                    let (model, bytes) = NativeModel::load(Path::new(&path))?;
                    let metadata = model.metadata();
                    let guard = crate::cache_limit::coordinator()
                        .register_with_cache_limit(bytes, cache_limit);
                    Ok((
                        (
                            Some(model),
                            Some(guard),
                            Some(PrefixCache::new(options.instruction_cache)),
                        ),
                        metadata,
                    ))
                },
                |state, command| match command {
                    Command::Generate {
                        text,
                        options,
                        control,
                        tx,
                        busy,
                    } => {
                        let _guard = RunGuard {
                            busy,
                            control: Some(control.clone()),
                        };
                        let result = state
                            .0
                            .as_ref()
                            .zip(state.2.as_mut())
                            .ok_or_else(|| Error::from_reason("TTS model disposed"))
                            .and_then(|(model, cache)| {
                                model.generate(
                                    &text,
                                    *options,
                                    &control.cancelled,
                                    cache,
                                    |samples| {
                                        if control.cancelled.load(Ordering::Acquire) {
                                            return Err(Error::from_reason("TTS cancelled"));
                                        }
                                        tx.send(Ok(Packet {
                                            samples,
                                            reason: None,
                                            synthesis_ms: None,
                                            first_pcm_ms: None,
                                        }))
                                        .map_err(|_| Error::from_reason("TTS consumer closed"))
                                    },
                                )
                            });
                        if !control.cancelled.load(Ordering::Acquire) {
                            let terminal = result.map(|stats| Packet {
                                samples: vec![],
                                reason: Some(stats.reason.into()),
                                synthesis_ms: Some(stats.synthesis_ms),
                                first_pcm_ms: stats.first_pcm_ms,
                            });
                            let _ = tx.send(terminal);
                        }
                    }
                    Command::Prepare {
                        audio,
                        sample_rate,
                        transcript,
                        cancelled,
                        reply,
                        busy,
                    } => {
                        let _guard = RunGuard {
                            busy,
                            control: None,
                        };
                        let result = state
                            .0
                            .as_mut()
                            .ok_or_else(|| Error::from_reason("TTS model disposed"))
                            .and_then(|m| {
                                m.prepare_voice(audio, sample_rate, transcript, &cancelled)
                            });
                        drop(_guard);
                        let _ = reply.send(result);
                    }
                    Command::Release { id, reply } => {
                        if let Some(model) = state.0.as_mut() {
                            model.release_voice(&id);
                        }
                        let _ = reply.send(Ok(()));
                    }
                    Command::Dispose(reply) => {
                        state.0.take();
                        state.1.take();
                        state.2.take();
                        let _ = reply.send(Ok(()));
                    }
                },
            );
            let metadata = rx.await.map_err(|_| {
                Error::from_reason("TTS model worker exited during initialization")
            })??;
            Ok(Self {
                thread,
                metadata,
                busy: Arc::new(AtomicBool::new(false)),
                disposed: AtomicBool::new(false),
                active: Mutex::new(None),
            })
        })
    }
    #[napi(getter)]
    pub fn metadata(&self) -> String {
        self.metadata.clone()
    }
    #[napi]
    pub fn start(&self, text: String, options_json: String) -> Result<TtsNativeStream> {
        if self.disposed.load(Ordering::Acquire) {
            return Err(Error::from_reason("TTS model disposed"));
        }
        let options: Options = serde_json::from_str(&options_json)
            .map_err(|e| Error::from_reason(format!("Invalid TTS options: {e}")))?;
        let capacity = options.buffer_chunks.unwrap_or(6);
        if capacity == 0 || capacity > 4096 {
            return Err(Error::from_reason("Invalid TTS buffer capacity"));
        }
        if self.busy.swap(true, Ordering::AcqRel) {
            return Err(Error::from_reason("TTS model busy"));
        }
        let (tx, receiver) = stream_channel(capacity);
        let control = Arc::new(Control {
            cancelled: AtomicBool::new(false),
            notify: Notify::new(),
            receiver: AsyncMutex::new(receiver),
            pulling: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            ended: Notify::new(),
        });
        match self.active.lock() {
            Ok(mut active) => {
                *active = Some(ActiveOperation::Generate(control.clone()));
            }
            Err(_) => {
                self.busy.store(false, Ordering::Release);
                return Err(Error::from_reason("TTS lifecycle lock poisoned"));
            }
        }
        let result = self.thread.send(Command::Generate {
            text,
            options: Box::new(options),
            control: control.clone(),
            tx,
            busy: self.busy.clone(),
        });
        if let Err(error) = result {
            if let Ok(mut active) = self.active.lock() {
                *active = None;
            }
            self.busy.store(false, Ordering::Release);
            return Err(error);
        }
        Ok(TtsNativeStream { control })
    }
    #[napi]
    pub fn prepare_voice<'env>(
        &self,
        env: &'env Env,
        audio: Float32Array,
        sample_rate: u32,
        transcript: String,
    ) -> Result<PromiseRaw<'env, String>> {
        if self.disposed.load(Ordering::Acquire) {
            return Err(Error::from_reason("TTS model disposed"));
        }
        if self.busy.swap(true, Ordering::AcqRel) {
            return Err(Error::from_reason("TTS model busy"));
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        match self.active.lock() {
            Ok(mut active) => *active = Some(ActiveOperation::Prepare(cancelled.clone())),
            Err(_) => {
                self.busy.store(false, Ordering::Release);
                return Err(Error::from_reason("TTS lifecycle lock poisoned"));
            }
        }
        let (reply, rx) = tokio::sync::oneshot::channel();
        if let Err(error) = self.thread.send(Command::Prepare {
            audio: audio.to_vec(),
            sample_rate,
            transcript,
            cancelled,
            reply,
            busy: self.busy.clone(),
        }) {
            if let Ok(mut active) = self.active.lock() {
                *active = None;
            }
            self.busy.store(false, Ordering::Release);
            return Err(error);
        }
        env.spawn_future(async move {
            rx.await
                .map_err(|_| Error::from_reason("TTS worker exited during voice preparation"))?
        })
    }
    #[napi]
    pub fn release_voice<'env>(&self, env: &'env Env, id: String) -> Result<PromiseRaw<'env, ()>> {
        if self.disposed.load(Ordering::Acquire) {
            return env.spawn_future(async { Ok(()) });
        }
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.thread.send(Command::Release { id, reply })?;
        env.spawn_future(async move {
            rx.await
                .map_err(|_| Error::from_reason("TTS worker exited during voice release"))?
        })
    }
    #[napi]
    pub fn dispose<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
        self.disposed.store(true, Ordering::Release);
        if let Some(control) = self.active.lock().unwrap_or_else(|e| e.into_inner()).take() {
            control.cancel();
        }
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.thread.send(Command::Dispose(tx))?;
        env.spawn_future(async move {
            rx.await
                .map_err(|_| Error::from_reason("TTS worker exited"))?
        })
    }
}
impl Drop for TtsNativeModel {
    fn drop(&mut self) {
        let active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(control) = active.as_ref() {
            control.cancel();
        }
    }
}
#[napi]
pub struct TtsNativeStream {
    control: Arc<Control>,
}
#[napi]
impl TtsNativeStream {
    #[napi]
    pub fn next<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, Option<TtsNativeChunk>>> {
        if self.control.pulling.swap(true, Ordering::AcqRel) {
            return Err(Error::from_reason(
                "Concurrent TTS next() calls are not supported",
            ));
        }
        let control = self.control.clone();
        match env.spawn_future(async move {
            let mut receiver = control.receiver.lock().await;
            let notified = control.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let result = if control.cancelled.load(Ordering::Acquire) {
                receiver.close();
                None
            } else {
                tokio::select! { value=receiver.recv()=>value, _=notified=>{receiver.close();None} }
            };
            control.pulling.store(false, Ordering::Release);
            result.transpose().map(|packet| {
                packet.map(|p| TtsNativeChunk {
                    samples: p.samples.into(),
                    finished: p.reason.is_some(),
                    finish_reason: p.reason,
                    synthesis_ms: p.synthesis_ms,
                    first_pcm_ms: p.first_pcm_ms,
                })
            })
        }) {
            Ok(promise) => Ok(promise),
            Err(error) => {
                self.control.pulling.store(false, Ordering::Release);
                Err(error)
            }
        }
    }
    #[napi]
    pub fn wait_finished<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
        let control = self.control.clone();
        env.spawn_future(async move {
            let notified = control.ended.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !control.finished.load(Ordering::Acquire) {
                notified.await;
            }
            Ok(())
        })
    }
    #[napi]
    pub fn cancel(&self) {
        self.control.cancel();
    }
}
impl Drop for TtsNativeStream {
    fn drop(&mut self) {
        self.control.cancel();
    }
}

#[cfg(test)]
mod cache_limit_tests {
    use super::parse_cache_limit;

    #[test]
    fn accepts_gib_and_zero_without_silently_clamping() {
        assert_eq!(parse_cache_limit(" 0.5 ").unwrap(), Some(1 << 29));
        assert_eq!(parse_cache_limit("1").unwrap(), Some(1 << 30));
        assert_eq!(parse_cache_limit("0").unwrap(), None);
        for invalid in ["", "1GB", "-1", "NaN", "inf", "1e100", "1e-30"] {
            assert!(parse_cache_limit(invalid).is_err(), "{invalid}");
        }
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[test]
    fn qwen3_tts_model_drop_cancels_reference_preparation() {
        // No model or MLX arrays are needed to exercise the native owner lifecycle.
        let (thread, ready) = ModelThread::spawn_with_init(|| Ok(((), ())), |_, _: Command| {});
        ready.blocking_recv().unwrap().unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let model = TtsNativeModel {
            thread,
            metadata: String::new(),
            busy: Arc::new(AtomicBool::new(true)),
            disposed: AtomicBool::new(false),
            active: Mutex::new(Some(ActiveOperation::Prepare(cancelled.clone()))),
        };
        drop(model);
        assert!(super::super::check_cancelled(&cancelled).is_err());
    }
}
