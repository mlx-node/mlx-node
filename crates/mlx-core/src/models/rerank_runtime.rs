//! Family-independent task ownership for inference-only rerankers.
use crate::{
    array,
    model_thread::{ModelThread, ResponseTx},
};
use napi::bindgen_prelude::*;
use serde::Deserialize;
use serde_json::Value;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::oneshot;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Request {
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: Option<usize>,
    #[serde(default)]
    pub truncate: bool,
}

pub(crate) trait Backend: Send + 'static {
    fn info(&self) -> &Value;
    fn rerank(&self, request: Request, cancel: &AtomicBool, shutdown: &AtomicBool)
    -> Result<Value>;
}

struct Command {
    request: Request,
    cancel: Arc<AtomicBool>,
    reply: ResponseTx<String>,
}
struct Handle {
    thread: Mutex<Option<ModelThread<Command>>>,
    shutdown: Arc<AtomicBool>,
    disposal: tokio::sync::Mutex<()>,
}
pub(crate) struct Runtime {
    handle: Arc<Handle>,
    pub info: String,
}
fn worker_error(message: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("rerank_worker_failed: {message}"))
}
impl Runtime {
    pub async fn load<B: Backend>(
        init: impl FnOnce() -> Result<B> + Send + 'static,
    ) -> Result<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let stopping = shutdown.clone();
        let (thread, ready) = ModelThread::spawn_with_init(
            move || {
                let model = init()?;
                let info = serde_json::to_string(model.info()).map_err(worker_error)?;
                Ok((model, info))
            },
            move |model, command: Command| {
                let result = model
                    .rerank(command.request, &command.cancel, &stopping)
                    .and_then(|value| serde_json::to_string(&value).map_err(worker_error));
                // A completion, including cancellation, is a GPU lifetime boundary.
                array::synchronize();
                let _ = command.reply.send(result);
            },
        );
        let info = await_reply(ready).await?;
        Ok(Self {
            info,
            handle: Arc::new(Handle {
                thread: Mutex::new(Some(thread)),
                shutdown,
                disposal: tokio::sync::Mutex::new(()),
            }),
        })
    }
    pub fn start(&self, request_json: String) -> Result<Task> {
        let request = serde_json::from_str(&request_json)
            .map_err(|e| Error::from_reason(format!("rerank_invalid_request: {e}")))?;
        let thread = self
            .handle
            .thread
            .lock()
            .map_err(|_| worker_error("thread lock poisoned"))?;
        if self.handle.shutdown.load(Ordering::Acquire) {
            return Err(Error::from_reason("rerank_closed: model disposed"));
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let (reply, receiver) = oneshot::channel();
        thread
            .as_ref()
            .ok_or_else(|| Error::from_reason("rerank_closed: model disposed"))?
            .send(Command {
                request,
                cancel: cancel.clone(),
                reply,
            })?;
        Ok(Task {
            cancel,
            receiver: Mutex::new(Some(receiver)),
        })
    }
    pub fn dispose<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
        let handle = self.handle.clone();
        handle.shutdown.store(true, Ordering::Release);
        env.spawn_future(async move {
            let _guard = handle.disposal.lock().await;
            let thread = handle
                .thread
                .lock()
                .map_err(|_| worker_error("thread lock poisoned"))?
                .take();
            if let Some(mut thread) = thread {
                spawn_blocking(move || thread.shutdown_and_join().map_err(worker_error))
                    .await
                    .map_err(worker_error)??;
            }
            Ok(())
        })
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        self.handle.shutdown.store(true, Ordering::Release);
    }
}

pub(crate) struct Task {
    cancel: Arc<AtomicBool>,
    receiver: Mutex<Option<oneshot::Receiver<Result<String>>>>,
}
impl Task {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
    pub fn result<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, String>> {
        let rx = self
            .receiver
            .lock()
            .map_err(|_| worker_error("task lock poisoned"))?
            .take()
            .ok_or_else(|| worker_error("task result already consumed"))?;
        env.spawn_future(await_reply(rx))
    }
}
impl Drop for Task {
    fn drop(&mut self) {
        self.cancel();
    }
}
async fn await_reply<T>(rx: oneshot::Receiver<Result<T>>) -> Result<T> {
    rx.await.map_err(|_| worker_error("model thread exited"))?
}
