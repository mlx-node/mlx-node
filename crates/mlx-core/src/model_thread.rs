//! Generic dedicated-thread infrastructure for model state ownership.
//!
//! Each model instance gets its own OS thread that owns all model state
//! (weights, KV caches, tokenizer). Commands are sent via an unbounded
//! MPSC channel and responses flow back through oneshot or streaming
//! channels. This keeps model state off the NAPI/Tokio threads and
//! avoids `Send + Sync` requirements on MLX arrays.

/// Oneshot sender for request–response commands.
pub type ResponseTx<T> = tokio::sync::oneshot::Sender<napi::Result<T>>;

/// Unbounded sender for streaming commands (e.g. token-by-token output).
pub type StreamTx<T> = tokio::sync::mpsc::UnboundedSender<napi::Result<T>>;

/// A dedicated OS thread that owns model state and processes commands.
///
/// Generic over `Cmd` so each model picks its command enum: the shared
/// `engine::cmd::ChatCmd` for chat-only families (gemma4), or a
/// per-family enum carrying extra variants (e.g. `Qwen3Cmd`).
pub struct ModelThread<Cmd: Send + 'static> {
    cmd_tx: Option<tokio::sync::mpsc::UnboundedSender<Cmd>>,
    _handle: Option<std::thread::JoinHandle<()>>,
}

impl<Cmd: Send + 'static> ModelThread<Cmd> {
    /// Spawn a dedicated model thread with an initialization phase.
    ///
    /// 1. The thread runs `init_fn` which returns `(State, InitResult)`.
    /// 2. `InitResult` is sent back to the caller via the returned oneshot receiver.
    /// 3. The thread then enters a command loop calling `handler` for each `Cmd`.
    ///
    /// If `init_fn` fails the error is sent via the oneshot and the thread exits.
    pub fn spawn_with_init<State, Init, InitResult, Handler>(
        init_fn: Init,
        mut handler: Handler,
    ) -> (
        Self,
        tokio::sync::oneshot::Receiver<napi::Result<InitResult>>,
    )
    where
        State: Send + 'static,
        Init: FnOnce() -> napi::Result<(State, InitResult)> + Send + 'static,
        InitResult: Send + 'static,
        Handler: FnMut(&mut State, Cmd) + Send + 'static,
    {
        let (cmd_tx, mut cmd_rx) = tokio::sync::mpsc::unbounded_channel::<Cmd>();
        let (init_tx, init_rx) = tokio::sync::oneshot::channel();

        let handle = std::thread::Builder::new()
            .name("mlx-model".into())
            .spawn(move || {
                let mut state = match init_fn() {
                    Ok((state, init_result)) => {
                        let _ = init_tx.send(Ok(init_result));
                        state
                    }
                    Err(e) => {
                        let _ = init_tx.send(Err(e));
                        return;
                    }
                };

                while let Some(cmd) = cmd_rx.blocking_recv() {
                    handler(&mut state, cmd);
                }
            })
            .expect("failed to spawn mlx-model thread");

        let thread = Self {
            cmd_tx: Some(cmd_tx),
            _handle: Some(handle),
        };
        (thread, init_rx)
    }

    /// Get a reference to the command sender.
    /// Training engines use this to send training commands directly.
    pub fn cmd_sender(&self) -> Option<&tokio::sync::mpsc::UnboundedSender<Cmd>> {
        self.cmd_tx.as_ref()
    }

    /// Send a command to the model thread.
    ///
    /// Returns an error if the channel is closed (thread has exited).
    pub fn send(&self, cmd: Cmd) -> napi::Result<()> {
        self.cmd_tx
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("Model thread is not running"))?
            .send(cmd)
            .map_err(|_| napi::Error::from_reason("Model thread has exited"))
    }

    /// Close the command channel and wait for the model thread to release its
    /// state.
    ///
    /// Normal NAPI teardown deliberately detaches in [`Drop`] so JavaScript GC
    /// never blocks on multi-gigabyte Metal cleanup. Real-weight tests sometimes
    /// need deterministic teardown before loading a second oracle model on a
    /// memory-constrained runner; this explicit path provides that stronger
    /// lifecycle guarantee without changing production drop behavior.
    pub(crate) fn shutdown_and_join(&mut self) -> Result<(), String> {
        if let Some(cmd_tx) = self.cmd_tx.as_ref() {
            let sender_count = cmd_tx.strong_count();
            if sender_count != 1 {
                return Err(format!(
                    "cannot join model thread while {sender_count} command senders are alive"
                ));
            }
        }
        self.cmd_tx.take();
        match self._handle.take() {
            Some(handle) => handle
                .join()
                .map_err(|_| "model thread panicked during shutdown".to_string()),
            None => Ok(()),
        }
    }
}

impl<Cmd: Send + 'static> Drop for ModelThread<Cmd> {
    fn drop(&mut self) {
        // Close the command channel so the thread's recv loop exits.
        // We intentionally do NOT join the thread here — dropping the
        // JoinHandle detaches it.  The thread will finish processing any
        // in-flight command, drop its state (freeing Metal resources),
        // and exit on its own.  Joining can block for seconds while MLX
        // tears down GPU allocations, which causes vitest fork workers
        // to time out and get killed.
        self.cmd_tx.take();
    }
}

/// Send a command and await the response asynchronously.
///
/// Use this from `#[napi]` async methods. Creates a oneshot channel,
/// builds the command via `make_cmd`, sends it, and awaits the reply.
pub async fn send_and_await<Cmd, T, F>(thread: &ModelThread<Cmd>, make_cmd: F) -> napi::Result<T>
where
    Cmd: Send + 'static,
    T: Send + 'static,
    F: FnOnce(ResponseTx<T>) -> Cmd,
{
    let (tx, rx) = tokio::sync::oneshot::channel();
    thread.send(make_cmd(tx))?;
    rx.await
        .map_err(|_| napi::Error::from_reason("Model thread exited unexpectedly"))?
}

/// Send a command and block until the response arrives.
///
/// Use this from synchronous NAPI methods (e.g. training ops that must
/// run sequentially). Same pattern as [`send_and_await`] but calls
/// `blocking_recv()` instead of `.await`.
pub fn send_and_block<Cmd, T, F>(thread: &ModelThread<Cmd>, make_cmd: F) -> napi::Result<T>
where
    Cmd: Send + 'static,
    T: Send + 'static,
    F: FnOnce(ResponseTx<T>) -> Cmd,
{
    let (tx, rx) = tokio::sync::oneshot::channel();
    thread.send(make_cmd(tx))?;
    rx.blocking_recv()
        .map_err(|_| napi::Error::from_reason("Model thread exited unexpectedly"))?
}

#[cfg(test)]
mod tests {
    use super::ModelThread;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    struct DropState(Arc<AtomicBool>);

    impl Drop for DropState {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn explicit_shutdown_requires_exclusive_sender_and_joins_state_drop() {
        let dropped = Arc::new(AtomicBool::new(false));
        let dropped_for_thread = dropped.clone();
        let (mut model_thread, init_rx) = ModelThread::<()>::spawn_with_init(
            move || Ok((DropState(dropped_for_thread), ())),
            |_, _| {},
        );
        init_rx
            .blocking_recv()
            .expect("model init channel closed")
            .expect("model init failed");

        let extra_sender = model_thread
            .cmd_sender()
            .expect("model sender missing")
            .clone();
        let err = model_thread
            .shutdown_and_join()
            .expect_err("an outstanding sender must prevent a blocking join");
        assert!(err.contains("2 command senders"), "unexpected error: {err}");
        assert!(!dropped.load(Ordering::SeqCst));

        drop(extra_sender);
        model_thread
            .shutdown_and_join()
            .expect("exclusive shutdown should join the worker");
        assert!(dropped.load(Ordering::SeqCst));
    }
}
