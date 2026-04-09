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
/// Generic over `Cmd` so each model defines its own command enum
/// (e.g. `Gemma4Cmd`, `Qwen3Cmd`).
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
