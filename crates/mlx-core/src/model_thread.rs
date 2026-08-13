//! Generic dedicated-thread infrastructure for model state ownership.
//!
//! Each model instance gets its own OS thread that owns all model state
//! (weights, KV caches, tokenizer). Commands are sent via an unbounded
//! MPSC channel and responses flow back through oneshot or bounded streaming
//! mailboxes. This keeps model state off the NAPI/Tokio threads, avoids
//! `Send + Sync` requirements on MLX arrays, and prevents a slow consumer
//! from turning token output into an unbounded native allocation.

/// Oneshot sender for request–response commands.
pub type ResponseTx<T> = tokio::sync::oneshot::Sender<napi::Result<T>>;

/// Deliver the one-shot model-thread initialization result from whichever
/// side wins the spawn lifecycle: the new thread after `Builder::spawn`
/// succeeds, or the caller when OS thread creation itself fails.
fn send_init_result<T>(
    slot: &std::sync::Mutex<Option<tokio::sync::oneshot::Sender<napi::Result<T>>>>,
    result: napi::Result<T>,
) {
    let mut sender = match slot.lock() {
        Ok(sender) => sender,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let Some(sender) = sender.take() {
        let _ = sender.send(result);
    }
}

/// Producer side of one request's bounded streaming mailbox.
///
/// Model execution runs on a dedicated OS thread, so once the mailbox is full
/// it is safe to block that producer until the async forwarding pump drains a
/// slot. The Tokio runtime is never blocked. If the consumer has gone away,
/// `send` returns the unsent item just like the standard channel sender.
#[derive(Debug)]
pub struct StreamTx<T> {
    inner: tokio::sync::mpsc::Sender<napi::Result<T>>,
}

impl<T> Clone for StreamTx<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> StreamTx<T> {
    pub fn send(
        &self,
        item: napi::Result<T>,
    ) -> Result<(), tokio::sync::mpsc::error::SendError<napi::Result<T>>> {
        match self.inner.try_send(item) {
            Ok(()) => Ok(()),
            Err(tokio::sync::mpsc::error::TrySendError::Closed(item)) => {
                Err(tokio::sync::mpsc::error::SendError(item))
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(item)) => {
                self.inner.blocking_send(item)
            }
        }
    }
}

/// Create one bounded request-output mailbox.
pub fn stream_channel<T>(
    capacity: usize,
) -> (StreamTx<T>, tokio::sync::mpsc::Receiver<napi::Result<T>>) {
    assert!(capacity > 0, "stream mailbox capacity must be non-zero");
    let (tx, rx) = tokio::sync::mpsc::channel(capacity);
    (StreamTx { inner: tx }, rx)
}

/// A dedicated OS thread that owns model state and processes commands.
///
/// Generic over `Cmd` so each model picks its command enum: the shared
/// `engine::cmd::ChatCmd` for chat-only families (gemma4), or a
/// per-family enum carrying extra variants (e.g. `Qwen3Cmd`).
pub struct ModelThread<Cmd: Send + 'static> {
    cmd_tx: Option<tokio::sync::mpsc::UnboundedSender<Cmd>>,
    _handle: Option<std::thread::JoinHandle<()>>,
}

/// Result of one scheduler-owned model-thread loop iteration.
pub(crate) enum LoopControl {
    Continue,
    Break,
}

impl<Cmd: Send + 'static> ModelThread<Cmd> {
    /// Spawn a model thread whose loop owns the command receiver.
    ///
    /// Unlike [`Self::spawn_with_init`], this variant does not consume one
    /// command and hand it to a whole-turn handler. The loop body may
    /// `blocking_recv` while idle and `try_recv` between active scheduler
    /// steps, which is the required shape for continuous batching without a
    /// polling thread or a busy wait.
    pub(crate) fn spawn_with_scheduler<State, Init, InitResult, LoopBody>(
        init_fn: Init,
        mut loop_body: LoopBody,
    ) -> (
        Self,
        tokio::sync::oneshot::Receiver<napi::Result<InitResult>>,
    )
    where
        Init: FnOnce() -> napi::Result<(State, InitResult)> + Send + 'static,
        InitResult: Send + 'static,
        LoopBody: FnMut(&mut State, &mut tokio::sync::mpsc::UnboundedReceiver<Cmd>) -> LoopControl
            + Send
            + 'static,
    {
        let (cmd_tx, mut cmd_rx) = tokio::sync::mpsc::unbounded_channel::<Cmd>();
        let (init_tx, init_rx) = tokio::sync::oneshot::channel();
        let init_tx = std::sync::Arc::new(std::sync::Mutex::new(Some(init_tx)));
        let thread_init_tx = std::sync::Arc::clone(&init_tx);
        let handle = match std::thread::Builder::new()
            .name("mlx-model".into())
            .spawn(move || {
                let mut state = match init_fn() {
                    Ok((state, init_result)) => {
                        send_init_result(&thread_init_tx, Ok(init_result));
                        state
                    }
                    Err(error) => {
                        send_init_result(&thread_init_tx, Err(error));
                        return;
                    }
                };
                while matches!(loop_body(&mut state, &mut cmd_rx), LoopControl::Continue) {}
            }) {
            Ok(handle) => Some(handle),
            Err(error) => {
                send_init_result(
                    &init_tx,
                    Err(napi::Error::from_reason(format!(
                        "failed to spawn mlx-model thread: {error}"
                    ))),
                );
                None
            }
        };

        (
            Self {
                cmd_tx: Some(cmd_tx),
                _handle: handle,
            },
            init_rx,
        )
    }

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
        let init_tx = std::sync::Arc::new(std::sync::Mutex::new(Some(init_tx)));
        let thread_init_tx = std::sync::Arc::clone(&init_tx);

        let handle = match std::thread::Builder::new()
            .name("mlx-model".into())
            .spawn(move || {
                let mut state = match init_fn() {
                    Ok((state, init_result)) => {
                        send_init_result(&thread_init_tx, Ok(init_result));
                        state
                    }
                    Err(e) => {
                        send_init_result(&thread_init_tx, Err(e));
                        return;
                    }
                };

                while let Some(cmd) = cmd_rx.blocking_recv() {
                    handler(&mut state, cmd);
                }
            }) {
            Ok(handle) => Some(handle),
            Err(error) => {
                send_init_result(
                    &init_tx,
                    Err(napi::Error::from_reason(format!(
                        "failed to spawn mlx-model thread: {error}"
                    ))),
                );
                None
            }
        };

        let thread = Self {
            cmd_tx: Some(cmd_tx),
            _handle: handle,
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
    use super::{LoopControl, ModelThread, ResponseTx, send_and_await, stream_channel};
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
    fn stream_mailbox_backpressures_at_its_fixed_capacity() {
        let (tx, mut rx) = stream_channel::<u32>(1);
        tx.send(Ok(1)).expect("first mailbox slot");

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let producer = std::thread::spawn(move || {
            entered_tx.send(()).expect("announce blocked send");
            tx.send(Ok(2)).expect("second mailbox slot after drain");
            done_tx.send(()).expect("announce completed send");
        });

        entered_rx.recv().expect("producer did not start");
        assert!(
            matches!(
                done_rx.try_recv(),
                Err(std::sync::mpsc::TryRecvError::Empty)
            ),
            "the producer advanced past a full one-item mailbox"
        );
        assert_eq!(
            rx.blocking_recv()
                .expect("first item missing")
                .expect("first item failed"),
            1
        );
        done_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("producer did not resume after the consumer drained a slot");
        assert_eq!(
            rx.blocking_recv()
                .expect("second item missing")
                .expect("second item failed"),
            2
        );
        producer.join().expect("producer panicked");
    }

    #[test]
    fn dropping_a_full_mailbox_receiver_wakes_the_blocked_producer() {
        let (tx, rx) = stream_channel::<u32>(1);
        tx.send(Ok(1)).expect("fill mailbox");

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let producer = std::thread::spawn(move || {
            entered_tx.send(()).expect("announce blocked send");
            result_tx
                .send(tx.send(Ok(2)))
                .expect("report blocked-send result");
        });

        entered_rx.recv().expect("producer did not start");
        assert!(
            matches!(
                result_rx.try_recv(),
                Err(std::sync::mpsc::TryRecvError::Empty)
            ),
            "the producer advanced past a full mailbox"
        );
        drop(rx);
        let error = result_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("receiver drop did not wake the blocked producer")
            .expect_err("closed mailbox send unexpectedly succeeded");
        assert_eq!(error.0.expect("unsent item changed"), 2);
        producer.join().expect("producer panicked");
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

    #[test]
    fn scheduler_spawn_hands_receiver_ownership_to_the_loop() {
        enum Cmd {
            Add(i32, ResponseTx<i32>),
            Stop,
        }
        let (mut model_thread, init_rx) = ModelThread::<Cmd>::spawn_with_scheduler(
            || Ok((40, ())),
            |state, receiver| match receiver.blocking_recv() {
                Some(Cmd::Add(value, reply)) => {
                    *state += value;
                    let _ = reply.send(Ok(*state));
                    LoopControl::Continue
                }
                Some(Cmd::Stop) | None => LoopControl::Break,
            },
        );
        init_rx
            .blocking_recv()
            .expect("model init channel closed")
            .expect("model init failed");
        let (reply, result) = tokio::sync::oneshot::channel();
        model_thread
            .send(Cmd::Add(2, reply))
            .expect("send scheduler command");
        assert_eq!(
            result
                .blocking_recv()
                .expect("reply channel closed")
                .expect("command failed"),
            42
        );
        model_thread.send(Cmd::Stop).expect("send stop");
        model_thread
            .shutdown_and_join()
            .expect("scheduler loop should join");
    }

    /// H1a shape pin: a reset-style command queued behind an in-flight
    /// turn must park a *future*, never the calling thread. This is the
    /// dispatch shape `chat_napi_thread_reset!` uses via
    /// [`send_and_await`] — the old `send_and_block` path would freeze
    /// the caller (the Node event loop in production) on the first
    /// `poll`-equivalent instead of returning `Pending`.
    ///
    /// Deterministic (no timers): the first command occupies the model
    /// thread until explicitly released, so the queued reset is
    /// provably pending while the caller keeps doing other work.
    #[test]
    fn queued_reset_parks_a_future_not_the_calling_thread() {
        enum Cmd {
            /// Simulated in-flight turn: holds the model thread until
            /// the test releases it.
            Occupy {
                release: std::sync::mpsc::Receiver<()>,
                reply: ResponseTx<()>,
            },
            /// Simulated `ResetCaches`: replies immediately once the
            /// model thread reaches it.
            Reset { reply: ResponseTx<()> },
        }

        let (model_thread, init_rx) = ModelThread::<Cmd>::spawn_with_init(
            || Ok(((), ())),
            |_, cmd| match cmd {
                Cmd::Occupy { release, reply } => {
                    release.recv().expect("test dropped the release sender");
                    let _ = reply.send(Ok(()));
                }
                Cmd::Reset { reply } => {
                    let _ = reply.send(Ok(()));
                }
            },
        );
        init_rx
            .blocking_recv()
            .expect("model init channel closed")
            .expect("model init failed");

        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (turn_tx, turn_rx) = tokio::sync::oneshot::channel();
        model_thread
            .send(Cmd::Occupy {
                release: release_rx,
                reply: turn_tx,
            })
            .expect("failed to enqueue the occupying turn");

        // A current-thread runtime models the Node event loop: anything
        // that blocks this thread blocks *everything* scheduled on it.
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("failed to build current-thread runtime");
        rt.block_on(async {
            let reset = send_and_await(&model_thread, |reply| Cmd::Reset { reply });
            futures::pin_mut!(reset);

            // Queued behind the occupied turn: polling returns Pending
            // instead of blocking the calling thread (`send_and_block`
            // would never return here).
            assert!(
                futures::poll!(reset.as_mut()).is_pending(),
                "reset resolved while the in-flight turn still held the model thread"
            );

            // The "event loop" stays free to run unrelated work while
            // the reset is parked.
            let other_work = async { 21 * 2 }.await;
            assert_eq!(other_work, 42);
            assert!(
                futures::poll!(reset.as_mut()).is_pending(),
                "reset must stay parked until the in-flight turn drains"
            );

            // Drain the turn; FIFO ordering then completes the reset.
            release_tx
                .send(())
                .expect("model thread dropped the occupy command");
            turn_rx
                .await
                .expect("turn reply channel closed")
                .expect("occupying turn failed");
            reset
                .await
                .expect("reset should resolve once the turn drains");
        });
    }
}
