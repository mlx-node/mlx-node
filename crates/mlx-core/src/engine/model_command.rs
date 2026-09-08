//! One command envelope for every chat model thread.
//!
//! Chat and scheduler telemetry have one representation and dispatch path.
//! Families define only their additional operations. Inference-only families
//! use `ModelCommand` directly, with an uninhabited extension type.

use std::convert::Infallible;

use crate::engine::SchedulerStatsJs;
use crate::engine::backend::ChatBackend;
use crate::engine::cmd::{ChatCmd, FromTrainCmd, TrainCmd, handle_chat_cmd};
use crate::model_thread::ResponseTx;

pub(crate) enum ModelCommand<C = Infallible> {
    Chat(Box<ChatCmd>),
    SchedulerStats { reply: ResponseTx<SchedulerStatsJs> },
    Family(C),
}

/// Executes a family's additional operation on its owning model thread.
/// These commands are ordered barriers in the shared scheduler.
pub(crate) trait FamilyCommand<B>: Send + 'static {
    fn execute(self, backend: &mut B);
}

impl<B> FamilyCommand<B> for Infallible {
    fn execute(self, _backend: &mut B) {
        match self {}
    }
}

impl<C> ModelCommand<C> {
    pub(crate) fn from_chat(command: ChatCmd) -> Self {
        Self::Chat(Box::new(command))
    }

    pub(crate) fn as_chat(&self) -> Option<&ChatCmd> {
        match self {
            Self::Chat(chat) => Some(chat),
            _ => None,
        }
    }

    pub(crate) fn into_chat(self) -> Result<ChatCmd, Self> {
        match self {
            Self::Chat(chat) => Ok(*chat),
            other => Err(other),
        }
    }

    pub(crate) fn scheduler_stats(reply: ResponseTx<SchedulerStatsJs>) -> Self {
        Self::SchedulerStats { reply }
    }

    pub(crate) fn into_scheduler_stats(self) -> Result<ResponseTx<SchedulerStatsJs>, Self> {
        match self {
            Self::SchedulerStats { reply } => Ok(reply),
            other => Err(other),
        }
    }
}

impl<C> From<C> for ModelCommand<C> {
    fn from(command: C) -> Self {
        Self::Family(command)
    }
}

impl<C: FromTrainCmd> FromTrainCmd for ModelCommand<C> {
    fn from_train(command: TrainCmd) -> Self {
        Self::Family(C::from_train(command))
    }
}

/// Shared dispatch for a thread without an active hybrid scheduler, including
/// random-weight checkpoint creation. The scheduler supplies its live telemetry
/// and owner-aware chat hook before reaching this fallback.
pub(crate) fn handle_model_command<B, C>(backend: &mut B, command: ModelCommand<C>)
where
    B: ChatBackend,
    C: FamilyCommand<B>,
{
    match command {
        ModelCommand::Chat(chat) => handle_chat_cmd(backend, *chat),
        ModelCommand::SchedulerStats { reply } => {
            let _ = reply.send(Ok(
                crate::engine::scheduler::SchedulerStats::default().to_js()
            ));
        }
        ModelCommand::Family(command) => command.execute(backend),
    }
}
