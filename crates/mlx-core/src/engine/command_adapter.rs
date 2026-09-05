//! Shared adapters for a family's typed model-thread commands.
//!
//! A family declares `Chat(ChatCmd)` (or `Chat(Box<ChatCmd>)`) and
//! `SchedulerStats { reply }`; its remaining variants stay ordered barriers.
//! Adding a family does not require copying the command-view implementations.

macro_rules! impl_scheduler_command {
    ($command:ty, $storage:ident) => {
        impl $crate::engine::cmd::FromChatCmd for $command {
            fn from_chat(command: $crate::engine::cmd::ChatCmd) -> Self {
                Self::Chat($crate::engine::command_adapter::wrap_chat!(
                    command, $storage
                ))
            }
        }

        impl $crate::engine::hybrid_scheduler::HybridSchedulerCommand for $command {
            fn as_chat(&self) -> Option<&$crate::engine::cmd::ChatCmd> {
                match self {
                    Self::Chat(chat) => Some(chat),
                    _ => None,
                }
            }

            fn into_chat(self) -> ::std::result::Result<$crate::engine::cmd::ChatCmd, Self> {
                match self {
                    Self::Chat(chat) => Ok($crate::engine::command_adapter::unwrap_chat!(
                        chat, $storage
                    )),
                    other => Err(other),
                }
            }

            fn scheduler_stats(
                reply: $crate::model_thread::ResponseTx<$crate::engine::SchedulerStatsJs>,
            ) -> Self {
                Self::SchedulerStats { reply }
            }

            fn into_scheduler_stats(
                self,
            ) -> ::std::result::Result<
                $crate::model_thread::ResponseTx<$crate::engine::SchedulerStatsJs>,
                Self,
            > {
                match self {
                    Self::SchedulerStats { reply } => Ok(reply),
                    other => Err(other),
                }
            }
        }
    };
}

macro_rules! unwrap_chat {
    ($chat:ident, direct) => {
        $chat
    };
    ($chat:ident, boxed) => {
        *$chat
    };
}

pub(crate) use impl_scheduler_command;
pub(crate) use unwrap_chat;

macro_rules! wrap_chat {
    ($chat:ident, direct) => {
        $chat
    };
    ($chat:ident, boxed) => {
        Box::new($chat)
    };
}
pub(crate) use wrap_chat;
