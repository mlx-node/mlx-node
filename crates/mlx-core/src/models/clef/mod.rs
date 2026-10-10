//! Text CLEF decision models sharing Qwen's native model-owning thread.
pub(crate) mod encoding;
pub(crate) mod head;
use crate::models::qwen3_5::{
    model::{Qwen3_5Model, Qwen35FamilyCommand},
    persistence,
};
use napi::{Error, Result};
use napi_derive::napi;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[napi]
pub struct ClefCancellation {
    pub(crate) flag: Arc<AtomicBool>,
}
#[napi]
impl ClefCancellation {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }
    #[napi]
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }
}
impl Default for ClefCancellation {
    fn default() -> Self {
        Self::new()
    }
}

/// Native implementation; use the typed ClefModel exported by @mlx-node/lm.
#[napi]
pub struct ClefModel {
    backbone: Qwen3_5Model,
}
#[napi]
impl ClefModel {
    #[napi]
    pub async fn load(path: String) -> Result<Self> {
        Ok(Self {
            backbone: persistence::load_clef_with_thread(&path).await?,
        })
    }
    /// Receives raw JSON to retain HTTP object ordering and number spelling.
    #[napi]
    pub async fn decide_json(
        &self,
        request: String,
        cancellation: Option<&ClefCancellation>,
    ) -> Result<String> {
        if request.len() > 10 * 1024 * 1024 {
            return Err(encoding::invalid("request exceeds 10 MiB"));
        }
        let request = serde_json::from_str::<encoding::Request>(&request)
            .map_err(|e| encoding::invalid(e.to_string()))?;
        let cancelled = cancellation.map(|c| c.flag.clone()).unwrap_or_default();
        crate::model_thread::send_and_await(&self.backbone.thread, |reply| {
            Qwen35FamilyCommand::DecideClef {
                request,
                cancelled,
                reply,
            }
        })
        .await
    }
}

pub(crate) fn check_cancel(flag: &AtomicBool) -> Result<()> {
    if flag.load(Ordering::Acquire) {
        Err(Error::from_reason("CLEF inference cancelled"))
    } else {
        Ok(())
    }
}
