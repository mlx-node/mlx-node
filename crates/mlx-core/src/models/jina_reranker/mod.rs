//! Inference-only listwise ranking. No chat sessions or process-global model state.
pub(crate) mod config;
mod inference;
pub(crate) mod weights;
use super::rerank_runtime::{Backend, Request, Runtime, Task};
use config::{Options, invalid};
use inference::Model;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::{path::PathBuf, sync::atomic::AtomicBool};

impl Backend for Model {
    fn info(&self) -> &serde_json::Value {
        &self.info
    }
    fn rerank(
        &self,
        request: Request,
        cancel: &AtomicBool,
        shutdown: &AtomicBool,
    ) -> Result<serde_json::Value> {
        self.rerank(request, cancel, shutdown)
    }
}
#[napi]
pub struct JinaRerankerModel {
    runtime: Runtime,
}
#[napi]
impl JinaRerankerModel {
    #[napi]
    pub fn load<'env>(
        env: &'env Env,
        model_path: String,
        options_json: Option<String>,
    ) -> Result<PromiseRaw<'env, JinaRerankerModel>> {
        let options: Options =
            serde_json::from_str(options_json.as_deref().unwrap_or("{}")).map_err(invalid)?;
        env.spawn_future(async move {
            Ok(Self {
                runtime: Runtime::load(move || Model::load(&PathBuf::from(model_path), options))
                    .await?,
            })
        })
    }
    #[napi(getter)]
    pub fn info_json(&self) -> String {
        self.runtime.info.clone()
    }
    #[napi]
    pub fn start_rerank(&self, request_json: String) -> Result<JinaRerankerTask> {
        Ok(JinaRerankerTask {
            task: self.runtime.start(request_json)?,
        })
    }
    #[napi]
    pub fn dispose<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
        self.runtime.dispose(env)
    }
}
#[napi]
pub struct JinaRerankerTask {
    task: Task,
}
#[napi]
impl JinaRerankerTask {
    #[napi]
    pub fn cancel(&self) {
        self.task.cancel();
    }
    #[napi]
    pub fn result<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, String>> {
        self.task.result(env)
    }
}
