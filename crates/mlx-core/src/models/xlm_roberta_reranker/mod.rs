//! Inference-only XLM-R cross-encoder ranking.
pub(crate) mod config;
mod inference;
pub(crate) mod weights;
use super::rerank_runtime::{Runtime, Task};
use config::{Options, invalid};
use inference::Model;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::path::PathBuf;

#[napi]
pub struct XlmRobertaRerankerModel {
    runtime: Runtime,
}
#[napi]
impl XlmRobertaRerankerModel {
    #[napi]
    pub fn load<'env>(
        env: &'env Env,
        model_path: String,
        options_json: Option<String>,
    ) -> Result<PromiseRaw<'env, XlmRobertaRerankerModel>> {
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
    pub fn start_rerank(&self, request_json: String) -> Result<XlmRobertaRerankerTask> {
        Ok(XlmRobertaRerankerTask {
            task: self.runtime.start(request_json)?,
        })
    }
    #[napi]
    pub fn dispose<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
        self.runtime.dispose(env)
    }
}
#[napi]
pub struct XlmRobertaRerankerTask {
    task: Task,
}
#[napi]
impl XlmRobertaRerankerTask {
    #[napi]
    pub fn cancel(&self) {
        self.task.cancel();
    }
    #[napi]
    pub fn result<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, String>> {
        self.task.result(env)
    }
}
