//! Product chat lane: one flat whole-turn session on its own model thread
//! (D-21). The store's S15 admission runs at load and at every turn start
//! (`Store::begin_turn`), and its bounded wired lease covers the turn.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use napi::{Error, Result};
use napi_derive::napi;

use super::attention::FakeQuant;
use super::model::{Session, Trace};
use super::store::{CachePlan, Store};
use crate::array::{DType, MxArray};
use crate::engine::backend::{ChatBackend, DecodeStep, ResetScope, SaveStateArgs, TurnSetup};
use crate::engine::model_command::{ModelCommand, handle_model_command};
use crate::engine::params::ThinkingPolicy;
use crate::model_thread::{LoopControl, ModelThread};
use crate::moe_stream::budget::{Admission, Limits};
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;
use crate::utils::gguf::{GgufMetaValue, embedded_gpt2_tokenizer_json, parse_gguf_reader};

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("deepseek_v41: {msg}"))
}

/// Every shard of the model at `path`: any shard of a `-NNNNN-of-MMMMM.gguf`
/// set (siblings found by name), a directory holding one such set or one
/// `.gguf`, or a single-file GGUF.
pub(crate) fn shard_paths(path: &Path) -> Result<Vec<PathBuf>> {
    let path = if path.is_dir() {
        let mut v: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|e| bad(format!("'{}': {e}", path.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
            .collect();
        v.sort();
        let first = v
            .iter()
            .find(|p| p.to_string_lossy().contains("-00001-of-"));
        match (first, v.len()) {
            (Some(f), _) => f.clone(),
            (None, 1) => v.swap_remove(0),
            _ => return Err(bad(format!("'{}': no single GGUF set", path.display()))),
        }
    } else {
        path.to_path_buf()
    };
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| bad("non-UTF8 model path"))?;
    let split = name
        .strip_suffix(".gguf")
        .and_then(|s| s.rsplit_once("-of-"))
        .and_then(|(head, total)| Some((head.rsplit_once('-')?, total)))
        .filter(|((_, idx), total)| {
            [idx, total]
                .iter()
                .all(|s| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit()))
        });
    let Some(((prefix, _), total)) = split else {
        return Ok(vec![path]);
    };
    let n: usize = total
        .parse()
        .map_err(|_| bad(format!("shard count in '{name}'")))?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let paths: Vec<PathBuf> = (1..=n)
        .map(|i| dir.join(format!("{prefix}-{i:05}-of-{total}.gguf")))
        .collect();
    if let Some(missing) = paths.iter().find(|p| !p.is_file()) {
        return Err(bad(format!("shard '{}' missing", missing.display())));
    }
    Ok(paths)
}

/// The GGUF-embedded tokenizer and chat template of shard 1 (header only,
/// F_NOCACHE), built in memory.
pub(crate) fn gguf_tokenizer(first: &Path) -> Result<(Qwen3Tokenizer, u32)> {
    let file = crate::moe_stream::io::open_nocache(first)?;
    let g = parse_gguf_reader(std::io::BufReader::new(file))?;
    let json = embedded_gpt2_tokenizer_json(&g.metadata)?
        .ok_or_else(|| bad("GGUF lacks a supported embedded tokenizer"))?;
    let bytes = serde_json::to_vec(&json).map_err(|e| bad(e.to_string()))?;
    let tok = tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| bad(e.to_string()))?;
    let template = match g.metadata.get("tokenizer.chat_template") {
        Some(GgufMetaValue::String(t)) => Some(t.clone()),
        _ => None,
    };
    let eos = g
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(GgufMetaValue::as_u32)
        .ok_or_else(|| bad("GGUF lacks tokenizer.ggml.eos_token_id"))?;
    Ok((Qwen3Tokenizer::from_tokenizer(tok, template), eos))
}

/// Admission and bank state as of the last turn edge (host values only).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Residency {
    pub admission: Option<Admission>,
    /// Bank pages resident after the last turn made the bank volatile: a
    /// second live wired lease (another model's turn) keeps them wired.
    pub bank_resident: u64,
    pub switches: u32,
    pub in_turn: bool,
}

/// One call seen by a test probe: start position, ids, engram row ids.
pub(crate) type ProbeCall = (usize, Vec<u32>, Vec<u32>);

pub(crate) struct Inner {
    pub(crate) store: Store,
    sess: Session,
    tokenizer: Arc<Qwen3Tokenizer>,
    eos: u32,
    saved: Vec<u32>,
    cancel: Option<Arc<AtomicBool>>,
    in_turn: bool,
    residency: Arc<Mutex<Residency>>,
    /// Test hook: every forward call, and per prefill the index of its call
    /// and the session's whole prompt (history + call ids).
    pub(crate) probe: Option<(Vec<ProbeCall>, Vec<(usize, Vec<u32>)>)>,
}

impl Inner {
    /// The product load: every shard of `path`, f32 activations with the
    /// reference fake quant, dense mode from `MLX_DSV41_DENSE` (q8 by
    /// default), S15 admission from the environment (auto budget, D-40).
    pub(crate) fn load(path: &Path, residency: Arc<Mutex<Residency>>) -> Result<Self> {
        let paths = shard_paths(path)?;
        let (tokenizer, eos) = gguf_tokenizer(&paths[0])?;
        let store = Store::open_with(
            &paths,
            DType::Float32,
            FakeQuant::ON,
            CachePlan::Admit(Limits::from_env()?),
        )?;
        Ok(Self::new(store, tokenizer, eos, residency))
    }

    pub(crate) fn new(
        store: Store,
        tokenizer: Qwen3Tokenizer,
        eos: u32,
        residency: Arc<Mutex<Residency>>,
    ) -> Self {
        let sess = store.model.new_session();
        let mut inner = Self {
            store,
            sess,
            tokenizer: Arc::new(tokenizer),
            eos,
            saved: Vec::new(),
            cancel: None,
            in_turn: false,
            residency,
            probe: None,
        };
        inner.publish(None);
        inner
    }

    fn publish(&mut self, bank_resident: Option<u64>) {
        let mut r = self
            .residency
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        r.admission = self.store.admission();
        r.switches = self.store.switches().len() as u32;
        r.in_turn = self.in_turn;
        if let Some(b) = bank_resident {
            r.bank_resident = b;
        }
    }

    fn begin(&mut self) -> Result<()> {
        if !self.in_turn {
            self.store.begin_turn()?;
            self.in_turn = true;
            self.publish(None);
        }
        Ok(())
    }

    fn end(&mut self) {
        if !self.in_turn {
            return;
        }
        self.in_turn = false;
        {
            let _ctx = StreamContext::new(Stream::generation());
            if let Err(e) = self.store.end_turn() {
                tracing::warn!("deepseek_v41 end of turn: {e}");
            }
        }
        let resident = self.store.bank.resident_bytes().unwrap_or(0);
        self.publish(Some(resident));
    }

    /// One call on the session: f32 logits of its last row `[1, vocab]`.
    fn call(&mut self, ids: &[u32], cancel: Option<&dyn Fn(usize) -> bool>) -> Result<MxArray> {
        let start = self.sess.tokens.len();
        let mut trace = self.probe.as_ref().map(|_| Trace::default());
        let logits = self
            .store
            .model
            .forward_last(&mut self.sess, ids, cancel, trace.as_mut())?;
        if let (Some((calls, _)), Some(t)) = (self.probe.as_mut(), trace) {
            calls.push((start, ids.to_vec(), t.engram_rows));
        }
        Ok(logits)
    }
}

pub(crate) struct Step<'a>(&'a mut Inner);

impl DecodeStep for Step<'_> {
    fn forward(&mut self, ids: &MxArray) -> Result<(MxArray, bool)> {
        let token = ids.item_at_int32(0)? as u32;
        self.forward_with_token(ids, token)
    }

    fn forward_with_token(&mut self, _ids: &MxArray, token: u32) -> Result<(MxArray, bool)> {
        Ok((self.0.call(&[token], None)?, false))
    }
}

impl ChatBackend for Inner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        Ok(Arc::clone(&self.tokenizer))
    }

    fn family_name(&self) -> &'static str {
        "deepseek_v41"
    }

    fn session_eos_id(&self, _tok: &Qwen3Tokenizer) -> Result<u32> {
        Ok(self.eos)
    }

    fn policy(&self) -> ThinkingPolicy {
        ThinkingPolicy::TemplateHonoring
    }

    fn cached_token_history(&self) -> &[u32] {
        &self.saved
    }

    fn reset_caches(&mut self, _scope: ResetScope) -> Result<()> {
        self.sess = self.store.model.new_session();
        self.saved.clear();
        Ok(())
    }

    /// The session keeps every forwarded token (`Session::tokens`), which the
    /// engram hash of the next call reads: a hit continues on the same
    /// history (D-27), anything else replays the whole prompt.
    fn verify_cache_prefix(&self, tokens: &[u32], reuse: bool) -> usize {
        let h = &self.saved;
        if reuse
            && !h.is_empty()
            && self.sess.tokens == *h
            && tokens.len() > h.len()
            && tokens.starts_with(h)
        {
            h.len()
        } else {
            0
        }
    }

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        if args.reuse_cache {
            self.saved = self.sess.tokens.clone();
        } else {
            self.sess = self.store.model.new_session();
            self.saved.clear();
        }
    }

    fn eval_caches(&self) -> Result<()> {
        Ok(())
    }

    /// `None`: the engine's own context would request the whole recommended
    /// working set; the store holds a lease bounded by the admission instead
    /// (`fixed + cache`, taken in `Store::begin_turn`).
    fn wired_limit_bytes(&self) -> Option<usize> {
        None
    }

    fn set_turn_cancel_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        let ending = flag.is_none();
        self.cancel = flag;
        if ending {
            self.end();
        }
    }

    fn prefill(&mut self, tokens: &[u32], stream: Stream) -> Result<MxArray> {
        if let Some((calls, prompts)) = self.probe.as_mut() {
            let full = self.sess.tokens.iter().chain(tokens).copied().collect();
            prompts.push((calls.len(), full));
        }
        self.begin()?;
        let _ctx = StreamContext::new(stream);
        let flag = self.cancel.clone();
        let cancel = move |_: usize| flag.as_ref().is_some_and(|f| f.load(Ordering::Relaxed));
        let out = self.call(tokens, Some(&cancel));
        match out {
            Err(_) if cancel(0) => Err(Error::from_reason("prefill cancelled")),
            o => o,
        }
    }

    type Decode<'a>
        = Step<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Step<'_>> {
        Ok(Step(self))
    }
}

#[napi(object)]
pub struct DeepseekV41ContextLimits {
    /// min(GGUF context, 16384): the v1 limit the admission budgets for.
    pub effective_window_tokens: u32,
    pub trained_window_tokens: u32,
}

/// The load-time and latest per-turn S15 admission, and the expert bank.
#[napi(object)]
pub struct DeepseekV41ResidencyInfo {
    /// `exact`, `f32`, `q8` or `mixed` (`MLX_DSV41_DENSE`).
    pub dense_mode: String,
    pub dense_bytes: f64,
    pub physical_bytes: Option<f64>,
    /// Other processes' (and other models') bytes at the last admission.
    pub others_bytes: Option<f64>,
    /// Fixed working set: dense, staging, KV and compute at the context
    /// limit, I/O buffers, driver and MLX pool reserves.
    pub fixed_bytes: Option<f64>,
    pub expert_cache_slots: Option<u32>,
    pub expert_cache_bytes: Option<f64>,
    pub staging_layers: Option<u32>,
    /// Bounded wired-limit request of each turn (`fixed + cache`).
    pub wired_limit_bytes: Option<f64>,
    /// Bank pages still resident after the last turn made the bank volatile;
    /// non-zero while another live wired lease keeps them wired.
    pub bank_resident_after_turn_bytes: f64,
    pub bank_switches: u32,
    pub in_turn: bool,
    /// MLX TF32 GEMMs (`MLX_ENABLE_TF32`, process-wide); the reference
    /// numerics were gated with it off.
    pub tf32: bool,
}

/// DeepSeek-V4.1 Flash (GGUF `deepseek41`) with SSD-streamed routed experts:
/// text-only chat, one session at a time, on one owned model thread.
#[napi]
pub struct DeepseekV41Model {
    thread: ModelThread<ModelCommand>,
    limits: (u32, u32),
    dense: (String, u64),
    residency: Arc<Mutex<Residency>>,
}

#[napi]
impl DeepseekV41Model {
    /// `model_path`: any shard of the split GGUF, its directory, or a
    /// single-file GGUF.
    #[napi]
    pub async fn load(model_path: String) -> Result<Self> {
        let residency = Arc::new(Mutex::new(Residency::default()));
        let shared = Arc::clone(&residency);
        let (thread, ready) = ModelThread::spawn_with_scheduler(
            move || {
                let inner = Inner::load(Path::new(&model_path), shared)?;
                let s = &inner.store;
                let limits = (s.size.ctx as u32, s.model.cfg.context_length as u32);
                let dense = (format!("{:?}", s.dense_mode).to_lowercase(), s.dense_bytes);
                Ok((inner, (limits, dense)))
            },
            |inner: &mut Inner, rx| match rx.blocking_recv() {
                Some(cmd) => {
                    handle_model_command(inner, cmd);
                    LoopControl::Continue
                }
                None => LoopControl::Break,
            },
        );
        let (limits, dense) = ready
            .await
            .map_err(|e| Error::from_reason(e.to_string()))??;
        Ok(Self {
            thread,
            limits,
            dense,
            residency,
        })
    }

    #[napi]
    pub fn context_limits(&self) -> DeepseekV41ContextLimits {
        DeepseekV41ContextLimits {
            effective_window_tokens: self.limits.0,
            trained_window_tokens: self.limits.1,
        }
    }

    #[napi]
    pub fn max_concurrent_sequences(&self) -> u32 {
        1
    }

    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        false
    }

    #[napi]
    pub fn residency_info(&self) -> DeepseekV41ResidencyInfo {
        let r = *self
            .residency
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let a = r.admission;
        DeepseekV41ResidencyInfo {
            dense_mode: self.dense.0.clone(),
            dense_bytes: self.dense.1 as f64,
            physical_bytes: a.map(|a| a.physical as f64),
            others_bytes: a.map(|a| a.others as f64),
            fixed_bytes: a.map(|a| a.fixed as f64),
            expert_cache_slots: a.map(|a| a.cache_slots as u32),
            expert_cache_bytes: a.map(|a| a.cache_bytes() as f64),
            staging_layers: a.map(|a| a.staging_layers as u32),
            wired_limit_bytes: a.map(|a| a.wired_limit() as f64),
            bank_resident_after_turn_bytes: r.bank_resident as f64,
            bank_switches: r.switches,
            in_turn: r.in_turn,
            tf32: std::env::var_os("MLX_ENABLE_TF32").is_none_or(|v| v != "0"),
        }
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: DeepseekV41Model,
    thread_cmd: crate::engine::model_command::ModelCommand,
    thread: direct,
    image_guard: text_only,
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
}
