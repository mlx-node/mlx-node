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
use super::stats::{self, Counters};
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
#[derive(Clone, Debug, Default)]
pub(crate) struct Residency {
    pub admission: Option<Admission>,
    /// Bank pages resident after the last turn made the bank volatile: a
    /// second live wired lease (another model's turn) keeps them wired.
    pub bank_resident: u64,
    pub switches: u32,
    pub in_turn: bool,
    /// The last saved turn: its rendered prompt ids and generated ids.
    pub turn_ids: Option<(Vec<u32>, Vec<u32>)>,
}

/// One turn's timings and counters for the `MLX_DSV41_STATS` line.
struct TurnStats {
    t0: std::time::Instant,
    c0: Counters,
    swap0: Option<u64>,
    level_min: Option<u64>,
    prompt: usize,
    ttft_s: f64,
    decode_s: Vec<f64>,
    /// Counters and clock at the end of the prefill (decode-only terms).
    dec0: Option<(Counters, std::time::Instant)>,
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
    turn_stats: Option<TurnStats>,
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
            turn_stats: None,
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

    /// The last saved turn's (prompt ids, generated ids).
    #[cfg(test)]
    pub(crate) fn turn_ids(&self) -> Option<(Vec<u32>, Vec<u32>)> {
        let r = self
            .residency
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        r.turn_ids.clone()
    }

    /// With `MLX_DSV41_STATS` set: evaluates the call's logits (timing) and
    /// appends one stats line for the turn so far, with decode-only terms.
    fn emit_stats(&mut self, logits: &MxArray, prefill: bool, t_call: std::time::Instant) {
        let Some(ts) = self.turn_stats.as_mut() else {
            return;
        };
        logits.eval();
        let dt = t_call.elapsed().as_secs_f64();
        let c_now = Counters::now(&self.store.bank);
        if prefill {
            ts.ttft_s = ts.t0.elapsed().as_secs_f64();
            ts.dec0 = Some((c_now, std::time::Instant::now()));
        } else {
            ts.decode_s.push(dt);
        }
        let (lvl, swap) = stats::host_pressure();
        ts.level_min = match (ts.level_min, lvl) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let c = c_now.since(&ts.c0);
        let (_, peak, _) = stats::rusage();
        let swap_d = swap.zip(ts.swap0).map_or(0, |(a, b)| a as i64 - b as i64);
        let mut line = stats::line(
            "chat",
            ts.prompt,
            1 + ts.decode_s.len(),
            ts.ttft_s,
            &ts.decode_s,
            &c,
            ts.t0.elapsed().as_secs_f64(),
            c.fill_bytes + c.engram_bytes,
            self.store.cache_slots(),
            peak,
            swap_d,
            ts.level_min,
        );
        if let (Some((cd0, td0)), Some(obj)) = (ts.dec0, line.as_object_mut()) {
            let d = c_now.since(&cd0);
            let n = ts.decode_s.len().max(1) as f64;
            let wall = td0.elapsed().as_secs_f64();
            let (io, eng) = (d.fill_ns as f64 * 1e-9, d.engram_ns as f64 * 1e-9);
            obj.insert("decode_tok".into(), ts.decode_s.len().into());
            obj.insert(
                "decode_tok_s_mean".into(),
                (ts.decode_s.len() as f64 / wall.max(1e-9)).into(),
            );
            obj.insert("decode_t_c_ms".into(), ((wall - io - eng) * 1e3 / n).into());
            obj.insert("decode_t_io_ms".into(), (io * 1e3 / n).into());
            obj.insert("decode_t_engram_ms".into(), (eng * 1e3 / n).into());
            obj.insert(
                "decode_t_router_wait_ms".into(),
                (d.router_ns as f64 * 1e-6 / n).into(),
            );
            obj.insert(
                "decode_h".into(),
                (d.hits as f64 / d.uses.max(1) as f64).into(),
            );
            obj.insert(
                "decode_read_gbps".into(),
                (d.fill_bytes as f64 / 1e9 / io.max(1e-9)).into(),
            );
        }
        if let Err(e) = stats::emit(&line) {
            tracing::warn!("deepseek_v41 stats: {e}");
        }
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
        let t = std::time::Instant::now();
        let logits = self.0.call(&[token], None)?;
        self.0.emit_stats(&logits, false, t);
        Ok((logits, false))
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
        {
            let mut r = self
                .residency
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            r.turn_ids = Some((args.save_tokens.to_vec(), args.generated_tokens.to_vec()));
        }
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
        let t0 = std::time::Instant::now();
        self.turn_stats = std::env::var_os("MLX_DSV41_STATS").map(|_| TurnStats {
            t0,
            c0: Counters::now(&self.store.bank),
            swap0: stats::host_pressure().1,
            level_min: None,
            prompt: tokens.len(),
            ttft_s: 0.0,
            decode_s: Vec::new(),
            dec0: None,
        });
        self.begin()?;
        let _ctx = StreamContext::new(stream);
        let flag = self.cancel.clone();
        let cancel = move |_: usize| flag.as_ref().is_some_and(|f| f.load(Ordering::Relaxed));
        let out = self.call(tokens, Some(&cancel));
        match out {
            Err(_) if cancel(0) => Err(Error::from_reason("prefill cancelled")),
            Ok(logits) => {
                self.emit_stats(&logits, true, t0);
                Ok(logits)
            }
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

/// The last saved turn of the session (S18 CLI-vs-Rust id gate).
#[napi(object)]
pub struct DeepseekV41TurnTokenIds {
    /// The rendered prompt the turn was given (whole conversation).
    pub prompt_ids: Vec<u32>,
    /// The generated ids, the stop token included when one ended the turn.
    pub output_ids: Vec<u32>,
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

    /// Prompt and generated ids of the last completed turn, if any.
    #[napi]
    pub fn last_turn_token_ids(&self) -> Option<DeepseekV41TurnTokenIds> {
        let r = self
            .residency
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        r.turn_ids.as_ref().map(|(p, o)| DeepseekV41TurnTokenIds {
            prompt_ids: p.clone(),
            output_ids: o.clone(),
        })
    }

    #[napi]
    pub fn residency_info(&self) -> DeepseekV41ResidencyInfo {
        let r = self
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
