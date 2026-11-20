//! Chat backend on one model thread, in one of two modes fixed at load:
//! flat (`MLX_SERVE_FORCE_SERIAL=1`, or no wave row cap): one whole-turn
//! sequence, as below; paged: the shared scheduler over owner cells and the
//! paged pool (`sched.rs`).
//!
//! ```text
//! verify  ─► Conv::verify: longest reusable prefix (sequence cut │ ladder entry) ─► plan
//! prefill ─► store lease (begin_turn) ─► Driver chunks on the grid ─► last-row logits
//!            cancelled: the finished chunks stay, the fail-closed reset keeps them
//! decode  ─► one row per step (RowKind::Decode), or an MTP turn (`spec.rs`)
//!            with the MTP layer, each trunk row closes the draft row before it
//! save    ─► history = the sequence's tokens, a Turn ladder entry
//! disk    ─► flat: the prompt's states and the turn state publish to the stored states
//!            (`persist.rs`); a prompt past the ladder's reach reads a stored state back
//! end     ─► store lease released (end_turn) when the cancel flag is cleared
//! ```
//!
//! The admission picks the context and the prefill chunk at load; every turn
//! re-admits at its start and holds a wired lease bounded by it.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::cell::Cell;
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use napi::{Error, Result};
use napi_derive::napi;

use super::finalize::{self, Hy4Tags};
use super::ladder::{Kind, Ladder};
use super::mtp::{LagRunner, Lagged, MtpLayer};
use super::persist::{self, BlockRuns, Disk, FlatTier};
use super::persist_writer::{Closer, Telemetry};
use super::prefill::{Driver, Grid, Plan, Prefilled, Runner, Seq};
use super::prefix_state::PrefixState;
use super::sched::{self, Paged, pass_keys};
use super::spec;
use super::stats::{self, Counters, Line, WaveLog, Waves};
use super::store::{Options, Store};
use crate::array::MxArray;
use crate::disk_tier::{DiskTierOptions, TierHandle};
use crate::engine::backend::{
    ChatBackend, DecodeStep, FinalizeArgs, MtpBackend, MtpTurnSetup, ResetScope, SaveStateArgs,
    TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::hybrid_scheduler::HybridSchedulerState;
use crate::engine::model_command::{ModelCommand, handle_model_command};
use crate::engine::params::{
    ChatParams, ModelGenerationDefaults, ThinkingPolicy, apply_generation_defaults,
    extract_chat_params,
};
use crate::engine::plan::{ExecutionPlan, PagedAttentionPlan};
use crate::engine::types::{ChatConfig, ChatResult};
use crate::engine::warmup::WarmupReport;
use crate::model_thread::{LoopControl, ModelThread};
use crate::models::deepseek_v41::chat::{gguf_tokenizer, shard_paths};
use crate::moe_stream::budget::Admission;
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;

pub const LABEL: &str = "HyV4Flash";
/// `<｜hy_end｜>`: the only stop id (`generation_config.json` `eos_token_id`).
pub const EOS: u32 = 120_025;
/// `<｜hy_start｜>`, the opening of every turn.
pub const HY_START: u32 = 120_000;
/// Deepest MTP draft a request may ask for.
pub const MTP_DEPTH_MAX: usize = 3;
pub const TEMPERATURE: f64 = 0.9;
pub const TOP_P: f64 = 1.0;
/// The engine's fail-closed arms key on this exact reason.
pub const PREFILL_CANCELLED: &str = "prefill cancelled";
/// Warmup prefill length: splits at replay row 64 into 64 + 32 rows so both
/// `qmm_t` and ragged/small paths compile.
const HY4_WARMUP_SLICE_ROWS: u32 = 96;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("hy_v4_flash: {msg}"))
}

/// `generation_config.json`: sampled at T 0.9, top_p 1.0, no top_k.
pub fn generation_defaults() -> ModelGenerationDefaults {
    ModelGenerationDefaults {
        temperature: Some(TEMPERATURE),
        top_p: Some(TOP_P),
        do_sample: Some(true),
        ..ModelGenerationDefaults::default()
    }
}

/// The request over the model defaults; MTP depth at most `MTP_DEPTH_MAX`,
/// never adaptive.
pub(crate) fn resolve(config: &ChatConfig, defaults: &ModelGenerationDefaults) -> ChatParams {
    let mut config = config.clone();
    apply_generation_defaults(&mut config, defaults);
    let mut params = extract_chat_params(&config);
    params.mtp_depth = params.mtp_depth.min(MTP_DEPTH_MAX);
    params.mtp_adaptive_depth = false;
    params
}

/// The GGUF's `tokenizer.ggml.eos_token_id` must be `EOS`.
pub fn check_eos(gguf_eos: u32) -> Result<u32> {
    if gguf_eos != EOS {
        return Err(bad(format!(
            "tokenizer.ggml.eos_token_id is {gguf_eos}, expected {EOS}"
        )));
    }
    Ok(EOS)
}

/// Ids of `<｜hy_start｜>assistant`: the assistant opening the replay row
/// sits before.
pub fn assistant_open(tok: &Qwen3Tokenizer) -> Result<Vec<u32>> {
    let start = tok
        .id_to_token(HY_START)
        .filter(|t| t.starts_with("<｜hy_start") && t.ends_with("｜>"))
        .ok_or_else(|| bad(format!("token {HY_START} is not <｜hy_start｜>")))?;
    let ids = tok.encode_sync(&format!("{start}assistant"), Some(false))?;
    match ids.first() {
        Some(&HY_START) if ids.len() > 1 => Ok(ids),
        _ => Err(bad(format!("{start}assistant encodes as {ids:?}"))),
    }
}

/// One conversation's sequence, its RAM ladder, the saved history and the
/// reuse plan `verify` made for the next prefill.
pub struct Conv<S: Seq> {
    seq: S,
    ladder: Ladder<S::Snap>,
    saved: Vec<u32>,
    plan: Cell<Option<Plan>>,
    /// A cancelled prefill left the sequence at a chunk end: the reset that
    /// follows inside the same turn keeps it.
    kept: bool,
    /// The row the last prefill started at.
    from: usize,
    /// The cache salt of the turns that built the rows held.
    salt: u64,
}

impl<S: Seq> Conv<S> {
    pub fn new(seq: S) -> Self {
        Self {
            seq,
            ladder: Ladder::new(),
            saved: Vec::new(),
            plan: Cell::new(None),
            kept: false,
            from: 0,
            salt: 0,
        }
    }

    /// The row the last finished prefill started at.
    pub fn prefilled_from(&self) -> usize {
        self.from
    }

    pub fn seq(&self) -> &S {
        &self.seq
    }

    pub fn seq_mut(&mut self) -> &mut S {
        &mut self.seq
    }

    pub fn ladder(&self) -> &Ladder<S::Snap> {
        &self.ladder
    }

    /// The committed history (`cached_token_history`).
    pub fn history(&self) -> &[u32] {
        &self.saved
    }

    pub fn kept(&self) -> bool {
        self.kept
    }

    /// Starts over on `seq`, rows of the cache salt `salt`: nothing else
    /// is kept.
    pub fn replace(&mut self, seq: S, salt: u64) {
        *self = Self::new(seq);
        self.salt = salt;
    }

    /// Whether the rows held were built under `salt`: another salt reuses
    /// none of them.
    pub fn reuses(&self, salt: u64) -> bool {
        self.salt == salt
    }

    /// The next prefill runs under `salt`: rows another salt built go.
    pub fn own(&mut self, salt: u64) -> Result<()> {
        if self.salt == salt {
            return Ok(());
        }
        self.salt = salt;
        self.kept = false;
        self.clear()
    }

    /// Rows of `tokens` the next prefill does not run again (`reuse` false:
    /// none); always below `tokens.len()`.
    pub fn verify<R: Runner<S>>(
        &self,
        runner: &R,
        grid: &Grid,
        tokens: &[u32],
        reuse: bool,
    ) -> usize {
        let driver = Driver {
            runner,
            grid,
            hidden: false,
        };
        let plan = driver.plan(&self.seq, &self.ladder, tokens, reuse);
        let rows = plan.reuse();
        self.plan.set(Some(plan));
        rows
    }

    /// Runs `tokens` (the prompt past the verified rows, or past the
    /// sequence) in chunks. A cancel keeps the finished chunks and fails
    /// with `PREFILL_CANCELLED`. `on_chunk` sees each finished chunk's rows.
    pub fn prefill<R: Runner<S>>(
        &mut self,
        runner: &R,
        grid: &Grid,
        tokens: &[u32],
        cancel: &dyn Fn(usize) -> bool,
        on_chunk: &mut dyn FnMut(Range<usize>),
    ) -> Result<R::Out> {
        let driver = Driver {
            runner,
            grid,
            hidden: false,
        };
        let plan = self.plan.take();
        self.kept = false;
        self.saved.clear();
        let out = driver.prefill(
            &mut self.seq,
            &mut self.ladder,
            plan,
            tokens,
            cancel,
            &mut |rows: Range<usize>, _: &R::Out| {
                on_chunk(rows);
                Ok(())
            },
        );
        match out {
            Ok(Prefilled::Done { out, from }) => {
                self.from = from;
                Ok(out)
            }
            Ok(Prefilled::Cancelled { .. }) => {
                self.kept = true;
                Err(Error::from_reason(PREFILL_CANCELLED))
            }
            Err(_) if cancel(0) => Err(Error::from_reason(PREFILL_CANCELLED)),
            Err(e) => Err(e),
        }
    }

    /// End of a turn: with `reuse` the sequence is the history and gets a
    /// Turn entry; without, the conversation is dropped.
    pub fn save(&mut self, reuse: bool) -> Result<()> {
        self.kept = false;
        self.plan.set(None);
        if !reuse {
            return self.clear();
        }
        self.saved.clear();
        self.saved.extend_from_slice(self.seq.tokens());
        if !self.seq.is_empty() {
            self.ladder.put(Kind::Turn, self.seq.snap());
        }
        Ok(())
    }

    /// `PrefixMiss` drops the history only (the prefill settles the
    /// sequence); `Command` drops everything, except the chunks a cancelled
    /// prefill of this turn kept.
    pub(crate) fn reset(&mut self, scope: ResetScope) -> Result<()> {
        self.saved.clear();
        match scope {
            ResetScope::PrefixMiss => Ok(()),
            ResetScope::Command if self.kept => {
                self.kept = false;
                self.plan.set(None);
                Ok(())
            }
            ResetScope::Command => {
                self.plan.set(None);
                self.clear()
            }
        }
    }

    /// The turn is over: a later reset drops everything.
    pub fn turn_over(&mut self) {
        self.kept = false;
    }

    fn clear(&mut self) -> Result<()> {
        self.saved.clear();
        self.ladder.clear();
        self.seq.truncate(0)
    }
}

/// Admission and bank state as of the last turn edge (host values only).
#[derive(Clone, Debug, Default)]
pub(crate) struct Residency {
    pub admission: Option<Admission>,
    pub cache_slots: [usize; 2],
    /// Bank pages resident after the last turn made the bank volatile: a
    /// second live wired lease (another model's turn) keeps them wired.
    pub bank_resident: u64,
    pub switches: u32,
    pub in_turn: bool,
    /// The bank's and the Engram table's counters since load.
    pub counters: Counters,
    pub waves: Waves,
    /// The stored states: `[with a RAM copy, their bytes, published,
    /// restored]`.
    pub prefix: [u64; 4],
    /// The last saved turn: its rendered prompt ids and generated ids.
    pub turn_ids: Option<(Vec<u32>, Vec<u32>)>,
}

/// One turn's clocks and counters for the `MLX_HY4_STATS` lines.
struct TurnStats {
    t0: Instant,
    c0: Counters,
    swap0: Option<u64>,
    level_min: Option<u64>,
    prompt: usize,
    ttft_s: f64,
    decode_s: Vec<f64>,
}

pub(crate) struct Inner {
    pub(crate) store: Store,
    pub(super) mtp: Option<MtpLayer>,
    conv: Conv<Lagged>,
    pub(super) grid: Grid,
    tokenizer: Arc<Qwen3Tokenizer>,
    tags: Hy4Tags,
    defaults: ModelGenerationDefaults,
    pub(super) cancel: Option<Arc<AtomicBool>>,
    pub(super) in_turn: bool,
    residency: Arc<Mutex<Residency>>,
    turn_stats: Option<TurnStats>,
    /// The paged mode's pool and owners; `None` in flat mode.
    pub(super) paged: Option<Paged>,
    /// The flat mode's stored states and their disk; `None` in paged mode
    /// or without a disk.
    pub(super) flat: Option<FlatTier>,
    pub(super) telemetry: Arc<Telemetry>,
    /// The paged mode's decode waves.
    pub(super) waves: WaveLog,
    /// The bank's counters when serving began; `residencyInfo` counts from
    /// here.
    pub(super) served_from: Counters,
    /// The cache salt of the turn about to verify.
    turn_salt: Cell<u64>,
}

impl Drop for Inner {
    /// A dispose writes the waiting states and drains the copies within one
    /// bound before the tier closes.
    fn drop(&mut self) {
        if let Some(f) = self.flat.as_mut() {
            f.dispose(self.conv.seq());
        }
    }
}

impl Inner {
    /// Every shard of `path`, the admission from the environment, flat mode.
    #[cfg(test)]
    pub(crate) fn load(
        path: &Path,
        max_context: Option<usize>,
        residency: Arc<Mutex<Residency>>,
    ) -> Result<Self> {
        Self::open(path, max_context, None, residency, true, None).map(|(inner, _)| inner)
    }

    /// `load` in flat mode with `serial`, else in paged mode when the wave
    /// row cap holds a padded Q8 call; the stored states on the disk tier of
    /// `tier`, which a paged load gets back to attach after its warm-up;
    /// every admission within `max_memory`.
    pub(crate) fn open(
        path: &Path,
        max_context: Option<usize>,
        max_memory: Option<u64>,
        residency: Arc<Mutex<Residency>>,
        serial: bool,
        tier: Option<&DiskTierOptions>,
    ) -> Result<(Self, Option<Disk>)> {
        let paths = shard_paths(path)?;
        let (tokenizer, eos) = gguf_tokenizer(&paths[0])?;
        check_eos(eos)?;
        let opts = Options {
            max_memory,
            ..Options::from_env(max_context, serial)?
        };
        let mut store = Store::open(&paths, &opts)?;
        let mtp = if store.mtp {
            let m = MtpLayer::load(&store.scan, &store.shards, &store.model)?;
            store.dense_bytes += m.bytes();
            Some(m)
        } else {
            None
        };
        store.files_unchanged()?;
        let mut inner = Self::new(store, mtp, tokenizer, residency)?;
        let disk = tier.and_then(|t| {
            persist::open(
                t,
                &paths,
                inner.store.model.cfg(),
                inner.mtp.is_some(),
                Arc::clone(&inner.telemetry),
            )
        });
        let held = sched::attach(&mut inner, serial, disk)?;
        inner.publish(None);
        Ok((inner, held))
    }

    pub(crate) fn new(
        store: Store,
        mtp: Option<MtpLayer>,
        tokenizer: Qwen3Tokenizer,
        residency: Arc<Mutex<Residency>>,
    ) -> Result<Self> {
        let tags = Hy4Tags::from_tokenizer(&tokenizer)?;
        let grid = Grid::new(store.chunk, assistant_open(&tokenizer)?)?;
        let seq = new_seq(&store, mtp.as_ref())?;
        let mut inner = Self {
            store,
            mtp,
            conv: Conv::new(seq),
            grid,
            tokenizer: Arc::new(tokenizer),
            tags,
            defaults: generation_defaults(),
            cancel: None,
            in_turn: false,
            residency,
            turn_stats: None,
            paged: None,
            flat: None,
            telemetry: Arc::new(Telemetry::default()),
            waves: WaveLog::default(),
            served_from: Counters::default(),
            turn_salt: Cell::new(0),
        };
        inner.publish(None);
        Ok(inner)
    }

    #[cfg(test)]
    pub(crate) fn conv(&self) -> &Conv<Lagged> {
        &self.conv
    }

    /// The last saved turn's (prompt ids, generated ids).
    #[cfg(test)]
    pub(crate) fn turn_ids(&self) -> Option<(Vec<u32>, Vec<u32>)> {
        self.residency().turn_ids.clone()
    }

    /// The stored states' disk tier of either mode.
    fn disk(&self) -> Option<&Disk> {
        match (&self.paged, &self.flat) {
            (Some(p), _) => p.disk.as_ref(),
            (None, Some(f)) => Some(&f.disk),
            (None, None) => None,
        }
    }

    /// The flat disk's pass over the sequence's rows.
    fn flat_tick(&mut self) {
        if let Some(f) = self.flat.as_mut() {
            f.tick(self.conv.seq());
        }
    }

    /// A found stored state of the prompt `tokens` ends read back into a
    /// new sequence, its blocks' keys returned; on a failed read the
    /// sequence starts over and the whole prompt comes back to prefill.
    fn flat_restore(
        &mut self,
        tokens: &[u32],
    ) -> Result<(Option<Vec<Vec<u64>>>, Option<Vec<u32>>)> {
        let Some(f) = self.flat.as_mut() else {
            return Ok((None, None));
        };
        let Some((_, prompt)) = f.found_prompt(tokens) else {
            return Ok((None, None));
        };
        let (store, mtp) = (&self.store, self.mtp.as_ref());
        let filled =
            f.restore(|row, state, runs| filled_seq(store, mtp, &prompt[..row], state, runs));
        match filled {
            Some((seq, keys)) => {
                f.restored();
                self.conv.replace(seq, self.turn_salt.get());
                Ok((Some(keys), None))
            }
            None => {
                f.forget();
                let seq = new_seq(&self.store, self.mtp.as_ref())?;
                self.conv.replace(seq, self.turn_salt.get());
                Ok((None, Some(prompt)))
            }
        }
    }

    /// After a prefill: the flat turn's keys from the row it started at,
    /// and the prompt's state rows published.
    fn flat_publish(&mut self, line: Option<Vec<Vec<u64>>>) {
        let Some(f) = self.flat.as_mut() else {
            return;
        };
        let from = self.conv.prefilled_from();
        let prompt = self.conv.seq().tokens();
        let ends = self.grid.ends(0, prompt);
        let pass = pass_keys(prompt, &ends, self.grid.chunk());
        f.begin_turn(prompt, from, line, &pass, from == 0 || ends.contains(&from));
        let rows = self.grid.states(from, prompt);
        f.publish_prompt(self.conv.seq(), self.conv.ladder(), &rows);
    }

    fn residency(&self) -> std::sync::MutexGuard<'_, Residency> {
        self.residency
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    pub(super) fn publish(&mut self, bank_resident: Option<u64>) {
        let admission = self.store.admission();
        let switches = self.store.lease.switches().len() as u32;
        let cache_slots = self.store.cache_slots();
        let in_turn = self.in_turn;
        let counters = Counters::now(&self.store.bank, &self.store.table).since(&self.served_from);
        let waves = self.waves.waves;
        let prefix = match (&self.paged, &self.flat) {
            (Some(p), _) => p.counts(),
            (None, Some(f)) => f.counts(),
            (None, None) => [0; 4],
        };
        let mut r = self.residency();
        r.admission = admission;
        r.switches = switches;
        r.cache_slots = cache_slots;
        r.in_turn = in_turn;
        r.counters = counters;
        r.waves = waves;
        r.prefix = prefix;
        if let Some(b) = bank_resident {
            r.bank_resident = b;
        }
    }

    /// Bytes of every live sequence's cache.
    fn kv_bytes(&self) -> u64 {
        self.paged_kv_bytes()
            .unwrap_or_else(|| self.conv.seq().nbytes() as u64)
    }

    pub(super) fn begin(&mut self) -> Result<()> {
        if !self.in_turn {
            self.store.begin_turn(self.kv_bytes())?;
            self.in_turn = true;
            self.publish(None);
        }
        Ok(())
    }

    pub(super) fn end(&mut self) {
        if !self.in_turn {
            return;
        }
        self.in_turn = false;
        {
            let _ctx = StreamContext::new(Stream::generation());
            if let Err(e) = self.store.end_turn() {
                tracing::warn!("hy_v4_flash end of turn: {e}");
            }
        }
        let resident = self.store.bank.resident_bytes().unwrap_or(0);
        self.publish(Some(resident));
    }

    /// A new empty sequence in place of the conversation.
    fn fresh(&mut self, why: &Error) {
        tracing::warn!("hy_v4_flash: {}; starting a new sequence", why.reason);
        if let Some(f) = self.flat.as_mut() {
            f.forget();
        }
        match new_seq(&self.store, self.mtp.as_ref()) {
            Ok(s) => self.conv.replace(s, self.turn_salt.get()),
            Err(e) => tracing::warn!("hy_v4_flash new sequence: {}", e.reason),
        }
    }

    fn start_stats(&mut self, prompt: usize) {
        self.turn_stats = std::env::var_os(stats::STATS_ENV).map(|_| TurnStats {
            t0: Instant::now(),
            c0: Counters::now(&self.store.bank, &self.store.table),
            swap0: stats::host_pressure().1,
            level_min: None,
            prompt,
            ttft_s: 0.0,
            decode_s: Vec::new(),
        });
    }

    /// With `MLX_HY4_STATS` set: evaluates the call's logits (timing) and
    /// appends one stats line for the turn so far.
    fn emit_stats(&mut self, logits: &MxArray, prefill: bool, t_call: Instant) {
        let Some(ts) = self.turn_stats.as_mut() else {
            return;
        };
        logits.eval();
        if prefill {
            ts.ttft_s = ts.t0.elapsed().as_secs_f64();
        } else {
            ts.decode_s.push(t_call.elapsed().as_secs_f64());
        }
        let (lvl, swap) = stats::host_pressure();
        ts.level_min = match (ts.level_min, lvl) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let (_, peak, _) = stats::rusage();
        let line = Line {
            stage: "chat",
            prompt_tok: ts.prompt,
            gen_tok: 1 + ts.decode_s.len(),
            ttft_s: ts.ttft_s,
            decode_s: &ts.decode_s,
            counters: Counters::now(&self.store.bank, &self.store.table).since(&ts.c0),
            wall_s: ts.t0.elapsed().as_secs_f64(),
            cache_slots: self.store.cache_slots(),
            ctx: self.store.ctx,
            chunk: self.grid.chunk(),
            peak_footprint: peak,
            mlx_cache_bytes: crate::array::get_cache_memory(),
            swapout_delta: swap.zip(ts.swap0).map_or(0, |(a, b)| a as i64 - b as i64),
            memstatus_min: ts.level_min,
        };
        if let Err(e) = stats::emit(&line.json()) {
            tracing::warn!("hy_v4_flash stats: {e}");
        }
    }
}

pub(crate) struct Step<'a>(&'a mut Inner);

impl DecodeStep for Step<'_> {
    fn forward(&mut self, ids: &MxArray) -> Result<(MxArray, bool)> {
        let token = ids.item_at_int32(0)? as u32;
        self.forward_with_token(ids, token)
    }

    fn forward_with_token(&mut self, _ids: &MxArray, token: u32) -> Result<(MxArray, bool)> {
        let t = Instant::now();
        let inner = &mut *self.0;
        let (logits, _) =
            inner
                .conv
                .seq_mut()
                .step(&inner.store.model, inner.mtp.as_ref(), token)?;
        inner.flat_tick();
        inner.emit_stats(&logits, false, t);
        Ok((logits, false))
    }
}

impl ChatBackend for Inner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        Ok(Arc::clone(&self.tokenizer))
    }

    fn family_name(&self) -> &'static str {
        "hy_v4_flash"
    }

    fn session_eos_id(&self, _tok: &Qwen3Tokenizer) -> Result<u32> {
        Ok(EOS)
    }

    fn policy(&self) -> ThinkingPolicy {
        ThinkingPolicy::TemplateHonoring
    }

    fn generation_defaults(&self) -> Option<&ModelGenerationDefaults> {
        Some(&self.defaults)
    }

    /// MTP is on unless the request turns it off.
    fn resolve_params(&self, config: &ChatConfig) -> ChatParams {
        let mut p = resolve(config, &self.defaults);
        if config.enable_mtp.is_none() {
            p.enable_mtp = self.mtp.is_some();
        }
        self.turn_salt.set(p.cache_salt);
        if let Some(f) = &self.flat {
            f.set_salt(p.cache_salt);
        }
        p
    }

    fn cached_token_history(&self) -> &[u32] {
        match &self.paged {
            Some(p) => &p.saved,
            None => self.conv.history(),
        }
    }

    fn finalize_turn(&self, args: FinalizeArgs<'_>) -> Result<ChatResult> {
        finalize::finalize(args, &self.tags)
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        if self.paged.is_some() {
            return self.paged_reset(matches!(scope, ResetScope::Command));
        }
        if matches!(scope, ResetScope::Command)
            && let Some(f) = self.flat.as_mut()
        {
            f.reset();
        }
        if let Err(e) = self.conv.reset(scope) {
            self.fresh(&e);
        }
        Ok(())
    }

    /// The longest prefix the sequence or a ladder entry still holds, below
    /// the prompt's end; the prefill takes the plan made here.
    fn verify_cache_prefix(&self, tokens: &[u32], reuse: bool) -> usize {
        if self.paged.is_some() {
            return self.paged_verify(tokens, reuse);
        }
        let runner = LagRunner {
            model: &self.store.model,
            mtp: self.mtp.as_ref(),
        };
        let ram = reuse && self.conv.reuses(self.turn_salt.get());
        let rows = self.conv.verify(&runner, &self.grid, tokens, ram);
        let Some(f) = &self.flat else {
            return rows;
        };
        if !reuse {
            f.clear_found();
            return rows;
        }
        let ends = self.grid.ends(0, tokens);
        let pass = pass_keys(tokens, &ends, self.grid.chunk());
        rows.max(f.find(tokens, &pass, rows))
    }

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        self.residency().turn_ids =
            Some((args.save_tokens.to_vec(), args.generated_tokens.to_vec()));
        if self.paged.is_some() {
            return;
        }
        let saved = self
            .conv
            .seq_mut()
            .settle()
            .and_then(|()| self.conv.save(args.reuse_cache));
        if let Err(e) = saved {
            self.fresh(&e);
            return;
        }
        if let Some(f) = self.flat.as_mut() {
            if args.reuse_cache {
                f.publish_turn(self.conv.seq());
            } else {
                f.forget();
            }
        }
    }

    fn eval_caches(&self) -> Result<()> {
        Ok(())
    }

    /// The admitted context: `Model::forward` refuses past it.
    fn context_token_limit(&self) -> Option<usize> {
        Some(self.store.ctx)
    }

    /// `None`: the store holds a lease bounded by the admission instead
    /// (`fixed + cache`, taken in `Store::begin_turn`).
    fn wired_limit_bytes(&self) -> Option<usize> {
        None
    }

    fn set_turn_cancel_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        let ending = flag.is_none();
        if self.paged.is_some() {
            self.cancel = flag;
            self.paged_flag(!ending);
            return;
        }
        self.cancel = flag;
        if ending {
            self.end();
            self.conv.turn_over();
            if let Some(f) = self.flat.as_mut() {
                f.end_turn(self.conv.seq());
            }
        }
    }

    fn prefill(&mut self, tokens: &[u32], stream: Stream) -> Result<MxArray> {
        self.start_stats(tokens.len());
        let t0 = Instant::now();
        self.begin()?;
        let _ctx = StreamContext::new(stream);
        let flag = self.cancel.clone();
        let cancel = move |_: usize| flag.as_ref().is_some_and(|f| f.load(Ordering::Relaxed));
        let (line, whole) = self.flat_restore(tokens)?;
        if let Err(e) = self.conv.own(self.turn_salt.get()) {
            self.fresh(&e);
        }
        let tokens = whole.as_deref().unwrap_or(tokens);
        let total = tokens.len();
        let mut ran = 0usize;
        let mut on_chunk = |rows: Range<usize>| {
            ran += rows.len();
            let done = u32::try_from(ran.min(total)).unwrap_or(u32::MAX);
            crate::engine::progress::with_turn_progress(|p| p.prefill_advanced(done));
        };
        let runner = LagRunner {
            model: &self.store.model,
            mtp: self.mtp.as_ref(),
        };
        let out = self
            .conv
            .prefill(&runner, &self.grid, tokens, &cancel, &mut on_chunk);
        let out = match out {
            Ok(out) => out,
            Err(e) => {
                if let Some(f) = self.flat.as_mut() {
                    f.forget();
                }
                return Err(e);
            }
        };
        self.flat_publish(line);
        let logits = out
            .logits
            .ok_or_else(|| bad("the prefill returned no logits"))?;
        self.emit_stats(&logits, true, t0);
        Ok(logits)
    }

    type Decode<'a>
        = Step<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Step<'_>> {
        Ok(Step(self))
    }

    fn execution_plan(&self) -> ExecutionPlan {
        if self.paged.is_some() {
            return ExecutionPlan {
                paged_attention: Some(PagedAttentionPlan {
                    supports_delta: true,
                }),
                speculative: self.mtp.as_ref().map(|_| spec::PAGED_PLAN),
                ..ExecutionPlan::TEXT_ONLY
            };
        }
        ExecutionPlan {
            speculative: self.mtp.as_ref().map(|_| spec::PLAN),
            ..ExecutionPlan::TEXT_ONLY
        }
    }

    fn run_paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        crate::engine::paged_turn::run_paged_turn(self, args)
    }

    fn run_speculative_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        spec::run_turn(self, args)
    }
}

impl MtpBackend for Inner {
    type MtpDecode<'a>
        = spec::Stepper<'a>
    where
        Self: 'a;

    fn begin_mtp_decode(&mut self, _setup: &MtpTurnSetup<'_>) -> Result<spec::Stepper<'_>> {
        if self.paged.is_some() {
            return spec::Stepper::paged(self);
        }
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| bad("an MTP turn without the MTP layer"))?;
        spec::Stepper::begin(&self.store.model, mtp, self.conv.seq_mut())
    }
}

/// An empty sequence of the admitted context, with a draft when the MTP
/// layer is loaded.
fn new_seq(store: &Store, mtp: Option<&MtpLayer>) -> Result<Lagged> {
    let trunk = store.model.new_session(store.ctx)?;
    Ok(Lagged::new(trunk, mtp.map(|m| m.new_draft(store.ctx))))
}

/// A new sequence holding the stored `state` of `tokens`, its blocks read
/// from `runs`.
fn filled_seq(
    store: &Store,
    mtp: Option<&MtpLayer>,
    tokens: &[u32],
    state: &PrefixState,
    runs: &mut BlockRuns<'_>,
) -> Result<Lagged> {
    let mut seq = new_seq(store, mtp)?;
    seq.fill(tokens, state, runs)?;
    Ok(seq)
}

#[napi(object)]
#[derive(Default)]
pub struct HyV4FlashLoadOptions {
    /// Context token limit (`--max-context`): the admission budgets for at
    /// most this many tokens (and at most 262,144).
    pub max_context_tokens: Option<i64>,
    /// The Metal memory budget (`--max-memory`): every admission, at load
    /// and at each turn, sizes the expert cache within it, and the memory
    /// governor's limit for this model is it.
    pub max_memory_bytes: Option<i64>,
    /// SSD cache for the stored states (`--max-cache-disk`): `'auto'` (the
    /// default: up to 32 GiB, a quarter of the volume's room), a byte count,
    /// or `'0'`, which turns the SSD cache off.
    pub cache_disk: Option<String>,
    /// The paged KV disk cache switch every family shares (`--no-persist-cache`):
    /// keep the stored states on SSD across restarts in a directory of the
    /// model's own under `cacheDir`; off, the cache is temporary: a directory
    /// under the temp root that takes copies only while memory drops states,
    /// removed at the stop. Unset ⇒ `MLX_PERSIST_PAGED_CACHE`, else off.
    pub persist_paged_cache: Option<bool>,
    /// The persistent cache's root (`--cache-dir`), default
    /// `~/.mlx-node/cache/prefix-cache`; needs `persistPagedCache`.
    pub cache_dir: Option<String>,
}

/// The last saved turn of the session (prompt and output ids).
#[napi(object)]
pub struct HyV4FlashTurnTokenIds {
    /// The rendered prompt the turn was given (whole conversation).
    pub prompt_ids: Vec<u32>,
    /// The generated ids, the stop token included when one ended the turn.
    pub output_ids: Vec<u32>,
}

#[napi(object)]
#[derive(Clone, Debug, PartialEq)]
pub struct HyV4FlashContextLimits {
    /// The context the load's admission budgets for.
    pub effective_window_tokens: u32,
    pub trained_window_tokens: u32,
    /// Rows of one prefill chunk.
    pub prefill_chunk_tokens: u32,
    /// The model's `modelId` in `memoryGovernorStatus().scheduler`; a
    /// flat-mode load does not register.
    pub governor_model_id: Option<f64>,
}

/// The load-time and latest per-turn admission, and the expert bank.
#[napi(object)]
pub struct HyV4FlashResidencyInfo {
    pub dense_bytes: f64,
    pub physical_bytes: Option<f64>,
    /// Other processes' (and other models') bytes at the last admission.
    pub others_bytes: Option<f64>,
    /// Fixed working set: dense, staging, KV and compute at the context
    /// limit, I/O buffers, driver and MLX pool reserves.
    pub fixed_bytes: Option<f64>,
    /// Expert cache slots of both classes.
    pub expert_cache_slots: u32,
    /// Expert cache slots per class (Q6_K down, Q4_K down).
    pub expert_cache_slots_by_class: Vec<u32>,
    pub expert_cache_bytes: Option<f64>,
    pub staging_layers: Option<u32>,
    /// Bounded wired-limit request of each turn (`fixed + cache`).
    pub wired_limit_bytes: Option<f64>,
    /// Bank pages still resident after the last turn made the bank volatile;
    /// non-zero while another live wired lease keeps them wired.
    pub bank_resident_after_turn_bytes: f64,
    pub bank_switches: u32,
    pub in_turn: bool,
    /// Routed expert uses since load served without a read (cache hits and
    /// still-valid staging copies), and the experts read into slots.
    pub slot_hits: f64,
    pub slot_misses: f64,
    /// Bytes read from the GGUF shards since load: expert fills and Engram
    /// rows; the page cache may serve part of them.
    pub checkpoint_bytes_read: f64,
    pub engram_bytes_read: f64,
    /// Stored prefix states with a RAM copy, and their bytes.
    pub prefix_states: f64,
    pub prefix_state_bytes: f64,
    /// States published since load, and turns that resumed from one.
    pub prefix_published: f64,
    pub prefix_restored: f64,
    /// The stored states' disk tier is open and takes writes.
    pub prefix_disk_enabled: bool,
    /// States waiting for their disk write, the one in flight included.
    pub prefix_disk_waiting: f64,
    /// States written whose whole chain is on disk.
    pub prefix_disk_durable: f64,
    /// Waiting states not written: their conversation went on.
    pub prefix_disk_unneeded: f64,
    /// States not written or dropped (chain incomplete, room, a failed
    /// write, a closed cache), and copies the writer refused.
    pub prefix_disk_refused: f64,
    /// Turns that resumed from a state read back from disk.
    pub prefix_disk_restored: f64,
    /// States a persistent cache took back at load: on disk only until a
    /// turn reads one back.
    pub prefix_disk_adopted: f64,
    /// States removed from disk: their chain was not there.
    pub prefix_disk_dropped_chain: f64,
    /// Block and segment copies waiting for the writer.
    pub prefix_disk_owed_blocks: f64,
    /// Bytes of states written.
    pub prefix_disk_state_bytes_written: f64,
    /// Paged mode: decode and verify waves since load.
    pub decode_waves: f64,
    /// Waves by rows: `[i]` counts the waves of `i + 1` rows, the last
    /// entry also every wider wave.
    pub wave_rows: Vec<f64>,
    /// Distinct experts per layer step of the waves, summed, and the rows
    /// times the routed experts each row picks: their ratio is the share
    /// of picks the lanes shared.
    pub union_experts: f64,
    pub row_experts: f64,
    /// Wave rows kept (one emitted token each).
    pub wave_tokens: f64,
    /// Expert bytes the waves read per kept row.
    pub miss_bytes_per_token: f64,
    /// `[n - 1]`: the share of expert uses served without a read in waves
    /// of `n` owners.
    pub hit_rate_by_lanes: Vec<f64>,
    /// Host syncs (evals and router reads) per wave.
    pub evals_per_wave: f64,
    /// Milliseconds of prefill slices run while another scheduled turn was open.
    pub prefill_chunk_stall_ms: f64,
}

/// The model thread's state: the flat whole-turn backend or the shared
/// scheduler over the paged mode.
enum Engine {
    Flat(Box<Inner>),
    Paged(Box<HybridSchedulerState<Inner>>),
}

/// Hy4-flash (GGUF `hy_v4_flash`) with SSD-streamed routed experts:
/// text-only chat on one owned model thread. Sequences of different cache
/// owners run together on the shared scheduler over a paged pool, unless
/// `MLX_SERVE_FORCE_SERIAL=1` at load keeps one whole turn at a time.
#[napi]
pub struct HyV4FlashModel {
    thread: ModelThread<ModelCommand>,
    limits: HyV4FlashContextLimits,
    dense_bytes: u64,
    residency: Arc<Mutex<Residency>>,
    paged: bool,
    lanes: usize,
    /// Weak: the tier goes with the model thread's state.
    closer: Option<std::sync::Weak<Closer>>,
    state_tier: TierHandle,
    telemetry: Arc<Telemetry>,
    warmup: Option<WarmupReport>,
}

#[napi]
impl HyV4FlashModel {
    /// `model_path`: any shard of the split GGUF, its directory, or a
    /// single-file GGUF.
    #[napi]
    pub async fn load(model_path: String, options: Option<HyV4FlashLoadOptions>) -> Result<Self> {
        let progress = crate::engine::load_progress::begin(&model_path);
        let cell = progress.cell();
        let options = options.unwrap_or_default();
        let max_context =
            crate::nn::rope_scaling::max_context_option(LABEL, options.max_context_tokens)?
                .map(|t| t as usize);
        let max_memory =
            crate::models::qwen3_5::load_plan::max_memory_option(LABEL, options.max_memory_bytes)?;
        let env = std::env::var(crate::cold_tier::PERSIST_PAGED_CACHE_ENV).ok();
        let (tier, disk) = persist::disk_plan(
            LABEL,
            options.cache_disk.as_deref(),
            options.persist_paged_cache,
            options.cache_dir.clone(),
            env.as_deref(),
        )?;
        let residency = Arc::new(Mutex::new(Residency::default()));
        let shared = Arc::clone(&residency);
        let serial = HybridSchedulerState::<Inner>::force_serial();
        let (thread, ready) = ModelThread::spawn_with_scheduler(
            move || {
                let tier = disk.then_some(&tier);
                let (inner, held) = Inner::open(
                    Path::new(&model_path),
                    max_context,
                    max_memory,
                    shared,
                    serial,
                    tier,
                )?;
                let s = &inner.store;
                let governor = inner.paged.as_ref().map(|p| p.governor.model_id());
                let limits = HyV4FlashContextLimits {
                    effective_window_tokens: u32::try_from(s.ctx).unwrap_or(u32::MAX),
                    trained_window_tokens: u32::try_from(s.model.cfg().context_length)
                        .unwrap_or(u32::MAX),
                    prefill_chunk_tokens: u32::try_from(s.chunk).unwrap_or(u32::MAX),
                    governor_model_id: governor.map(|id| id as f64),
                };
                let dense = s.dense_bytes;
                let lanes = inner.paged.as_ref().map(|p| p.lanes);
                let tier_disk = held.as_ref().or(inner.disk());
                let closer = tier_disk.map(Disk::closer);
                let state_tier = tier_disk.map(Disk::tier_handle).unwrap_or_default();
                let telemetry = Arc::clone(&inner.telemetry);
                let (engine, warmup) = match lanes {
                    Some(_) => {
                        let (mut state, warmup) = crate::engine::warmup::serving_scheduler_minimal(
                            inner,
                            &cell,
                            true,
                            HY4_WARMUP_SLICE_ROWS,
                        )?;
                        sched::serve(&mut state.inner, held);
                        (Engine::Paged(Box::new(state)), warmup)
                    }
                    None => (Engine::Flat(Box::new(inner)), None),
                };
                Ok((
                    engine,
                    (
                        limits, dense, lanes, closer, state_tier, telemetry, warmup, governor,
                    ),
                ))
            },
            |engine: &mut Engine, rx| match engine {
                Engine::Paged(state) => state.drive(rx),
                Engine::Flat(inner) => match rx.blocking_recv() {
                    Some(ModelCommand::MemoryControl) => {
                        inner.flat_tick();
                        LoopControl::Continue
                    }
                    Some(cmd) => {
                        handle_model_command(&mut **inner, cmd);
                        LoopControl::Continue
                    }
                    None => LoopControl::Break,
                },
            },
        );
        let (limits, dense, lanes, closer, state_tier, telemetry, warmup, governor) = ready
            .await
            .map_err(|e| Error::from_reason(e.to_string()))??;
        drop(progress);
        if let (Some(sender), Some(model)) = (thread.cmd_sender(), governor) {
            crate::engine::model_command::register_memory_waker(model, &sender);
        }
        if let (Some(sender), Some(c)) = (
            thread.cmd_sender(),
            closer.as_ref().and_then(std::sync::Weak::upgrade),
        ) {
            let sender = sender.downgrade();
            c.set_wake(Arc::new(move || {
                if let Some(sender) = sender.upgrade() {
                    let _ = sender.send(ModelCommand::MemoryControl);
                }
            }));
        }
        Ok(Self {
            thread,
            limits,
            dense_bytes: dense,
            residency,
            paged: lanes.is_some(),
            lanes: lanes.unwrap_or(1),
            closer,
            state_tier,
            telemetry,
            warmup,
        })
    }

    /// What the load's warm-up did; `null` for a flat-mode load, which runs
    /// none.
    #[napi(getter)]
    pub fn warmup(&self) -> Option<WarmupReport> {
        self.warmup.clone()
    }

    /// Closes the stored states' SSD cache at a clean stop: on a persistent
    /// cache it asks the model thread to write the waiting states, the
    /// newest first, then drains every copy, until `deadlineMs`
    /// (milliseconds on `Date.now()`'s clock, so models closed in turn share
    /// one deadline); a temporary cache is removed. The model thread gets
    /// all of the time but the last second (at most half of it). The cache
    /// takes no write and removes nothing afterwards. Resolves true only
    /// when no state was left waiting and every copy and write landed by
    /// the deadline; true without a cache, or after `dispose`, which closed
    /// it. A call while another close runs gets that close's outcome.
    #[napi]
    pub async fn close_cache(&self, deadline_ms: f64) -> Result<bool> {
        let Some(closer) = self.closer.as_ref().and_then(std::sync::Weak::upgrade) else {
            return Ok(true);
        };
        let deadline = crate::models::qwen3_5::dflash2_state::disk::instant_at(deadline_ms);
        napi::bindgen_prelude::spawn_blocking(move || closer.close(deadline))
            .await
            .map_err(|e| Error::from_reason(format!("{LABEL} closeCache: {e}")))
    }

    /// The stored states' SSD cache: `'persistent'` or `'temp'` while it is
    /// open, `'off'` without one, after `closeCache`, and once a write failed
    /// for anything but room (it takes no write after).
    #[napi(ts_return_type = "'persistent' | 'temp' | 'off'")]
    pub fn state_tier_mode(&self) -> String {
        match self.telemetry.mode(&self.state_tier) {
            1 => "persistent",
            2 => "temp",
            _ => "off",
        }
        .to_string()
    }

    /// The directory name of the stored states' open persistent SSD cache
    /// under its cache root, read from the cache itself; null without one,
    /// for a temporary cache, and after `closeCache` or `dispose`.
    #[napi]
    pub fn state_tier_namespace(&self) -> Option<String> {
        self.state_tier.namespace()
    }

    /// Frees the weights, the bank and the sequence now: the model thread
    /// drops them and exits; every later call fails.
    #[napi]
    pub async fn dispose(&self) -> Result<()> {
        self.thread.dispose().await
    }

    #[napi]
    pub fn context_limits(&self) -> HyV4FlashContextLimits {
        self.limits.clone()
    }

    /// 1 under `MLX_SERVE_FORCE_SERIAL` (read per call) or in flat mode,
    /// else the scheduler's lanes.
    #[napi]
    pub fn max_concurrent_sequences(&self) -> u32 {
        sched::napi_concurrency(HybridSchedulerState::<Inner>::force_serial(), self.lanes)
    }

    /// Whether the load runs in paged mode (fixed at load).
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged
    }

    /// Prompt and generated ids of the last completed turn, if any.
    #[napi]
    pub fn last_turn_token_ids(&self) -> Option<HyV4FlashTurnTokenIds> {
        let r = self
            .residency
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        r.turn_ids.as_ref().map(|(p, o)| HyV4FlashTurnTokenIds {
            prompt_ids: p.clone(),
            output_ids: o.clone(),
        })
    }

    #[napi]
    pub fn residency_info(&self) -> HyV4FlashResidencyInfo {
        let r = self
            .residency
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let a = r.admission;
        let slots = r.cache_slots;
        let (c, w, t) = (&r.counters, &r.waves, &*self.telemetry);
        let load = |n: &std::sync::atomic::AtomicU64| n.load(Ordering::Relaxed) as f64;
        HyV4FlashResidencyInfo {
            dense_bytes: self.dense_bytes as f64,
            physical_bytes: a.map(|a| a.physical as f64),
            others_bytes: a.map(|a| a.others as f64),
            fixed_bytes: a.map(|a| a.fixed as f64),
            expert_cache_slots: u32::try_from(slots[0] + slots[1]).unwrap_or(u32::MAX),
            expert_cache_slots_by_class: slots
                .iter()
                .map(|&n| u32::try_from(n).unwrap_or(u32::MAX))
                .collect(),
            expert_cache_bytes: a.map(|a| a.cache_bytes() as f64),
            staging_layers: a.map(|a| a.staging_layers as u32),
            wired_limit_bytes: a.map(|a| a.wired_limit() as f64),
            bank_resident_after_turn_bytes: r.bank_resident as f64,
            bank_switches: r.switches,
            in_turn: r.in_turn,
            slot_hits: c.uses.saturating_sub(c.fill_slots) as f64,
            slot_misses: c.fill_slots as f64,
            checkpoint_bytes_read: (c.fill_bytes + c.engram_bytes) as f64,
            engram_bytes_read: c.engram_bytes as f64,
            prefix_states: r.prefix[0] as f64,
            prefix_state_bytes: r.prefix[1] as f64,
            prefix_published: r.prefix[2] as f64,
            prefix_restored: r.prefix[3] as f64,
            prefix_disk_enabled: t.mode(&self.state_tier) != 0,
            prefix_disk_waiting: load(&t.waiting),
            prefix_disk_durable: load(&t.durable),
            prefix_disk_unneeded: load(&t.unneeded),
            prefix_disk_refused: load(&t.refused),
            prefix_disk_restored: load(&t.disk_restored),
            prefix_disk_adopted: load(&t.adopted),
            prefix_disk_dropped_chain: load(&t.dropped_chain),
            prefix_disk_owed_blocks: load(&t.owed),
            prefix_disk_state_bytes_written: load(&t.state_bytes_written),
            decode_waves: w.decode as f64,
            wave_rows: w.rows.iter().map(|&n| n as f64).collect(),
            union_experts: w.union_experts as f64,
            row_experts: w.row_experts as f64,
            wave_tokens: w.tokens as f64,
            miss_bytes_per_token: w.miss_bytes_per_token(),
            hit_rate_by_lanes: w.hit_rate_by_lanes().to_vec(),
            evals_per_wave: w.evals_per_wave(),
            prefill_chunk_stall_ms: w.prefill_stall_ns as f64 * 1e-6,
        }
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: HyV4FlashModel,
    thread_cmd: crate::engine::model_command::ModelCommand,
    thread: direct,
    image_guard: text_only,
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
}
