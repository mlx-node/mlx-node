//! The whole text model (`model.py` `Transformer.forward`): embed, expand to
//! `hc` copies, engram + block per layer, collapse with the last FFN `pre`,
//! final norm, f32 head. One call is layer-outer over all its rows.

use std::cell::RefCell;
use std::sync::Arc;

use napi::{Error, Result};

use super::attention::{FakeQuant, rms_norm};
use super::block::Block;
use super::cache::{Candidates, CompressedCache, ForwardShared, TopK, WindowCache};
use super::config::DeepseekV41Config;
use super::engram::{Engram, EngramRows};
use super::engram_hash::EngramHasher;
use super::moe::{ExpertBank, Experts, Moe, MoeWeights, Routes};
use super::ops::hc_pre;
use super::store::{Dense, Weight};
use crate::array::{DType, MxArray};
use crate::moe_stream::bank::StagingBank;

thread_local! {
    static SITES: RefCell<Option<Vec<(&'static str, usize, DType)>>> = const { RefCell::new(None) };
}

/// Records `a`'s dtype at a §3 fp32 site while `record_fp32_sites` is on.
pub(crate) fn fp32_site(site: &'static str, layer: usize, a: &MxArray) -> Result<()> {
    if SITES.with(|s| s.borrow().is_none()) {
        return Ok(());
    }
    let dt = a.dtype()?;
    SITES.with(|s| {
        if let Some(v) = s.borrow_mut().as_mut() {
            v.push((site, layer, dt));
        }
    });
    Ok(())
}

/// Runs `f` with the fp32-site recorder on (this thread) and returns what it saw.
pub fn record_fp32_sites<T>(f: impl FnOnce() -> T) -> (T, Vec<(&'static str, usize, DType)>) {
    SITES.with(|s| *s.borrow_mut() = Some(Vec::new()));
    let out = f();
    let seen = SITES.with(|s| s.borrow_mut().take()).unwrap_or_default();
    (out, seen)
}

/// Per-layer debug dump of one call.
#[derive(Default)]
pub struct Trace {
    /// Engram table rows `[pos][engram layer][col]` (global ids).
    pub engram_rows: Vec<u32>,
    pub routes: Vec<Routes>,
    /// The index source's top-k after each layer that is an index source.
    pub topk: Vec<Option<TopK>>,
    pub candidates: Vec<Option<Candidates>>,
    /// Index scores `[s * T]` before the final top-k at index sources, when
    /// `Model::export_scores` is on.
    pub index_scores: Vec<Option<Vec<f32>>>,
    /// Each block's output stream `[s, hc, dim]` when `keep_out`.
    pub keep_out: bool,
    pub out: Vec<MxArray>,
}

/// One sequence's state. A failed or cancelled call leaves it as it was
/// before the call.
#[derive(Clone)]
pub struct Session {
    pub tokens: Vec<u32>,
    pub(crate) wins: Vec<WindowCache>,
    pub(crate) sources: Vec<Option<CompressedCache>>,
}

pub struct Model {
    pub cfg: DeepseekV41Config,
    act: DType,
    embed: Dense,
    blocks: Vec<Block>,
    engrams: Vec<Option<(usize, Engram)>>,
    hasher: Option<EngramHasher>,
    table: Option<Box<dyn EngramRows>>,
    norm: MxArray,
    head: Dense,
    /// Context limit a call may reach (session tokens + call ids); the store
    /// sets the admitted context (D-24), `None` = no limit.
    pub ctx_limit: Option<usize>,
    /// Long-prompt layer skip in `forward_last` (bitwise, see `skip_from`);
    /// tests turn it off to compare.
    pub layer_skip: bool,
}

impl Model {
    /// `get` returns a dense GGUF tensor, `bank` a routed-expert tensor
    /// (not called with `stream`: every layer then gathers from that shared
    /// bank); `table` serves engram rows. `act` is the activation dtype.
    /// The calling thread runs every later call of this model: it is set to
    /// true f32 GEMMs here.
    pub fn from_gguf(
        cfg: DeepseekV41Config,
        act: DType,
        fq: FakeQuant,
        mut get: impl FnMut(&str) -> Result<Dense>,
        mut bank: impl FnMut(&str) -> Result<ExpertBank>,
        table: Option<Box<dyn EngramRows>>,
        stream: Option<Arc<StagingBank>>,
    ) -> Result<Self> {
        // the reference's f32 sites need true f32; NAX GPUs run f32 GEMMs in
        // TF32 under MLX's default unless the thread opts out
        crate::array::set_thread_tf32(Some(false));
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        let mut engrams = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let mw = match &stream {
                Some(st) => MoeWeights::with_experts(l, &mut get, Experts::Stream(Arc::clone(st)))?,
                None => MoeWeights::from_gguf(l, &mut get, &mut bank)?,
            };
            let moe = Moe::new(&cfg, l, mw)?;
            blocks.push(Block::from_gguf(&cfg, l, fq, moe, &mut get)?);
            let e = cfg
                .engram
                .as_ref()
                .and_then(|e| e.layers.iter().position(|&x| x == l));
            engrams.push(match e {
                Some(e) => Some((
                    e,
                    Engram::from_gguf(l, cfg.dim, cfg.hc_mult, cfg.norm_eps, &mut get)?,
                )),
                None => None,
            });
        }
        let hasher = cfg.engram.as_ref().map(EngramHasher::new).transpose()?;
        if hasher.is_some() != table.is_some() {
            return Err(Error::from_reason(
                "engram layers need a table (and only they)",
            ));
        }
        Ok(Self {
            act,
            embed: get("token_embd.weight")?,
            norm: get("output_norm.weight")?.array()?,
            head: get("output.weight")?,
            blocks,
            engrams,
            hasher,
            table,
            cfg,
            ctx_limit: None,
            layer_skip: true,
        })
    }

    /// Stage P5: the first row of layer `l`'s output that anyone reads when
    /// only the call's last row needs logits (`s` rows). From the last
    /// layer that compresses its own KV (`z`, the last compressor and
    /// index-key owner) on, a layer's per-row output feeds only the next
    /// layers' queries and their window KV: the last layer needs 1 row,
    /// each layer below it 127 (window - 1) more, so layer `l >= z` needs
    /// `1 + (n_layers - 1 - l) * (window - 1)` rows. The compressor, index
    /// keys and window KV take every row's INPUT, never a skipped output,
    /// so the session state and every later call are unchanged (oracle
    /// proof: DECISIONS P5-1). 0 = all rows.
    pub fn skip_from(&self, l: usize, s: usize) -> usize {
        let c = &self.cfg;
        let Some(z) = (0..c.n_layers).filter(|&k| c.kv_src[k] == Some(k)).max() else {
            return 0;
        };
        if l < z || l >= c.n_layers || c.window_size == 0 {
            return 0;
        }
        let need = 1 + (c.n_layers - 1 - l) * (c.window_size - 1);
        s.saturating_sub(need)
    }

    /// Index sources return their scores into `Trace::index_scores`
    /// (diagnostics: `s * T` host floats per call).
    pub fn export_scores(&mut self, on: bool) {
        for b in &mut self.blocks {
            if let super::block::Attn::Compressed(c) = &mut b.attn {
                c.export_scores = on;
            }
        }
    }

    pub fn new_session(&self) -> Session {
        let c = &self.cfg;
        Session {
            tokens: Vec::new(),
            wins: (0..c.n_layers)
                .map(|_| WindowCache::new(c.window_size))
                .collect(),
            sources: (0..c.n_layers)
                .map(|l| {
                    (c.kv_src[l] == Some(l))
                        .then(|| CompressedCache::new(c.compress_ratios[l] as usize))
                })
                .collect(),
        }
    }

    /// Runs `ids` at the end of `sess` and returns f32 logits `[ids.len(),
    /// vocab]`. `cancel(layer)` is asked before each layer; a cancelled or
    /// failed call restores the session.
    pub fn forward(
        &self,
        sess: &mut Session,
        ids: &[u32],
        cancel: Option<&dyn Fn(usize) -> bool>,
        trace: Option<&mut Trace>,
    ) -> Result<MxArray> {
        let before = sess.clone();
        let out = self.forward_inner(sess, ids, cancel, trace, false);
        if out.is_err() {
            *sess = before;
        }
        out
    }

    /// `forward`, but only the last row's f32 logits `[1, vocab]` (a prompt
    /// prefill: no `[ids.len(), vocab]` logits block).
    pub fn forward_last(
        &self,
        sess: &mut Session,
        ids: &[u32],
        cancel: Option<&dyn Fn(usize) -> bool>,
        trace: Option<&mut Trace>,
    ) -> Result<MxArray> {
        let before = sess.clone();
        let out = self.forward_inner(sess, ids, cancel, trace, true);
        if out.is_err() {
            *sess = before;
        }
        out
    }

    /// Largest scratch chunk any compressed attention has used (indexer
    /// scores, sparse attention), bytes.
    pub fn scratch_peak(&self) -> usize {
        self.blocks
            .iter()
            .filter_map(|b| match &b.attn {
                super::block::Attn::Compressed(c) => Some(c.scratch_peak()),
                super::block::Attn::Window(_) => None,
            })
            .max()
            .unwrap_or(0)
    }

    fn forward_inner(
        &self,
        sess: &mut Session,
        ids: &[u32],
        cancel: Option<&dyn Fn(usize) -> bool>,
        mut trace: Option<&mut Trace>,
        last_only: bool,
    ) -> Result<MxArray> {
        let (s, start) = (ids.len() as i64, sess.tokens.len());
        let (hc, dim) = (self.cfg.hc_mult as i64, self.cfg.dim as i64);
        if s == 0 {
            return Err(Error::from_reason("deepseek_v41: empty call"));
        }
        if let Some(limit) = self.ctx_limit
            && start + ids.len() > limit
        {
            return Err(Error::from_reason(format!(
                "deepseek_v41: {start} + {} tokens exceed the {limit}-token context limit",
                ids.len()
            )));
        }
        let rows = self.embed.shape_at(0)?;
        if let Some(bad) = ids.iter().find(|&&id| i64::from(id) >= rows) {
            return Err(Error::from_reason(format!(
                "deepseek_v41: token id {bad} outside the {rows}-row embedding"
            )));
        }
        let mut all = sess.tokens.clone();
        all.extend_from_slice(ids);
        let hash = self.hasher.as_ref().map(|h| h.rows(&all, start));
        if let (Some(t), Some(h)) = (trace.as_deref_mut(), &hash) {
            t.engram_rows = h.clone();
        }
        let tok = MxArray::from_uint32(ids, &[s])?;
        let h = self
            .embed
            .rows(&tok)?
            .astype(self.act)?
            .reshape(&[s, 1, dim])?
            .broadcast_to(&[s, hc, dim])?;
        let mut one_hot = vec![0f32; (s * hc) as usize];
        one_hot
            .iter_mut()
            .step_by(hc as usize)
            .for_each(|v| *v = 1.0);
        let pre_mix = MxArray::from_float32(&one_hot, &[s, hc])?;
        let mut shared = ForwardShared::default();
        let (h, pre_mix) = self.run_layers(
            sess,
            start,
            hash.as_deref(),
            0..self.cfg.n_layers,
            (h, pre_mix),
            &mut shared,
            cancel,
            trace,
            last_only && self.layer_skip,
        )?;
        let tp = super::stats::prof_start();
        let x = rms_norm(&hc_pre(&h, &pre_mix)?, &self.norm, self.cfg.norm_eps)?
            .astype(DType::Float32)?;
        let x = if last_only {
            x.slice(&[s - 1, 0], &[s, dim])?
        } else {
            x
        };
        fp32_site("head_input", self.cfg.n_layers, &x)?;
        let logits = self.head_logits(&x)?;
        MxArray::eval_arrays_with_context(&[&logits], "deepseek_v41 head")?;
        super::stats::prof_end(super::stats::Phase::Head, tp, &[])?;
        sess.tokens = all;
        Ok(logits)
    }

    /// Layers `layers` of one call at `start` on the stream `h` `[s, hc,
    /// dim]` with `pre_mix` `[s, hc]`; `hash` holds the call's engram rows.
    #[allow(clippy::too_many_arguments)]
    fn run_layers(
        &self,
        sess: &mut Session,
        start: usize,
        hash: Option<&[u32]>,
        layers: std::ops::Range<usize>,
        (mut h, mut pre_mix): (MxArray, MxArray),
        shared: &mut ForwardShared,
        cancel: Option<&dyn Fn(usize) -> bool>,
        mut trace: Option<&mut Trace>,
        skip: bool,
    ) -> Result<(MxArray, MxArray)> {
        let end = layers.end;
        let s = h.shape_at(0)? as usize;
        let from = |l: usize| if skip { self.skip_from(l, s) } else { 0 };
        for l in layers {
            let blk = &self.blocks[l];
            if cancel.is_some_and(|c| c(l)) {
                return Err(Error::from_reason(format!(
                    "deepseek_v41: cancelled before layer {l}"
                )));
            }
            if let (Some((e, eng)), Some(rows), Some(table)) = (&self.engrams[l], hash, &self.table)
            {
                let n = self.hasher.as_ref().map_or(1, |x| x.n_layers());
                let cols = self.hasher.as_ref().map_or(0, |x| x.n_cols());
                let ids: Vec<u32> = rows
                    .chunks(n * cols)
                    .flat_map(|p| p[e * cols..(e + 1) * cols].iter().copied())
                    .collect();
                let tp = super::stats::prof_start();
                h = eng.forward(&h, &table.rows(*e, &ids)?)?;
                super::stats::prof_end(super::stats::Phase::Engram, tp, &[&h])?;
            }
            let src = match self.cfg.kv_src[l] {
                Some(k) if self.cfg.compress_ratios[l] != 0 => sess.sources[k].as_mut(),
                _ => None,
            };
            let next = (l + 1 < end).then(|| (&self.blocks[l + 1].moe, from(l + 1)));
            let (x, pm, routes, scores) = blk.forward(
                &h,
                start,
                &pre_mix,
                &mut sess.wins[l],
                src,
                shared,
                next,
                from(l),
            )?;
            MxArray::eval_arrays_with_context(&[&x, &pm], "deepseek_v41 block")?;
            (h, pre_mix) = (x, pm);
            if let Some(t) = trace.as_deref_mut() {
                t.routes.push(routes);
                t.index_scores.push(scores);
                let idx_src = self.cfg.topk_src[l] == Some(l);
                t.topk
                    .push(if idx_src { shared.topk.clone() } else { None });
                let cand = self
                    .cfg
                    .candidate
                    .as_ref()
                    .is_some_and(|c| c.source_layer == l);
                t.candidates.push(if cand {
                    shared.candidates.clone()
                } else {
                    None
                });
                if t.keep_out {
                    t.out.push(h.clone());
                }
            }
        }
        Ok((h, pre_mix))
    }

    /// Test seam (layer groups vs oracle dumps): runs only `layers` of the
    /// call `ids` on an injected stream `h` `[s, hc, dim]` (activation dtype)
    /// and `pre_mix` `[s, hc]` (f32), with `shared` as handed down by the
    /// earlier layers; KV sources outside `layers` must already hold this
    /// call's state in `sess`. Returns the stream and the next pre-mix (no
    /// head). A failed call restores the session.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_layers(
        &self,
        sess: &mut Session,
        ids: &[u32],
        layers: std::ops::Range<usize>,
        h: MxArray,
        pre_mix: MxArray,
        mut shared: ForwardShared,
        cancel: Option<&dyn Fn(usize) -> bool>,
        mut trace: Option<&mut Trace>,
    ) -> Result<(MxArray, MxArray)> {
        let before = sess.clone();
        let (s, start) = (ids.len() as i64, sess.tokens.len());
        let (hc, dim) = (self.cfg.hc_mult as i64, self.cfg.dim as i64);
        let want = [s, hc, dim];
        if s == 0 || layers.is_empty() || layers.end > self.cfg.n_layers {
            return Err(Error::from_reason(format!(
                "deepseek_v41: layers {layers:?} for {s} ids"
            )));
        }
        if h.shape()?.as_ref() != want || pre_mix.shape()?.as_ref() != [s, hc] {
            return Err(Error::from_reason("deepseek_v41: injected stream shape"));
        }
        let mut all = sess.tokens.clone();
        all.extend_from_slice(ids);
        let hash = self.hasher.as_ref().map(|x| x.rows(&all, start));
        if let (Some(t), Some(x)) = (trace.as_deref_mut(), &hash) {
            t.engram_rows = x.clone();
        }
        let out = self.run_layers(
            sess,
            start,
            hash.as_deref(),
            layers,
            (h, pre_mix),
            &mut shared,
            cancel,
            trace,
            false,
        );
        match out {
            Ok(v) => {
                sess.tokens = all;
                Ok(v)
            }
            Err(e) => {
                *sess = before;
                Err(e)
            }
        }
    }

    /// f32 logits of `x [s, dim]` (f32). A Q8_0 head is dequantized
    /// `HEAD_CHUNK` vocab rows at a time; an f32 head larger than one chunk
    /// takes the same chunks (bitwise == the dequantized one); a `Qmm` head
    /// is one `quantized_matmul`.
    fn head_logits(&self, x: &MxArray) -> Result<MxArray> {
        let w = &self.head;
        let vocab = w.shape_at(0)?;
        match w {
            Dense::Array(a) if vocab <= HEAD_CHUNK => {
                return x.matmul(&a.astype(DType::Float32)?.transpose(Some(&[1, 0]))?);
            }
            Dense::Qmm { .. } => return w.mm_t(x),
            _ => {}
        }
        let mut parts = Vec::new();
        let mut lo = 0;
        while lo < vocab {
            let hi = (lo + HEAD_CHUNK).min(vocab);
            let p = x.matmul(&w.row_range(lo, hi)?.f32()?.transpose(Some(&[1, 0]))?)?;
            MxArray::eval_arrays_with_context(&[&p], "deepseek_v41 head chunk")?;
            parts.push(p);
            lo = hi;
        }
        match parts.len() {
            1 => Ok(parts.swap_remove(0)),
            _ => MxArray::concatenate_many(parts.iter().collect(), Some(1)),
        }
    }
}

/// Vocab rows per dequantized head chunk (16384 x 5120 f32 = 320 MiB).
const HEAD_CHUNK: i64 = 16384;
