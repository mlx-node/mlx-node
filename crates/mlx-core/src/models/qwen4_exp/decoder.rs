use super::{config::Config, math, weights::Store};
use crate::array::{DType, MxArray, scaled_dot_product_attention};
use crate::nn::Activations;
use napi::{Error, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
#[path = "attention_window.rs"]
mod attention_window;
#[path = "batch.rs"]
mod batch;
#[path = "device_routes.rs"]
mod device_routes;
#[path = "draft_window.rs"]
mod draft_window;
#[path = "scheduled_batch.rs"]
mod scheduled_batch;

/// Slot banks for a window's expert groups: a lease per group, then the
/// release of that group's output before the slots serve the next group.
pub(super) trait GroupBanks {
    fn lease(
        &mut self,
        experts: &[u32],
    ) -> Result<Option<([Arc<super::weights::Weight>; 3], Vec<u32>, usize)>>;
    fn release(&mut self, output: &MxArray) -> Result<()>;
}

struct AttentionProjections {
    qg: MxArray,
    k: MxArray,
    v: MxArray,
    iq: MxArray,
    ik: MxArray,
}

struct GdnProjections {
    qkv: MxArray,
    z: MxArray,
    a: MxArray,
    b: MxArray,
}
impl GdnProjections {
    fn row(&self, row: usize) -> Result<Self> {
        let slice = |a: &MxArray| a.slice_axis(1, row as i64, row as i64 + 1);
        Ok(Self {
            qkv: slice(&self.qkv)?,
            z: slice(&self.z)?,
            a: slice(&self.a)?,
            b: slice(&self.b)?,
        })
    }
}
impl AttentionProjections {
    fn row(&self, row: usize) -> Result<Self> {
        let slice = |a: &MxArray| a.slice_axis(1, row as i64, row as i64 + 1);
        Ok(Self {
            qg: slice(&self.qg)?,
            k: slice(&self.k)?,
            v: slice(&self.v)?,
            iq: slice(&self.iq)?,
            ik: slice(&self.ik)?,
        })
    }
}

const DRAFT_HISTORY_STEP: usize = 256;
const DRAFT_HEAD_ROWS: usize = 3;

#[derive(Clone, Default)]
pub struct LayerCache {
    conv: Option<MxArray>,
    recurrent: Option<MxArray>,
    keys: Option<MxArray>,
    values: Option<MxArray>,
    /// Draft history: the rows of `keys`/`values` this cache owns. Clones
    /// share one buffer and write in place past their own rows, so no cache
    /// may read past its own rows.
    draft_rows: Option<usize>,
    index_tail: Vec<MxArray>,
    /// `[blocks, indexer_head_dim]`. Snapshots and replay share handles, so
    /// appending must build a new array, never write into this one.
    index_keys: Option<MxArray>,
    ple_conv: Option<MxArray>,
}
impl LayerCache {
    fn index_block_count(&self) -> Result<usize> {
        match &self.index_keys {
            Some(keys) => Ok(keys.shape_at(0)? as usize),
            None => Ok(0),
        }
    }
    fn append_index_keys(&mut self, keys: MxArray) -> Result<&MxArray> {
        let keys = match &self.index_keys {
            Some(old) => MxArray::concatenate(old, &keys, 0)?,
            None => keys,
        };
        Ok(self.index_keys.insert(keys))
    }
    /// Write draft K/V rows after this cache's rows. A buffer too small for
    /// them is replaced in this cache only: other holders keep reading theirs.
    pub(super) fn append_draft_kv(
        &mut self,
        k: &MxArray,
        v: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let rows = self.draft_rows.unwrap_or(0);
        let end = rows + k.shape_at(2)? as usize;
        let capacity = match &self.keys {
            Some(keys) => keys.shape_at(2)? as usize,
            None => 0,
        };
        if end > capacity {
            let target = (end + DRAFT_HEAD_ROWS).next_multiple_of(DRAFT_HISTORY_STEP);
            let grow = |old: &Option<MxArray>, update: &MxArray| -> Result<MxArray> {
                let shape = [
                    update.shape_at(0)?,
                    update.shape_at(1)?,
                    (target - rows) as i64,
                    update.shape_at(3)?,
                ];
                let fresh = MxArray::zeros(&shape, Some(update.dtype()?))?;
                match old {
                    Some(old) if rows > 0 => {
                        MxArray::concatenate(&old.slice_axis(2, 0, rows as i64)?, &fresh, 2)
                    }
                    _ => Ok(fresh),
                }
            };
            self.keys = Some(grow(&self.keys, k)?);
            self.values = Some(grow(&self.values, v)?);
        }
        let (Some(keys), Some(values)) = (self.keys.as_mut(), self.values.as_mut()) else {
            return Err(Error::from_reason(
                "Qwen4 MTP draft history lost its buffer",
            ));
        };
        keys.slice_assign_axis_inplace(2, rows as i64, end as i64, k)?;
        values.slice_assign_axis_inplace(2, rows as i64, end as i64, v)?;
        self.draft_rows = Some(end);
        Ok((keys.clone(), values.clone()))
    }
    /// (rows, K buffer, V buffer) of a draft history.
    #[cfg(test)]
    pub(super) fn draft_kv(&self) -> Option<(usize, &MxArray, &MxArray)> {
        Some((self.draft_rows?, self.keys.as_ref()?, self.values.as_ref()?))
    }
    /// (K/V rows, indexer blocks, indexer tail rows).
    #[cfg(test)]
    pub(super) fn history_shape(&self) -> (usize, usize, usize) {
        let rows = self.draft_rows.unwrap_or_else(|| {
            self.keys
                .as_ref()
                .map_or(0, |keys| keys.shape_at(2).unwrap() as usize)
        });
        (
            rows,
            self.index_block_count().unwrap(),
            self.index_tail.len(),
        )
    }
    pub(super) fn arrays(&self) -> impl Iterator<Item = &MxArray> {
        self.conv
            .iter()
            .chain(self.recurrent.iter())
            .chain(self.ple_conv.iter())
            .chain(self.keys.iter())
            .chain(self.values.iter())
            .chain(self.index_keys.iter())
            .chain(self.index_tail.iter())
    }
}
#[derive(Clone, Default)]
pub struct DecoderState {
    caches: Vec<LayerCache>,
    history: Vec<u32>,
    positions: Vec<[i64; 3]>,
    rope_delta: i64,
    media_digests: Vec<crate::engine::cache::ImageCacheDigest>,
    last_chunk_hidden: Vec<MxArray>,
    cancelled: Option<Arc<AtomicBool>>,
}
impl DecoderState {
    pub(super) fn arrays(&self) -> impl Iterator<Item = &MxArray> {
        self.caches
            .iter()
            .flat_map(LayerCache::arrays)
            .chain(self.last_chunk_hidden.iter())
    }
}
pub struct Decoder {
    pub prefill_chunk_size: usize,
    /// A test that sizes admission through the prefill window keeps its
    /// window across the turn opens that re-derive it from the plan.
    #[cfg(test)]
    pub(super) prefill_window_pin: Option<usize>,
    // Like TrackFastModel's retained configuration, share immutable metadata
    // across forward helpers instead of copying its strings/lists per layer.
    pub config: Arc<Config>,
    pub weights: Store,
    pub caches: Vec<LayerCache>,
    pub history: Vec<u32>,
    pub positions: Vec<[i64; 3]>,
    pub rope_delta: i64,
    pub media_digests: Vec<crate::engine::cache::ImageCacheDigest>,
    pub cancelled: Option<Arc<AtomicBool>>,
    pub paged: Option<crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter>,
    paged_chunk_start: usize,
    scope: String,
    position_override: Option<usize>,
    rotary_tables: std::cell::RefCell<math::RotaryWindowCache>,
    verification: Option<Vec<DecoderState>>,
    device_routes: Option<Vec<(usize, MxArray)>>,
    device_route_cooldown: usize,
    /// Layers whose routes a pass read back to the host one at a time.
    #[cfg(test)]
    pub(super) route_readbacks: u64,
    #[cfg(test)]
    pub(super) submit_log: std::cell::RefCell<Vec<String>>,
    pub last_chunk_hidden: Vec<MxArray>,
    /// The final residual streams of the last layer-major group, one per
    /// window, until its views are taken.
    group_streams: Vec<MxArray>,
    /// Tests pin group mode on or off in place of the live rule.
    #[cfg(test)]
    pub(super) group_pin: Option<bool>,
    /// Tentative commits taken, each with the windows it covered.
    #[cfg(test)]
    pub(super) tentative_commits: Vec<usize>,
    /// Every (rows, elapsed) sample handed to the tripwire, in order.
    #[cfg(test)]
    pub(super) pass_samples: Vec<(usize, std::time::Duration)>,
    /// Runs after every layer sweep of a group, with the layer; tests trip
    /// the wire or cancel between two sweeps through it.
    #[cfg(test)]
    pub(super) after_sweep: Option<Box<dyn FnMut(&mut Decoder, usize) + Send>>,
    /// The weight budget this store registers with the process-wide cache
    /// limit; re-registered whenever the residency changes, between turns or
    /// between two passes of one request.
    cache_limit_guard: Option<crate::cache_limit::CacheLimitGuard>,
    /// Runs after every observed pass; tests cut the admitted window between
    /// two passes of one prompt through it.
    #[cfg(test)]
    pub(super) after_pass: Option<Box<dyn FnMut(&mut Decoder) + Send>>,
}
impl Decoder {
    /// Register the budget in force with the process-wide cache limit.
    pub fn register_cache_limit(&mut self) {
        self.cache_limit_guard = None;
        self.cache_limit_guard =
            Some(crate::cache_limit::coordinator().register(self.weights.cache_budget()));
    }

    /// The prefill window follows the plan in force (mapped or fallback).
    pub fn refresh_prefill_window(&mut self) {
        let window = super::memory::prefill_window(&self.config, &self.weights.plan);
        #[cfg(test)]
        let window = self.prefill_window_pin.unwrap_or(window);
        self.prefill_chunk_size = window;
        self.weights
            .live
            .prefill_chunk_tokens
            .store(self.prefill_chunk_size as u64, Ordering::Release);
    }

    /// The store changed its residency: the budget it registers and the
    /// prefill window it admits follow.
    pub fn residency_changed(&mut self) {
        self.register_cache_limit();
        self.refresh_prefill_window();
    }

    /// One main-network pass over `rows` rows took `elapsed`. A pass that
    /// trips the wire moves the store to slots before the next one; the
    /// request goes on with its state intact, and `residencyInfo()` records
    /// the position it left the mapped path at.
    pub(super) fn observe_pass(&mut self, rows: usize, elapsed: std::time::Duration) {
        #[cfg(test)]
        self.pass_samples.push((rows, elapsed));
        match self.weights.observe_pass(rows, elapsed) {
            Ok(None) => {}
            Ok(Some(_)) => {
                self.weights
                    .live
                    .fallback_token_index
                    .store(self.history.len() as u64, Ordering::Release);
                self.residency_changed();
            }
            Err(error) => {
                tracing::warn!("qwen4 residency: slot fallback not applied: {error}")
            }
        }
        #[cfg(test)]
        if let Some(mut hook) = self.after_pass.take() {
            hook(self);
            self.after_pass = Some(hook);
        }
    }

    pub fn prefill_slice_size(&self) -> usize {
        self.prefill_chunk_size
    }

    pub fn new(config: Config, weights: Store) -> Result<Self> {
        // Validate the checkpoint topology before the first tensor payload read.
        let prefill_chunk_size = super::memory::prefill_window(&config, &weights.plan);
        let d = Self {
            prefill_chunk_size,
            #[cfg(test)]
            prefill_window_pin: None,
            caches: (0..config.num_hidden_layers)
                .map(|_| LayerCache::default())
                .collect(),
            config: Arc::new(config),
            weights,
            history: Vec::new(),
            positions: Vec::new(),
            rope_delta: 0,
            media_digests: Vec::new(),
            cancelled: None,
            paged: None,
            paged_chunk_start: 0,
            scope: String::new(),
            position_override: None,
            rotary_tables: Default::default(),
            verification: None,
            device_routes: None,
            device_route_cooldown: 0,
            #[cfg(test)]
            route_readbacks: 0,
            #[cfg(test)]
            submit_log: Default::default(),
            last_chunk_hidden: Vec::new(),
            group_streams: Vec::new(),
            #[cfg(test)]
            group_pin: None,
            #[cfg(test)]
            tentative_commits: Vec::new(),
            #[cfg(test)]
            pass_samples: Vec::new(),
            #[cfg(test)]
            after_sweep: None,
            cache_limit_guard: None,
            #[cfg(test)]
            after_pass: None,
        };
        d.preflight()?;
        Ok(d)
    }
    pub fn snapshot(&self) -> DecoderState {
        DecoderState {
            caches: self.caches.clone(),
            history: self.history.clone(),
            positions: self.positions.clone(),
            rope_delta: self.rope_delta,
            media_digests: self.media_digests.clone(),
            last_chunk_hidden: self.last_chunk_hidden.clone(),
            cancelled: self.cancelled.clone(),
        }
    }
    #[cfg(test)]
    pub(super) fn index_keys(&self, layer: usize) -> Option<&MxArray> {
        self.caches[layer].index_keys.as_ref()
    }
    pub fn take_state(&mut self) -> DecoderState {
        let state = DecoderState {
            caches: std::mem::take(&mut self.caches),
            history: std::mem::take(&mut self.history),
            positions: std::mem::take(&mut self.positions),
            rope_delta: self.rope_delta,
            media_digests: self.media_digests.clone(),
            last_chunk_hidden: std::mem::take(&mut self.last_chunk_hidden),
            cancelled: self.cancelled.take(),
        };
        self.reset();
        state
    }
    pub fn restore_state(&mut self, state: DecoderState) {
        self.rotary_tables.get_mut().clear();
        self.caches = state.caches;
        self.history = state.history;
        self.positions = state.positions;
        self.rope_delta = state.rope_delta;
        self.media_digests = state.media_digests;
        self.last_chunk_hidden = state.last_chunk_hidden;
        self.cancelled = state.cancelled;
    }
    pub fn reset(&mut self) {
        self.rotary_tables.get_mut().clear();
        self.verification = None;
        self.device_routes = None;
        self.device_route_cooldown = 0;
        self.history.clear();
        self.positions.clear();
        self.rope_delta = 0;
        self.media_digests.clear();
        self.last_chunk_hidden.clear();
        self.group_streams.clear();
        self.caches = (0..self.config.num_hidden_layers)
            .map(|_| LayerCache::default())
            .collect();
    }
    fn rope(&self, x: &MxArray, pos: usize) -> Result<MxArray> {
        if !self.scope.is_empty() || self.positions.is_empty() {
            return math::rope(x, pos, self.config.rope_dims(), self.config.rope_theta());
        }
        let v = &self.config.rope_parameters;
        let p = self
            .positions
            .get(pos)
            .copied()
            .unwrap_or([pos as i64 + self.rope_delta; 3]);
        let sections =
            std::array::from_fn(|i| v["mrope_section"][i].as_u64().unwrap_or(0) as usize);
        let interleaved = v["mrope_interleaved"].as_bool().unwrap_or(true);
        math::mrope(
            x,
            p,
            self.config.rope_dims(),
            self.config.rope_theta(),
            sections,
            interleaved,
        )
    }
    #[cfg(test)]
    pub(super) fn rope_row(&self, x: &MxArray, pos: usize) -> Result<MxArray> {
        self.rope(x, pos)
    }
    fn gguf(&self) -> bool {
        self.weights.gguf && self.scope.is_empty()
    }
    fn key(&self, hf: &str, gg: &str) -> String {
        if self.gguf() {
            gg.into()
        } else {
            format!("{}{hf}", self.scope)
        }
    }
    fn dense(&mut self, hf: &str, gg: &str) -> Result<MxArray> {
        self.weights.dense(&self.key(hf, gg))
    }
    fn linear(&mut self, x: &MxArray, hf: &str, gg: &str) -> Result<MxArray> {
        self.weights.linear(&self.key(hf, gg), x)
    }
    fn linear_pair(
        &mut self,
        x: &MxArray,
        a: (&str, &str),
        b: (&str, &str),
    ) -> Result<(MxArray, MxArray)> {
        self.weights
            .linear_pair(x, &self.key(a.0, a.1), &self.key(b.0, b.1))
    }
    /// Row-exact projections of a verification window (`[1, R, K]`, `R <= 4`):
    /// row `j` carries the bits of the single-row projection. Outside a
    /// verification pass a window is a prefill window, whatever its row
    /// count, and takes the batched projection: the row kernels admit four
    /// rows, and every other row would run a single-row projection of its own
    /// (a 6k prompt dispatched 1.5 M of them).
    fn linear_rows(&mut self, x: &MxArray, hf: &str, gg: &str) -> Result<MxArray> {
        if self.verification.is_none() {
            return self.linear(x, hf, gg);
        }
        self.weights.linear_rows(&self.key(hf, gg), x)
    }
    fn linear_pair_rows(
        &mut self,
        x: &MxArray,
        a: (&str, &str),
        b: (&str, &str),
    ) -> Result<(MxArray, MxArray)> {
        if self.verification.is_none() {
            return self.linear_pair(x, a, b);
        }
        self.weights
            .linear_pair_rows(x, &self.key(a.0, a.1), &self.key(b.0, b.1))
    }
    fn norm(
        &mut self,
        x: &MxArray,
        hf: &str,
        gg: &str,
        group: usize,
        centered: bool,
    ) -> Result<MxArray> {
        let w = self.dense(hf, gg)?;
        math::norm(
            x,
            &w,
            group,
            self.config.rms_norm_eps,
            centered && !self.gguf(),
        )
    }
    pub fn check_cancelled(&self) -> Result<()> {
        if self
            .cancelled
            .as_ref()
            .is_some_and(|c| c.load(Ordering::Relaxed))
        {
            return Err(Error::from_reason("prefill cancelled"));
        }
        Ok(())
    }
    fn shape(&self, hf: &str, gg: &str, want: &[usize]) -> Result<()> {
        let key = self.key(hf, gg);
        let shape = &self.weights.descriptor(&key)?.shape;
        if shape != want {
            return Err(Error::from_reason(format!(
                "qwen4_exp tensor {key}: expected {want:?}, got {shape:?}"
            )));
        }
        Ok(())
    }
    fn preflight(&self) -> Result<()> {
        let c = &self.config;
        let h = c.hidden_size;
        let hc = h * c.hc_count;
        self.shape(
            "embed_tokens.weight",
            "token_embd.weight",
            &[c.vocab_size, h],
        )?;
        self.shape("lm_head.weight", "output.weight", &[c.vocab_size, h])?;
        for (hf, gg, shape) in [
            (
                "hyper_connection_mixer.hc_norm.weight",
                "output_hc_norm.weight",
                vec![hc],
            ),
            (
                "hyper_connection_mixer.input_mix_weight_down.weight",
                "output_hc_down.weight",
                vec![c.hc_lowrank, hc],
            ),
            (
                "hyper_connection_mixer.input_mix_weight_up.weight",
                "output_hc_up.weight",
                vec![hc, c.hc_lowrank],
            ),
        ] {
            self.shape(hf, gg, &shape)?;
        }
        for i in 0..c.num_hidden_layers {
            for (hf, gg) in [
                ("attn_hyper_connection", "hc_attn"),
                ("mlp_hyper_connection", "hc_ffn"),
            ] {
                self.shape(
                    &format!("layers.{i}.{hf}.input_mix_weight_down.weight"),
                    &format!("blk.{i}.{gg}_down.weight"),
                    &[c.hc_lowrank, hc],
                )?;
                self.shape(
                    &format!("layers.{i}.{hf}.input_mix_weight_up.weight"),
                    &format!("blk.{i}.{gg}_up.weight"),
                    &[hc, c.hc_lowrank],
                )?;
                self.shape(
                    &format!("layers.{i}.{hf}.hc_norm.weight"),
                    &format!("blk.{i}.{gg}_norm.weight"),
                    &[hc],
                )?;
                self.shape(
                    &format!("layers.{i}.{hf}.block_inject_weight.weight"),
                    &format!("blk.{i}.{gg}_inject.weight"),
                    &[c.hc_count, hc],
                )?;
            }
            let kd = c.linear_num_key_heads * c.linear_key_head_dim;
            let vd = c.linear_num_value_heads * c.linear_value_head_dim;
            let p = format!("layers.{i}");
            let g = format!("blk.{i}");
            if c.linear(i) {
                for (hf, gg, shape) in [
                    (
                        "linear_attn.in_proj_qkv.weight",
                        "attn_qkv.weight",
                        vec![2 * kd + vd, h],
                    ),
                    (
                        "linear_attn.in_proj_z.weight",
                        "attn_gate.weight",
                        vec![vd, h],
                    ),
                    (
                        "linear_attn.in_proj_a.weight",
                        "ssm_alpha.weight",
                        vec![c.linear_num_value_heads, h],
                    ),
                    (
                        "linear_attn.in_proj_b.weight",
                        "ssm_beta.weight",
                        vec![c.linear_num_value_heads, h],
                    ),
                    ("linear_attn.out_proj.weight", "ssm_out.weight", vec![h, vd]),
                    ("linear_attn.A_log", "ssm_a", vec![c.linear_num_value_heads]),
                    (
                        "linear_attn.dt_bias",
                        "ssm_dt.bias",
                        vec![c.linear_num_value_heads],
                    ),
                    (
                        "linear_attn.norm.weight",
                        "ssm_norm.weight",
                        vec![c.linear_value_head_dim],
                    ),
                ] {
                    self.shape(&format!("{p}.{hf}"), &format!("{g}.{gg}"), &shape)?;
                }
                self.shape(
                    &format!("{p}.linear_attn.conv1d.weight"),
                    &format!("{g}.ssm_conv1d.weight"),
                    &if self.gguf() {
                        vec![2 * kd + vd, c.linear_conv_kernel_dim]
                    } else {
                        vec![2 * kd + vd, 1, c.linear_conv_kernel_dim]
                    },
                )?;
            } else {
                for (hf, gg, shape) in [
                    (
                        "self_attn.q_proj.weight",
                        "attn_q.weight",
                        vec![2 * c.num_attention_heads * c.head_dim, h],
                    ),
                    (
                        "self_attn.k_proj.weight",
                        "attn_k.weight",
                        vec![c.num_key_value_heads * c.head_dim, h],
                    ),
                    (
                        "self_attn.v_proj.weight",
                        "attn_v.weight",
                        vec![c.num_key_value_heads * c.head_dim, h],
                    ),
                    (
                        "self_attn.o_proj.weight",
                        "attn_output.weight",
                        vec![h, c.num_attention_heads * c.head_dim],
                    ),
                    (
                        "self_attn.q_norm.weight",
                        "attn_q_norm.weight",
                        vec![c.head_dim],
                    ),
                    (
                        "self_attn.k_norm.weight",
                        "attn_k_norm.weight",
                        vec![c.head_dim],
                    ),
                    (
                        "self_attn.indexer.q_layernorm.weight",
                        "indexer.q_norm.weight",
                        vec![c.indexer_head_dim],
                    ),
                    (
                        "self_attn.indexer.k_layernorm.weight",
                        "indexer.k_norm.weight",
                        vec![c.indexer_head_dim],
                    ),
                ] {
                    self.shape(&format!("{p}.{hf}"), &format!("{g}.{gg}"), &shape)?;
                }
                if self.gguf() {
                    self.shape(
                        "",
                        &format!("{g}.indexer.q_proj.weight"),
                        &[c.indexer_n_heads * c.indexer_head_dim, h],
                    )?;
                    self.shape(
                        "",
                        &format!("{g}.indexer.k_proj.weight"),
                        &[c.indexer_head_dim, h],
                    )?;
                } else {
                    self.shape(
                        &format!("{p}.self_attn.indexer.index_qk_proj.weight"),
                        "",
                        &[(c.indexer_n_heads + 1) * c.indexer_head_dim, h],
                    )?;
                }
            }
            for (hf, gg, shape) in [
                (
                    "shared_expert.gate_proj.weight",
                    "ffn_gate_shexp.weight",
                    vec![c.shared_expert_intermediate_size, h],
                ),
                (
                    "shared_expert.up_proj.weight",
                    "ffn_up_shexp.weight",
                    vec![c.shared_expert_intermediate_size, h],
                ),
                (
                    "shared_expert.down_proj.weight",
                    "ffn_down_shexp.weight",
                    vec![h, c.shared_expert_intermediate_size],
                ),
            ] {
                self.shape(&format!("{p}.mlp.{hf}"), &format!("{g}.{gg}"), &shape)?;
            }
            self.shape(
                &format!("{p}.mlp.shared_expert_gate.weight"),
                &format!("{g}.ffn_gate_inp_shexp.weight"),
                &if self.gguf() { vec![h] } else { vec![1, h] },
            )?;
            if c.ple_layer_ids.contains(&(i + 1)) {
                for (hf, gg, shape) in [
                    (
                        "key_proj.weight",
                        "ple_key.weight",
                        vec![hc, c.ple_embed_dim],
                    ),
                    (
                        "value_proj.weight",
                        "ple_value.weight",
                        vec![h, c.ple_embed_dim],
                    ),
                    ("norm_key.weight", "ple_norm_key.weight", vec![hc]),
                    ("norm_query.weight", "ple_norm_query.weight", vec![hc]),
                    ("norm_conv.weight", "ple_norm_conv.weight", vec![hc]),
                ] {
                    self.shape(&format!("{p}.ple.{hf}"), &format!("{g}.{gg}"), &shape)?;
                }
                self.shape(
                    &format!("{p}.ple.conv1d.weight"),
                    &format!("{g}.ple_conv1d.weight"),
                    &if self.gguf() {
                        vec![hc, c.ple_conv_kernel_size]
                    } else {
                        vec![hc, 1, c.ple_conv_kernel_size]
                    },
                )?;
                let heads = (c.ngram_size - 1) * c.heads_per_ngram;
                let mut rows = 0usize;
                if self.gguf() {
                    let table = self.weights.descriptor("per_layer_token_embd.weight")?;
                    if table.shape.len() != 2 || table.width()? != c.ple_embed_dim / heads {
                        return Err(Error::from_reason("Invalid GGUF PLE shape"));
                    }
                    rows = table.rows()?;
                } else {
                    for shard in 0..c.split_ngram_parts {
                        let table = self.weights.descriptor(&format!(
                            "{p}.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
                        ))?;
                        if table.shape.len() != 2 || table.width()? != c.ple_embed_dim / heads {
                            return Err(Error::from_reason("Invalid PLE shard shape"));
                        }
                        rows = rows
                            .checked_add(table.rows()?)
                            .ok_or_else(|| Error::from_reason("PLE row count overflow"))?;
                    }
                }
                let constants = |hf: &str, gg: &str| {
                    if self.gguf() {
                        super::gguf::integers(&self.weights.metadata, &format!("qwen4exp.ple.{gg}"))
                    } else {
                        self.weights
                            .integers(&format!("{p}.ple.ple_embedding.{hf}"))
                    }
                };
                let sizes = constants("ngram_heads_vocab_sizes", "head_vocab_sizes")?;
                let offsets = constants("ngram_heads_offsets", "head_offsets")?;
                let multipliers = constants("layer_multipliers", "layer_multipliers")?;
                math::hash_ids(
                    0,
                    &[],
                    c.eos_token_id,
                    c.ngram_size,
                    c.heads_per_ngram,
                    &multipliers,
                    &sizes,
                    &offsets,
                )?;
                let mut end = 0u64;
                for (&size, &offset) in sizes.iter().zip(&offsets) {
                    if offset != end {
                        return Err(Error::from_reason(
                            "PLE hash vocabularies overlap or contain gaps",
                        ));
                    }
                    end = end
                        .checked_add(size)
                        .ok_or_else(|| Error::from_reason("PLE vocabulary size overflow"))?;
                }
                if end > rows as u64 {
                    return Err(Error::from_reason(
                        "PLE table does not cover its hashed vocabulary",
                    ));
                }
            }
            self.shape(
                &format!("layers.{i}.mlp.gate.weight"),
                &format!("blk.{i}.ffn_gate_inp.weight"),
                &[c.num_experts, h],
            )?;
            if self.gguf() {
                self.shape(
                    "",
                    &format!("blk.{i}.ffn_gate_exps.weight"),
                    &[c.num_experts, c.moe_intermediate_size, h],
                )?;
                self.shape(
                    "",
                    &format!("blk.{i}.ffn_up_exps.weight"),
                    &[c.num_experts, c.moe_intermediate_size, h],
                )?;
            } else {
                self.shape(
                    &format!("layers.{i}.mlp.experts.gate_up_proj"),
                    "",
                    &[c.num_experts, 2 * c.moe_intermediate_size, h],
                )?;
            }
            self.shape(
                &format!("layers.{i}.mlp.experts.down_proj"),
                &format!("blk.{i}.ffn_down_exps.weight"),
                &[c.num_experts, h, c.moe_intermediate_size],
            )?;
        }
        Ok(())
    }
    fn hyper(
        &mut self,
        x: &MxArray,
        hf: &str,
        gg: &str,
        inject: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        self.hyper_with_norm(x, hf, gg, inject, None)
    }

    fn hyper_with_norm(
        &mut self,
        x: &MxArray,
        hf: &str,
        gg: &str,
        inject: bool,
        normed: Option<&MxArray>,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let c = self.config.clone();
        let n = if let Some(normed) = normed {
            normed.clone()
        } else {
            self.norm(
                x,
                &format!("{hf}.hc_norm.weight"),
                &format!("{gg}_norm.weight"),
                c.hidden_size,
                true,
            )?
        };
        let activated = if self.gguf()
            && c.hc_count == 4
            && c.hidden_size == 2560
            && (x.shape()?[1] >= 1024 || x.shape()?[1] == 1)
        {
            let down_key = self.key(
                &format!("{hf}.input_mix_weight_down.weight"),
                &format!("{gg}_down.weight"),
            );
            self.weights
                .resident_bank(&down_key)
                .map(|w| w.mixer_act(&n))
                .transpose()?
                .flatten()
        } else {
            None
        };
        let is_activated = activated.is_some();
        let (down, inject_projection) = if let Some(activated) = activated {
            let gate = if inject {
                Some(self.linear(
                    &n,
                    &format!("{hf}.block_inject_weight.weight"),
                    &format!("{gg}_inject.weight"),
                )?)
            } else {
                None
            };
            (activated, gate)
        } else if inject {
            let (down, gate) = self.linear_pair(
                &n,
                (
                    &format!("{hf}.input_mix_weight_down.weight"),
                    &format!("{gg}_down.weight"),
                ),
                (
                    &format!("{hf}.block_inject_weight.weight"),
                    &format!("{gg}_inject.weight"),
                ),
            )?;
            (down, Some(gate))
        } else {
            (
                self.linear(
                    &n,
                    &format!("{hf}.input_mix_weight_down.weight"),
                    &format!("{gg}_down.weight"),
                )?,
                None,
            )
        };
        let down = if is_activated {
            down
        } else {
            Activations::silu(&down.div_scalar(c.hc_count as f64)?)?
        };
        let key = self.key(
            &format!("{hf}.input_mix_weight_up.weight"),
            &format!("{gg}_up.weight"),
        );
        let fused = if self.gguf() && c.hc_count == 4 && c.hidden_size == 2560 {
            self.weights
                .resident_bank(&key)
                .map(|w| w.hyper_up(&down, &n))
                .transpose()?
                .flatten()
        } else {
            None
        };
        let mixed = if let Some(mixed) = fused {
            mixed
        } else {
            let up = self.weights.linear(&key, &down)?;
            let combined = if self.gguf()
                && crate::engine::persistence::compiled_forward_backend_available()
            {
                unsafe { mlx_sys::mlx_qwen4_prefill_hc_mix(up.as_raw_ptr(), n.as_raw_ptr()) }
            } else {
                std::ptr::null_mut()
            };
            if combined.is_null() {
                math::sigmoid_mul(&up, &n)?
                    .reshape(&[1, x.shape()?[1], c.hc_count as i64, c.hidden_size as i64])?
                    .mean(Some(&[-2]), Some(false))?
            } else {
                MxArray::from_handle(combined, "Qwen4 prefill mixer combine")?
            }
        };
        let gate = if inject {
            let w = inject_projection.ok_or_else(|| {
                Error::from_reason("Qwen4 hyper-connection is missing its injection projection")
            })?;
            Some(Activations::sigmoid(&w.div_scalar(c.hc_count as f64)?)?.mul_scalar(2.0)?)
        } else {
            None
        };
        Ok((mixed, gate))
    }

    // Normalized results stay on the forward's call stack, with no
    // cache/frontier state to restore after a device-route replay.
    fn inject_for_mlp(
        &mut self,
        x: &MxArray,
        branch: &MxArray,
        gate: &MxArray,
        layer: usize,
        normalize: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        self.inject_for_hyper(
            x,
            branch,
            gate,
            &format!("layers.{layer}.mlp_hyper_connection.hc_norm.weight"),
            &format!("blk.{layer}.hc_ffn_norm.weight"),
            normalize,
        )
    }

    fn inject_for_hyper(
        &mut self,
        x: &MxArray,
        branch: &MxArray,
        gate: &MxArray,
        hf: &str,
        gg: &str,
        normalize: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        if normalize
            && self.gguf()
            && self.config.hidden_size == 2560
            && self.config.hc_count == 4
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            let w = self.dense(hf, gg)?;
            let (mut stream, mut normed) = (std::ptr::null_mut(), std::ptr::null_mut());
            if unsafe {
                mlx_sys::mlx_qwen4_inject_norm(
                    x.as_raw_ptr(),
                    branch.as_raw_ptr(),
                    gate.as_raw_ptr(),
                    w.as_raw_ptr(),
                    self.config.rms_norm_eps,
                    &mut stream,
                    &mut normed,
                )
            } {
                return Ok((
                    MxArray::from_handle(stream, "Qwen4 fused injection stream")?,
                    Some(MxArray::from_handle(
                        normed,
                        "Qwen4 fused injection normalization",
                    )?),
                ));
            }
        }
        Ok((Self::inject(x, branch, gate)?, None))
    }
    fn submit(&self, arrays: &[&MxArray], context: &str) -> Result<()> {
        #[cfg(test)]
        self.submit_log.borrow_mut().push(context.to_owned());
        if crate::engine::persistence::compiled_forward_backend_available() {
            MxArray::async_eval_arrays(arrays);
            Ok(())
        } else {
            MxArray::eval_arrays_with_context(arrays, context)
        }
    }
    fn submit_state(&self, arrays: &[&MxArray], context: &str) -> Result<()> {
        // The reference submits whole layer groups. Inside a checked device
        // transaction, its final join also completes every auxiliary state
        // before bank readers are released or a failed window is replayed.
        if self.deferred_state_submission() {
            Ok(())
        } else {
            self.submit(arrays, context)
        }
    }
    fn inject(x: &MxArray, y: &MxArray, g: &MxArray) -> Result<MxArray> {
        if crate::engine::persistence::compiled_forward_backend_available() {
            return MxArray::from_handle(
                unsafe {
                    mlx_sys::mlx_qwen4_inject(x.as_raw_ptr(), y.as_raw_ptr(), g.as_raw_ptr())
                },
                "Qwen4 compiled residual injection",
            );
        }
        x.add(
            &y.expand_dims(-2)?
                .mul(&g.expand_dims(-1)?)?
                .reshape(&x.shape()?)?,
        )
    }
    fn gdn_projections(&mut self, x: &MxArray, p: &str, g: &str) -> Result<GdnProjections> {
        let (qkv, z) = self.linear_pair_rows(
            x,
            (
                &format!("{p}.in_proj_qkv.weight"),
                &format!("{g}.attn_qkv.weight"),
            ),
            (
                &format!("{p}.in_proj_z.weight"),
                &format!("{g}.attn_gate.weight"),
            ),
        )?;
        let z = z.reshape(&[
            1,
            x.shape_at(1)?,
            self.config.linear_num_value_heads as i64,
            self.config.linear_value_head_dim as i64,
        ])?;
        let a = self
            .linear_rows(
                x,
                &format!("{p}.in_proj_a.weight"),
                &format!("{g}.ssm_alpha.weight"),
            )?
            .astype(DType::Float32)?;
        let b = self
            .linear_rows(
                x,
                &format!("{p}.in_proj_b.weight"),
                &format!("{g}.ssm_beta.weight"),
            )?
            .astype(DType::Float32)?;
        Ok(GdnProjections { qkv, z, a, b })
    }

    fn gdn(&mut self, x: &MxArray, i: usize, cache: &mut LayerCache) -> Result<MxArray> {
        let p = format!("layers.{i}.linear_attn");
        let g = format!("blk.{i}");
        let projections = self.gdn_projections(x, &p, &g)?;
        let out = self.gdn_mix(x.dtype()?, &projections, i, cache)?;
        self.linear(
            &out,
            &format!("{p}.out_proj.weight"),
            &format!("{g}.ssm_out.weight"),
        )
    }

    /// The stateful middle of one GDN row: the conv history and the recurrent
    /// state advance by exactly one token.
    fn gdn_mix(
        &mut self,
        dtype: DType,
        projections: &GdnProjections,
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        let c = self.config.clone();
        let p = format!("layers.{i}.linear_attn");
        let g = format!("blk.{i}");
        let nh = c.linear_num_value_heads as i64;
        let kh = c.linear_num_key_heads as i64;
        let kd = c.linear_key_head_dim as i64;
        let vd = c.linear_value_head_dim as i64;
        let keydim = kh * kd;
        let GdnProjections { qkv, z, a, b } = projections;
        let conv = self.dense(
            &format!("{p}.conv1d.weight"),
            &format!("{g}.ssm_conv1d.weight"),
        )?;
        if self.gguf()
            && (kh, nh, kd, vd) == (16, 48, 128, 128)
            && (2..=8).contains(&c.linear_conv_kernel_dim)
            && c.rms_norm_eps.is_finite()
            && c.rms_norm_eps > 0.0
            && c.output_gate_type == "sigmoid"
            && qkv.dtype()? == DType::BFloat16
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            let scale = self.dense(&format!("{p}.A_log"), &format!("{g}.ssm_a"))?;
            let dt = self.dense(&format!("{p}.dt_bias"), &format!("{g}.ssm_dt.bias"))?;
            let w = self.dense(&format!("{p}.norm.weight"), &format!("{g}.ssm_norm.weight"))?;
            let history = match &cache.conv {
                Some(s) => s.clone(),
                None => MxArray::zeros(
                    &[(c.linear_conv_kernel_dim - 1) as i64, 2 * keydim + nh * vd],
                    Some(qkv.dtype()?),
                )?,
            };
            let state = match &cache.recurrent {
                Some(s) => s.clone(),
                None => MxArray::zeros(&[1, nh, vd, kd], Some(DType::Float32))?,
            };
            let (out, state, history) = math::complete_gdn(
                qkv,
                z,
                a,
                b,
                &conv,
                &history,
                &scale,
                &dt,
                &state,
                &w,
                c.rms_norm_eps,
            )?;
            self.submit_state(&[&out, &state, &history], "qwen4::decoder::complete_gdn")?;
            cache.recurrent = Some(state);
            cache.conv = Some(history);
            return Ok(out);
        }
        let qkv = math::conv_with_completion(
            qkv,
            &conv,
            &mut cache.conv,
            c.linear_conv_kernel_dim,
            1,
            self.verification.is_some(),
        )?;
        let q = math::l2(&qkv.slice_axis(2, 0, keydim)?.reshape(&[1, kh, kd])?)?
            .mul_scalar((kd as f64).powf(-0.5))?;
        let k = math::l2(
            &qkv.slice_axis(2, keydim, 2 * keydim)?
                .reshape(&[1, kh, kd])?,
        )?;
        let v = qkv
            .slice_axis(2, 2 * keydim, 2 * keydim + nh * vd)?
            .reshape(&[1, nh, vd])?
            .astype(DType::Float32)?;
        let (q, k) = if self.gguf() {
            (q, k)
        } else {
            (
                q.repeat((nh / kh) as i32, 1)?,
                k.repeat((nh / kh) as i32, 1)?,
            )
        };
        let q = q.astype(DType::Float32)?.reshape(&[1, -1, 1, kd])?;
        let k = k.astype(DType::Float32)?.reshape(&[1, -1, 1, kd])?;
        let a_log = self
            .dense(&format!("{p}.A_log"), &format!("{g}.ssm_a"))?
            .astype(DType::Float32)?;
        let a_scale = if self.gguf() {
            a_log
        } else {
            a_log.exp()?.negative()?
        };
        let dt = self
            .dense(&format!("{p}.dt_bias"), &format!("{g}.ssm_dt.bias"))?
            .astype(DType::Float32)?;
        let (decay, beta) = math::gdn_gates(a, b, &a_scale, &dt)?;
        let decay = decay.reshape(&[1, nh, 1, 1])?;
        let state = match &cache.recurrent {
            Some(s) => s.clone(),
            None => MxArray::zeros(&[1, nh, vd, kd], Some(DType::Float32))?,
        };
        let (out, state) =
            math::recurrent_step(&q, &k, &v, &decay, &beta.reshape(&[1, nh, 1])?, &state)?;
        let out = out.astype(dtype)?;
        self.submit_state(&[&state], "qwen4::decoder::state")?;
        cache.recurrent = Some(state);
        let w = self.dense(&format!("{p}.norm.weight"), &format!("{g}.ssm_norm.weight"))?;
        let out = math::norm(&out, &w, c.linear_value_head_dim, c.rms_norm_eps, false)?
            .astype(DType::Float32)?;
        let z = z.astype(DType::Float32)?;
        let out = if c.output_gate_type == "sigmoid" {
            math::sigmoid_mul(&z, &out)?
        } else {
            math::swiglu(&z, &out)?
        };
        out.astype(dtype)?.reshape(&[1, 1, nh * vd])
    }
    fn attention(&mut self, x: &MxArray, i: usize, cache: &mut LayerCache) -> Result<MxArray> {
        let projections = self.attention_projections(x, i)?;
        self.attention_projected(x, i, cache, &projections, true)
    }

    fn attention_projections(&mut self, x: &MxArray, i: usize) -> Result<AttentionProjections> {
        let p = format!("layers.{i}.self_attn");
        let g = format!("blk.{i}");
        let qg = self.linear_rows(
            x,
            &format!("{p}.q_proj.weight"),
            &format!("{g}.attn_q.weight"),
        )?;
        let (k, v) = self.linear_pair_rows(
            x,
            (&format!("{p}.k_proj.weight"), &format!("{g}.attn_k.weight")),
            (&format!("{p}.v_proj.weight"), &format!("{g}.attn_v.weight")),
        )?;
        let ih = self.config.indexer_n_heads as i64;
        let id = self.config.indexer_head_dim as i64;
        let (iq, ik) = if self.gguf() {
            (
                self.linear_rows(x, "", &format!("{g}.indexer.q_proj.weight"))?,
                self.linear_rows(x, "", &format!("{g}.indexer.k_proj.weight"))?,
            )
        } else {
            let qk = self.linear_rows(x, &format!("{p}.indexer.index_qk_proj.weight"), "")?;
            (
                qk.slice_axis(2, 0, ih * id)?,
                qk.slice_axis(2, ih * id, (ih + 1) * id)?,
            )
        };
        Ok(AttentionProjections { qg, k, v, iq, ik })
    }

    /// Projection matrices can span a prompt window while rotary positions,
    /// sparse selection and recurrent/indexer state still advance causally.
    fn attention_projected(
        &mut self,
        x: &MxArray,
        i: usize,
        cache: &mut LayerCache,
        projections: &AttentionProjections,
        project_output: bool,
    ) -> Result<MxArray> {
        let c = self.config.clone();
        let p = format!("layers.{i}.self_attn");
        let g = format!("blk.{i}");
        let pos = self.position_override.unwrap_or(self.history.len());
        let hd = c.head_dim as i64;
        let nh = c.num_attention_heads as i64;
        let kh = c.num_key_value_heads as i64;
        let qg = projections.qg.reshape(&[1, 1, nh, 2 * hd])?;
        let q = qg.slice_axis(3, 0, hd)?;
        let q = self.attention_norm_rotary(
            &q.transpose(Some(&[0, 2, 1, 3]))?,
            &format!("{p}.q_norm.weight"),
            &format!("{g}.attn_q_norm.weight"),
            pos,
            1,
            false,
        )?;
        let k = self.attention_norm_rotary(
            &projections
                .k
                .reshape(&[1, 1, kh, hd])?
                .transpose(Some(&[0, 2, 1, 3]))?,
            &format!("{p}.k_norm.weight"),
            &format!("{g}.attn_k_norm.weight"),
            pos,
            1,
            false,
        )?;
        let v = projections
            .v
            .reshape(&[1, 1, kh, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let (keys, values) = if cache.draft_rows.is_some() {
            cache.append_draft_kv(&k, &v)?
        } else {
            let keys = match &cache.keys {
                Some(old) => MxArray::concatenate(old, &k, 2)?,
                None => k,
            };
            let values = match &cache.values {
                Some(old) => MxArray::concatenate(old, &v, 2)?,
                None => v,
            };
            (keys, values)
        };
        self.submit_state(&[&keys, &values], "qwen4::decoder::kv")?;
        cache.keys = Some(keys.clone());
        cache.values = Some(values.clone());
        let ih = c.indexer_n_heads as i64;
        let id = c.indexer_head_dim as i64;
        let (iq, ik) = (&projections.iq, &projections.ik);
        cache.index_tail.push(ik.reshape(&[1, id])?);
        if cache.index_tail.len() == c.indexer_compress_ratio {
            let raw = MxArray::concatenate_many(cache.index_tail.iter().collect(), Some(0))?
                .astype(DType::Float32)?
                .mean(Some(&[0]), Some(true))?
                .astype(x.dtype()?)?;
            let raw = self.norm(
                &raw,
                &format!("{p}.indexer.k_layernorm.weight"),
                &format!("{g}.indexer.k_norm.weight"),
                c.indexer_head_dim,
                true,
            )?;
            let block = self.rope(&raw, pos + 1 - c.indexer_compress_ratio)?;
            let keys = cache.append_index_keys(block)?;
            self.submit_state(&[keys], "qwen4::decoder::block")?;
            cache.index_tail.clear();
        }
        let topk = c.indexer_budget / c.indexer_compress_ratio;
        let nb = cache.index_block_count()?;
        let selected = if let Some(index_keys) = cache.index_keys.as_ref().filter(|_| nb > topk) {
            let iq = iq.reshape(&[ih, id])?;
            let iq = self.norm(
                &iq,
                &format!("{p}.indexer.q_layernorm.weight"),
                &format!("{g}.indexer.q_norm.weight"),
                c.indexer_head_dim,
                true,
            )?;
            let iq = self.rope(&iq, pos)?.astype(DType::Float32)?;
            let blocks = index_keys.astype(DType::Float32)?;
            let scores = iq
                .matmul(&blocks.transpose(None)?)?
                .clip(Some(0.0), None)?
                .sum(Some(&[0]), Some(false))?;
            // ReLU produces exact ties. Use the reference MLX partition, not
            // a CPU sort with a different tie rule, to preserve selected blocks.
            let scores = scores
                .div_scalar((c.indexer_head_dim as f64).sqrt())?
                .reshape(&[1, 1, -1])?;
            let n = nb as i64;
            let indices = scores
                .argpartition(-(topk as i32), Some(-1))?
                .slice_axis(2, n - topk as i64, n)?
                .reshape(&[-1])?
                .astype(DType::Int32)?
                .sort(Some(0))?;
            let offsets = MxArray::arange(
                0.,
                c.indexer_compress_ratio as f64,
                None,
                Some(DType::Int32),
            )?;
            let ids = indices
                .expand_dims(-1)?
                .mul_scalar(c.indexer_compress_ratio as f64)?
                .astype(DType::Int32)?
                .add(&offsets)?
                .reshape(&[-1])?;
            let tail = MxArray::arange(
                (nb * c.indexer_compress_ratio) as f64,
                (pos + 1) as f64,
                None,
                Some(DType::Int32),
            )?;
            MxArray::concatenate(&ids, &tail, 0)?
        } else {
            MxArray::arange(0., (pos + 1) as f64, None, Some(DType::Int32))?
        };
        let (keys, values) = if let Some(adapter) = &self.paged {
            let layer = (0..i).filter(|&l| !self.config.linear(l)).count() as u32;
            super::paged::gather_window(
                adapter,
                layer,
                &selected,
                &keys,
                &values,
                self.paged_chunk_start,
            )?
        } else {
            (keys.take(&selected, 2)?, values.take(&selected, 2)?)
        };
        let out =
            scaled_dot_product_attention(&q, &keys, &values, (c.head_dim as f64).powf(-0.5), None)?;
        let out = math::attention_output(&out, &qg)?;
        if project_output {
            self.linear(
                &out,
                &format!("{p}.o_proj.weight"),
                &format!("{g}.attn_output.weight"),
            )
        } else {
            Ok(out)
        }
    }
    fn moe(&mut self, x: &MxArray, i: usize) -> Result<MxArray> {
        let main = self.gguf();
        if i == 0 {
            self.weights.wire_mapped_banks(main)?;
        } else {
            self.weights.keep_wired(i, main)?;
        }
        let c = self.config.clone();
        let p = format!("layers.{i}.mlp");
        let g = format!("blk.{i}");
        let logits = self.linear(
            x,
            &format!("{p}.gate.weight"),
            &format!("{g}.ffn_gate_inp.weight"),
        )?;
        let (selected, scores) =
            if let Some(routes) = math::singleton_routes(&logits, c.num_experts_per_tok)? {
                routes
            } else {
                let probs = Activations::softmax_precise(&logits, Some(-1))?;
                let ne = c.num_experts as i64;
                let top = c.num_experts_per_tok as i64;
                let selected =
                    probs
                        .argpartition(-(top as i32), Some(-1))?
                        .slice_axis(2, ne - top, ne)?;
                let scores = probs.take_along_axis(&selected, -1)?;
                let scores = scores.div(&scores.sum(Some(&[-1]), Some(true))?)?;
                (selected, scores)
            };
        if !self.scope.is_empty()
            && let Some(sum) = self.mtp_resident_experts(x, &selected, &scores)?
        {
            let (shared, gate) = self.shared_expert_parts(x, i)?;
            return sum
                .astype(x.dtype()?)?
                .add(&math::sigmoid_mul(&gate, &shared)?);
        }
        if let Some(out) = self.tentative_shared_experts(x, &selected, &scores, i)? {
            return Ok(out);
        }
        let device = self.tentative_experts(x, &selected, &scores, i)?;
        let resident = if device.is_some() {
            device
        } else {
            self.resident_experts(x, &selected, &scores, i)?
        };
        let ids = if resident.is_none() {
            self.read_routes(&selected)?
        } else {
            Vec::new()
        };
        let sum = if let Some(sum) = resident {
            sum
        } else if let Some(sum) = self.selected_experts(x, &ids, &scores, i)? {
            sum
        } else {
            let mut weighted = Vec::new();
            for (j, &e) in ids.iter().enumerate() {
                let e = e as usize;
                self.check_cancelled()?;
                let m = c.moe_intermediate_size;
                let (gate, up) = if self.gguf() {
                    (
                        self.weights
                            .read(&format!("{g}.ffn_gate_exps.weight"), e * m, m)?
                            .linear(x)?,
                        self.weights
                            .read(&format!("{g}.ffn_up_exps.weight"), e * m, m)?
                            .linear(x)?,
                    )
                } else {
                    let key = self.key(&format!("{p}.experts.gate_up_proj"), "");
                    let both = self.weights.read(&key, e * 2 * m, 2 * m)?.linear(x)?;
                    (
                        both.slice_axis(2, 0, m as i64)?,
                        both.slice_axis(2, m as i64, 2 * m as i64)?,
                    )
                };
                let h = math::swiglu(&gate, &up)?;
                let key = self.key(
                    &format!("{p}.experts.down_proj"),
                    &format!("{g}.ffn_down_exps.weight"),
                );
                let out = self
                    .weights
                    .read(&key, e * c.hidden_size, c.hidden_size)?
                    .linear(&h)?;
                let value = out.mul(&scores.slice_axis(2, j as i64, j as i64 + 1)?)?;
                MxArray::eval_arrays_with_context(&[&value], "qwen4::decoder::value")?;
                weighted.push(value);
            }
            MxArray::concatenate_many(weighted.iter().collect(), Some(1))?
                .sum(Some(&[1]), Some(true))?
        };
        let (shared, gate) = self.shared_expert_parts(x, i)?;
        sum.astype(x.dtype()?)?
            .add(&math::sigmoid_mul(&gate, &shared)?)
    }
    /// A layer's routes on the host: the pass waits here for the GPU.
    fn read_routes(&mut self, selected: &MxArray) -> Result<Vec<u32>> {
        #[cfg(test)]
        {
            self.route_readbacks += 1;
        }
        Ok(selected.to_uint32()?.to_vec())
    }
    fn shared_expert_parts(&mut self, x: &MxArray, i: usize) -> Result<(MxArray, MxArray)> {
        let p = format!("layers.{i}.mlp");
        let g = format!("blk.{i}");
        let (gate, up) = self.linear_pair_rows(
            x,
            (
                &format!("{p}.shared_expert.gate_proj.weight"),
                &format!("{g}.ffn_gate_shexp.weight"),
            ),
            (
                &format!("{p}.shared_expert.up_proj.weight"),
                &format!("{g}.ffn_up_shexp.weight"),
            ),
        )?;
        let activation = math::swiglu(&gate, &up)?;
        let shared = self.linear_rows(
            &activation,
            &format!("{p}.shared_expert.down_proj.weight"),
            &format!("{g}.ffn_down_shexp.weight"),
        )?;
        drop(activation);
        let gate = self.linear_rows(
            x,
            &format!("{p}.shared_expert_gate.weight"),
            &format!("{g}.ffn_gate_inp_shexp.weight"),
        )?;
        Ok((shared, gate))
    }

    fn ple(
        &mut self,
        x: &MxArray,
        token: u32,
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        self.ple_window(x, &[token], i, cache)
    }

    /// `hyper_with_norm` over a `[1, R, 4H]` verify window. The admitted
    /// mixer banks stream once through the row-batched kernels; any other
    /// configuration runs the single-row mixer per row.
    fn hyper_rows(
        &mut self,
        x: &MxArray,
        hf: &str,
        gg: &str,
        inject: bool,
        normed: Option<&MxArray>,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let rows = x.shape_at(1)?;
        if rows == 1 {
            return self.hyper_with_norm(x, hf, gg, inject, normed);
        }
        let c = self.config.clone();
        let down_key = self.key(
            &format!("{hf}.input_mix_weight_down.weight"),
            &format!("{gg}_down.weight"),
        );
        let up_key = self.key(
            &format!("{hf}.input_mix_weight_up.weight"),
            &format!("{gg}_up.weight"),
        );
        if self.gguf()
            && c.hc_count == 4
            && c.hidden_size == 2560
            && let (Some(down_bank), Some(up_bank)) = (
                self.weights.resident_bank(&down_key),
                self.weights.resident_bank(&up_key),
            )
        {
            let n = match normed {
                Some(normed) => normed.clone(),
                None => self.norm(
                    x,
                    &format!("{hf}.hc_norm.weight"),
                    &format!("{gg}_norm.weight"),
                    c.hidden_size,
                    true,
                )?,
            };
            if let Some(down) = math::rows::mixer_act(&down_bank, &n)? {
                let injection = if inject {
                    Some(self.linear_rows(
                        &n,
                        &format!("{hf}.block_inject_weight.weight"),
                        &format!("{gg}_inject.weight"),
                    )?)
                } else {
                    None
                };
                if let Some(mixed) = math::rows::hyper_up(&up_bank, &down, &n)? {
                    let gate = injection
                        .map(|w| -> Result<MxArray> {
                            Activations::sigmoid(&w.div_scalar(c.hc_count as f64)?)?.mul_scalar(2.0)
                        })
                        .transpose()?;
                    return Ok((mixed, gate));
                }
            }
        }
        let mut mixed = Vec::with_capacity(rows as usize);
        let mut gates = Vec::with_capacity(rows as usize);
        for j in 0..rows {
            let normed = normed.map(|n| n.slice_axis(1, j, j + 1)).transpose()?;
            let (m, gate) =
                self.hyper_with_norm(&x.slice_axis(1, j, j + 1)?, hf, gg, inject, normed.as_ref())?;
            mixed.push(m);
            gates.extend(gate);
        }
        let gate = if inject {
            Some(MxArray::concatenate_many(gates.iter().collect(), Some(1))?)
        } else {
            None
        };
        Ok((
            MxArray::concatenate_many(mixed.iter().collect(), Some(1))?,
            gate,
        ))
    }

    /// `moe` over a `[1, R, H]` verify window: the routes of all rows stay on
    /// the device over the mapped pack, otherwise one routing readback, then
    /// the routed and shared experts through the row-batched slot kernels.
    /// Without those kernels every row runs the single-row MoE.
    fn moe_rows(&mut self, x: &MxArray, i: usize) -> Result<MxArray> {
        let rows = x.shape_at(1)?;
        if rows == 1 {
            return self.moe(x, i);
        }
        let main = self.gguf();
        if i == 0 {
            self.weights.wire_mapped_banks(main)?;
        } else {
            self.weights.keep_wired(i, main)?;
        }
        let c = self.config.clone();
        if main {
            let logits = self.linear_rows(
                x,
                &format!("layers.{i}.mlp.gate.weight"),
                &format!("blk.{i}.ffn_gate_inp.weight"),
            )?;
            if let Some((selected, scores)) =
                math::rows::singleton_routes(&logits, c.num_experts_per_tok)?
            {
                if let Some(out) = self.tentative_shared_expert_rows(x, &selected, &scores, i)? {
                    return Ok(out);
                }
                let mut sum = self.resident_experts(x, &selected, &scores, i)?;
                if sum.is_none() {
                    let ids = self.read_routes(&selected)?;
                    if let Some(out) = self.slotted_shared_experts(x, &ids, &scores, i)? {
                        return Ok(out);
                    }
                    sum = self.selected_experts(x, &ids, &scores, i)?;
                }
                if let Some(sum) = sum {
                    let (shared, gate) = self.shared_expert_parts(x, i)?;
                    return sum
                        .astype(x.dtype()?)?
                        .add(&math::sigmoid_mul(&gate, &shared)?);
                }
            }
        }
        let parts = (0..rows)
            .map(|j| self.moe(&x.slice_axis(1, j, j + 1)?, i))
            .collect::<Result<Vec<_>>>()?;
        MxArray::concatenate_many(parts.iter().collect(), Some(1))
    }

    /// The routed and shared experts of the verify rows in one kernel over
    /// the host-checked slots: the device-route decode kernel with the
    /// routes already known.
    fn slotted_shared_experts(
        &mut self,
        x: &MxArray,
        ids: &[u32],
        scores: &MxArray,
        i: usize,
    ) -> Result<Option<MxArray>> {
        let c = self.config.clone();
        if c.num_experts_per_tok != 10
            || c.hidden_size != 2560
            || c.moe_intermediate_size != 640
            || x.dtype()? != DType::BFloat16
        {
            return Ok(None);
        }
        let shared = ["gate", "up", "down"].map(|part| {
            self.weights
                .resident_bank(&format!("blk.{i}.ffn_{part}_shexp.weight"))
        });
        let [Some(gate), Some(up), Some(down)] = shared else {
            return Ok(None);
        };
        let shared = [gate, up, down];
        let Some((banks, slots, _)) =
            self.weights
                .expert_slots(i, c.num_experts, c.num_experts_per_tok, ids)?
        else {
            return Ok(None);
        };
        let shared_gate = self.linear_rows(
            x,
            &format!("layers.{i}.mlp.shared_expert_gate.weight"),
            &format!("blk.{i}.ffn_gate_inp_shexp.weight"),
        )?;
        let slots = MxArray::from_uint32(&slots, &[slots.len() as i64])?;
        let Some(out) =
            math::rows::routed_shared_experts(x, &slots, scores, &banks, &shared, &shared_gate)?
        else {
            return Ok(None);
        };
        self.weights.finish_expert_slots(i, &out)?;
        Ok(Some(out))
    }

    /// One decoder layer over the `[1, R, 4H]` verify window. Projections,
    /// mixers, router, experts and output projections read their weights
    /// once for all rows through the row-exact kernels; PLE, GDN and
    /// attention state advance row by row. `saved[j]` is this layer's cache
    /// after row `j`: the frontier a commit of `j + 1` tokens restores.
    fn verify_layer(
        &mut self,
        x: &MxArray,
        tokens: &[u32],
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<(MxArray, Vec<LayerCache>)> {
        let base = self.history.len();
        let rows = tokens.len();
        let mut saved = vec![LayerCache::default(); rows];
        let row = |a: &MxArray, j: usize| a.slice_axis(1, j as i64, j as i64 + 1);
        let x = if self.config.ple_layer_ids.contains(&(i + 1)) {
            let dtype = x.dtype()?;
            let mut embeddings = Vec::with_capacity(rows);
            for &token in tokens {
                self.check_cancelled()?;
                embeddings.push(self.ple_embeddings(&[token], i, dtype)?);
                self.history.push(token);
            }
            self.history.truncate(base);
            let embeddings = MxArray::concatenate_many(embeddings.iter().collect(), Some(1))?;
            let (key, value) = self.ple_projections(&embeddings, i)?;
            let mut parts = Vec::with_capacity(rows);
            for (j, saved) in saved.iter_mut().enumerate() {
                self.check_cancelled()?;
                let input = row(x, j)?;
                let ple = self.ple_mix(&input, &row(&key, j)?, &row(&value, j)?, i, cache)?;
                parts.push(input.add(&ple)?);
                saved.ple_conv = cache.ple_conv.clone();
            }
            MxArray::concatenate_many(parts.iter().collect(), Some(1))?
        } else {
            x.clone()
        };
        let (mixed, gate) = self.hyper_rows(
            &x,
            &format!("layers.{i}.attn_hyper_connection"),
            &format!("blk.{i}.hc_attn"),
            true,
            None,
        )?;
        let branch = if self.config.linear(i) {
            let p = format!("layers.{i}.linear_attn");
            let g = format!("blk.{i}");
            let projections = self.gdn_projections(&mixed, &p, &g)?;
            let dtype = mixed.dtype()?;
            let mut outputs = Vec::with_capacity(rows);
            for (j, saved) in saved.iter_mut().enumerate() {
                self.check_cancelled()?;
                outputs.push(self.gdn_mix(dtype, &projections.row(j)?, i, cache)?);
                saved.conv = cache.conv.clone();
                saved.recurrent = cache.recurrent.clone();
            }
            let out = MxArray::concatenate_many(outputs.iter().collect(), Some(1))?;
            self.linear_rows(
                &out,
                &format!("{p}.out_proj.weight"),
                &format!("{g}.ssm_out.weight"),
            )?
        } else {
            let projections = self.attention_projections(&mixed, i)?;
            let mut outputs = Vec::with_capacity(rows);
            for (j, (&token, saved)) in tokens.iter().zip(saved.iter_mut()).enumerate() {
                self.check_cancelled()?;
                outputs.push(self.attention_projected(
                    &row(&mixed, j)?,
                    i,
                    cache,
                    &projections.row(j)?,
                    false,
                )?);
                self.history.push(token);
                if self.paged.is_none() {
                    saved.keys = cache.keys.clone();
                    saved.values = cache.values.clone();
                }
                saved.index_tail = cache.index_tail.clone();
                saved.index_keys = cache.index_keys.clone();
            }
            self.history.truncate(base);
            let out = MxArray::concatenate_many(outputs.iter().collect(), Some(1))?;
            self.linear_rows(
                &out,
                &format!("layers.{i}.self_attn.o_proj.weight"),
                &format!("blk.{i}.attn_output.weight"),
            )?
        };
        let gate = gate
            .ok_or_else(|| Error::from_reason("Qwen4 attention is missing its injection gate"))?;
        let (x, normed) = self.inject_for_mlp(&x, &branch, &gate, i, true)?;
        let (mixed, gate) = self.hyper_rows(
            &x,
            &format!("layers.{i}.mlp_hyper_connection"),
            &format!("blk.{i}.hc_ffn"),
            true,
            normed.as_ref(),
        )?;
        let branch = self.moe_rows(&mixed, i)?;
        let gate =
            gate.ok_or_else(|| Error::from_reason("Qwen4 MLP is missing its injection gate"))?;
        Ok((Self::inject(&x, &branch, &gate)?, saved))
    }

    /// `[multipliers, head sizes, head offsets]` of PLE layer `i`.
    fn ple_constants(&self, i: usize) -> Result<[Vec<u64>; 3]> {
        let p = format!("layers.{i}.ple");
        let constants = |name: &str| -> Result<Vec<u64>> {
            if self.gguf() {
                super::gguf::integers(&self.weights.metadata, &format!("qwen4exp.ple.{name}"))
            } else {
                self.weights.integers(&format!(
                    "{p}.ple_embedding.{}",
                    match name {
                        "head_offsets" => "ngram_heads_offsets",
                        "head_vocab_sizes" => "ngram_heads_vocab_sizes",
                        _ => name,
                    }
                ))
            }
        };
        Ok([
            constants("layer_multipliers")?,
            constants("head_vocab_sizes")?,
            constants("head_offsets")?,
        ])
    }

    /// Hash ids of `tokens` continuing the current history, in gather order.
    fn ple_hash_ids(&self, i: usize, tokens: &[u32]) -> Result<Vec<u64>> {
        self.ple_hash_ids_after(i, &self.history, tokens)
    }

    /// The hash ids of `tokens` following `history` (the ngram context is
    /// its tail).
    fn ple_hash_ids_after(&self, i: usize, history: &[u32], tokens: &[u32]) -> Result<Vec<u64>> {
        let c = &self.config;
        let [multipliers, sizes, offsets] = self.ple_constants(i)?;
        let mut previous = history[history.len().saturating_sub(c.ngram_size - 1)..].to_vec();
        let mut ids = Vec::new();
        for &token in tokens {
            self.check_cancelled()?;
            ids.extend(math::hash_ids(
                token,
                &previous,
                c.eos_token_id,
                c.ngram_size,
                c.heads_per_ngram,
                &multipliers,
                &sizes,
                &offsets,
            )?);
            previous.push(token);
        }
        Ok(ids)
    }

    /// The table row behind each hash id: the one GGUF table, or the
    /// safetensors shard that holds it.
    fn ple_row_keys(&self, i: usize, ids: &[u64]) -> Result<Vec<(String, usize)>> {
        if self.gguf() {
            return Ok(ids
                .iter()
                .map(|&id| ("per_layer_token_embd.weight".to_string(), id as usize))
                .collect());
        }
        let p = format!("layers.{i}.ple");
        let shards = (0..self.config.split_ngram_parts)
            .map(|shard| {
                let key = format!("{p}.ple_embedding.ngram_embedding.shard_{shard}.weight");
                let count = self.weights.descriptor(&key)?.rows()?;
                Ok((key, count))
            })
            .collect::<Result<Vec<_>>>()?;
        ids.iter()
            .map(|&id| {
                let mut local = id as usize;
                for (key, count) in &shards {
                    if local < *count {
                        return Ok((key.clone(), local));
                    }
                    local -= count;
                }
                Err(Error::from_reason("PLE row outside checkpoint vocabulary"))
            })
            .collect()
    }

    /// Hand the reader threads the PLE rows `tokens` will need at their PLE
    /// layer, before the layers ahead of it are encoded. The embedding row
    /// is needed at once, so the model thread reads it itself.
    fn prefetch_ple_rows(&mut self, tokens: &[u32]) -> Result<()> {
        let keys = self.ple_row_keys_after(&self.history.clone(), tokens)?;
        self.weights.prefetch_rows(&keys)
    }

    /// The PLE rows of `tokens` following `history`, over every PLE layer.
    fn ple_row_keys_after(&self, history: &[u32], tokens: &[u32]) -> Result<Vec<(String, usize)>> {
        let mut keys = Vec::new();
        for &layer in &self.config.ple_layer_ids {
            let Some(i) = layer.checked_sub(1) else {
                continue;
            };
            let ids = self.ple_hash_ids_after(i, history, tokens)?;
            keys.extend(self.ple_row_keys(i, &ids)?);
        }
        Ok(keys)
    }

    /// A prefill window starts: request this window's rows, then, queued
    /// behind them, the rows of the window after it, so the reader threads
    /// fetch those while this window's layers run instead of the model
    /// thread reading them one `pread` at a time at the next window's start
    /// (1,024 embedding rows and up to 16k PLE rows per window). Two
    /// requests, not one sorted batch: the rows this window needs now must
    /// not queue behind rows needed a second later. A window given its
    /// embeddings (media) has no embedding rows to read.
    fn prefetch_window_rows(&mut self, tokens: &[u32], next: &[u32], embed: bool) -> Result<()> {
        let embedding = self.key("embed_tokens.weight", "token_embd.weight");
        let mut keys = Vec::new();
        if embed {
            keys.extend(tokens.iter().map(|&t| (embedding.clone(), t as usize)));
        }
        let history = self.history.clone();
        keys.extend(self.ple_row_keys_after(&history, tokens)?);
        self.weights.prefetch_rows(&keys)?;
        if next.is_empty() {
            return Ok(());
        }
        let mut keys = Vec::new();
        if embed {
            keys.extend(next.iter().map(|&t| (embedding.clone(), t as usize)));
        }
        let mut through = history;
        through.extend_from_slice(tokens);
        keys.extend(self.ple_row_keys_after(&through, next)?);
        self.weights.prefetch_rows(&keys)
    }

    fn ple_window(
        &mut self,
        x: &MxArray,
        tokens: &[u32],
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        let embeddings = self.ple_embeddings(tokens, i, x.dtype()?)?;
        let (key, value) = self.ple_projections(&embeddings, i)?;
        self.ple_mix(x, &key, &value, i, cache)
    }

    /// The `[1, T, ple_embed_dim]` embeddings of `tokens` following the history.
    fn ple_embeddings(&mut self, tokens: &[u32], i: usize, dtype: DType) -> Result<MxArray> {
        let c = self.config.clone();
        let t = tokens.len() as i64;
        let ids = self.ple_hash_ids(i, tokens)?;
        let mut groups: std::collections::BTreeMap<String, (Vec<usize>, Vec<usize>)> =
            Default::default();
        for (position, (name, row)) in self.ple_row_keys(i, &ids)?.into_iter().enumerate() {
            let (rows, positions) = groups.entry(name).or_default();
            rows.push(row);
            positions.push(position);
        }
        let emb = if groups.len() == 1
            && let Some((name, (rows, _))) = groups.pop_first()
        {
            self.weights.lookup_rows(&name, &rows)?
        } else {
            let mut rows = vec![None; ids.len()];
            for (name, (indices, positions)) in groups {
                let bank = self.weights.lookup_rows(&name, &indices)?;
                for (row, position) in positions.into_iter().enumerate() {
                    rows[position] = Some(bank.slice_axis(0, row as i64, row as i64 + 1)?);
                }
            }
            let rows = rows
                .iter()
                .map(|row| {
                    row.as_ref().ok_or_else(|| {
                        Error::from_reason("Qwen4 PLE lookup did not populate every embedding row")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            MxArray::concatenate_many(rows, Some(0))?
        };
        emb.reshape(&[1, t, c.ple_embed_dim as i64])?.astype(dtype)
    }

    /// The PLE key and value projections of a `[1, T, ple_embed_dim]` window.
    fn ple_projections(&mut self, emb: &MxArray, i: usize) -> Result<(MxArray, MxArray)> {
        let p = format!("layers.{i}.ple");
        let g = format!("blk.{i}");
        let key = self.weights.linear_ple_window(
            &self.key(
                &format!("{p}.key_proj.weight"),
                &format!("{g}.ple_key.weight"),
            ),
            emb,
        )?;
        let value = self.weights.linear_ple_window(
            &self.key(
                &format!("{p}.value_proj.weight"),
                &format!("{g}.ple_value.weight"),
            ),
            emb,
        )?;
        Ok((key, value))
    }

    /// The PLE gate and convolution over `x` (`[1, T, 4H]`) given the
    /// window's key and value projections.
    fn ple_mix(
        &mut self,
        x: &MxArray,
        key: &MxArray,
        value: &MxArray,
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        let c = self.config.clone();
        let t = x.shape_at(1)?;
        let p = format!("layers.{i}.ple");
        let g = format!("blk.{i}");
        let key = self
            .norm(
                key,
                &format!("{p}.norm_key.weight"),
                &format!("{g}.ple_norm_key.weight"),
                c.hidden_size,
                true,
            )?
            .reshape(&[1, t, c.hc_count as i64, c.hidden_size as i64])?;
        let query = self
            .norm(
                x,
                &format!("{p}.norm_query.weight"),
                &format!("{g}.ple_norm_query.weight"),
                c.hidden_size,
                true,
            )?
            .reshape(&[1, t, c.hc_count as i64, c.hidden_size as i64])?;
        let score = key
            .mul(&query)?
            .sum(Some(&[-1]), Some(true))?
            .div_scalar((c.hidden_size as f64).sqrt())?;
        let score = score
            .sign()?
            .mul(&score.abs()?.clip(Some(1e-6), None)?.sqrt()?)?;
        let value = Activations::sigmoid(&score)?
            .mul(&value.expand_dims(-2)?)?
            .reshape(&x.shape()?)?;
        let normalized = self.norm(
            &value,
            &format!("{p}.norm_conv.weight"),
            &format!("{g}.ple_norm_conv.weight"),
            c.hidden_size,
            true,
        )?;
        let weight = self.dense(
            &format!("{p}.conv1d.weight"),
            &format!("{g}.ple_conv1d.weight"),
        )?;
        let convolved = math::conv_window(
            &normalized,
            &weight,
            &mut cache.ple_conv,
            c.ple_conv_kernel_size,
            c.ngram_size,
        )?;
        value.add(&convolved)
    }
    pub fn step(&mut self, token: u32) -> Result<MxArray> {
        self.rotary_tables.get_mut().clear();
        let started = std::time::Instant::now();
        let result = self.forward_with_device_routes(token, true);
        self.rotary_tables.get_mut().clear();
        // A read failure or cancellation can happen after earlier layers have
        // advanced. Discard the entire prefix instead of reusing partial state.
        if result.is_err() {
            self.reset();
        } else {
            self.observe_pass(1, started.elapsed());
        }
        result
    }

    fn layer(
        &mut self,
        x: MxArray,
        token: u32,
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        let (x, normed) = self.attention_block_for_mlp(x, token, i, cache, true)?;
        let (mixed, gate) = self.hyper_with_norm(
            &x,
            &format!("layers.{i}.mlp_hyper_connection"),
            &format!("blk.{i}.hc_ffn"),
            true,
            normed.as_ref(),
        )?;
        let branch = self.moe(&mixed, i)?;
        let gate =
            gate.ok_or_else(|| Error::from_reason("Qwen4 MLP is missing its injection gate"))?;
        Self::inject(&x, &branch, &gate)
    }

    fn attention_block(
        &mut self,
        x: MxArray,
        token: u32,
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        Ok(self.attention_block_for_mlp(x, token, i, cache, false)?.0)
    }

    fn attention_block_for_mlp(
        &mut self,
        mut x: MxArray,
        token: u32,
        i: usize,
        cache: &mut LayerCache,
        normalize: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        if self.config.ple_layer_ids.contains(&(i + 1)) {
            x = x.add(&self.ple(&x, token, i, cache)?)?;
        }
        let (mixed, gate) = self.hyper_with_norm(
            &x,
            &format!("layers.{i}.attn_hyper_connection"),
            &format!("blk.{i}.hc_attn"),
            true,
            None,
        )?;
        let branch = if self.config.linear(i) {
            self.gdn(&mixed, i, cache)?
        } else {
            self.attention(&mixed, i, cache)?
        };
        let gate = gate
            .ok_or_else(|| Error::from_reason("Qwen4 attention is missing its injection gate"))?;
        self.inject_for_mlp(&x, &branch, &gate, i, normalize)
    }

    /// The token's embedding row as a lazy array: an LRU hit or an import
    /// the consuming layer submits, never a drain at the token start.
    pub fn embed_token(&mut self, token: u32) -> Result<MxArray> {
        let key = self.key("embed_tokens.weight", "token_embd.weight");
        self.weights
            .lookup_rows(&key, &[token as usize])?
            .reshape(&[1, 1, self.config.hidden_size as i64])
    }

    /// Traverse a bounded prompt chunk layer by layer, sharing weight reads
    /// and projections across its tokens. Verification retains singleton
    /// arithmetic and captures state at every accepted frontier.
    #[cfg(test)]
    pub fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        embeddings: Option<&MxArray>,
        project_logits: bool,
    ) -> Result<MxArray> {
        self.prefill_chunk_ahead(tokens, &[], embeddings, project_logits)
    }

    /// `prefill_chunk` whose caller knows the window after this one:
    /// `next`'s rows are requested while this window runs.
    pub fn prefill_chunk_ahead(
        &mut self,
        tokens: &[u32],
        next: &[u32],
        embeddings: Option<&MxArray>,
        project_logits: bool,
    ) -> Result<MxArray> {
        if cfg!(target_os = "macos") {
            self.weights.check_forward_headroom()?;
        }
        self.rotary_tables.get_mut().clear();
        let started = std::time::Instant::now();
        let result =
            self.prefill_group_with_device_routes(&[tokens], next, embeddings, project_logits);
        self.rotary_tables.get_mut().clear();
        if result.is_err() {
            self.reset();
        } else {
            self.observe_pass(tokens.len(), started.elapsed());
        }
        result
    }
    pub fn verify_chunk(&mut self, tokens: &[u32]) -> Result<(MxArray, Vec<DecoderState>)> {
        if tokens.is_empty() || tokens.len() > 4 {
            return Err(Error::from_reason("Qwen4 verification width must be 1-4"));
        }
        self.rotary_tables.get_mut().clear();
        let started = std::time::Instant::now();
        self.verification = Some(Vec::new());
        if self.verify_routes_on_device() {
            self.device_routes = Some(Vec::with_capacity(self.config.num_hidden_layers));
        }
        let result = (|| {
            if cfg!(target_os = "macos") {
                self.weights.check_forward_headroom()?;
            }
            self.prefill_chunk_inner(tokens, &[], None, false)?;
            // The pass stays open through the head: these rows are row-exact
            // too, and `linear_rows` is batched outside a verification pass.
            let window =
                MxArray::concatenate_many(self.last_chunk_hidden.iter().collect(), Some(1))?;
            let (mixed, _) =
                self.hyper_rows(&window, "hyper_connection_mixer", "output_hc", false, None)?;
            let logits = self
                .linear_rows(&mixed, "lm_head.weight", "output.weight")?
                .reshape(&[tokens.len() as i64, -1])?;
            let states = self.verification.take().ok_or_else(|| {
                Error::from_reason("Qwen4 verification lost its accepted-frontier snapshots")
            })?;
            let routes = self.device_routes.take();
            let tape = routes
                .as_ref()
                .filter(|routes| !routes.is_empty())
                .map(|routes| {
                    MxArray::concatenate_many(routes.iter().map(|(_, ids)| ids).collect(), Some(0))
                })
                .transpose()?;
            // Commit rolls the page cursor back and restores one of these
            // states: every pool write and frontier array must be complete.
            let completed: Vec<_> = std::iter::once(&logits)
                .chain(states.iter().flat_map(DecoderState::arrays))
                .chain(self.caches.iter().flat_map(LayerCache::arrays))
                .chain(tape.iter())
                .collect();
            MxArray::async_eval_arrays(&completed);
            if let Some(adapter) = &mut self.paged {
                adapter
                    .eval_pending_pool_writes()
                    .map_err(Error::from_reason)?;
            }
            MxArray::eval_arrays_with_context(&completed, "qwen4::decoder::verify")?;
            if let (Some(routes), Some(tape)) = (routes, tape) {
                let layers: Vec<_> = routes.iter().map(|(layer, _)| *layer).collect();
                if !self.weights.commit_device_routes(
                    &layers,
                    &tape.to_uint32()?,
                    tokens.len() * self.config.num_experts_per_tok,
                )? {
                    return Err(Error::from_reason(
                        "Qwen4 verification routes lost their mapped banks",
                    ));
                }
            }
            Ok((logits, states))
        })();
        self.rotary_tables.get_mut().clear();
        match result {
            Ok(verified) => {
                // Only after the route commit: a pass that trips the wire
                // drops the mapped banks the tape is checked against.
                self.observe_pass(tokens.len(), started.elapsed());
                Ok(verified)
            }
            Err(error) => {
                self.reset();
                Err(error)
            }
        }
    }

    fn prefill_chunk_inner(
        &mut self,
        tokens: &[u32],
        next: &[u32],
        embeddings: Option<&MxArray>,
        project_logits: bool,
    ) -> Result<MxArray> {
        self.prefill_group_inner(&[tokens], next, embeddings, project_logits)
    }

    /// A test event (`submit_log`): the group's frontier evaluated, the
    /// route commit, end, trip, switch and the tail's start, in the order
    /// they ran.
    fn note(&self, event: &str) {
        #[cfg(test)]
        self.submit_log.borrow_mut().push(event.to_owned());
        #[cfg(not(test))]
        let _ = event;
    }

    /// The windows of `tokens` a group runs and the tail that runs after it:
    /// slices of the admitted window, except a final one shorter than both
    /// the slice and 256 rows, which fails the tentative gate and would drain
    /// the GPU inside every sweep.
    pub fn group_windows<'t>(&self, tokens: &'t [u32]) -> (Vec<&'t [u32]>, &'t [u32]) {
        let slice = self.prefill_slice_size().max(1);
        let mut windows: Vec<&[u32]> = tokens.chunks(slice).collect();
        let tail = match windows.last() {
            Some(last) if last.len() < slice && last.len() < 256 => {
                let tail = *last;
                windows.pop();
                tail
            }
            _ => &tokens[tokens.len()..],
        };
        (windows, tail)
    }

    /// Whether a prompt of `suffix` new tokens runs as one layer-major
    /// group: the turn opened over a streaming pack, the carried window path
    /// serves it, every window is a tentative-eligible slice, and the W
    /// residual streams fit beside the mapped turn floor. Read at admission
    /// and again at execution: a trip applied in between runs the prompt one
    /// window at a time.
    pub(super) fn group_mode(&self, suffix: usize) -> bool {
        let slice = self.prefill_slice_size().max(1);
        let windows = suffix / slice + usize::from(suffix % slice >= 256);
        if windows < 2 || self.verification.is_some() {
            return false;
        }
        #[cfg(test)]
        if let Some(pin) = self.group_pin {
            return pin;
        }
        if !self.weights.plan.mapped || !self.weights.streaming_pass() || slice < 256 {
            return false;
        }
        let Ok(host) = super::memory::host_memory() else {
            return false;
        };
        let need = 22_400 * suffix as u64
            + super::memory::prefill_scratch(&self.config, super::MAX_PREFILL_CHUNK)
            + super::memory::MAPPED_TURN_FLOOR;
        need <= host.available()
    }

    /// A whole prompt: one layer-major group and its tail in group mode,
    /// else one window at a time with the next window's rows requested while
    /// this one runs. The window is read before every pass: a pass that
    /// leaves the mapped path shrinks it for the passes after it.
    pub fn prefill_prompt(&mut self, tokens: &[u32]) -> Result<MxArray> {
        if self.group_mode(tokens.len()) {
            let (windows, tail) = self.group_windows(tokens);
            return Ok(self.prefill_group(&windows, tail, true, false)?.0);
        }
        let mut logits = None;
        let mut start = 0;
        while start < tokens.len() {
            let window = self.prefill_slice_size();
            let end = (start + window).min(tokens.len());
            let next_end = (end + window).min(tokens.len());
            logits = Some(self.prefill_chunk_ahead(
                &tokens[start..end],
                &tokens[end..next_end],
                None,
                end == tokens.len(),
            )?);
            start = end;
        }
        logits.ok_or_else(|| Error::from_reason("Empty qwen4_exp prefill"))
    }

    /// One layer-major group over `windows`, then `tail` as its own pass on
    /// the group's final state. Returns the logits of the last row and, when
    /// `views` is set, one hidden view per token of every window (what the
    /// MTP observer takes per window); `last_chunk_hidden` holds the last
    /// pass's views as after any pass. An error resets the decoder.
    pub fn prefill_group(
        &mut self,
        windows: &[&[u32]],
        tail: &[u32],
        project_logits: bool,
        views: bool,
    ) -> Result<(MxArray, Vec<Vec<MxArray>>)> {
        if windows.is_empty() {
            return Err(Error::from_reason("Qwen4 prefill group has no windows"));
        }
        if cfg!(target_os = "macos") {
            self.weights.check_forward_headroom()?;
        }
        let rows: Vec<usize> = windows.iter().map(|w| w.len()).collect();
        let total: usize = rows.iter().sum();
        self.rotary_tables.get_mut().clear();
        self.rotary_tables.get_mut().reserve(windows.len());
        if windows.len() > 1 {
            // Writes a failed turn left lazy would otherwise ride on this
            // group's first gather of the same layer.
            if let Some(adapter) = &mut self.paged {
                adapter
                    .eval_pending_pool_writes()
                    .map_err(Error::from_reason)?;
            }
            self.weights.begin_group();
        }
        let started = std::time::Instant::now();
        let result = self.prefill_group_with_device_routes(
            windows,
            tail,
            None,
            project_logits && tail.is_empty(),
        );
        self.rotary_tables.get_mut().clear();
        self.weights.end_group();
        let logits = match result {
            Ok(logits) => logits,
            Err(error) => {
                self.group_streams.clear();
                self.reset();
                return Err(error);
            }
        };
        self.note("qwen4::group::end");
        let streams = std::mem::take(&mut self.group_streams);
        let views = if views && windows.len() > 1 {
            streams
                .iter()
                .zip(&rows)
                .map(|(stream, &n)| {
                    (0..n)
                        .map(|j| stream.slice_axis(1, j as i64, j as i64 + 1))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?
        } else if views {
            vec![self.last_chunk_hidden.clone()]
        } else {
            Vec::new()
        };
        let elapsed = started.elapsed();
        if windows.len() > 1 {
            for &n in &rows {
                self.observe_pass(n, elapsed.mul_f64(n as f64 / total as f64));
            }
        } else {
            self.observe_pass(total, elapsed);
        }
        if tail.is_empty() {
            return Ok((logits, views));
        }
        self.note("qwen4::tail::start");
        let logits = self.prefill_chunk_ahead(tail, &[], None, project_logits)?;
        Ok((logits, views))
    }

    /// Window `w` of a group starts a layer: `history` is the prompt before
    /// it and the page cursor its base, as at the start of its own pass.
    fn set_cursor(&mut self, base: usize, windows: &[&[u32]], w: usize) {
        self.history.truncate(base);
        for window in &windows[..w] {
            self.history.extend_from_slice(window);
        }
        self.paged_chunk_start = self.history.len();
    }

    /// A group starts: every window's embedding rows, then every window's
    /// PLE rows, then the tail's, three requests in that order so sweep 0
    /// never queues behind sweep 1's rows and the tail's behind neither.
    fn prefetch_group_rows(&mut self, windows: &[&[u32]], tail: &[u32]) -> Result<()> {
        let embedding = self.key("embed_tokens.weight", "token_embd.weight");
        let keys: Vec<_> = windows
            .iter()
            .flat_map(|w| w.iter().map(|&t| (embedding.clone(), t as usize)))
            .collect();
        self.weights.prefetch_rows(&keys)?;
        let mut history = self.history.clone();
        let mut keys = Vec::new();
        for window in windows {
            keys.extend(self.ple_row_keys_after(&history, window)?);
            history.extend_from_slice(window);
        }
        self.weights.prefetch_rows(&keys)?;
        if tail.is_empty() {
            return Ok(());
        }
        let mut keys: Vec<_> = tail
            .iter()
            .map(|&t| (embedding.clone(), t as usize))
            .collect();
        keys.extend(self.ple_row_keys_after(&history, tail)?);
        self.weights.prefetch_rows(&keys)
    }

    /// Layer `i` of a window of `rows` tokens at `first` is done: a full
    /// attention layer's keys and values go to the pool, lazily, and are
    /// evaluated where a plain pass evaluates them.
    #[allow(clippy::too_many_arguments)]
    fn write_window_kv(
        &mut self,
        i: usize,
        cache: &mut LayerCache,
        rows: usize,
        first: usize,
        floor: usize,
        async_window: bool,
        submit_span: usize,
        defer_final: bool,
    ) -> Result<()> {
        if self.config.linear(i) {
            return Ok(());
        }
        let Some(adapter) = &mut self.paged else {
            return Ok(());
        };
        let layer = (0..i).filter(|&l| !self.config.linear(l)).count() as u32;
        let shape = [
            rows as i64,
            self.config.num_key_value_heads as i64,
            self.config.head_dim as i64,
        ];
        let k = cache
            .keys
            .take()
            .ok_or_else(|| Error::from_reason("Qwen4 prefill produced no attention keys"))?
            .transpose(Some(&[0, 2, 1, 3]))?
            .reshape(&shape)?;
        let v = cache
            .values
            .take()
            .ok_or_else(|| Error::from_reason("Qwen4 prefill produced no attention values"))?
            .transpose(Some(&[0, 2, 1, 3]))?
            .reshape(&shape)?;
        super::paged::write_rows(adapter, layer, &k, &v, first as u32, floor as u32)?;
        let layers = self.config.num_hidden_layers;
        if self.verification.is_none()
            && ((rows > 1 && !async_window)
                || (i + 1).is_multiple_of(submit_span)
                || i + 1 == layers)
            && !(defer_final && i + 1 == layers)
        {
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
        }
        Ok(())
    }

    /// A trip is held at the end of a sweep: the group's frontier (every
    /// stream, the tape so far, the caches, the pending touches) is evaluated
    /// and committed first, so nothing lazy reads a mapped bank, then the
    /// store leaves the mapped path and the remaining sweeps run plain on
    /// slots over the same partition.
    fn switch_mid_group(&mut self, streams: &[MxArray], base: usize) -> Result<()> {
        let outputs: Vec<&MxArray> = streams.iter().collect();
        match self.device_routes.take() {
            Some(routes) => {
                if !self.commit_tentative(&outputs, &routes)? {
                    return Err(Error::from_reason(
                        "Qwen4 prefill group routes lost their mapped banks",
                    ));
                }
            }
            None => {
                let touches = self.weights.take_pending_touches();
                let mut completed = outputs;
                completed.extend(self.pending_state_arrays());
                completed.extend(touches.iter());
                MxArray::eval_arrays_with_context(&completed, "qwen4::group::frontier")?;
                if let Some(adapter) = &mut self.paged {
                    adapter
                        .eval_pending_pool_writes()
                        .map_err(Error::from_reason)?;
                }
            }
        }
        self.note("qwen4::group::frontier");
        self.note("qwen4::group::trip");
        match self.weights.apply_held_trip() {
            Ok(Some(_)) => {
                self.note("qwen4::group::switch");
                self.weights.end_group();
                self.weights
                    .live
                    .fallback_token_index
                    .store(base as u64, Ordering::Release);
                self.residency_changed();
            }
            Ok(None) => {}
            Err(error) => {
                tracing::warn!("qwen4 residency: slot fallback not applied: {error}")
            }
        }
        Ok(())
    }

    /// One pass over `windows`, `for layer { for window }`. A group of one
    /// window is the window-major pass; a group of more runs every window of
    /// a layer before the next layer, each window at its own cursor, so the
    /// GDN, PLE, indexer and page state flows from (layer, w) to
    /// (layer, w + 1) as a window-major run carries it between passes.
    fn prefill_group_inner(
        &mut self,
        windows: &[&[u32]],
        next: &[u32],
        embeddings: Option<&MxArray>,
        project_logits: bool,
    ) -> Result<MxArray> {
        let base = self.history.len();
        let started = std::time::Instant::now();
        let rows = (self.weights.lookup.prefetched(), self.weights.lookup.late());
        let group = windows.len() > 1;
        let tokens = *windows
            .first()
            .ok_or_else(|| Error::from_reason("Qwen4 prefill group has no windows"))?;
        let total: usize = windows.iter().map(|w| w.len()).sum();
        let layers = self.config.num_hidden_layers;
        let mut submit_span = self.decode_submit_span();
        let mut async_window = windows.iter().all(|w| w.len() > 8) && self.async_device_prefill();
        let mut defer_final = self.deferred_forward_completion();
        if windows
            .iter()
            .any(|w| w.is_empty() || w.len() > super::MAX_PREFILL_CHUNK)
            || base + total > self.config.effective_context_limit()
            || windows
                .iter()
                .flat_map(|w| w.iter())
                .any(|&t| t as usize >= self.config.vocab_size)
        {
            return Err(Error::from_reason(
                "Qwen4 prefill chunk exceeds token/context budget",
            ));
        }
        if let Some(e) = embeddings {
            if group {
                return Err(Error::from_reason(
                    "Qwen4 prefill group takes no media embeddings",
                ));
            }
            if *e.shape()? != [1, tokens.len() as i64, self.config.hidden_size as i64] {
                return Err(Error::from_reason("Qwen4 media embedding shape mismatch"));
            }
        }
        self.paged_chunk_start = base;
        if let Some(adapter) = &mut self.paged {
            if adapter.request_tokens() != self.history {
                return Err(Error::from_reason(
                    "Qwen4 paged/recurrent frontier mismatch",
                ));
            }
            if group {
                adapter
                    .record_tokens(&windows.concat())
                    .map_err(Error::from_reason)?;
            } else {
                adapter.record_tokens(tokens).map_err(Error::from_reason)?;
            }
        }
        if group {
            self.prefetch_group_rows(windows, next)?;
        } else {
            self.prefetch_window_rows(tokens, next, embeddings.is_none())?;
        }
        let carry = tokens.len() > 1 && self.verification.is_none();
        if group && !carry {
            return Err(Error::from_reason(
                "Qwen4 prefill group needs the carried window path",
            ));
        }
        let mut hidden = Vec::with_capacity(if carry { 0 } else { tokens.len() });
        let embedding_key = self.key("embed_tokens.weight", "token_embd.weight");
        let hc = self.config.hc_count as i32;
        let window_embedding = if embeddings.is_none() && tokens.len() > 1 {
            Some(
                self.weights
                    .lookup_rows(
                        &embedding_key,
                        &tokens.iter().map(|&t| t as usize).collect::<Vec<_>>(),
                    )?
                    .reshape(&[1, tokens.len() as i64, self.config.hidden_size as i64])?,
            )
        } else {
            None
        };
        // One residual stream per window; the other windows' streams start
        // at their layer-0 step, from the rows the fetcher holds by then.
        let mut streams: Vec<MxArray> = Vec::with_capacity(windows.len());
        let mut window: Option<MxArray> = None;
        if carry {
            let embeddings = embeddings.or(window_embedding.as_ref()).ok_or_else(|| {
                Error::from_reason("Qwen4 prefill window is missing its token embeddings")
            })?;
            streams.push(MxArray::tile(embeddings, &[1, 1, hc])?);
        } else {
            for (j, &token) in tokens.iter().enumerate() {
                self.check_cancelled()?;
                let row = match embeddings.or(window_embedding.as_ref()) {
                    Some(e) => e.slice_axis(1, j as i64, j as i64 + 1)?,
                    None => self.embed_token(token)?,
                };
                hidden.push(MxArray::tile(&row, &[1, 1, hc])?);
            }
        }
        if self.verification.is_some() {
            self.verification = Some(
                (0..tokens.len())
                    .map(|j| DecoderState {
                        caches: (0..self.config.num_hidden_layers)
                            .map(|_| LayerCache::default())
                            .collect(),
                        history: self
                            .history
                            .iter()
                            .chain(tokens[..=j].iter())
                            .copied()
                            .collect(),
                        positions: self.positions.clone(),
                        rope_delta: self.rope_delta,
                        media_digests: self.media_digests.clone(),
                        last_chunk_hidden: Vec::new(),
                        cancelled: self.cancelled.clone(),
                    })
                    .collect(),
            );
        }
        if self.verification.is_some() {
            window = Some(MxArray::concatenate_many(hidden.iter().collect(), Some(1))?);
            hidden.clear();
        }
        for i in 0..layers {
            self.check_cancelled()?;
            let mut cache = std::mem::take(&mut self.caches[i]);
            if carry {
                for (w, tokens_w) in windows.iter().enumerate() {
                    if group {
                        if w > 0 {
                            self.check_cancelled()?;
                        }
                        self.set_cursor(base, windows, w);
                    }
                    let input = if i == 0 && w > 0 {
                        let rows: Vec<usize> = tokens_w.iter().map(|&t| t as usize).collect();
                        let e = self.weights.lookup_rows(&embedding_key, &rows)?.reshape(&[
                            1,
                            tokens_w.len() as i64,
                            self.config.hidden_size as i64,
                        ])?;
                        let stream = MxArray::tile(&e, &[1, 1, hc])?;
                        streams.push(stream.clone());
                        stream
                    } else {
                        streams[w].clone()
                    };
                    let (stream, normed) =
                        self.attention_matrix_for_mlp(&input, tokens_w, i, &mut cache, true)?;
                    streams[w] = self.mlp_matrix_for_attention(&stream, i, normed.as_ref())?;
                    let first = self.paged_chunk_start;
                    self.write_window_kv(
                        i,
                        &mut cache,
                        tokens_w.len(),
                        first,
                        base,
                        async_window,
                        submit_span,
                        defer_final,
                    )?;
                }
            } else if self.verification.is_some() {
                let input = window
                    .take()
                    .ok_or_else(|| Error::from_reason("Qwen4 verification lost its window"))?;
                let (stream, saved) = self.verify_layer(&input, tokens, i, &mut cache)?;
                self.submit(&[&stream], "qwen4::decoder::verify")?;
                window = Some(stream);
                if let Some(states) = &mut self.verification {
                    for (state, saved) in states.iter_mut().zip(saved) {
                        state.caches[i] = saved;
                    }
                }
            } else {
                for (j, &token) in tokens.iter().enumerate() {
                    self.check_cancelled()?;
                    let next = self.layer(hidden[j].clone(), token, i, &mut cache)?;
                    if defer_final && i + 1 == self.config.num_hidden_layers {
                        // TrackFastModel returns a lazy final mixer/head. The
                        // checked route commit joins it with all state/readers.
                    } else if tokens.len() == 1
                        && (i + 1) % submit_span != 0
                        && i + 1 < self.config.num_hidden_layers
                    {
                        if self.decode_should_submit(i + 1) {
                            self.submit(&[&next], "qwen4::decoder::next")?;
                        }
                    } else {
                        MxArray::eval_arrays_with_context(&[&next], "qwen4::decoder::next")?;
                    }
                    hidden[j] = next;
                    self.history.push(token);
                }
            }
            self.history.truncate(base);
            if group {
                self.paged_chunk_start = base;
            }
            if !carry {
                self.write_window_kv(
                    i,
                    &mut cache,
                    tokens.len(),
                    base,
                    base,
                    async_window,
                    submit_span,
                    defer_final,
                )?;
            }
            self.caches[i] = cache;
            if tokens.len() == 1
                && self.verification.is_none()
                && ((i + 1) % submit_span == 0 || i + 1 == self.config.num_hidden_layers)
                && !(defer_final && i + 1 == self.config.num_hidden_layers)
                && let Some(adapter) = &mut self.paged
            {
                adapter
                    .eval_pending_pool_writes()
                    .map_err(Error::from_reason)?;
            }
            super::memory::maintain_freelist(self.weights.plan.physical_bytes);
            if group {
                #[cfg(test)]
                if let Some(mut hook) = self.after_sweep.take() {
                    hook(self, i);
                    self.after_sweep = Some(hook);
                }
                if self.weights.observe_sweep() {
                    self.switch_mid_group(&streams, base)?;
                    submit_span = self.decode_submit_span();
                    async_window =
                        windows.iter().all(|w| w.len() > 8) && self.async_device_prefill();
                    defer_final = self.deferred_forward_completion();
                }
            }
        }
        let last_tokens = windows[windows.len() - 1];
        let final_window = if carry {
            streams.last().cloned()
        } else {
            window
        };
        if let Some(window) = final_window {
            // Only public/MTP frontier capture needs token views. Keep one
            // matrix across layer boundaries instead of rebuilding it twice
            // per layer from thousands of token-sized lazy array objects.
            hidden = (0..last_tokens.len())
                .map(|j| window.slice_axis(1, j as i64, j as i64 + 1))
                .collect::<Result<Vec<_>>>()?;
        }
        if let Some(states) = &mut self.verification {
            for (state, h) in states.iter_mut().zip(&hidden) {
                state.last_chunk_hidden = vec![h.clone()];
            }
        }
        let last_hidden = hidden
            .last()
            .ok_or_else(|| Error::from_reason("Qwen4 prefill produced no hidden rows"))?
            .clone();
        self.last_chunk_hidden = hidden;
        if self.verification.is_some() {
            self.history.extend_from_slice(tokens);
            return Ok(last_hidden);
        }
        let (mixed, _) = self.hyper(&last_hidden, "hyper_connection_mixer", "output_hc", false)?;
        let logits = if project_logits {
            self.linear(&mixed, "lm_head.weight", "output.weight")?
        } else {
            mixed
        };
        if !defer_final {
            MxArray::eval_arrays_with_context(&[&logits], "qwen4::decoder::logits")?;
        }
        tracing::debug!(
            "qwen4 prefill {}: {total} tokens at {base} in {:.1?}, rows prefetched +{} late +{}",
            if group {
                format!("group of {} windows", windows.len())
            } else {
                "window".to_owned()
            },
            started.elapsed(),
            self.weights.lookup.prefetched() - rows.0,
            self.weights.lookup.late() - rows.1
        );

        for window in windows {
            self.history.extend_from_slice(window);
        }
        self.group_streams = if group { streams } else { Vec::new() };
        Ok(logits)
    }

    fn forward_token_inner(&mut self, token: u32, project_logits: bool) -> Result<MxArray> {
        self.check_cancelled()?;
        let submit_span = self.decode_submit_span();
        let defer_final = self.deferred_forward_completion();
        if token as usize >= self.config.vocab_size
            || self.history.len() >= self.config.effective_context_limit()
        {
            return Err(Error::from_reason(
                "qwen4_exp token/context outside configured limits",
            ));
        }
        self.prefetch_ple_rows(&[token])?;
        let row = self.embed_token(token)?;
        let mut x = MxArray::tile(&row, &[1, 1, self.config.hc_count as i32])?;
        for i in 0..self.config.num_hidden_layers {
            self.check_cancelled()?;
            let mut cache = std::mem::take(&mut self.caches[i]);
            x = self.layer(x, token, i, &mut cache)?;
            if defer_final && i + 1 == self.config.num_hidden_layers {
                // The validated route commit completes the final head too.
            } else if (i + 1) % submit_span != 0 && i + 1 < self.config.num_hidden_layers {
                if self.decode_should_submit(i + 1) {
                    self.submit(&[&x], "qwen4::decoder::x")?;
                }
            } else {
                MxArray::eval_arrays_with_context(&[&x], "qwen4::decoder::x")?;
            }
            self.caches[i] = cache;
            // Keep reusable buffers bounded after all layer consumers complete.
            super::memory::maintain_freelist(self.weights.plan.physical_bytes);
        }
        self.last_chunk_hidden = vec![x.clone()];
        let (hidden, _) = self.hyper(&x, "hyper_connection_mixer", "output_hc", false)?;
        let logits = if project_logits {
            self.linear(&hidden, "lm_head.weight", "output.weight")?
        } else {
            hidden
        };
        if !defer_final {
            MxArray::eval_arrays_with_context(&[&logits], "qwen4::decoder::logits")?;
        }
        self.history.push(token);
        Ok(logits)
    }
}
