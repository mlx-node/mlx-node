//! Pre-allocated CUDA GPU KV cache for Inkling-2-Mini (40 SWA circular rings + 8 Global slabs).
//!
//! Layout matches the Wave 2 CUDA primitives (`kv_conv_prep_store_cuda`, `sdpa_few_cuda`,
//! `sdpa_prefill_cuda`, and `kv_compact_rows`):
//! - `k`, `v`: `[1, num_kv_heads, cap, head_dim]` (`BFloat16` or `Int8`, allocated on GPU `device == 0`)
//! - `k_scales`, `v_scales`: optional `[1, num_kv_heads, cap, 1]` (`Float32`, present iff `Int8`)
//!
//! Memory architecture:
//! - **40 SWA visits (`is_local = true`)**: fixed circular ring buffer of capacity `2048`
//!   (`1024` retained sliding window + up to `1024` chunk tokens without overwriting any active
//!   window slot). Zero reallocations during decode keep GPU device pointers stable for CUDA Graphs.
//! - **8 Global visits (`is_local = false`)**: growable flat slab (`ring_cap = 0`), starting at
//!   capacity `4096` and doubling on demand via `alloc_gpu_uninit` + slice copy.

use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};
use crate::transformer::KvFormat;

use super::config::Inkling2Config;
use super::short_conv::{Inkling2ShortConvState, Inkling2ShortConvTape};

/// Circular ring buffer capacity for SWA visits (`1024` window + `1024` max chunk).
pub(crate) const CUDA_SWA_RING_CAP: usize = 2048;

/// Initial capacity for growable Global flat KV slabs.
pub(crate) const CUDA_GLOBAL_INITIAL_CAP: usize = 4096;

/// Storage format for Inkling-2-Mini CUDA KV caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Inkling2CudaKvFormat {
    Bf16,
    Int8,
}

impl Inkling2CudaKvFormat {
    /// Resolve the active CUDA KV format from `MLX_INKLING2_CUDA_KV_FORMAT=bf16|int8`
    /// (highest priority), then load options, defaulting to **`Bf16`**.
    pub(crate) fn resolve(requested: Option<KvFormat>) -> Self {
        let env = std::env::var("MLX_INKLING2_CUDA_KV_FORMAT").ok();
        Self::resolve_with_env(requested, env.as_deref())
    }

    pub(crate) fn resolve_with_env(
        requested: Option<KvFormat>,
        env_override: Option<&str>,
    ) -> Self {
        if let Some(raw) = env_override {
            match raw.trim().to_ascii_lowercase().as_str() {
                "bf16" | "bfloat16" => return Self::Bf16,
                "int8" | "i8" | "q8" => return Self::Int8,
                _ => {}
            }
        }
        match requested {
            Some(KvFormat::Int8) => Self::Int8,
            Some(KvFormat::Bf16) | None => Self::Bf16,
        }
    }

    #[inline]
    pub(crate) fn kv_dtype(self) -> DType {
        match self {
            Self::Bf16 => DType::BFloat16,
            Self::Int8 => DType::Int8,
        }
    }

    #[inline]
    pub(crate) fn is_int8(self) -> bool {
        matches!(self, Self::Int8)
    }

    #[inline]
    pub(crate) fn as_kv_format(self) -> KvFormat {
        match self {
            Self::Bf16 => KvFormat::Bf16,
            Self::Int8 => KvFormat::Int8,
        }
    }
}

/// Allocate a settled GPU-resident (`device == 0`) uninitialized KV buffer when CUDA is active,
/// failing loudly if GPU allocation fails, or a settled zero-initialized array on CPU / non-CUDA
/// test hosts.
fn alloc_kv_buffer_with(
    shape: &[i64],
    dtype: DType,
    cuda_available: bool,
    gpu_alloc: impl FnOnce(&[i64], DType) -> Result<Option<MxArray>>,
) -> Result<MxArray> {
    if cuda_available {
        return gpu_alloc(shape, dtype)?.ok_or_else(|| {
            Error::from_reason(format!(
                "Inkling2 CUDA KV buffer allocation failed for shape {shape:?} dtype {dtype:?}"
            ))
        });
    }
    let buf = MxArray::zeros(shape, Some(dtype))?;
    MxArray::eval_arrays(&[&buf])?;
    Ok(buf)
}

fn alloc_kv_buffer(shape: &[i64], dtype: DType) -> Result<MxArray> {
    alloc_kv_buffer_with(
        shape,
        dtype,
        crate::accelerator::cuda_ops_available(),
        MxArray::alloc_gpu_uninit,
    )
}

/// Write `gathered` (`[1, H, keep, D]`) into `buf` starting at `write_offset`, splitting across
/// the end of the ring when `ring_cap > 0 && write_offset + keep > ring_cap`.
fn scatter_rows_ring(
    buf: &mut MxArray,
    gathered: &MxArray,
    write_offset: usize,
    keep: usize,
    ring_cap: usize,
) -> Result<()> {
    if ring_cap == 0 || write_offset + keep <= ring_cap {
        buf.slice_assign_axis_inplace(
            2,
            write_offset as i64,
            (write_offset + keep) as i64,
            gathered,
        )
    } else {
        let first = ring_cap - write_offset;
        let head = gathered.slice_axis(2, 0, first as i64)?;
        let tail = gathered.slice_axis(2, first as i64, keep as i64)?;
        buf.slice_assign_axis_inplace(2, write_offset as i64, ring_cap as i64, &head)?;
        buf.slice_assign_axis_inplace(2, 0, (keep - first) as i64, &tail)
    }
}

/// Per-visit pre-allocated CUDA KV cache (circular ring for SWA, growable slab for Global).
#[derive(Clone)]
pub(crate) struct Inkling2CudaVisitCache {
    pub(crate) k: MxArray,
    pub(crate) v: MxArray,
    pub(crate) k_scales: Option<MxArray>,
    pub(crate) v_scales: Option<MxArray>,
    /// Absolute logical position of the oldest retained token (index `0` of active cache).
    pub(crate) kv_start_pos: usize,
    /// Physical slot index of the oldest retained token in a circular ring (`0` for Global slabs).
    pub(crate) kv_phys_base: usize,
    /// Number of committed prefix tokens currently retained (`<= sliding_window` for SWA).
    pub(crate) cache_len: usize,
    /// Total tokens ever committed into this visit cache.
    pub(crate) total_seen: usize,
    /// Whether this visit uses sliding-window attention.
    pub(crate) is_local: bool,
    /// Sliding window size (`1024` for Inkling-2-Mini).
    pub(crate) sliding_window: usize,
    /// Circular ring capacity (`2048` for SWA, `0` for linear Global slab).
    pub(crate) ring_cap: usize,
    /// Physical allocated slot capacity along axis 2.
    pub(crate) cap: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) kv_format: Inkling2CudaKvFormat,
}

impl Inkling2CudaVisitCache {
    pub(crate) fn new(
        num_kv_heads: usize,
        head_dim: usize,
        is_local: bool,
        sliding_window: usize,
        kv_format: Inkling2CudaKvFormat,
    ) -> Result<Self> {
        let (ring_cap, cap) = if is_local {
            let rc = sliding_window
                .saturating_add(super::cuda_forward::CUDA_PREFILL_CHUNK)
                .max(CUDA_SWA_RING_CAP);
            (rc, rc)
        } else {
            (0, CUDA_GLOBAL_INITIAL_CAP)
        };
        Self::with_capacity(
            num_kv_heads,
            head_dim,
            is_local,
            sliding_window,
            ring_cap,
            cap,
            kv_format,
        )
    }

    fn with_capacity(
        num_kv_heads: usize,
        head_dim: usize,
        is_local: bool,
        sliding_window: usize,
        ring_cap: usize,
        cap: usize,
        kv_format: Inkling2CudaKvFormat,
    ) -> Result<Self> {
        let kv_shape = [1, num_kv_heads as i64, cap as i64, head_dim as i64];
        let k = alloc_kv_buffer(&kv_shape, kv_format.kv_dtype())?;
        let v = alloc_kv_buffer(&kv_shape, kv_format.kv_dtype())?;
        let (k_scales, v_scales) = if kv_format.is_int8() {
            let s_shape = [1, num_kv_heads as i64, cap as i64, 1];
            (
                Some(alloc_kv_buffer(&s_shape, DType::Float32)?),
                Some(alloc_kv_buffer(&s_shape, DType::Float32)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            k,
            v,
            k_scales,
            v_scales,
            kv_start_pos: 0,
            kv_phys_base: 0,
            cache_len: 0,
            total_seen: 0,
            is_local,
            sliding_window,
            ring_cap,
            cap,
            num_kv_heads,
            head_dim,
            kv_format,
        })
    }

    /// Physical starting slot where the next batch of tokens should be written.
    #[inline]
    pub(crate) fn write_offset(&self) -> usize {
        if self.ring_cap > 0 {
            (self.kv_phys_base + self.cache_len) % self.ring_cap
        } else {
            self.cache_len
        }
    }

    /// Ensure the backing KV buffers can hold `add_tokens` additional tokens starting at
    /// [`Self::write_offset`].
    ///
    /// - For SWA circular rings (`ring_cap > 0`): zero-allocation fast path as long as
    ///   `cache_len + add_tokens <= ring_cap`.
    /// - For Global linear slabs (`ring_cap == 0`): doubles capacity via `alloc_gpu_uninit` +
    ///   contiguous prefix copy and immediate eval.
    pub(crate) fn ensure_capacity_for(&mut self, add_tokens: usize) -> Result<()> {
        let needed = self.cache_len.saturating_add(add_tokens);
        if self.ring_cap > 0 {
            if needed > self.ring_cap {
                return Err(Error::from_reason(format!(
                    "Inkling2CudaVisitCache SWA ring overflow: cache_len={} + add_tokens={} > ring_cap={}",
                    self.cache_len, add_tokens, self.ring_cap
                )));
            }
            return Ok(());
        }

        if needed <= self.cap {
            return Ok(());
        }

        let mut new_cap = self.cap.max(CUDA_GLOBAL_INITIAL_CAP);
        while new_cap < needed {
            new_cap = new_cap.saturating_mul(2);
            if new_cap == usize::MAX {
                return Err(Error::from_reason(
                    "Inkling2CudaVisitCache Global slab capacity overflow",
                ));
            }
        }

        let kv_shape = [
            1,
            self.num_kv_heads as i64,
            new_cap as i64,
            self.head_dim as i64,
        ];
        let mut new_k = alloc_kv_buffer(&kv_shape, self.kv_format.kv_dtype())?;
        let mut new_v = alloc_kv_buffer(&kv_shape, self.kv_format.kv_dtype())?;
        let (mut new_ks, mut new_vs) = if self.kv_format.is_int8() {
            let s_shape = [1, self.num_kv_heads as i64, new_cap as i64, 1];
            (
                Some(alloc_kv_buffer(&s_shape, DType::Float32)?),
                Some(alloc_kv_buffer(&s_shape, DType::Float32)?),
            )
        } else {
            (None, None)
        };

        if self.cache_len > 0 {
            let mut old_pending: Vec<&MxArray> = Vec::with_capacity(4);
            if !self.k.is_available() {
                old_pending.push(&self.k);
            }
            if !self.v.is_available() {
                old_pending.push(&self.v);
            }
            if let Some(ref ks) = self.k_scales
                && !ks.is_available()
            {
                old_pending.push(ks);
            }
            if let Some(ref vs) = self.v_scales
                && !vs.is_available()
            {
                old_pending.push(vs);
            }
            if !old_pending.is_empty() {
                MxArray::eval_arrays_with_context(&old_pending, "inkling2 cuda kv pre-grow eval")?;
            }

            let len = self.cache_len as i64;
            let k_src = self.k.slice_axis(2, 0, len)?;
            let v_src = self.v.slice_axis(2, 0, len)?;
            new_k.slice_assign_axis_inplace(2, 0, len, &k_src)?;
            new_v.slice_assign_axis_inplace(2, 0, len, &v_src)?;

            let mut to_eval: Vec<&MxArray> = Vec::with_capacity(4);
            to_eval.push(&new_k);
            to_eval.push(&new_v);
            if let (Some(nks), Some(oks)) = (new_ks.as_mut(), self.k_scales.as_ref()) {
                let ks_src = oks.slice_axis(2, 0, len)?;
                nks.slice_assign_axis_inplace(2, 0, len, &ks_src)?;
                to_eval.push(nks);
            }
            if let (Some(nvs), Some(ovs)) = (new_vs.as_mut(), self.v_scales.as_ref()) {
                let vs_src = ovs.slice_axis(2, 0, len)?;
                nvs.slice_assign_axis_inplace(2, 0, len, &vs_src)?;
                to_eval.push(nvs);
            }
            MxArray::eval_arrays_with_context(&to_eval, "inkling2 cuda kv grow copy")?;
        }

        self.k = new_k;
        self.v = new_v;
        self.k_scales = new_ks;
        self.v_scales = new_vs;
        self.cap = new_cap;
        Ok(())
    }

    /// O(1) linear cursor commit after `accepted` contiguous tokens were written starting at
    /// `write_offset()`. Advances the SWA circular window when `cache_len > sliding_window`.
    #[inline]
    pub(crate) fn commit_linear(&mut self, accepted: usize) {
        self.cache_len += accepted;
        self.total_seen += accepted;
        if self.is_local && self.ring_cap > 0 && self.cache_len > self.sliding_window {
            let drop = self.cache_len - self.sliding_window;
            self.kv_start_pos += drop;
            self.kv_phys_base = (self.kv_phys_base + drop) % self.ring_cap;
            self.cache_len = self.sliding_window;
        }
    }

    /// Commit an accepted speculative tree branch (`accepted_nodes` ordered oldest-to-newest).
    ///
    /// - Contiguous fast path (`is_contiguous == true`): delegates directly to [`Self::commit_linear`].
    /// - Non-contiguous non-wrapping path (`keep <= 16` and no wrap past `ring_cap`): runs race-free
    ///   in-place `kernels::kv::kv_compact_rows` on `(k, v)` and `(k_scales, v_scales)`.
    /// - Ring-wrapping or fallback path: gathers accepted rows cleanly into temporaries, evaluates
    ///   them to prevent read-after-write hazards, then scatters back into up to 2 contiguous
    ///   ring intervals before advancing the cursor.
    pub(crate) fn commit_tree(
        &mut self,
        accepted_nodes: &[u32],
        is_contiguous: bool,
    ) -> Result<()> {
        let keep = accepted_nodes.len();
        if keep == 0 {
            return Ok(());
        }
        if is_contiguous {
            self.commit_linear(keep);
            return Ok(());
        }

        let write_off = self.write_offset();
        let max_node = accepted_nodes.iter().copied().max().unwrap_or(0) as usize;
        let max_slot = if self.ring_cap > 0 {
            self.ring_cap
        } else {
            self.cap
        };

        if self.ring_cap == 0 && (write_off + max_node >= self.cap || write_off + keep > self.cap) {
            return Err(Error::from_reason(format!(
                "Inkling2CudaVisitCache::commit_tree out of bounds: write_off={write_off} max_node={max_node} keep={keep} cap={}",
                self.cap
            )));
        }

        let no_wrap = write_off + max_node < max_slot && write_off + keep <= max_slot;
        if no_wrap && keep <= 16 {
            let mut nodes_i32 = Vec::with_capacity(keep);
            for &node in accepted_nodes {
                nodes_i32.push(i32::try_from(node).map_err(|_| {
                    Error::from_reason(format!(
                        "Inkling2CudaVisitCache::commit_tree: node {node} exceeds i32::MAX"
                    ))
                })?);
            }
            let nodes_arr = MxArray::from_int32(&nodes_i32, &[keep as i64])?;
            let base_i32 = i32::try_from(write_off).map_err(|_| {
                Error::from_reason(format!(
                    "Inkling2CudaVisitCache::commit_tree: write_off {write_off} exceeds i32::MAX"
                ))
            })?;

            if let Some((new_k, new_v)) =
                crate::kernels::kv::kv_compact_rows(&self.k, &self.v, base_i32, &nodes_arr)?
            {
                let mut scales_compacted = true;
                if let (Some(ks), Some(vs)) = (&self.k_scales, &self.v_scales) {
                    if let Some((new_ks, new_vs)) =
                        crate::kernels::kv::kv_compact_rows(ks, vs, base_i32, &nodes_arr)?
                    {
                        self.k_scales = Some(new_ks);
                        self.v_scales = Some(new_vs);
                    } else {
                        scales_compacted = false;
                    }
                }
                if scales_compacted {
                    self.k = new_k;
                    self.v = new_v;
                    self.commit_linear(keep);
                    return Ok(());
                }
            }
        }

        // Race-free ring-safe gather -> eval -> scatter fallback (also handles ring wrap & CPU tests).
        let mut src_slots = Vec::with_capacity(keep);
        for &node in accepted_nodes {
            let slot = if self.ring_cap > 0 {
                (write_off + node as usize) % self.ring_cap
            } else {
                write_off + node as usize
            };
            src_slots.push(i32::try_from(slot).map_err(|_| {
                Error::from_reason(format!(
                    "Inkling2CudaVisitCache::commit_tree: slot {slot} exceeds i32::MAX"
                ))
            })?);
        }
        let src_idx = MxArray::from_int32(&src_slots, &[keep as i64])?;

        let k_sel = self.k.take(&src_idx, 2)?;
        let v_sel = self.v.take(&src_idx, 2)?;
        let ks_sel = match &self.k_scales {
            Some(ks) => Some(ks.take(&src_idx, 2)?),
            None => None,
        };
        let vs_sel = match &self.v_scales {
            Some(vs) => Some(vs.take(&src_idx, 2)?),
            None => None,
        };

        let mut gathered_refs: Vec<&MxArray> = Vec::with_capacity(4);
        gathered_refs.push(&k_sel);
        gathered_refs.push(&v_sel);
        if let Some(ref ks) = ks_sel {
            gathered_refs.push(ks);
        }
        if let Some(ref vs) = vs_sel {
            gathered_refs.push(vs);
        }
        MxArray::eval_arrays_with_context(&gathered_refs, "inkling2 cuda kv commit_tree gather")?;

        scatter_rows_ring(&mut self.k, &k_sel, write_off, keep, self.ring_cap)?;
        scatter_rows_ring(&mut self.v, &v_sel, write_off, keep, self.ring_cap)?;
        if let (Some(ks), Some(gks)) = (self.k_scales.as_mut(), &ks_sel) {
            scatter_rows_ring(ks, gks, write_off, keep, self.ring_cap)?;
        }
        if let (Some(vs), Some(gvs)) = (self.v_scales.as_mut(), &vs_sel) {
            scatter_rows_ring(vs, gvs, write_off, keep, self.ring_cap)?;
        }

        let mut settled_refs: Vec<&MxArray> = Vec::with_capacity(4);
        settled_refs.push(&self.k);
        settled_refs.push(&self.v);
        if let Some(ref ks) = self.k_scales {
            settled_refs.push(ks);
        }
        if let Some(ref vs) = self.v_scales {
            settled_refs.push(vs);
        }
        MxArray::eval_arrays_with_context(&settled_refs, "inkling2 cuda kv commit_tree scatter")?;

        self.commit_linear(keep);
        Ok(())
    }
}

/// Complete 48-visit CUDA KV + ShortConv state for one active Inkling-2-Mini session.
#[derive(Clone)]
pub(crate) struct Inkling2CudaCache {
    pub(crate) visits: Vec<Inkling2CudaVisitCache>,
    pub(crate) conv_state: Inkling2ShortConvState,
    pub(crate) pos_offset: usize,
    #[allow(dead_code)]
    pub(crate) kv_format: Inkling2CudaKvFormat,
}

impl Inkling2CudaCache {
    pub(crate) fn new(config: &Inkling2Config, kv_format: Inkling2CudaKvFormat) -> Result<Self> {
        let schedule = config.visits();
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let sliding_window = config.sliding_window;
        let mut visits = Vec::with_capacity(schedule.len());
        for lv in &schedule {
            visits.push(Inkling2CudaVisitCache::new(
                num_kv_heads,
                head_dim,
                lv.is_local,
                sliding_window,
                kv_format,
            )?);
        }
        let kv_dim = num_kv_heads * head_dim;
        let mut conv_state = Inkling2ShortConvState::new(
            schedule.len(),
            config.sconv_kernel_size,
            kv_dim,
            config.hidden_size,
            DType::BFloat16,
        )?;
        let conv_refs: Vec<&MxArray> = conv_state
            .visits
            .iter()
            .flat_map(|v| [&v.k, &v.v, &v.attn_post, &v.mlp_post])
            .collect();
        MxArray::eval_arrays_with_context(&conv_refs, "inkling2 cuda conv_state init")?;
        conv_state.promote_cuda_device()?;
        Ok(Self {
            visits,
            conv_state,
            pos_offset: 0,
            kv_format,
        })
    }

    /// Collect all state arrays (KV caches, optional INT8 scales, and ShortConv histories) that
    /// are currently unevaluated (`!is_available()`).
    pub(crate) fn collect_pending_state_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        for v in &self.visits {
            if !v.k.is_available() {
                out.push(&v.k);
            }
            if !v.v.is_available() {
                out.push(&v.v);
            }
            if let Some(ref ks) = v.k_scales
                && !ks.is_available()
            {
                out.push(ks);
            }
            if let Some(ref vs) = v.v_scales
                && !vs.is_available()
            {
                out.push(vs);
            }
        }
        for c in &self.conv_state.visits {
            if !c.k.is_available() {
                out.push(&c.k);
            }
            if !c.v.is_available() {
                out.push(&c.v);
            }
            if !c.attn_post.is_available() {
                out.push(&c.attn_post);
            }
            if !c.mlp_post.is_available() {
                out.push(&c.mlp_post);
            }
        }
    }

    /// Evaluate every KV cache buffer, INT8 scale buffer, and ShortConv history across all visits.
    pub(crate) fn eval_state(&self) -> Result<()> {
        let mut arrays: Vec<&MxArray> = Vec::with_capacity(self.visits.len() * 8);
        for v in &self.visits {
            arrays.push(&v.k);
            arrays.push(&v.v);
            if let Some(ref ks) = v.k_scales {
                arrays.push(ks);
            }
            if let Some(ref vs) = v.v_scales {
                arrays.push(vs);
            }
        }
        for c in &self.conv_state.visits {
            arrays.push(&c.k);
            arrays.push(&c.v);
            arrays.push(&c.attn_post);
            arrays.push(&c.mlp_post);
        }
        MxArray::eval_arrays_with_context(&arrays, "inkling2 cuda cache eval_state")
    }

    /// Evaluate only state arrays that are not yet settled (`!is_available()`).
    pub(crate) fn eval_pending_state(&self) -> Result<()> {
        let mut pending: Vec<&MxArray> = Vec::new();
        self.collect_pending_state_arrays(&mut pending);
        if !pending.is_empty() {
            MxArray::eval_arrays_with_context(&pending, "inkling2 cuda cache eval_pending_state")?;
        }
        Ok(())
    }

    /// Commit `accepted` linear tokens across all visits and advance `pos_offset`.
    pub(crate) fn commit_linear(&mut self, accepted: usize) {
        for v in &mut self.visits {
            v.commit_linear(accepted);
        }
        self.pos_offset += accepted;
    }

    /// Commit an accepted speculative tree branch across all 48 KV caches and advance `pos_offset`.
    ///
    /// - Contiguous fast path (`is_contiguous == true`): O(1) cursor advance on every visit.
    /// - Non-contiguous CUDA fast path (`keep <= 16`): batches all non-wrapping visits' `(k, v)`
    ///   (plus `(k_scales, v_scales)` when INT8) into a single `compact_kv_batch_cuda` call
    ///   (1 CUDA launch for 96 BF16 caches / 2 launches for 192 INT8 caches instead of 48..96).
    ///   Aliased outputs remain lazy so `collect_pending_state_arrays` / `eval_pending_state` /
    ///   `eval_boundary()` evaluate them in one batch alongside ShortConv tape commits.
    ///   Any SWA visit whose tree range wraps around `ring_cap` falls back cleanly per visit.
    pub(crate) fn commit_tree(
        &mut self,
        accepted_nodes: &[u32],
        is_contiguous: bool,
    ) -> Result<()> {
        let keep = accepted_nodes.len();
        if keep == 0 {
            return Ok(());
        }
        if is_contiguous {
            self.commit_linear(keep);
            return Ok(());
        }

        let max_node = accepted_nodes.iter().copied().max().unwrap_or(0) as usize;
        for v in &self.visits {
            let write_off = v.write_offset();
            if v.ring_cap == 0 && (write_off + max_node >= v.cap || write_off + keep > v.cap) {
                return Err(Error::from_reason(format!(
                    "Inkling2CudaVisitCache::commit_tree out of bounds: write_off={write_off} max_node={max_node} keep={keep} cap={}",
                    v.cap
                )));
            }
        }

        if crate::kernels::inkling2::available() && keep <= 16 && !self.visits.is_empty() {
            let mut nodes_i32 = Vec::with_capacity(keep);
            for &node in accepted_nodes {
                nodes_i32.push(i32::try_from(node).map_err(|_| {
                    Error::from_reason(format!(
                        "Inkling2CudaCache::commit_tree: node {node} exceeds i32::MAX"
                    ))
                })?);
            }
            let nodes_arr = MxArray::from_int32(&nodes_i32, &[keep as i64])?;

            #[derive(Clone, Copy)]
            enum Slot {
                K,
                V,
                Ks,
                Vs,
            }

            let expected_h_kv = self.visits[0].num_kv_heads;
            let mut batched_visits: Vec<usize> = Vec::with_capacity(self.visits.len());
            let mut fallback_visits: Vec<usize> = Vec::new();
            let mut batch_slots: Vec<(usize, Slot, i32)> =
                Vec::with_capacity(self.visits.len() * 4);

            for (idx, v) in self.visits.iter().enumerate() {
                let write_off = v.write_offset();
                let max_slot = if v.ring_cap > 0 { v.ring_cap } else { v.cap };
                let no_wrap = write_off + max_node < max_slot && write_off + keep <= max_slot;
                let scales_paired = v.k_scales.is_some() == v.v_scales.is_some();
                if no_wrap
                    && scales_paired
                    && v.num_kv_heads == expected_h_kv
                    && let Ok(off_i32) = i32::try_from(write_off)
                {
                    batched_visits.push(idx);
                    batch_slots.push((idx, Slot::K, off_i32));
                    batch_slots.push((idx, Slot::V, off_i32));
                    if v.k_scales.is_some() {
                        batch_slots.push((idx, Slot::Ks, off_i32));
                        batch_slots.push((idx, Slot::Vs, off_i32));
                    }
                } else {
                    fallback_visits.push(idx);
                }
            }

            if !batch_slots.is_empty() {
                const MAX_BATCH: usize = 192;
                let mut all_ok = true;
                let mut compacted_outs: Vec<MxArray> = Vec::with_capacity(batch_slots.len());
                for chunk in batch_slots.chunks(MAX_BATCH) {
                    let mut cache_refs: Vec<&MxArray> = Vec::with_capacity(chunk.len());
                    let mut offsets: Vec<i32> = Vec::with_capacity(chunk.len());
                    for &(v_idx, slot, off_i32) in chunk {
                        let v = &self.visits[v_idx];
                        let arr = match slot {
                            Slot::K => &v.k,
                            Slot::V => &v.v,
                            Slot::Ks => v.k_scales.as_ref().ok_or_else(|| {
                                Error::from_reason(
                                    "Inkling2CudaCache::commit_tree: missing k_scales for batched slot",
                                )
                            })?,
                            Slot::Vs => v.v_scales.as_ref().ok_or_else(|| {
                                Error::from_reason(
                                    "Inkling2CudaCache::commit_tree: missing v_scales for batched slot",
                                )
                            })?,
                        };
                        cache_refs.push(arr);
                        offsets.push(off_i32);
                    }
                    if let Some(outs) = crate::kernels::inkling2::compact_kv_batch_cuda(
                        &cache_refs,
                        &offsets,
                        &nodes_arr,
                    )? && outs.len() == chunk.len()
                    {
                        compacted_outs.extend(outs);
                    } else {
                        all_ok = false;
                        break;
                    }
                }

                if all_ok {
                    for ((v_idx, slot, _), out_arr) in batch_slots.into_iter().zip(compacted_outs) {
                        let v = &mut self.visits[v_idx];
                        match slot {
                            Slot::K => v.k = out_arr,
                            Slot::V => v.v = out_arr,
                            Slot::Ks => v.k_scales = Some(out_arr),
                            Slot::Vs => v.v_scales = Some(out_arr),
                        }
                    }
                    for v_idx in batched_visits {
                        self.visits[v_idx].commit_linear(keep);
                    }
                    for v_idx in fallback_visits {
                        self.visits[v_idx].commit_tree(accepted_nodes, false)?;
                    }
                    self.pos_offset += keep;
                    return Ok(());
                }
            }
        }

        for v in &mut self.visits {
            v.commit_tree(accepted_nodes, is_contiguous)?;
        }
        self.pos_offset += keep;
        Ok(())
    }

    /// Commit an accepted speculative tree branch across all 48 KV caches AND ShortConv tapes.
    #[allow(dead_code)]
    pub(crate) fn commit_tree_with_tape(
        &mut self,
        accepted_nodes: &[u32],
        is_contiguous: bool,
        tape: &Inkling2ShortConvTape,
    ) -> Result<()> {
        self.commit_tree(accepted_nodes, is_contiguous)?;
        tape.commit_tree(accepted_nodes, is_contiguous, &mut self.conv_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row_data(cap: usize, heads: usize, dim: usize, base: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(heads * cap * dim);
        for h in 0..heads {
            for s in 0..cap {
                for d in 0..dim {
                    out.push(base + (h as f32) * 100.0 + (s as f32) * 1.0 + (d as f32) * 0.01);
                }
            }
        }
        out
    }

    #[test]
    fn test_cuda_kv_format_defaults_to_bf16_and_honors_int8_overrides() {
        assert_eq!(
            Inkling2CudaKvFormat::resolve_with_env(None, None),
            Inkling2CudaKvFormat::Bf16
        );
        assert_eq!(
            Inkling2CudaKvFormat::resolve_with_env(Some(KvFormat::Bf16), None),
            Inkling2CudaKvFormat::Bf16
        );
        assert_eq!(
            Inkling2CudaKvFormat::resolve_with_env(Some(KvFormat::Int8), None),
            Inkling2CudaKvFormat::Int8
        );
        assert_eq!(
            Inkling2CudaKvFormat::resolve_with_env(None, Some("int8")),
            Inkling2CudaKvFormat::Int8
        );
        assert_eq!(
            Inkling2CudaKvFormat::resolve_with_env(Some(KvFormat::Int8), Some("bf16")),
            Inkling2CudaKvFormat::Bf16
        );
    }

    #[test]
    fn test_swa_ring_cursor_wrap_around() {
        let mut vc = Inkling2CudaVisitCache::with_capacity(
            2,
            4,
            true,
            8,
            16,
            16,
            Inkling2CudaKvFormat::Bf16,
        )
        .unwrap();

        assert_eq!(vc.write_offset(), 0);
        assert_eq!(vc.cache_len, 0);
        assert_eq!(vc.kv_start_pos, 0);
        assert_eq!(vc.kv_phys_base, 0);

        // Prefill 6 tokens (< window=8)
        vc.ensure_capacity_for(6).unwrap();
        vc.commit_linear(6);
        assert_eq!(vc.cache_len, 6);
        assert_eq!(vc.kv_start_pos, 0);
        assert_eq!(vc.kv_phys_base, 0);
        assert_eq!(vc.write_offset(), 6);

        // Add 6 more -> total 12, drops 4, retains 8 at phys_base=4..12, next write_offset=12
        vc.ensure_capacity_for(6).unwrap();
        vc.commit_linear(6);
        assert_eq!(vc.cache_len, 8);
        assert_eq!(vc.kv_start_pos, 4);
        assert_eq!(vc.kv_phys_base, 4);
        assert_eq!(vc.write_offset(), 12);
        assert_eq!(vc.total_seen, 12);

        // Add 8 more (max chunk = ring_cap - window) -> wraps around ring_cap=16!
        // Written to slots 12,13,14,15,0,1,2,3; retained window is last 8 = slots 12..16 + 0..4!
        vc.ensure_capacity_for(8).unwrap();
        vc.commit_linear(8);
        assert_eq!(vc.cache_len, 8);
        assert_eq!(vc.kv_start_pos, 12);
        assert_eq!(vc.kv_phys_base, 12);
        assert_eq!(vc.write_offset(), 4);
        assert_eq!(vc.total_seen, 20);

        // Overflow check: 8 + 9 > 16 must fail cleanly
        assert!(vc.ensure_capacity_for(9).is_err());
    }

    #[test]
    fn test_global_slab_grow_preserves_bf16_and_int8_contents() {
        for fmt in [Inkling2CudaKvFormat::Bf16, Inkling2CudaKvFormat::Int8] {
            let mut vc = Inkling2CudaVisitCache::with_capacity(2, 4, false, 8, 0, 4, fmt).unwrap();
            let raw_k = make_row_data(4, 2, 4, 1.0);
            let raw_v = make_row_data(4, 2, 4, 50.0);
            vc.k = if fmt.is_int8() {
                let ints: Vec<i8> = raw_k.iter().map(|&x| (x as i32 % 120) as i8).collect();
                MxArray::from_int8(&ints, &[1, 2, 4, 4]).unwrap()
            } else {
                MxArray::from_float32(&raw_k, &[1, 2, 4, 4])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap()
            };
            vc.v = if fmt.is_int8() {
                let ints: Vec<i8> = raw_v.iter().map(|&x| (x as i32 % 120) as i8).collect();
                MxArray::from_int8(&ints, &[1, 2, 4, 4]).unwrap()
            } else {
                MxArray::from_float32(&raw_v, &[1, 2, 4, 4])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap()
            };
            if fmt.is_int8() {
                let s_vals: Vec<f32> = (0..8).map(|i| 0.1 + i as f32 * 0.05).collect();
                vc.k_scales = Some(MxArray::from_float32(&s_vals, &[1, 2, 4, 1]).unwrap());
                vc.v_scales = Some(
                    MxArray::from_float32(
                        &s_vals.iter().map(|x| x * 2.0).collect::<Vec<_>>(),
                        &[1, 2, 4, 1],
                    )
                    .unwrap(),
                );
            }
            vc.commit_linear(3);

            let exp_k =
                vc.k.slice_axis(2, 0, 3)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();
            let exp_v =
                vc.v.slice_axis(2, 0, 3)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();
            let exp_ks = vc
                .k_scales
                .as_ref()
                .map(|s| s.slice_axis(2, 0, 3).unwrap().to_float32().unwrap());

            // Request +4 tokens -> 3 + 4 = 7 > 4 -> grows to at least CUDA_GLOBAL_INITIAL_CAP
            vc.ensure_capacity_for(4).unwrap();
            assert!(vc.cap >= 7);
            assert_eq!(vc.k.shape().unwrap()[2] as usize, vc.cap);

            let got_k =
                vc.k.slice_axis(2, 0, 3)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();
            let got_v =
                vc.v.slice_axis(2, 0, 3)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();
            assert_eq!(got_k.as_ref(), exp_k.as_ref());
            assert_eq!(got_v.as_ref(), exp_v.as_ref());
            if let (Some(ks), Some(eks)) = (vc.k_scales.as_ref(), exp_ks) {
                let gks = ks.slice_axis(2, 0, 3).unwrap().to_float32().unwrap();
                assert_eq!(gks.as_ref(), eks.as_ref());
            }
        }
    }

    #[test]
    fn test_commit_tree_non_wrapping_and_ring_wrapping_bf16_and_int8() {
        for fmt in [Inkling2CudaKvFormat::Bf16, Inkling2CudaKvFormat::Int8] {
            // 1. Non-wrapping linear Global slab
            let mut linear =
                Inkling2CudaVisitCache::with_capacity(2, 4, false, 8, 0, 16, fmt).unwrap();
            linear.commit_linear(3); // write_offset = 3
            let mut k_vals = vec![0i8; 2 * 16 * 4];
            let mut v_vals = vec![0i8; 2 * 16 * 4];
            let mut ks_vals = vec![0.0f32; 2 * 16];
            let mut vs_vals = vec![0.0f32; 2 * 16];
            for h in 0..2 {
                for s in 0..16 {
                    for d in 0..4 {
                        let idx = (h * 16 + s) * 4 + d;
                        k_vals[idx] = ((h * 32 + s * 2 + d) % 120) as i8;
                        v_vals[idx] = -(((h * 24 + s * 3 + d) % 120) as i8);
                    }
                    ks_vals[h * 16 + s] = (h as f32) * 10.0 + s as f32 + 0.25;
                    vs_vals[h * 16 + s] = (h as f32) * 20.0 + s as f32 + 0.5;
                }
            }
            if fmt.is_int8() {
                linear.k = MxArray::from_int8(&k_vals, &[1, 2, 16, 4]).unwrap();
                linear.v = MxArray::from_int8(&v_vals, &[1, 2, 16, 4]).unwrap();
                linear.k_scales = Some(MxArray::from_float32(&ks_vals, &[1, 2, 16, 1]).unwrap());
                linear.v_scales = Some(MxArray::from_float32(&vs_vals, &[1, 2, 16, 1]).unwrap());
            } else {
                let kf: Vec<f32> = k_vals.iter().map(|&x| f32::from(x)).collect();
                let vf: Vec<f32> = v_vals.iter().map(|&x| f32::from(x)).collect();
                linear.k = MxArray::from_float32(&kf, &[1, 2, 16, 4])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                linear.v = MxArray::from_float32(&vf, &[1, 2, 16, 4])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
            }

            // Accept non-contiguous nodes [0, 2, 5] starting at write_offset = 3 -> slots 3, 5, 8
            linear.commit_tree(&[0, 2, 5], false).unwrap();
            assert_eq!(linear.cache_len, 6);
            let got_k = linear
                .k
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            let got_v = linear
                .v
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            for (r, &node) in [0usize, 2, 5].iter().enumerate() {
                let dst_slot = 3 + r;
                let src_slot = 3 + node;
                for h in 0..2 {
                    for d in 0..4 {
                        assert_eq!(
                            got_k[(h * 16 + dst_slot) * 4 + d],
                            f32::from(k_vals[(h * 16 + src_slot) * 4 + d])
                        );
                        assert_eq!(
                            got_v[(h * 16 + dst_slot) * 4 + d],
                            f32::from(v_vals[(h * 16 + src_slot) * 4 + d])
                        );
                    }
                }
            }
            if let (Some(ks), Some(vs)) = (&linear.k_scales, &linear.v_scales) {
                let gks = ks.to_float32().unwrap();
                let gvs = vs.to_float32().unwrap();
                for (r, &node) in [0usize, 2, 5].iter().enumerate() {
                    for h in 0..2 {
                        assert_eq!(gks[h * 16 + 3 + r], ks_vals[h * 16 + 3 + node]);
                        assert_eq!(gvs[h * 16 + 3 + r], vs_vals[h * 16 + 3 + node]);
                    }
                }
            }

            // 2. Ring-wrapping SWA circular ring (ring_cap=16, window=8, write_offset=14)
            let mut ring =
                Inkling2CudaVisitCache::with_capacity(2, 4, true, 8, 16, 16, fmt).unwrap();
            ring.commit_linear(14); // cache_len=8, kv_phys_base=6, write_offset=14
            assert_eq!(ring.write_offset(), 14);
            if fmt.is_int8() {
                ring.k = MxArray::from_int8(&k_vals, &[1, 2, 16, 4]).unwrap();
                ring.v = MxArray::from_int8(&v_vals, &[1, 2, 16, 4]).unwrap();
                ring.k_scales = Some(MxArray::from_float32(&ks_vals, &[1, 2, 16, 1]).unwrap());
                ring.v_scales = Some(MxArray::from_float32(&vs_vals, &[1, 2, 16, 1]).unwrap());
            } else {
                let kf: Vec<f32> = k_vals.iter().map(|&x| f32::from(x)).collect();
                let vf: Vec<f32> = v_vals.iter().map(|&x| f32::from(x)).collect();
                ring.k = MxArray::from_float32(&kf, &[1, 2, 16, 4])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                ring.v = MxArray::from_float32(&vf, &[1, 2, 16, 4])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
            }

            // Nodes [0, 3, 5] at write_offset=14 -> src slots [14, 1, 3] -> dst slots [14, 15, 0]
            ring.commit_tree(&[0, 3, 5], false).unwrap();
            assert_eq!(ring.cache_len, 8);
            assert_eq!(ring.kv_phys_base, 9);
            assert_eq!(ring.write_offset(), 1);
            let got_k = ring.k.astype(DType::Float32).unwrap().to_float32().unwrap();
            let got_v = ring.v.astype(DType::Float32).unwrap().to_float32().unwrap();
            for (r, &node) in [0usize, 3, 5].iter().enumerate() {
                let dst_slot = (14 + r) % 16;
                let src_slot = (14 + node) % 16;
                for h in 0..2 {
                    for d in 0..4 {
                        assert_eq!(
                            got_k[(h * 16 + dst_slot) * 4 + d],
                            f32::from(k_vals[(h * 16 + src_slot) * 4 + d])
                        );
                        assert_eq!(
                            got_v[(h * 16 + dst_slot) * 4 + d],
                            f32::from(v_vals[(h * 16 + src_slot) * 4 + d])
                        );
                    }
                }
            }
            if let (Some(ks), Some(vs)) = (&ring.k_scales, &ring.v_scales) {
                let gks = ks.to_float32().unwrap();
                let gvs = vs.to_float32().unwrap();
                for (r, &node) in [0usize, 3, 5].iter().enumerate() {
                    let dst_slot = (14 + r) % 16;
                    let src_slot = (14 + node) % 16;
                    for h in 0..2 {
                        assert_eq!(gks[h * 16 + dst_slot], ks_vals[h * 16 + src_slot]);
                        assert_eq!(gvs[h * 16 + dst_slot], vs_vals[h * 16 + src_slot]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_cuda_cache_batched_commit_tree_with_mixed_wrapping_and_pending_state() {
        for fmt in [Inkling2CudaKvFormat::Bf16, Inkling2CudaKvFormat::Int8] {
            let mut k_vals = vec![0i8; 2 * 16 * 4];
            let mut v_vals = vec![0i8; 2 * 16 * 4];
            let mut ks_vals = vec![0.0f32; 2 * 16];
            let mut vs_vals = vec![0.0f32; 2 * 16];
            for h in 0..2 {
                for s in 0..16 {
                    for d in 0..4 {
                        let idx = (h * 16 + s) * 4 + d;
                        k_vals[idx] = ((h * 32 + s * 2 + d) % 120) as i8;
                        v_vals[idx] = -(((h * 24 + s * 3 + d) % 120) as i8);
                    }
                    ks_vals[h * 16 + s] = (h as f32) * 10.0 + s as f32 + 0.25;
                    vs_vals[h * 16 + s] = (h as f32) * 20.0 + s as f32 + 0.5;
                }
            }

            let populate = |vc: &mut Inkling2CudaVisitCache| {
                if fmt.is_int8() {
                    vc.k = MxArray::from_int8(&k_vals, &[1, 2, 16, 4]).unwrap();
                    vc.v = MxArray::from_int8(&v_vals, &[1, 2, 16, 4]).unwrap();
                    vc.k_scales = Some(MxArray::from_float32(&ks_vals, &[1, 2, 16, 1]).unwrap());
                    vc.v_scales = Some(MxArray::from_float32(&vs_vals, &[1, 2, 16, 1]).unwrap());
                } else {
                    let kf: Vec<f32> = k_vals.iter().map(|&x| f32::from(x)).collect();
                    let vf: Vec<f32> = v_vals.iter().map(|&x| f32::from(x)).collect();
                    vc.k = MxArray::from_float32(&kf, &[1, 2, 16, 4])
                        .unwrap()
                        .astype(DType::BFloat16)
                        .unwrap();
                    vc.v = MxArray::from_float32(&vf, &[1, 2, 16, 4])
                        .unwrap()
                        .astype(DType::BFloat16)
                        .unwrap();
                }
            };

            // Visit 0: non-wrapping linear Global slab (write_offset = 3)
            let mut v0 = Inkling2CudaVisitCache::with_capacity(2, 4, false, 8, 0, 16, fmt).unwrap();
            v0.commit_linear(3);
            populate(&mut v0);
            // Visit 1: non-wrapping SWA ring (write_offset = 5)
            let mut v1 = Inkling2CudaVisitCache::with_capacity(2, 4, true, 8, 16, 16, fmt).unwrap();
            v1.commit_linear(5);
            populate(&mut v1);
            // Visit 2: wrapping SWA ring (write_offset = 14, nodes [0, 3, 5] wrap past 16!)
            let mut v2 = Inkling2CudaVisitCache::with_capacity(2, 4, true, 8, 16, 16, fmt).unwrap();
            v2.commit_linear(14);
            populate(&mut v2);

            let conv_state = Inkling2ShortConvState::new(3, 4, 8, 16, DType::BFloat16).unwrap();
            let mut cache = Inkling2CudaCache {
                visits: vec![v0, v1, v2],
                conv_state,
                pos_offset: 3,
                kv_format: fmt,
            };
            cache.eval_state().unwrap();

            cache.commit_tree(&[0, 3, 5], false).unwrap();
            assert_eq!(cache.pos_offset, 6);
            if crate::kernels::inkling2::available() {
                // Batched visits 0 & 1 produce lazy aliased outputs picked up by eval_pending_state!
                let mut pending = Vec::new();
                cache.collect_pending_state_arrays(&mut pending);
                let expected_pending = if fmt.is_int8() { 8 } else { 4 };
                assert!(
                    pending.len() >= expected_pending,
                    "expected >= {expected_pending} lazy batched KV outputs, got {}",
                    pending.len()
                );
            }
            cache.eval_pending_state().unwrap();

            for (v_idx, write_off, ring_cap) in [(0usize, 3usize, 0usize), (1, 5, 16), (2, 14, 16)]
            {
                let vc = &cache.visits[v_idx];
                let got_k = vc.k.astype(DType::Float32).unwrap().to_float32().unwrap();
                let got_v = vc.v.astype(DType::Float32).unwrap().to_float32().unwrap();
                for (r, &node) in [0usize, 3, 5].iter().enumerate() {
                    let dst_slot = if ring_cap > 0 {
                        (write_off + r) % ring_cap
                    } else {
                        write_off + r
                    };
                    let src_slot = if ring_cap > 0 {
                        (write_off + node) % ring_cap
                    } else {
                        write_off + node
                    };
                    for h in 0..2 {
                        for d in 0..4 {
                            assert_eq!(
                                got_k[(h * 16 + dst_slot) * 4 + d],
                                f32::from(k_vals[(h * 16 + src_slot) * 4 + d]),
                                "visit={v_idx} fmt={fmt:?} K mismatch"
                            );
                            assert_eq!(
                                got_v[(h * 16 + dst_slot) * 4 + d],
                                f32::from(v_vals[(h * 16 + src_slot) * 4 + d]),
                                "visit={v_idx} fmt={fmt:?} V mismatch"
                            );
                        }
                    }
                }
                if let (Some(ks), Some(vs)) = (&vc.k_scales, &vc.v_scales) {
                    let gks = ks.to_float32().unwrap();
                    let gvs = vs.to_float32().unwrap();
                    for (r, &node) in [0usize, 3, 5].iter().enumerate() {
                        let dst_slot = if ring_cap > 0 {
                            (write_off + r) % ring_cap
                        } else {
                            write_off + r
                        };
                        let src_slot = if ring_cap > 0 {
                            (write_off + node) % ring_cap
                        } else {
                            write_off + node
                        };
                        for h in 0..2 {
                            assert_eq!(gks[h * 16 + dst_slot], ks_vals[h * 16 + src_slot]);
                            assert_eq!(gvs[h * 16 + dst_slot], vs_vals[h * 16 + src_slot]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn alloc_kv_buffer_with_fails_closed_on_cuda_and_evaluates_on_cpu() {
        let err = alloc_kv_buffer_with(&[1, 1, 2, 4], DType::BFloat16, true, |_, _| Ok(None))
            .err()
            .expect("CUDA allocation returning None must return Err instead of falling back");
        assert!(
            err.to_string()
                .contains("Inkling2 CUDA KV buffer allocation failed"),
            "unexpected error: {err}"
        );

        let gpu_buf = alloc_kv_buffer_with(&[1, 1, 2, 4], DType::BFloat16, true, |shape, dtype| {
            let arr = MxArray::zeros(shape, Some(dtype))?;
            MxArray::eval_arrays(&[&arr])?;
            Ok(Some(arr))
        })
        .expect("CUDA allocation returning Some(buf) should succeed");
        assert!(gpu_buf.is_available());
        assert_eq!(gpu_buf.shape().unwrap().as_ref(), [1, 1, 2, 4]);
        assert_eq!(gpu_buf.dtype().unwrap(), DType::BFloat16);

        let cpu_buf = alloc_kv_buffer_with(&[1, 1, 2, 4], DType::BFloat16, false, |_, _| {
            panic!("gpu_alloc must not run when cuda_available is false")
        })
        .expect("non-CUDA fallback should return a settled zero-initialized buffer");
        assert!(cpu_buf.is_available());
        assert_eq!(cpu_buf.shape().unwrap().as_ref(), [1, 1, 2, 4]);
        assert_eq!(cpu_buf.dtype().unwrap(), DType::BFloat16);
    }
}
