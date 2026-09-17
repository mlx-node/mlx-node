//! QSA pages store only full-attention K/V. GDN, PLE and the compressed
//! indexer remain request-owned; a KV-only prefix is never admitted.
use super::{Inner, Step};
use crate::array::MxArray;
use crate::engine::backend::{PagedBackend, PagedPrefix};
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, HybridStepExecutor, NoRestoreTicket,
};
use crate::models::qwen4_exp::runtime_flags;
use crate::stream::Stream;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};
use napi::{Error, Result};
use std::sync::{Arc, Mutex};

pub fn create(config: &super::config::Config) -> Result<PagedKVCacheAdapter> {
    let cfg = mlx_paged_attn::PagedAttentionConfig {
        block_size: 32,
        gpu_memory_mb: 1024,
        head_size: config.head_dim as u32,
        num_kv_heads: config.num_key_value_heads as u32,
        num_layers: (0..config.num_hidden_layers)
            .filter(|&i| !config.linear(i))
            .count() as u32,
        use_fp8_cache: Some(false),
        max_seq_len: Some(config.effective_context_limit() as u32),
        max_batch_size: Some(4),
    };
    cfg.validate().map_err(Error::from_reason)?;
    let count = cfg
        .calculate_num_blocks()
        .min((config.effective_context_limit() as u32).div_ceil(32) * 4);
    let allocator = Arc::new(Mutex::new(mlx_paged_attn::BlockAllocator::new(
        count, count, 32,
    )));
    let pool = mlx_paged_attn::LayerKVPool::new(
        cfg,
        count,
        count,
        mlx_paged_attn::metal::MetalDtype::BFloat16,
    )
    .map_err(|e| Error::from_reason(e.to_string()))?;
    PagedKVCacheAdapter::new(allocator, Arc::new(pool), 32).map_err(Error::from_reason)
}

/// The native bridge compacts strided inputs and preserves offsets for
/// contiguous read-only K/V views. An explicit zero selects independent
/// copies for comparison; mutable pool and metadata constraints stay strict.
pub(super) fn write_rows(
    adapter: &mut PagedKVCacheAdapter,
    layer: u32,
    keys: &MxArray,
    values: &MxArray,
    base: u32,
) -> Result<()> {
    let keys = keys.astype(crate::array::DType::BFloat16)?;
    let values = values.astype(crate::array::DType::BFloat16)?;
    let views = !runtime_flags::is_zero(c"MLX_QWEN4_PAGED_VIEWS");
    let keys = if views { keys } else { keys.deep_copy()? };
    let values = if views { values } else { values.deep_copy()? };
    adapter
        .update_keys_values_native(layer, &keys, &values, base)
        .map_err(Error::from_reason)
}

/// GPU gather from the native paged layout, without materializing the full
/// sequence or reading its K/V through CPU memory. Only integer addresses
/// cross the host boundary. Selected rows are ordered by logical position.
pub fn gather_selected(
    adapter: &PagedKVCacheAdapter,
    layer: u32,
    tokens: &[i32],
    heads: usize,
    dim: usize,
) -> Result<(MxArray, MxArray)> {
    let table = adapter
        .block_table()
        .ok_or_else(|| Error::from_reason("Qwen4 missing page table"))?;
    let bs = adapter.block_size() as usize;
    let slots = tokens
        .iter()
        .map(|&t| {
            let slot = u32::try_from(t)
                .ok()
                .and_then(|t| table.absolute_slot_index(t))
                .ok_or_else(|| Error::from_reason("Qwen4 selected an unallocated cache row"))?;
            i32::try_from(slot).map_err(|_| Error::from_reason("Qwen4 page offset overflow"))
        })
        .collect::<Result<Vec<_>>>()?;
    let slots = MxArray::from_int32(&slots, &[slots.len() as i64])?;
    let k = adapter.key_pool_array(layer).map_err(Error::from_reason)?;
    let v = adapter
        .value_pool_array(layer)
        .map_err(Error::from_reason)?;
    let mut ko = std::ptr::null_mut();
    let mut vo = std::ptr::null_mut();
    let ok = unsafe {
        mlx_sys::mlx_qwen4_gather_pages(
            k.as_raw_ptr(),
            v.as_raw_ptr(),
            slots.as_raw_ptr(),
            heads as i32,
            dim as i32,
            bs as i32,
            &mut ko,
            &mut vo,
        )
    };
    if !ok {
        return Err(Error::from_reason("Qwen4 native page gather failed"));
    }
    Ok((
        MxArray::from_handle(ko, "Qwen4 gathered keys")?,
        MxArray::from_handle(vo, "Qwen4 gathered values")?,
    ))
}

pub struct Prefix {
    pub cached: usize,
    pub suffix: usize,
}
impl PagedPrefix for Prefix {
    fn effective_cached_prefix_len(&self) -> usize {
        self.cached
    }
    fn suffix_len(&self) -> usize {
        self.suffix
    }
}
impl PagedBackend for Inner {
    type PagedDecode<'a> = Step<'a>;
    type PrefixState = Prefix;
    fn prime_prefix_state(
        &mut self,
        tokens: &[u32],
        reuse: bool,
        _: usize,
        _: &[u64],
        salt: u64,
    ) -> Result<Prefix> {
        let seq = self.active_seq.unwrap_or(0);
        self.activate_paged_seq(seq)?;
        let warm = reuse
            && !self.decoder.history.is_empty()
            && tokens.len() > self.decoder.history.len()
            && tokens.starts_with(&self.decoder.history);
        if !warm {
            self.decoder.reset();
            self.saved_history.clear();
        }
        let adapter = self
            .decoder
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen4 prefix preparation has no paged cache"))?;
        // skip_lookup disallows KV-only prefix restores: recurrent/indexer/PLE
        // state cannot be reconstructed from attention pages.
        let plan = adapter
            .prepare_turn_with_max_cache_hit_tokens(
                seq,
                tokens,
                tokens.len() as u32,
                warm,
                &[],
                salt,
                true,
                tokens.len().saturating_sub(1) as u32,
            )
            .map_err(Error::from_reason)?;
        if plan.cached_prefix_len as usize != self.decoder.history.len() {
            adapter
                .reset_for_new_request(seq)
                .map_err(Error::from_reason)?;
            adapter
                .allocate_suffix_blocks(tokens.len() as u32)
                .map_err(Error::from_reason)?;
            self.decoder.reset();
            return Ok(Prefix {
                cached: 0,
                suffix: tokens.len(),
            });
        }
        Ok(Prefix {
            cached: plan.cached_prefix_len as usize,
            suffix: plan.suffix_len as usize,
        })
    }
    fn paged_prefill(&mut self, tokens: &[u32], _: &Prefix, stream: Stream) -> Result<MxArray> {
        let _context = crate::stream::StreamContext::new(stream);
        let mut logits = None;
        let chunk_size = self.decoder.prefill_slice_size();
        for (i, chunk) in tokens.chunks(chunk_size).enumerate() {
            let base = self.decoder.history.len();
            let embeddings = if let Some(media) = &self.media_prefill {
                self.decoder.positions = media.positions.clone();
                self.decoder.rope_delta = media.delta;
                self.decoder.media_digests = media.digests.clone();
                Some(
                    media
                        .embeddings
                        .slice_axis(1, base as i64, (base + chunk.len()) as i64)?,
                )
            } else {
                None
            };
            logits = Some(self.decoder.prefill_chunk(
                chunk,
                embeddings.as_ref(),
                (i + 1) * chunk_size >= tokens.len(),
            )?);
            let hidden = self.decoder.last_chunk_hidden.clone();
            self.observe_mtp(self.active_seq.unwrap_or(0), chunk, &hidden)?;
        }
        logits
            .ok_or_else(|| Error::from_reason("Qwen4 empty paged prefill"))?
            .squeeze(Some(&[1]))
    }
    fn begin_paged_decode(&mut self) -> Result<Step<'_>> {
        Ok(Step(self))
    }
    fn admit_paged_speculative_decode(
        &mut self,
        _: &crate::engine::params::ChatParams,
    ) -> Result<bool> {
        // The scheduled lane owns proposal/verification transactions. An
        // exclusive legacy/media barrier serves target AR without opening a
        // second, competing speculative cache owner.
        Ok(false)
    }
    fn finalize_paged_turn(&mut self, reuse: bool, _: u64) {
        if let Some(adapter) = &mut self.decoder.paged {
            let result = if reuse {
                adapter.finalize_turn_keep_live_no_prefix()
            } else {
                adapter.release_request().map(|_| ())
            };
            if result.is_err() {
                self.decoder.reset();
                self.saved_history.clear();
            }
        }
    }
    fn save_paged_history(&mut self, _: &[u32], _: &[u32], _: bool, reuse: bool) -> Result<()> {
        // The scalar decoder consumes only forwarded tokens, including on a
        // length exit. Save that exact frontier, never an unconsumed sample.
        self.saved_history = if reuse {
            self.decoder.history.clone()
        } else {
            self.decoder.reset();
            Vec::new()
        };
        Ok(())
    }
    fn abort_paged_turn(&mut self) {
        if let Some(adapter) = &mut self.decoder.paged {
            let _ = adapter.release_request();
        }
        self.decoder.reset();
        self.saved_history.clear();
    }
}

impl Inner {
    pub(super) fn activate_exclusive_seq(&mut self, seq: SeqId) -> Result<()> {
        // Media runs behind a scheduler barrier, outside ordinary admission.
        // Bound retained auxiliary rows before creating another request there.
        if !self.has_scheduled_recurrent(seq)
            && self.rows.len() + usize::from(self.active_seq.is_some()) >= self.scheduler_capacity()
        {
            return Err(Error::from_reason(
                "Qwen4 exclusive owner capacity reached; releaseCacheOwner or resetCaches before adding another image session",
            ));
        }
        self.activate_paged_seq(seq)
    }

    fn park_row(&mut self) {
        if let Some(seq) = self.active_seq.take() {
            self.rows.insert(seq, self.decoder.take_state());
        }
    }
    fn activate_row(&mut self, seq: SeqId) -> Result<()> {
        if self.active_seq != Some(seq) {
            self.park_row();
            if let Some(state) = self.rows.remove(&seq) {
                self.decoder.restore_state(state);
            }
            self.active_seq = Some(seq);
        }
        self.saved_history = self.decoder.history.clone();
        Ok(())
    }
}
impl HybridSchedulerBackend for Inner {
    type FamilyCommand = std::convert::Infallible;
    type RestoreTicket = NoRestoreTicket;
    type OwnerState = Vec<u32>;
    type StepExecutor<'a> = HybridStepExecutor<'a, Self>;
    const SCHEDULER_NAME: &'static str = "qwen4_exp";
    fn scheduler_wired_bytes(&self) -> Option<usize> {
        if runtime_flags::is_zero(c"MLX_QWEN4_PREFILL_WIRED") {
            return None;
        }
        let pages = self.decoder.paged.as_ref().map_or(Some(0), |p| {
            p.bytes_per_block()
                .ok()?
                .checked_mul(u64::from(p.block_capacity()))
        })?;
        let bytes = self
            .decoder
            .weights
            .cache_budget()
            .checked_add(pages)?
            .checked_add(super::memory::WORKING_BYTES)?;
        usize::try_from(bytes).ok()
    }
    fn paged_adapter(&self) -> Option<&PagedKVCacheAdapter> {
        self.decoder.paged.as_ref()
    }
    fn paged_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter> {
        self.decoder.paged.as_mut()
    }
    fn supports_scheduled_speculation(&self) -> bool {
        self.has_mtp
    }
    fn supports_adaptive_scheduled_speculation(&self) -> bool {
        self.has_mtp
    }
    fn scheduled_verification_budget(
        &mut self,
    ) -> Option<&mut crate::engine::verification_budget::ScheduledVerificationBudget> {
        Some(&mut self.mtp.budget)
    }
    fn begin_scheduled_speculation(&mut self, seq: SeqId, position: u32) -> Result<bool> {
        self.begin_mtp(seq, position)
    }
    fn reserve_scheduled_speculation(&mut self, seq: SeqId, queries: usize) -> Result<bool> {
        self.activate_paged_seq(seq)?;
        if self.decoder.history.len() + queries > self.decoder.config.effective_context_limit() {
            return Ok(false);
        }
        Ok(self
            .decoder
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen4 speculative reservation has no paged cache"))?
            .reserve_rows(queries as u32)
            .is_ok())
    }
    fn scheduled_draft_state_bytes(&self, tokens: u32) -> u64 {
        4 * self.recurrent_state_bytes()
            + tokens as u64
                * self.decoder.config.num_key_value_heads as u64
                * self.decoder.config.head_dim as u64
                * 8
    }
    fn propose_scheduled(
        &mut self,
        seq: SeqId,
        anchor: u32,
        cap: usize,
        params: &crate::engine::params::ChatParams,
        rng: &mut dyn rand::Rng,
        confidence: bool,
    ) -> Result<crate::engine::backend::DsparkProposal> {
        self.activate_paged_seq(seq)?;
        self.propose_mtp(seq, anchor, cap, params, rng, confidence)
    }
    fn run_scheduled_verify(
        &mut self,
        rows: &[crate::engine::hybrid_scheduler::ScheduledVerifyRow],
    ) -> Result<MxArray> {
        self.verify_mtp(rows)
    }
    fn commit_scheduled_verify(
        &mut self,
        rows: &[crate::engine::hybrid_scheduler::ScheduledVerifyCommit],
    ) -> Result<Vec<Result<()>>> {
        self.commit_mtp(rows)
    }
    fn release_scheduled_speculation(&mut self, seq: SeqId) {
        self.mtp.finish(
            seq,
            (self.active_seq == Some(seq)).then_some(self.decoder.history.len()),
        );
    }
    fn scheduler_capacity(&self) -> usize {
        4
    }
    fn scheduler_prefill_slice_tokens(&self) -> u32 {
        self.decoder.prefill_slice_size() as u32
    }
    fn max_position_embeddings(&self) -> i32 {
        self.decoder.config.effective_context_limit() as i32
    }
    fn recurrent_state_bytes(&self) -> u64 {
        let c = &self.decoder.config;
        // Include the maximum indexer history and convolution windows as well
        // as F32 recurrent matrices in scheduler admission accounting.
        let linear = (0..c.num_hidden_layers).filter(|&i| c.linear(i)).count();
        let full = c.num_hidden_layers - linear;
        (linear
            * (c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim * 4
                + (2 * c.linear_num_key_heads * c.linear_key_head_dim
                    + c.linear_num_value_heads * c.linear_value_head_dim)
                    * c.linear_conv_kernel_dim
                    * 4)
            + full * c.indexer_head_dim * c.effective_context_limit() / c.indexer_compress_ratio
                * 4
            + c.ple_layer_ids.len()
                * c.hc_count
                * c.hidden_size
                * c.ple_conv_kernel_size
                * c.ngram_size
                * 4) as u64
    }
    fn scheduled_recurrent_bytes(&self) -> u64 {
        (self.rows.len() + usize::from(self.active_seq.is_some())) as u64
            * self.recurrent_state_bytes()
    }
    fn has_scheduled_recurrent(&self, seq: SeqId) -> bool {
        self.active_seq == Some(seq) || self.rows.contains_key(&seq)
    }
    fn activate_scheduled_recurrent(&mut self, seq: SeqId) -> Result<()> {
        self.activate_row(seq)
    }
    fn activate_paged_seq(&mut self, seq: SeqId) -> Result<()> {
        self.activate_row(seq)?;
        let adapter = self
            .decoder
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen4 paged cache unavailable"))?;
        if adapter.block_table_for(seq).is_some() {
            adapter.activate_request(seq)
        } else {
            adapter.begin_request(seq)
        }
        .map_err(Error::from_reason)
    }
    fn park_active_scheduled_recurrent(&mut self) -> Result<()> {
        self.park_row();
        Ok(())
    }
    fn release_scheduled_recurrent_for(&mut self, seq: SeqId) {
        self.rows.remove(&seq);
        self.mtp.release(seq);
        if self.active_seq == Some(seq) {
            self.decoder.reset();
            self.active_seq = None;
            self.saved_history.clear();
        }
    }
    fn run_paged_decode_step_batched(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        let mut seen = std::collections::HashSet::new();
        if rows.is_empty() || rows.iter().any(|(seq, _)| !seen.insert(*seq)) {
            return Err(Error::from_reason(
                "Qwen4 decode requires a nonempty batch of distinct sequences",
            ));
        }
        if rows.len() > self.scheduler_capacity() {
            return Err(Error::from_reason(
                "Qwen4 scheduled batch exceeds owner capacity",
            ));
        }
        if rows.len() > 1 && !runtime_flags::is_zero(c"MLX_QWEN4_BATCH_DECODE") {
            return self.decode_shared_layers(rows);
        }
        let mut logits = Vec::with_capacity(rows.len());
        for &(seq, token) in rows {
            self.activate_paged_seq(seq)?;
            // The shared scheduler slices [batch, 1, vocabulary] before
            // sampling each row; retain the singleton token dimension.
            logits.push(self.decoder.step(token)?);
            let hidden = self.decoder.last_chunk_hidden.clone();
            self.observe_mtp(seq, &[token], &hidden)?;
        }
        MxArray::concatenate_many(logits.iter().collect(), Some(0))
    }
    fn replace_cached_token_history(&mut self, history: Vec<u32>) {
        self.saved_history = history;
    }
    fn install_owner_state(&mut self, seq: SeqId, state: &Vec<u32>) {
        // Admission reads media identity and live history before it allocates
        // pages. Select the matching auxiliary row at that same boundary.
        let _ = self.activate_row(seq);
        self.saved_history = state.clone();
    }
    fn owner_tokens(state: &Vec<u32>) -> &[u32] {
        state
    }
    fn capture_owner_state(&mut self, _: SeqId) -> Vec<u32> {
        self.saved_history.clone()
    }
    fn build_scheduled_prefix(
        &self,
        _: &Prefix,
        cached: usize,
        suffix: usize,
        _: Vec<u32>,
        _: bool,
    ) -> Prefix {
        Prefix { cached, suffix }
    }
    fn execute_chat_barrier(
        &mut self,
        command: crate::engine::cmd::ChatCmd,
        owners: crate::engine::hybrid_scheduler::SchedulerOwnerContext<'_, Vec<u32>>,
    ) {
        use crate::engine::cmd::{ChatCmd, handle_chat_cmd};
        let owner = match &command {
            ChatCmd::SessionStart { config, .. }
            | ChatCmd::SessionContinue { config, .. }
            | ChatCmd::SessionContinueTool { config, .. }
            | ChatCmd::StreamSessionStart { config, .. }
            | ChatCmd::StreamSessionContinue { config, .. }
            | ChatCmd::StreamSessionContinueTool { config, .. } => {
                Some(config.cache_owner_id.clone().unwrap_or_default())
            }
            _ => None,
        };
        let Some(owner) = owner else {
            handle_chat_cmd(self, command);
            return;
        };
        let seq = *owners
            .owner_sequences
            .entry(owner.clone())
            .or_insert_with(|| {
                let seq = *owners.next_seq_id;
                *owners.next_seq_id += 1;
                seq
            });
        if let Err(error) = self.activate_exclusive_seq(seq) {
            match command {
                ChatCmd::SessionStart { reply, .. }
                | ChatCmd::SessionContinue { reply, .. }
                | ChatCmd::SessionContinueTool { reply, .. } => {
                    let _ = reply.send(Err(error));
                }
                ChatCmd::StreamSessionStart { stream_tx, .. }
                | ChatCmd::StreamSessionContinue { stream_tx, .. }
                | ChatCmd::StreamSessionContinueTool { stream_tx, .. } => {
                    let _ = stream_tx.send(Err(error));
                }
                _ => unreachable!(),
            }
            owners.owner_sequences.remove(&owner);
            owners.owner_states.remove(&owner);
            return;
        }
        handle_chat_cmd(self, command);
        if self.saved_history.is_empty() {
            // Admission/media failures can return before run_paged_turn's abort
            // hook. Empty rows have no reusable frontier and must not occupy an
            // owner slot (including successful requests with reuse disabled).
            if let Some(adapter) = &mut self.decoder.paged {
                let _ = adapter.release_request_for(seq);
            }
            self.release_scheduled_recurrent_for(seq);
            owners.owner_sequences.remove(&owner);
            owners.owner_states.remove(&owner);
        } else {
            owners
                .owner_states
                .insert(owner, self.saved_history.clone());
            self.park_row();
        }
    }
    fn step_executor(&mut self) -> Self::StepExecutor<'_> {
        HybridStepExecutor::new(self)
    }
}

/// Sparse logical indices stay on GPU. One gather combines prior pages and
/// the uncommitted current chunk without a CPU readback or split/concatenation.
pub(super) fn gather_window(
    adapter: &PagedKVCacheAdapter,
    layer: u32,
    tokens: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    base: usize,
) -> Result<(MxArray, MxArray)> {
    let table = adapter
        .block_table()
        .ok_or_else(|| Error::from_reason("Qwen4 missing page table"))?;
    let blocks = crate::transformer::paged_kv_cache_adapter::build_decode_block_ids(table);
    if blocks.len() * (adapter.block_size() as usize) < base {
        return Err(Error::from_reason(
            "Qwen4 page table does not cover its retained prefix",
        ));
    }
    let blocks = MxArray::from_int32(&blocks, &[blocks.len() as i64])?;
    let kp = adapter.key_pool_array(layer).map_err(Error::from_reason)?;
    let vp = adapter
        .value_pool_array(layer)
        .map_err(Error::from_reason)?;
    let (mut k, mut v) = (std::ptr::null_mut(), std::ptr::null_mut());
    let ok = unsafe {
        mlx_sys::mlx_qwen4_gather_window(
            kp.as_raw_ptr(),
            vp.as_raw_ptr(),
            blocks.as_raw_ptr(),
            tokens.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            base as i32,
            adapter.block_size() as i32,
            &mut k,
            &mut v,
        )
    };
    if !ok {
        return Err(Error::from_reason("Qwen4 GPU sparse page gather failed"));
    }
    Ok((
        MxArray::from_handle(k, "sparse page keys")?,
        MxArray::from_handle(v, "sparse page values")?,
    ))
}

impl Inner {
    fn decode_shared_layers(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        let result = (|| {
            let mut hidden = Vec::with_capacity(rows.len());
            for &(seq, token) in rows {
                self.activate_paged_seq(seq)?;
                hidden.push(self.decoder.scheduled_start(token)?);
            }
            for layer in 0..self.decoder.config.num_hidden_layers {
                for (j, &(seq, token)) in rows.iter().enumerate() {
                    self.activate_paged_seq(seq)?;
                    hidden[j] =
                        self.decoder
                            .scheduled_attention(hidden[j].clone(), token, layer)?;
                }
                hidden = self.decoder.scheduled_mlp(&hidden, layer)?;
                super::memory::maintain_freelist(self.decoder.weights.plan.physical_bytes);
            }
            let mut logits = Vec::with_capacity(rows.len());
            for (h, &(seq, token)) in hidden.iter().zip(rows) {
                self.activate_paged_seq(seq)?;
                logits.push(self.decoder.scheduled_finish(h, token)?);
                self.observe_mtp(seq, &[token], std::slice::from_ref(h))?;
            }
            MxArray::concatenate_many(logits.iter().collect(), Some(0))
        })();
        if result.is_err() {
            // A failed layer has no trusted frontier in any participating row.
            for &(seq, _) in rows {
                if let Some(adapter) = &mut self.decoder.paged
                    && adapter.activate_request(seq).is_ok()
                {
                    let _ = adapter.release_request();
                }
                self.release_scheduled_recurrent_for(seq);
            }
        }
        result
    }
}
