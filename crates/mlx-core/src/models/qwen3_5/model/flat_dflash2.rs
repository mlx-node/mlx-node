//! Off-Metal DFlash2 turns: cold tapped prefill plus eager flat verify.

use super::*;
use crate::engine::backend::{TurnOutput, WholeTurnArgs};
use crate::engine::dspark_turn::{FlatDsparkBackend, FlatDsparkPrefix, run_flat_dspark_turn};
use crate::engine::params::ChatParams;
use crate::models::forward::PREFILL_CANCELLED;
use crate::models::qwen3_5::dflash2::DFlash2ContextCache;
use crate::models::qwen3_5::dflash2_turn::DFlash2TurnState;

impl Qwen35Inner {
    fn flat_dflash2_prefill(
        &mut self,
        tokens: &[u32],
        _generation_stream: Stream,
    ) -> Result<(MxArray, DFlash2TurnState)> {
        if tokens.is_empty() {
            return Err(Error::from_reason(
                "Qwen3.8 DFlash2 prefill needs at least one token",
            ));
        }
        let tap_layers = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 has no loaded DFlash2 companion"))?
            .config
            .target_layers
            .clone();
        let config = &self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 has no loaded DFlash2 companion"))?
            .config;
        let retained_start = tokens.len().saturating_sub(config.sliding_window - 1);
        let mut context = DFlash2ContextCache::new(config);
        // Scan all prompt tokens up front so the turn's narrow/wide draft head
        // mode is established before the first decode proposal.
        context.observe_tokens(tokens);
        let mut next_position = 0i32;
        let mut last_logits = None;
        // On CUDA with in-register KQuantPrefill and fused FlashAttention-2 D=256,
        // peak chunk scratch is small (~400 MiB at 4096 rows), so 4096-row balanced
        // chunks stay within the 4096-row GDN/norm kernel caps while cutting full-model
        // weight reads from 3 passes to 2 passes on 6k prompts (3110 + 3109).
        let base_chunk = if crate::accelerator::cuda_ops_available() {
            4096usize
        } else {
            usize::try_from(PREFILL_STEP_SIZE).unwrap_or(2048).max(1)
        };
        let chunk_size = if crate::accelerator::cuda_ops_available() {
            let chunks = tokens.len().div_ceil(base_chunk).max(1);
            tokens.len().div_ceil(chunks)
        } else if tokens.len() > base_chunk {
            let rem = tokens.len() % base_chunk;
            if rem > 0 && rem <= base_chunk / 4 {
                let balanced_chunks = tokens.len() / base_chunk;
                tokens.len().div_ceil(balanced_chunks)
            } else {
                base_chunk
            }
        } else {
            base_chunk
        };
        let chunk_count = tokens.len().div_ceil(chunk_size);
        for (index, chunk) in tokens.chunks(chunk_size).enumerate() {
            if self
                .turn_cancel
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                return Err(Error::from_reason(PREFILL_CANCELLED));
            }
            let is_final = index + 1 == chunk_count;
            let input = MxArray::from_uint32(chunk, &[1, chunk.len() as i64])?;
            let (logits, taps, _) = forward_dflash2_eager_with_taps(
                self,
                &input,
                &tap_layers,
                false,
                if is_final {
                    DFlash2LogitsSpan::LastRow
                } else {
                    DFlash2LogitsSpan::None
                },
            )?;
            // Proposals can only read the retained sliding-window tail. Older
            // rows still advance provenance, but their draft FC and K/V work is dead.
            let chunk_start = next_position;
            let skip = retained_start
                .saturating_sub(chunk_start as usize)
                .min(chunk.len());
            if skip > 0 {
                context.record_only(&chunk[..skip])?;
            }
            if skip < chunk.len() {
                let taps = if skip == 0 {
                    taps
                } else {
                    taps.iter()
                        .map(|tap| tap.slice_axis(1, skip as i64, chunk.len() as i64))
                        .collect::<Result<Vec<_>>>()?
                };
                let draft = self
                    .dflash2
                    .as_ref()
                    .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?;
                let fused = draft.fuse_context(&taps)?;
                let append_base = chunk_start
                    .checked_add(i32::try_from(skip).map_err(|_| {
                        Error::from_reason("Qwen3.8 DFlash2 prefill position exceeds i32")
                    })?)
                    .ok_or_else(|| {
                        Error::from_reason("Qwen3.8 DFlash2 prefill position overflow")
                    })?;
                context.append(draft, &fused, append_base, &chunk[skip..])?;
            }
            next_position = chunk_start
                .checked_add(i32::try_from(chunk.len()).map_err(|_| {
                    Error::from_reason("Qwen3.8 DFlash2 prefill position exceeds i32")
                })?)
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 prefill position overflow"))?;
            last_logits = Some(logits);
            if index + 1 < chunk_count {
                async_eval_layer_caches(&self.caches);
            }
        }
        Ok((
            last_logits
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 produced no prefill logits"))?,
            DFlash2TurnState {
                context,
                next_position,
            },
        ))
    }

    fn flat_dflash2_fail_closed(&mut self, error: Error) -> Error {
        self.clear_cuda_dense_prefill();
        let _ = self.reset_caches_sync();
        error
    }

    /// Off-Metal DFlash2 starts cold on every turn. The full prompt rebuilds
    /// both the flat target state and the draft window, so no parked paged
    /// state or prior draft lineage can be reused accidentally.
    pub(crate) fn flat_dflash2_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        run_flat_dspark_turn(self, args)
    }
}

impl FlatDsparkBackend for Qwen35Inner {
    fn prepare_flat_dspark_turn(
        &mut self,
        tokens: &[u32],
        _params: &ChatParams,
        _is_delta: bool,
    ) -> Result<FlatDsparkPrefix> {
        let block_size = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 has no loaded DFlash2 companion"))?
            .config
            .block_size;
        self.reset_caches_sync()?;
        self.init_caches_sync()?;
        Ok(FlatDsparkPrefix {
            prefill_tokens: tokens.to_vec(),
            cached_prefix_len: 0,
            prior_cached_len: 0,
            block_size,
        })
    }

    fn flat_prefill_with_draft_state(
        &mut self,
        prefix: &FlatDsparkPrefix,
        stream: Stream,
    ) -> Result<MxArray> {
        let (last_logits, state) = self.flat_dflash2_prefill(&prefix.prefill_tokens, stream)?;
        self.dflash2_turn_state = Some(state);
        Ok(last_logits)
    }

    fn eval_after_flat_prefill(&self) -> Result<()> {
        ChatBackend::eval_caches(self)?;
        if let Some(state) = self.dflash2_turn_state.as_ref() {
            let mut draft_arrays: Vec<&MxArray> = Vec::new();
            state.context.collect_arrays(&mut draft_arrays);
            if !draft_arrays.is_empty() {
                MxArray::eval_arrays_with_context(&draft_arrays, "flat_dflash2_prefill_draft")?;
            }
        }
        Ok(())
    }

    fn fail_closed_flat_dspark(&mut self, error: Error) -> Error {
        self.flat_dflash2_fail_closed(error)
    }

    fn finish_flat_dspark_turn(
        &mut self,
        tokens: &[u32],
        generated_tokens: &[u32],
        _finish_reason: &str,
        last_in_cache: bool,
        _is_delta: bool,
        reuse_cache: bool,
        _stream: Stream,
    ) -> Result<()> {
        // Retain dense prefill weights across successful turns. Packed verify
        // uses separate operands, and failure cleanup releases the dense cache.

        // The GDN state cannot materialize an unverified boundary token after
        // the fact. Keep history only through the token present in the caches;
        // a later continuation re-renders the boundary token from the template.
        let saved_generated = if !last_in_cache && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            generated_tokens
        };
        if reuse_cache {
            let mut history = tokens.to_vec();
            history.extend_from_slice(saved_generated);
            self.cached_token_history = history;
            self.cached_image_key = None;
            self.cached_rope_deltas = None;
            self.cached_paged_image_token_positions.clear();
        } else {
            self.reset_caches_sync()?;
        }
        Ok(())
    }
}
