//! QSA pages store only full-attention K/V. GDN, PLE and the compressed
//! indexer remain request-owned; a KV-only prefix is never admitted. A
//! prefix resumes only where a stored state holds that request-owned cell
//! (`prefix_state`) and the prefix cache still holds every block below it.
use super::{Inner, Step, prefix_state};
use crate::array::MxArray;
use crate::engine::backend::{PagedBackend, PagedPrefix};
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, HybridStepExecutor, NoRestoreTicket, ScheduledStateCapacity,
};
use crate::engine::paged_stepper::PagedStepper;
use crate::models::qwen3_5::dflash2_state::store::{EntryId, StoreHit};
use crate::stream::Stream;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};
use crate::vision::qwen::prompt::IMAGE_TOKEN_ID;
use napi::{Error, Result};
use std::sync::{Arc, Mutex};

/// The pool of the memory plan: it starts at the runway and grows by
/// extents to the plan's pages, each growth probed against `ceiling`.
pub fn create(
    config: &super::config::Config,
    plan: &crate::engine::memory_plan::PoolPlan,
    ceiling: u64,
) -> Result<PagedKVCacheAdapter> {
    let cfg = pool_config(config, plan.max_blocks)?;
    let mut adapter = build(config, cfg, plan.runway_blocks, plan.max_blocks)?;
    if plan.runway_blocks < plan.max_blocks {
        adapter.set_extent_growth(
            crate::models::qwen3_5::dflash2_decode::runway_pool_extent_blocks(
                adapter.bytes_per_block().map_err(Error::from_reason)?,
            ),
        );
    }
    adapter.set_memory_ceiling(ceiling);
    Ok(adapter)
}

/// A pool of `initial` blocks that grows to `max`, for fixtures.
#[cfg(test)]
pub fn create_with_blocks(
    config: &super::config::Config,
    initial: u32,
    max: u32,
) -> Result<PagedKVCacheAdapter> {
    let cfg = pool_config(config, max)?;
    build(config, cfg, initial, max)
}

/// A fixture's pool of four sequences of its trained window (at most
/// 32,768 tokens) within 1 GiB, at that size from the start.
#[cfg(test)]
pub fn create_fixture(config: &super::config::Config) -> Result<PagedKVCacheAdapter> {
    let (_, bytes_per_block) = pool_block(config)?;
    let count = ((1u64 << 30) / bytes_per_block.max(1))
        .min(u64::from(config.max_position_embeddings.min(32_768) as u32).div_ceil(32) * 4)
        as u32;
    create_with_blocks(config, count, count)
}

/// The Flash Next page (Splash `QwenHybridLayout::kvLayout`, `SelectedBlocks`):
/// token-major K and V, and the indexer's pooled block keys in the page.
fn page_layout(config: &super::config::Config) -> mlx_paged_attn::PageLayout {
    mlx_paged_attn::PageLayout {
        values_token_major: true,
        index_rows: (32 / config.indexer_compress_ratio.max(1)) as u32,
        index_dimension: config.indexer_head_dim as u32,
    }
}

/// The pool's block dtype and physical bytes per block in the load's
/// format: the priced page (Splash `bytesPerModelPage`).
fn pool_block(config: &super::config::Config) -> Result<(mlx_paged_attn::metal::MetalDtype, u64)> {
    let dtype = match config.kv_format {
        crate::transformer::KvFormat::Int8 => mlx_paged_attn::metal::MetalDtype::Int8,
        crate::transformer::KvFormat::Bf16 => mlx_paged_attn::metal::MetalDtype::BFloat16,
    };
    let layers = super::load_plan::full_attention_layers(config) as u32;
    let bytes = mlx_paged_attn::profile::bytes_per_block(
        layers,
        config.num_key_value_heads as u32,
        config.head_dim as u32,
        32,
        dtype,
    )
    .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok((
        dtype,
        bytes + u64::from(layers) * page_layout(config).index_bytes(),
    ))
}

fn pool_config(
    config: &super::config::Config,
    max_blocks: u32,
) -> Result<mlx_paged_attn::PagedAttentionConfig> {
    let (_, bytes_per_block) = pool_block(config)?;
    let cfg = mlx_paged_attn::PagedAttentionConfig {
        block_size: 32,
        // `validate` refuses less than `MIN_GPU_MEMORY_MB`; the block counts
        // size the pool.
        gpu_memory_mb: (u64::from(max_blocks) * bytes_per_block)
            .div_ceil(1 << 20)
            .clamp(
                u64::from(mlx_paged_attn::MIN_GPU_MEMORY_MB),
                u64::from(u32::MAX),
            ) as u32,
        head_size: config.head_dim as u32,
        num_kv_heads: config.num_key_value_heads as u32,
        num_layers: super::load_plan::full_attention_layers(config) as u32,
        max_seq_len: Some(config.served_context_tokens()),
        max_batch_size: Some(4),
    };
    cfg.validate().map_err(Error::from_reason)?;
    Ok(cfg)
}

fn build(
    config: &super::config::Config,
    cfg: mlx_paged_attn::PagedAttentionConfig,
    initial: u32,
    max: u32,
) -> Result<PagedKVCacheAdapter> {
    let (dtype, _) = pool_block(config)?;
    let allocator = Arc::new(Mutex::new(mlx_paged_attn::BlockAllocator::new(
        initial, max, 32,
    )));
    let pool =
        mlx_paged_attn::LayerKVPool::new_with_layout(cfg, initial, max, dtype, page_layout(config))
            .map_err(|e| Error::from_reason(e.to_string()))?;
    PagedKVCacheAdapter::new(allocator, Arc::new(pool), 32).map_err(Error::from_reason)
}

/// The native bridge compacts strided inputs and preserves offsets for
/// contiguous read-only K/V views. An explicit zero selects independent
/// copies for comparison; mutable pool and metadata constraints stay strict.
/// The rows go to positions `base..` of the tokens recorded for this
/// request, never below `floor` (the tokens recorded before this pass): a
/// layer-major group records all its windows first and writes each window at
/// its own base.
pub(super) fn write_rows(
    adapter: &mut PagedKVCacheAdapter,
    layer: u32,
    keys: &MxArray,
    values: &MxArray,
    base: u32,
    floor: u32,
) -> Result<()> {
    let keys = keys.astype(crate::array::DType::BFloat16)?;
    let values = values.astype(crate::array::DType::BFloat16)?;
    adapter
        .update_keys_values_native_recorded(layer, &keys, &values, base, floor)
        .map_err(Error::from_reason)
}

/// GPU gather from the native paged layout, without materializing the full
/// sequence or reading its K/V through CPU memory. Only integer addresses
/// cross the host boundary. Selected rows are ordered by logical position.
/// An int8 pool comes back as BF16 rows (`dequantize_kv_rows` bits).
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
    let (ks, vs) = adapter
        .scale_pool_arrays(layer)
        .map_err(Error::from_reason)?;
    let mut ko = std::ptr::null_mut();
    let mut vo = std::ptr::null_mut();
    let ok = unsafe {
        mlx_sys::mlx_qwen4_gather_pages(
            k.as_raw_ptr(),
            v.as_raw_ptr(),
            ks.as_raw_ptr(),
            vs.as_raw_ptr(),
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
    type PagedDecode<'a> = PagedStepper<Step<'a>>;
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
        self.close_prefix_turn(seq);
        // A live continuation needs the row's whole history (decode rows
        // included) as the prompt's prefix; it publishes no state.
        let warm = reuse
            && !self.decoder.history.is_empty()
            && tokens.len() > self.decoder.history.len()
            && tokens.starts_with(&self.decoder.history);
        if !warm {
            self.decoder.reset();
            self.saved_history.clear();
        }
        // Token ids do not identify an image: an image row neither resumes
        // nor publishes, and keeps the plain window walk.
        let text =
            !warm && self.media_prefill.is_none() && !tokens.contains(&(IMAGE_TOKEN_ID as u32));
        let grid = self.decoder.prefill_slice_size();
        // Every text prompt, reused or not, cuts its passes where a cold
        // prefill of it does.
        let (cuts, ends) = if text {
            let cuts = prefix_state::boundaries(tokens, &self.prefix_states.open);
            let ends = prefix_state::pass_ends(tokens.len(), &cuts, |row| {
                self.decoder.prefill_rows_at(row)
            });
            (cuts, ends)
        } else {
            (Vec::new(), Vec::new())
        };
        let publish = text && reuse;
        let keys = if publish {
            prefix_state::pass_keys(tokens, &ends, grid)
        } else {
            Vec::new()
        };
        let hit = if publish {
            let adapter =
                self.decoder.paged.as_ref().ok_or_else(|| {
                    Error::from_reason("Qwen4 prefix preparation has no paged cache")
                })?;
            self.prefix_states.prune(adapter, grid);
            self.prefix_states.lookup(tokens, &keys, salt)
        } else {
            None
        };
        // After a command reset the purged prefix is prefilled again, as the
        // adapter skips its own cold restore once.
        let skip_disk = self
            .disk
            .as_mut()
            .is_some_and(super::prefix_disk::Disk::take_skip);
        let mut from_disk = None;
        let hit = if publish && !skip_disk && self.disk.is_some() {
            match self.restore_from_disk(tokens, &ends, &keys, salt, hit) {
                Some((deeper, id)) => {
                    from_disk = Some(id);
                    Some(deeper)
                }
                None => hit,
            }
        } else {
            hit
        };
        let adapter = self
            .decoder
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen4 prefix preparation has no paged cache"))?;
        // Without a stored state the lookup stays off: GDN, indexer and PLE
        // state cannot be rebuilt from attention pages.
        let plan = adapter
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                seq,
                tokens,
                tokens.len() as u32,
                warm,
                &keys,
                salt,
                hit.is_none(),
                hit.map_or(tokens.len().saturating_sub(1), |hit| hit.depth) as u32,
            )
            .map_err(Error::from_reason)?;
        let expected = hit.map_or(self.decoder.history.len(), |hit| hit.depth);
        let mut resumed = None;
        let mut cached = 0;
        if plan.cached_prefix_len as usize != expected {
            if let Some(id) = from_disk {
                self.prefix_states.store.remove(id);
            }
            adapter
                .reset_for_new_request(seq)
                .map_err(Error::from_reason)?;
            adapter
                .allocate_suffix_blocks(tokens.len() as u32)
                .map_err(Error::from_reason)?;
            self.decoder.reset();
        } else if let Some(hit) = hit {
            cached = hit.depth;
            let entry = self
                .prefix_states
                .store
                .get(hit.id)
                .ok_or_else(|| Error::from_reason("Qwen4 stored state vanished at restore"))?;
            self.decoder
                .restore_cell(entry.state(), &tokens[..hit.depth])?;
            // The state's draft history replaces whatever an earlier turn of
            // this sequence left: the restored turn drafts as a cold one.
            match (&entry.state().draft, self.has_mtp) {
                (Some(draft), true) => self.mtp.restore(seq, draft, &tokens[..hit.depth], grid)?,
                _ => self.mtp.forget(seq),
            }
            let mark = entry.mark();
            self.prefix_states.store.touch(hit.id);
            self.prefix_states.counters.restored += 1;
            resumed = Some(mark);
            tracing::debug!(
                target: "mlx_core::qwen4_exp::prefix",
                seq,
                cached = hit.depth,
                prompt = tokens.len(),
                "resumed from a stored state"
            );
        } else {
            cached = expected;
        }
        if text && resumed.is_none() && self.has_mtp {
            // On a load with the MTP head every published state carries the
            // draft history, so a cold prefill records it from row 0.
            if publish {
                self.mtp.start(seq);
            } else if self.mtp.owner_frontier(seq).is_some_and(|row| row != 0) {
                self.mtp.forget(seq);
            }
        }
        if text {
            self.open_prefix_turn(
                seq,
                tokens,
                prefix_state::PassPlan { cuts, ends, keys },
                salt,
                cached,
                resumed,
                publish,
            );
        }
        Ok(Prefix {
            cached,
            suffix: tokens.len() - cached,
        })
    }
    fn paged_prefill(&mut self, tokens: &[u32], _: &Prefix, stream: Stream) -> Result<MxArray> {
        let _context = crate::stream::StreamContext::new(stream);
        let seq = self.active_seq.unwrap_or(0);
        let base = self.decoder.history.len();
        let grid = self.decoder.prefill_slice_size();
        let runs = self
            .prefix_states
            .turns
            .get(&seq)
            .filter(|_| self.media_prefill.is_none())
            .map(|turn| turn.runs(base, base + tokens.len(), grid));
        let logits = if let Some(runs) = runs {
            // Group mode follows the direct-fill state at execution time; a
            // group holds only whole windows of the turn's passes.
            let mut logits = None;
            for run in runs {
                let (from, to) = (run[0].0, run[run.len() - 1].1);
                logits = Some(if run.len() > 1 && self.decoder.group_mode(to - from) {
                    let windows: Vec<&[u32]> = run
                        .iter()
                        .map(|&(start, end)| &tokens[start - base..end - base])
                        .collect();
                    let logits =
                        self.prefill_group_run(seq, &windows, to == base + tokens.len())?;
                    let cut_ok = self
                        .prefix_states
                        .turns
                        .get(&seq)
                        .is_some_and(|turn| turn.cold_pass(from, to));
                    self.reach_prefix_point(seq, to, cut_ok);
                    logits
                } else {
                    self.prefill_windows(seq, tokens, from - base, to - base)?
                });
            }
            logits.ok_or_else(|| Error::from_reason("Qwen4 empty paged prefill"))?
        } else if self.media_prefill.is_none() && self.decoder.group_mode(tokens.len()) {
            self.prefill_group_part(seq, tokens)?
        } else {
            self.prefill_windows(seq, tokens, 0, tokens.len())?
        };
        self.close_prefill_owner(seq);
        logits.squeeze(Some(&[1]))
    }
    fn begin_paged_decode(&mut self) -> Result<Self::PagedDecode<'_>> {
        Ok(PagedStepper(Step(self)))
    }
    fn admit_paged_speculative_decode(
        &mut self,
        _: &crate::engine::params::ChatParams,
    ) -> Result<bool> {
        // The scheduled lane owns proposal/verification transactions. An owned
        // turn routed here (cold continue of an unknown owner,
        // MLX_SERVE_FORCE_SERIAL) serves target AR without opening a second,
        // competing speculative cache owner. An ownerless one never gets here:
        // `run_paged_turn` refuses it before prefill.
        Ok(false)
    }
    /// The row stays live without publishing its blocks: rows a decode wrote
    /// are not what a cold prefill writes. States published during the
    /// prefill stay.
    fn finalize_paged_turn(&mut self, reuse: bool, _: u64) {
        self.close_prefix_turn(self.active_seq.unwrap_or(0));
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
    /// States published before the failure stay: each was published only
    /// once its blocks were registered.
    fn abort_paged_turn(&mut self) {
        self.close_prefix_turn(self.active_seq.unwrap_or(0));
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

    /// The pool's share of the dynamic budget now.
    fn pool_charge(&self) -> super::memory::PoolCharge {
        let c = &self.decoder.config;
        let (telemetry, physical) = self
            .decoder
            .paged
            .as_ref()
            .and_then(|p| Some((p.block_telemetry().ok()?, p.bytes_per_block().ok()?)))
            .unwrap_or_default();
        let total_blocks = u64::from(telemetry.total_blocks);
        super::memory::PoolCharge {
            total_blocks,
            max_blocks: u64::from(telemetry.max_total_blocks),
            physical_block_bytes: physical,
            growth_copy_bytes: total_blocks.saturating_mul(physical)
                / (super::load_plan::full_attention_layers(c) as u64).max(1),
        }
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
impl Inner {
    /// Opens `seq`'s text turn of `tokens`: its pass ends, and when it may
    /// publish, its points past `resume`, with the block of the state it
    /// resumed from in use.
    #[allow(clippy::too_many_arguments)]
    fn open_prefix_turn(
        &mut self,
        seq: SeqId,
        tokens: &[u32],
        passes: prefix_state::PassPlan,
        salt: u64,
        resume: usize,
        resumed: Option<u64>,
        publish: bool,
    ) {
        let grid = self.decoder.prefill_slice_size();
        let states = &mut self.prefix_states;
        let prefix_state::PassPlan {
            cuts,
            ends,
            mut keys,
        } = passes;
        let points = if publish {
            let replay = prefix_state::replay_row(&cuts, &ends, tokens.len());
            prefix_state::plan_points(resume, replay, &ends, grid, states.interval)
        } else {
            Vec::new()
        };
        let last = points.last().map_or(0, |point| point.row);
        let blocks = last / prefix_state::BLOCK_ROWS;
        keys.truncate(blocks);
        let chain = if blocks == 0 || keys.len() < blocks {
            Vec::new()
        } else {
            crate::models::qwen3_5::gdn_checkpoint_store::compute_paged_prefix_block_hashes(
                tokens,
                last as u32,
                prefix_state::BLOCK_ROWS as u32,
                &keys,
                salt,
            )
            .unwrap_or_default()
        };
        let uses = resumed.into_iter().collect::<Vec<_>>();
        for &block in &uses {
            states.store.use_state(block);
        }
        states.turns.insert(
            seq,
            prefix_state::Turn {
                salt,
                prompt_len: tokens.len(),
                resume,
                ends,
                lineage: tokens[..last].to_vec(),
                canonical: chain.len() == blocks,
                chain,
                keys,
                points,
                cursor: 0,
                uses,
                checkpoint: None,
                replay: None,
            },
        );
        if publish
            && let Some(disk) = self.disk.as_mut()
            && let Some(turn) = self.prefix_states.turns.get(&seq)
        {
            disk.start(seq, &turn.lineage, &turn.keys, &turn.chain, salt, grid);
        }
    }

    /// Ends `seq`'s turn: its replay state is parked for the disk, then the
    /// states it used become the newest of their class.
    pub(super) fn close_prefix_turn(&mut self, seq: SeqId) {
        if let (Some(disk), Some(adapter)) = (self.disk.as_mut(), self.decoder.paged.as_mut()) {
            let replay = self
                .prefix_states
                .turns
                .get(&seq)
                .and_then(|turn| turn.replay);
            if let Some((id, row)) = replay
                && let Some(entry) = self.prefix_states.store.get(id)
            {
                disk.park(seq, entry.state(), row, adapter);
            }
            disk.close(seq);
            disk.top_up_all(adapter);
        }
        self.prefix_states.end_turn(seq);
        self.telemetry.sync(&self.prefix_states);
    }

    /// After a decode step: copies out what the running turns owe the
    /// disk, within its queue, at most every few steps.
    pub(super) fn top_up_disk(&mut self) {
        if let (Some(disk), Some(adapter)) = (self.disk.as_mut(), self.decoder.paged.as_mut()) {
            disk.top_up_step(adapter);
        }
    }

    /// Splash's restart restore: the deepest state of `tokens` on disk past
    /// the memory `hit`, whose chain is whole, read back into the memory
    /// store under the room and live checks of a publication, with the pool
    /// grown for the blocks the adapter then restores. The state's id and
    /// its hit; `None` keeps `hit`.
    fn restore_from_disk(
        &mut self,
        tokens: &[u32],
        ends: &[usize],
        keys: &[Vec<u64>],
        salt: u64,
        hit: Option<StoreHit>,
    ) -> Option<(StoreHit, EntryId)> {
        use crate::models::qwen3_5::dflash2_state::store::{Lineage, StateClass, lineage_bytes};
        let grid = self.decoder.prefill_slice_size();
        let floor = hit.map_or(0, |hit| hit.depth);
        let candidate = {
            let disk = self.disk.as_ref()?;
            let adapter = self.decoder.paged.as_ref()?;
            disk.candidate(
                tokens,
                ends,
                keys,
                salt,
                floor,
                adapter,
                self.has_mtp.then_some(grid),
            )?
        };
        let row = candidate.row;
        let blocks = row / prefix_state::BLOCK_ROWS;
        let need =
            super::memory::state_bytes(&self.decoder.config, row as u64, grid as u64, self.has_mtp)
                .saturating_add(lineage_bytes(row, blocks));
        let Some((victims, freed)) = self.prefix_states.room_for(self.state_room(), need) else {
            self.prefix_states.counters.refused_room += 1;
            return None;
        };
        let live = super::memory::host_memory().is_ok_and(|host| {
            super::memory::state_fits_live(
                &host,
                crate::array::memory::anonymous_active_memory(),
                self.decoder
                    .weights
                    .active_limit(crate::cache_limit::coordinator().registered_pool_bytes()),
                need.saturating_sub(freed),
                self.decoder.weights.plan.use_direct_turn_floor(),
            )
        });
        if !live {
            self.prefix_states.counters.refused_live += 1;
            return None;
        }
        let cell = self.disk.as_ref()?.load(
            &candidate,
            self.decoder.caches.len(),
            grid,
            self.has_mtp,
            need.saturating_mul(2).saturating_add(1 << 20),
        )?;
        for id in victims {
            if self.prefix_states.store.remove(id) {
                self.prefix_states.counters.dropped_memory += 1;
            }
        }
        let inserted = self
            .prefix_states
            .store
            .insert(
                Lineage {
                    tokens: &tokens[..row],
                    extra_keys: &keys[..blocks],
                    salt,
                },
                cell,
                StateClass::Ordinary,
            )
            .ok()?;
        let hit = self
            .prefix_states
            .lookup(tokens, keys, salt)
            .filter(|hit| hit.id == inserted.id && hit.depth == row);
        let Some(hit) = hit else {
            self.prefix_states.store.remove(inserted.id);
            return None;
        };
        let adapter = self.decoder.paged.as_mut()?;
        let hot = crate::models::qwen3_5::gdn_checkpoint_store::compute_paged_prefix_block_hashes(
            tokens,
            row as u32,
            prefix_state::BLOCK_ROWS as u32,
            &keys[..blocks],
            salt,
        )
        .unwrap_or_default();
        let cached = hot
            .iter()
            .take_while(|&&hash| adapter.prefix_cached(hash))
            .count();
        adapter.grow_for(u32::try_from(blocks - cached.min(blocks)).unwrap_or(u32::MAX));
        tracing::debug!(
            target: "mlx_core::qwen4_exp::prefix",
            row,
            prompt = tokens.len(),
            "read a stored state back from disk"
        );
        Some((hit, inserted.id))
    }

    /// What stored states may hold now (see [`super::memory::state_room`]).
    pub(super) fn state_room(&self) -> u64 {
        #[cfg(test)]
        if let Some(room) = self.prefix_states.room_override {
            return room;
        }
        let c = &self.decoder.config;
        let held = (self.rows.len() + usize::from(self.active_seq.is_some())) as u64
            * super::memory::row_state_bytes(c)
            + self.mtp.held_bytes(c);
        super::memory::state_room(
            self.memory_plan.dynamic_budget_bytes,
            super::memory::prefill_scratch(
                c,
                self.decoder.prefill_chunk_size,
                self.decoder.context_window,
            ),
            &self.pool_charge(),
            held,
        )
    }

    /// A pass of `seq` ended at row `end`; `cut_ok` tells whether it was the
    /// pass a cold prefill runs there. A planned point there is published
    /// while every pass so far was.
    fn reach_prefix_point(&mut self, seq: SeqId, end: usize, cut_ok: bool) {
        let Some(turn) = self.prefix_states.turns.get_mut(&seq) else {
            return;
        };
        let Some(point) = turn.reach(end, cut_ok) else {
            return;
        };
        if !turn.canonical {
            self.prefix_states.counters.refused_off_grid += 1;
            return;
        }
        self.publish_prefix_point(seq, point);
    }

    /// Splash `Engine::publishReachedStateBoundaries` for one point: the
    /// states to drop for room are chosen (checkpoints first), the request's
    /// blocks below the point are registered, then those states go and the
    /// cell is stored. A refusal drops nothing, costs reuse and never fails
    /// the turn. A checkpoint retires the turn's previous one; the replay
    /// point is in use until the turn ends. A block already cached under its
    /// pass key came from the same pass over the same tokens, so the cache
    /// keeping the first one gives this state the rows it was computed on.
    fn publish_prefix_point(&mut self, seq: SeqId, point: prefix_state::Point) {
        use crate::models::qwen3_5::dflash2_state::store::{Lineage, StateClass, lineage_bytes};
        let row = point.row;
        let blocks = row / prefix_state::BLOCK_ROWS;
        let grid = self.decoder.prefill_slice_size();
        let Some(adapter) = self.decoder.paged.as_ref() else {
            return;
        };
        // Prune before registering: a state whose chain lost a block must
        // be gone before that hash can be registered again by another run.
        self.prefix_states.prune(adapter, grid);
        let Some(turn) = self.prefix_states.turns.get_mut(&seq) else {
            return;
        };
        if turn.chain.len() < blocks || turn.keys.len() < blocks {
            self.prefix_states.counters.refused_register += 1;
            turn.canonical = false;
            return;
        }
        let need =
            super::memory::state_bytes(&self.decoder.config, row as u64, grid as u64, self.has_mtp)
                .saturating_add(lineage_bytes(row, blocks));
        let room = self.state_room();
        let Some((victims, freed)) = self.prefix_states.room_for(room, need) else {
            self.prefix_states.counters.refused_room += 1;
            return;
        };
        let live = super::memory::host_memory().is_ok_and(|host| {
            super::memory::state_fits_live(
                &host,
                crate::array::memory::anonymous_active_memory(),
                self.decoder
                    .weights
                    .active_limit(crate::cache_limit::coordinator().registered_pool_bytes()),
                need.saturating_sub(freed),
                self.decoder.weights.plan.use_direct_turn_floor(),
            )
        });
        if !live {
            self.prefix_states.counters.refused_live += 1;
            return;
        }
        // On a load with the MTP head a state never goes without the draft
        // history the cold prefill has there.
        let draft = if self.has_mtp {
            match self
                .mtp
                .capture(seq, row, grid, self.decoder.config.indexer_compress_ratio)
            {
                Ok(Some(draft)) => Some(draft),
                Ok(None) | Err(_) => {
                    self.prefix_states.counters.refused_draft += 1;
                    return;
                }
            }
        } else {
            None
        };
        let Some(adapter) = self.decoder.paged.as_mut() else {
            return;
        };
        let states = &mut self.prefix_states;
        let Some(turn) = states.turns.get_mut(&seq) else {
            return;
        };
        let registered = adapter
            .register_committed_blocks(row, &turn.keys[..blocks], turn.salt)
            .is_ok()
            && turn.chain[..blocks]
                .iter()
                .all(|&hash| adapter.prefix_cached(hash));
        if !registered {
            states.counters.refused_register += 1;
            turn.canonical = false;
            return;
        }
        for id in victims {
            if states.store.remove(id) {
                states.counters.dropped_memory += 1;
            }
        }
        let cell = match self.decoder.cell(grid) {
            Ok(Some(cell)) => prefix_state::Cell { draft, ..cell },
            Ok(None) | Err(_) => {
                states.counters.refused_register += 1;
                return;
            }
        };
        let class = if point.checkpoint {
            StateClass::Checkpoint
        } else {
            StateClass::Ordinary
        };
        let inserted = match states.store.insert(
            Lineage {
                tokens: &turn.lineage[..row],
                extra_keys: &turn.keys[..blocks],
                salt: turn.salt,
            },
            cell,
            class,
        ) {
            Ok(inserted) => inserted,
            Err(_) => {
                states.counters.refused_register += 1;
                return;
            }
        };
        states.counters.published += 1;
        if point.checkpoint {
            if let Some(old) = turn.checkpoint.replace(inserted.id)
                && old != inserted.id
                && states.store.checkpoint_state(old).is_some()
                && states.store.remove(old)
            {
                states.counters.retired += 1;
            }
        } else if let Some(mark) = states.store.get(inserted.id).map(|entry| entry.mark()) {
            states.store.use_state(mark);
            turn.uses.push(mark);
            turn.replay = Some((inserted.id, row));
        }
        if let (Some(disk), Some(adapter)) = (self.disk.as_mut(), self.decoder.paged.as_mut()) {
            let draft = states
                .store
                .get(inserted.id)
                .and_then(|entry| entry.state().draft.as_ref());
            disk.published(seq, row, draft);
            disk.top_up(seq, adapter);
        }
        self.telemetry.sync(states);
        tracing::debug!(
            target: "mlx_core::qwen4_exp::prefix",
            seq,
            row,
            checkpoint = point.checkpoint,
            states = states.store.bytes().entries,
            bytes = states.store.bytes().state_bytes,
            "published a stored state"
        );
    }

    /// One layer-major group over `tokens` (and its tail), observed by the
    /// MTP owner per window: a prompt without a text turn.
    fn prefill_group_part(&mut self, seq: SeqId, tokens: &[u32]) -> Result<MxArray> {
        let (windows, tail) = self.decoder.group_windows(tokens);
        let tracking = self.mtp_tracking(seq);
        let (logits, views) = self.decoder.prefill_group(&windows, tail, true, tracking)?;
        if tracking {
            for (window, hidden) in windows.iter().zip(&views) {
                self.observe_mtp(seq, window, hidden)?;
            }
            if !tail.is_empty() {
                let hidden = self.decoder.last_chunk_hidden.clone();
                self.observe_mtp(seq, tail, &hidden)?;
            }
        }
        Ok(logits)
    }

    /// One layer-major group over a run of the turn's passes, observed by
    /// the MTP owner per pass.
    fn prefill_group_run(&mut self, seq: SeqId, windows: &[&[u32]], last: bool) -> Result<MxArray> {
        let tracking = self.mtp_tracking(seq);
        let (logits, views) = self.decoder.prefill_group(windows, &[], last, tracking)?;
        if tracking {
            for (window, hidden) in windows.iter().zip(&views) {
                self.observe_mtp(seq, window, hidden)?;
            }
        }
        Ok(logits)
    }

    /// `tokens[from..to]` one pass at a time, each ending at the turn's next
    /// pass end or where the admitted window, read before every pass, stops
    /// it; the logits of the last pass.
    fn prefill_windows(
        &mut self,
        seq: SeqId,
        tokens: &[u32],
        from: usize,
        to: usize,
    ) -> Result<MxArray> {
        let mut logits = None;
        let mut start = from;
        while start < to {
            let window = self.decoder.prefill_pass_window();
            let base = self.decoder.history.len();
            let planned = self
                .prefix_states
                .turns
                .get(&seq)
                .and_then(|turn| turn.next_end(base))
                .map_or(window, |end| (end - base).min(window));
            let end = (start + planned.max(1)).min(to);
            let chunk = &tokens[start..end];
            let next = &tokens[end..(end + window).min(tokens.len())];
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
            logits = Some(self.decoder.prefill_chunk_ahead(
                chunk,
                next,
                embeddings.as_ref(),
                end == tokens.len(),
            )?);
            let hidden = self.decoder.last_chunk_hidden.clone();
            self.observe_mtp(seq, chunk, &hidden)?;
            let reached = base + chunk.len();
            let cut_ok = self
                .prefix_states
                .turns
                .get(&seq)
                .is_some_and(|turn| turn.cold_pass(base, reached));
            self.reach_prefix_point(seq, reached, cut_ok);
            start = end;
        }
        logits.ok_or_else(|| Error::from_reason("Qwen4 empty paged prefill"))
    }

    /// The prefill of a text turn reached its prompt's end: an owner the
    /// scheduler did not begin speculation on only recorded the draft
    /// history for the states, which are published by now.
    fn close_prefill_owner(&mut self, seq: SeqId) {
        let done = self
            .prefix_states
            .turns
            .get(&seq)
            .is_some_and(|turn| self.decoder.history.len() >= turn.prompt_len);
        if done && !self.mtp.speculative(seq) {
            self.mtp.forget(seq);
        }
    }

    /// The model's wired lease: the store's budget and the plan's dynamic
    /// budget, which holds the pool, the rows' state and the transient.
    pub(super) fn wired_bytes(&self) -> usize {
        self.decoder
            .weights
            .cache_budget()
            .checked_add(self.memory_plan.dynamic_budget_bytes)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .unwrap_or(usize::MAX)
    }
}

impl HybridSchedulerBackend for Inner {
    type FamilyCommand = std::convert::Infallible;
    type RestoreTicket = NoRestoreTicket;
    type OwnerState = Vec<u32>;
    type StepExecutor<'a> = HybridStepExecutor<'a, Self>;
    const SCHEDULER_NAME: &'static str = "qwen4_exp";
    fn wired_residency(&self) -> Option<Arc<crate::stream::WiredResidency>> {
        self.wired.clone()
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
    fn ownerless_session_in_owner_tables(&self) -> bool {
        true
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
        if self.decoder.history.len() + queries > self.decoder.context_window {
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
    fn propose_scheduled(
        &mut self,
        seq: SeqId,
        anchor: u32,
        cap: usize,
        params: &crate::engine::params::ChatParams,
        rng: &mut dyn rand::Rng,
        confidence: bool,
    ) -> Result<crate::engine::backend::DsparkProposal> {
        self.in_decode_pass(|inner| {
            inner.activate_paged_seq(seq)?;
            inner.propose_mtp(seq, anchor, cap, params, rng, confidence)
        })
    }
    fn run_scheduled_verify(
        &mut self,
        rows: &[crate::engine::hybrid_scheduler::ScheduledVerifyRow],
    ) -> Result<MxArray> {
        self.in_decode_pass(|inner| inner.verify_mtp(rows))
    }
    fn commit_scheduled_verify(
        &mut self,
        rows: &[crate::engine::hybrid_scheduler::ScheduledVerifyCommit],
    ) -> Result<Vec<Result<()>>> {
        self.in_decode_pass(|inner| inner.commit_mtp(rows))
    }
    fn release_scheduled_speculation(&mut self, seq: SeqId) {
        self.mtp.finish(
            seq,
            (self.active_seq == Some(seq)).then_some(self.decoder.history.len()),
        );
    }
    fn scheduler_capacity(&self) -> usize {
        super::LANES
    }
    fn scheduled_turn_opened(&mut self, seq: SeqId) {
        let first = self.open_turns.is_empty() && !self.exclusive_turn;
        if self.open_turns.insert(seq) {
            self.open_turn(first);
        }
    }
    fn scheduled_turn_closed(&mut self, seq: SeqId) {
        self.open_turns.remove(&seq);
        self.close_prefix_turn(seq);
    }
    fn scheduler_prefill_slice_tokens(&self) -> u32 {
        self.decoder.prefill_slice_size() as u32
    }
    /// A text turn's slices end at its pass ends, the ends a cold prefill of
    /// its prompt cuts at.
    fn scheduler_prefill_ends(
        &self,
        seq: SeqId,
        prompt_tokens: u32,
        cached_prefix: u32,
    ) -> Option<Vec<u32>> {
        let turn = self.prefix_states.turns.get(&seq)?;
        if turn.prompt_len != prompt_tokens as usize {
            return None;
        }
        turn.ends
            .iter()
            .filter(|&&end| end > cached_prefix as usize)
            .map(|&end| u32::try_from(end).ok())
            .collect()
    }
    fn scheduler_output_budget_is_rest_of_window(&self) -> bool {
        true
    }
    /// Splash `Cache::ensureTokens`: the pool starts at its runway and a
    /// turn books its pages as it writes them.
    fn scheduler_books_on_demand(&self) -> bool {
        true
    }
    fn planned_turn_window(&self, _speculative: bool) -> Option<u32> {
        let pool = self.memory_plan.pool.as_ref()?;
        Some(pool.lane.window.maximum().min(self.context_cap()))
    }
    fn scheduler_whole_prompt_prefill(&self, prompt_tokens: u32, cached_prefix: u32) -> bool {
        self.media_prefill.is_none()
            && self
                .decoder
                .group_mode(prompt_tokens.saturating_sub(cached_prefix) as usize)
    }
    fn max_position_embeddings(&self) -> i32 {
        self.context_cap() as i32
    }
    fn scheduled_state_capacity(&self) -> Option<ScheduledStateCapacity> {
        let c = &self.decoder.config;
        let row = super::memory::row_state_bytes(c);
        Some(ScheduledStateCapacity {
            budget: super::memory::scheduled_state_budget(
                self.memory_plan.dynamic_budget_bytes,
                super::memory::prefill_scratch(
                    c,
                    self.decoder.prefill_chunk_size,
                    self.decoder.context_window,
                ),
                &self.pool_charge(),
            ),
            held: (self.rows.len() + usize::from(self.active_seq.is_some())) as u64 * row
                + self.mtp.held_bytes(c)
                + self.prefix_states.bytes(),
        })
    }
    /// The state budget is charged for every block the pool holds, and the
    /// pool only grows: a refused state need takes stored states first (rows
    /// outrank them), then the pool's free extents back (Splash
    /// `Cache::releaseExtent`), cached pages last.
    fn reclaim_for_scheduled_state(&mut self, shortfall: u64) -> bool {
        if self.prefix_states.reclaim(shortfall) > 0 {
            return true;
        }
        let Some(paged) = self.decoder.paged.as_mut() else {
            return false;
        };
        paged.try_release_pool_extents(shortfall, false) > 0
            || paged.try_release_pool_extents(shortfall, true) > 0
    }
    fn scheduled_state_candidate_bytes(&self, seq: SeqId, tokens: u32, speculative: bool) -> u64 {
        let c = &self.decoder.config;
        let row = if self.has_scheduled_recurrent(seq) {
            0
        } else {
            super::memory::row_state_bytes(c)
        };
        // On a load with the MTP head an autoregressive text prefill keeps
        // the draft history of its rows (a full window of them waiting as
        // hidden rows at most) until its prompt is in.
        let held = self.mtp.owner_frontier(seq).unwrap_or(0) as u64;
        let rows = u64::from(tokens).saturating_sub(held);
        let draft = if speculative {
            super::memory::draft_state_bytes(c, rows)
                + rows * super::memory::draft_index_bytes_per_token(c)
        } else if self.has_mtp {
            let waiting = rows.min(self.decoder.prefill_slice_size() as u64 + 1);
            super::memory::draft_owner_bytes(c, rows - waiting, waiting, false)
                .max(super::memory::draft_owner_bytes(c, rows, 0, false))
        } else {
            0
        };
        row + draft
    }
    fn scheduled_state_held_bytes(&self, seq: SeqId) -> u64 {
        let c = &self.decoder.config;
        u64::from(self.has_scheduled_recurrent(seq)) * super::memory::row_state_bytes(c)
            + self.mtp.owner_held_bytes(c, seq)
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
    /// A preempted turn's blocks go back without publication: its rows past
    /// the last stored state include decode rows, which are not what a cold
    /// prefill writes. Its replay resumes from its stored states.
    fn preempt_scheduled_cache(
        &mut self,
        seq: SeqId,
        _: u64,
        _: crate::engine::scheduler::PreemptionMode,
    ) -> Result<()> {
        self.close_prefix_turn(seq);
        let adapter = self.decoder.paged.as_mut().ok_or_else(|| {
            Error::from_reason("qwen4_exp cache manager is unavailable during preemption")
        })?;
        if adapter.block_table_for(seq).is_some() {
            adapter
                .release_request_for(seq)
                .map_err(Error::from_reason)?;
        }
        Ok(())
    }
    fn release_scheduled_recurrent_for(&mut self, seq: SeqId) {
        self.close_prefix_turn(seq);
        self.rows.remove(&seq);
        self.mtp.release(seq);
        if self.active_seq == Some(seq) {
            self.decoder.reset();
            self.active_seq = None;
            self.saved_history.clear();
        }
    }
    fn run_paged_decode_step_batched(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        let logits = self.in_decode_pass(|inner| inner.decode_step_rows(rows))?;
        self.top_up_disk();
        Ok(logits)
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
                    crate::engine::backend::ChunkSink::send(&stream_tx, Err(error));
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
/// Int8 pool rows come back as BF16 values (`dequantize_kv_rows` bits).
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
    let blocks = crate::transformer::paged_kv_cache_adapter::build_decode_block_ids(table, 0);
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
    let (ks, vs) = adapter
        .scale_pool_arrays(layer)
        .map_err(Error::from_reason)?;
    let (mut k, mut v) = (std::ptr::null_mut(), std::ptr::null_mut());
    let ok = unsafe {
        mlx_sys::mlx_qwen4_gather_window(
            kp.as_raw_ptr(),
            vp.as_raw_ptr(),
            ks.as_raw_ptr(),
            vs.as_raw_ptr(),
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
    /// Splash `Engine::stop` ends a lane at its pass boundary; see
    /// `Decoder::decode_pass`.
    fn in_decode_pass<R>(&mut self, pass: impl FnOnce(&mut Self) -> R) -> R {
        let prior = std::mem::replace(&mut self.decoder.decode_pass, true);
        let result = pass(self);
        self.decoder.decode_pass = prior;
        result
    }
    fn decode_step_rows(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
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
        if rows.len() > 1 {
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
    fn decode_shared_layers(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        let started = std::time::Instant::now();
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
        } else {
            self.decoder.observe_pass(rows.len(), started.elapsed());
        }
        result
    }
}
