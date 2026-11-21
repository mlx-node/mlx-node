//! The paged pool of the trunk's indexed layers, one full group (Hy4 has no
//! recurrent state):
//!
//! ```text
//! page (32 rows, 1 head, token-major, f16)   per indexed trunk layer
//!   K plane     [32, half]   latent cols 0..half          half = kv_row / 2
//!   V plane     [32, half]   latent cols half..kv_row
//!   index plane [32 / kpool, indexer.head_dim]   pooled keys of the block's groups
//! write : latent [n, kv_row] f16 -> two half rows, one store
//! read  : slot = block_table[t / 32]·32 + t % 32 (-1 kept), take K ‖ take V
//! ```
//!
//! Sliding windows, the `tok_k` tails and the MTP layer stay with each
//! sequence. A 32-row block holds whole groups, so a page depends only on
//! its own tokens.

use std::sync::{Arc, Mutex};

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{BlockAllocator, LayerKVPool, PageLayout, PagedAttentionConfig};
use napi::{Error, Result};

use super::cache::ROW_DTYPE;
use super::config::{Attn, HyV4FlashConfig};
use super::store::LANES_MAX;
use crate::array::{DType, MxArray};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

/// Rows per page.
pub const BLOCK: u32 = 32;
/// Dtype of the latent planes and of the index plane.
pub const PAGE_DTYPE: MetalDtype = MetalDtype::Float16;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 paged pool: {msg}"))
}

/// The pool layer of each trunk layer: the indexed layers in order, `None`
/// for a sliding layer.
pub fn pool_layers(cfg: &HyV4FlashConfig) -> Vec<Option<u32>> {
    let mut next = 0;
    cfg.trunk()
        .iter()
        .map(|l| {
            (l.attn == Attn::Indexed).then(|| {
                next += 1;
                next - 1
            })
        })
        .collect()
}

/// The sizes of one page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageGeometry {
    /// Indexed trunk layers.
    pub layers: u32,
    /// Columns per plane: half the latent row.
    pub half: u32,
    /// Pooled groups per page.
    pub index_rows: u32,
    pub index_dim: u32,
}

impl PageGeometry {
    pub fn of(cfg: &HyV4FlashConfig) -> Result<Self> {
        let row = cfg.mla.kv_row();
        let kpool = cfg.indexer.kpool;
        if !row.is_multiple_of(2) {
            return Err(bad(format!(
                "latent row {row} does not split in two planes"
            )));
        }
        if kpool == 0 || !(BLOCK as usize).is_multiple_of(kpool) {
            return Err(bad(format!(
                "groups of {kpool} do not tile a {BLOCK}-row page"
            )));
        }
        let layers = pool_layers(cfg).iter().flatten().count();
        let to_u32 = |v: usize, what: &str| {
            u32::try_from(v).map_err(|_| bad(format!("{what} {v} exceeds u32")))
        };
        Ok(Self {
            layers: to_u32(layers, "layer count")?,
            half: to_u32(row / 2, "plane width")?,
            index_rows: BLOCK / to_u32(kpool, "kpool")?,
            index_dim: to_u32(cfg.indexer.head_dim, "indexer width")?,
        })
    }

    pub fn layout(&self) -> PageLayout {
        PageLayout {
            values_token_major: true,
            keys_token_major: false,
            index_rows: self.index_rows,
            index_dimension: self.index_dim,
        }
    }

    /// Bytes of one page over every pool layer.
    pub fn block_bytes(&self) -> Result<u64> {
        let planes =
            mlx_paged_attn::profile::bytes_per_block(self.layers, 1, self.half, BLOCK, PAGE_DTYPE)
                .map_err(|e| bad(e.to_string()))?;
        Ok(planes + u64::from(self.layers) * self.layout().index_bytes())
    }

    /// The pool config of `max_blocks` pages for sequences of `context` rows.
    pub fn config(&self, max_blocks: u32, context: usize) -> Result<PagedAttentionConfig> {
        let cfg = PagedAttentionConfig {
            block_size: BLOCK,
            // `validate` refuses less than `MIN_GPU_MEMORY_MB`; the block
            // counts size the pool.
            gpu_memory_mb: (u64::from(max_blocks) * self.block_bytes()?)
                .div_ceil(1 << 20)
                .clamp(
                    u64::from(mlx_paged_attn::MIN_GPU_MEMORY_MB),
                    u64::from(u32::MAX),
                ) as u32,
            head_size: self.half,
            num_kv_heads: 1,
            num_layers: self.layers,
            max_seq_len: Some(u32::try_from(context).map_err(|_| bad("context exceeds u32"))?),
            max_batch_size: Some(LANES_MAX as u32),
        };
        cfg.validate().map_err(bad)?;
        Ok(cfg)
    }
}

/// The adapter over a pool of `initial` pages that grows by extents to
/// `max`, each growth probed against `ceiling` bytes.
pub fn create(
    g: &PageGeometry,
    context: usize,
    initial: u32,
    max: u32,
    ceiling: u64,
) -> Result<PagedKVCacheAdapter> {
    let cfg = g.config(max, context)?;
    let allocator = Arc::new(Mutex::new(BlockAllocator::new(initial, max, BLOCK)));
    let pool =
        LayerKVPool::new_with_index_dtype(cfg, initial, max, PAGE_DTYPE, g.layout(), PAGE_DTYPE)
            .map_err(bad)?;
    let mut adapter = PagedKVCacheAdapter::new(allocator, Arc::new(pool), BLOCK).map_err(bad)?;
    if initial < max {
        adapter.set_extent_growth(
            crate::models::qwen3_5::dflash2_decode::runway_pool_extent_blocks(g.block_bytes()?),
        );
    }
    adapter.set_memory_ceiling(ceiling);
    Ok(adapter)
}

/// Writes the latent rows `kv [n, 2·half]` (stored as f16) of positions
/// `base..base + n` of the active request to pool layer `layer`: columns
/// `0..half` to the K plane, the rest to the V plane. The rows must be
/// recorded, none below `floor`.
pub fn write_latent(
    adapter: &mut PagedKVCacheAdapter,
    layer: u32,
    kv: &MxArray,
    base: u32,
    floor: u32,
) -> Result<()> {
    let half = i64::from(adapter.layer_kv_pool().config().head_size);
    let d = kv.shape()?.as_ref().to_vec();
    let &[n, w] = d.as_slice() else {
        return Err(bad(format!("latent rows {d:?} must be [n, {}]", 2 * half)));
    };
    if w != 2 * half || n <= 0 {
        return Err(bad(format!("latent rows {d:?} for planes of {half}")));
    }
    let rows = kv.astype(ROW_DTYPE)?;
    let k = rows.slice(&[0, 0], &[n, half])?.reshape(&[n, 1, half])?;
    let v = rows
        .slice(&[0, half], &[n, 2 * half])?
        .reshape(&[n, 1, half])?;
    adapter
        .update_keys_values_native_recorded(layer, &k, &v, base, floor)
        .map_err(bad)
}

/// The K and V planes of pool layer `layer` as last written, each
/// `[pages·32, half]`, rows by slot id.
pub fn planes(adapter: &PagedKVCacheAdapter, layer: u32) -> Result<(MxArray, MxArray)> {
    let half = i64::from(adapter.layer_kv_pool().config().head_size);
    let rows = |plane: MxArray| -> Result<MxArray> {
        let d = plane.shape()?.as_ref().to_vec();
        match d.as_slice() {
            &[pages, 1, b, w] if b == i64::from(BLOCK) && w == half => {
                plane.reshape(&[pages * b, half])
            }
            s => Err(bad(format!(
                "plane {s:?} is not a token-major page of {half}"
            ))),
        }
    };
    Ok((
        rows(adapter.key_pool_array(layer).map_err(bad)?)?,
        rows(adapter.value_pool_array(layer).map_err(bad)?)?,
    ))
}

/// Page ids `[pages]` i32 of a block table; refused when a slot id would
/// not fit i32.
pub fn page_ids(ids: &[u32]) -> Result<MxArray> {
    let top = ids.iter().max().map_or(0, |&b| u64::from(b) + 1) * u64::from(BLOCK);
    if ids.is_empty() || top > i32::MAX as u64 {
        return Err(bad(format!(
            "{} pages ending at slot {top} do not map to i32 slots",
            ids.len()
        )));
    }
    let pages = ids
        .iter()
        .map(|&b| i32::try_from(b).map_err(|_| bad(format!("page {b} exceeds i32"))))
        .collect::<Result<Vec<_>>>()?;
    MxArray::from_int32(&pages, &[pages.len() as i64])
}

/// Logical rows `idx` (`-1` = no key) -> slot ids `pages[t / 32]·32 + t %
/// 32` (i32) of the same shape, `-1` kept. Every listed row must lie on a
/// page of `pages`.
pub fn slot_ids(pages: &MxArray, idx: &MxArray) -> Result<MxArray> {
    let idx = idx.astype(DType::Int32)?;
    let zero = MxArray::from_int32(&[0], &[1])?;
    let block = MxArray::from_int32(&[BLOCK as i32], &[1])?;
    let row = idx.maximum(&zero)?;
    let page = pages.take(&row.floor_divide(&block)?, 0)?;
    let slot = page.mul(&block)?.add(&row.remainder(&block)?)?;
    idx.greater_equal(&zero)?
        .where_(&slot, &MxArray::from_int32(&[-1], &[1])?)
}

/// The active request's pages on the GPU, to turn logical rows into slot
/// ids.
pub struct SlotMap {
    pages: MxArray,
    table: (u64, u64),
}

impl SlotMap {
    /// The map of the active request's pages now; build it after the call's
    /// rows are recorded.
    pub fn of(adapter: &PagedKVCacheAdapter) -> Result<Self> {
        let t = adapter
            .block_table()
            .ok_or_else(|| bad("slot map with no active request"))?;
        Ok(Self {
            pages: page_ids(&t.block_ids())?,
            table: (t.metadata_identity(), t.physical_revision()),
        })
    }

    /// Validated `[pages]` i32 page table of the active request. Refused once
    /// the active request's pages differ from the ones the map was built on.
    pub fn pages(&self, adapter: &PagedKVCacheAdapter) -> Result<&MxArray> {
        let now = adapter
            .block_table()
            .map(|t| (t.metadata_identity(), t.physical_revision()));
        if now != Some(self.table) {
            return Err(bad("slot map of other pages than the active request's"));
        }
        Ok(&self.pages)
    }

    /// `slot_ids` over this map's pages. Refused once the active request's
    /// pages differ from the ones the map was built on.
    pub fn slots(&self, adapter: &PagedKVCacheAdapter, idx: &MxArray) -> Result<MxArray> {
        slot_ids(self.pages(adapter)?, idx)
    }
}
