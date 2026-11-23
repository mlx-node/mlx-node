//! Bytes of the disk tier: a KV block of the indexed trunk layers, a stored
//! state, and an MTP segment.
//!
//! ```text
//! block, per indexed layer   keys   = latent cols 0..half of rows 32i..32i+32, f16 native
//!                            values = latent cols half..2·half of those rows ‖ pooled keys of their groups
//!                            the pool page's blobs (`read_block_all_layers`), from either source
//! state                      header (magic, format, row, per layer kind + len, draft rows)
//!                            then every array's dtype and shape, then their bytes: one fixed span
//! segment                    header, K/V [32, row] f16, pooled keys [32 / kpool, d] f16
//! ```
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::sync::Arc;

use mlx_paged_attn::{ColdCacheLayout, ColdLayerBlock};
use napi::Result;

use super::cache::{WINDOW_SLACK, WindowRingSnapshot};
use super::config::{Attn, HyV4FlashConfig};
use super::indexer::{IndexerGeometry, TokTail};
use super::paged::{BLOCK, PAGE_DTYPE, PageGeometry};
use super::prefill::BLOCK_ROWS;
use super::prefix_state::{DraftState, OpenRows, PrefixState, Segment, StateLayer};
use crate::array::MxArray;
use crate::disk_tier::binary_codec::{Reader, Sink, Span, Writer};

const STATE_MAGIC: &[u8; 4] = b"HY4S";
const SEGMENT_MAGIC: &[u8; 4] = b"HY4G";
/// Payload format of a state and a segment object.
pub const FORMAT: u32 = 1;

/// The cold layout of one block of the pool `g` describes.
pub fn block_layout(g: &PageGeometry) -> ColdCacheLayout {
    let plane = BLOCK as usize * g.half as usize * 2;
    ColdCacheLayout {
        block_size: BLOCK,
        num_layers: g.layers,
        num_kv_heads: 1,
        head_size: g.half,
        cache_dtype: format!("{PAGE_DTYPE:?}"),
        key_bytes_per_layer: plane,
        value_bytes_per_layer: plane + g.index_rows as usize * g.index_dim as usize * 2,
    }
}

/// One layer's page blobs from the f16 bits of a block's latent rows
/// `latent [32, 2·half]` and its pooled keys `pooled [index_rows, d]`.
pub fn split_block(latent: &[u16], pooled: &[u16], half: usize) -> ColdLayerBlock {
    let rows = latent.len() / (2 * half).max(1);
    let mut keys = Vec::with_capacity(rows * half * 2);
    let mut values = Vec::with_capacity(rows * half * 2 + pooled.len() * 2);
    for row in latent.chunks_exact(2 * half) {
        for &v in &row[..half] {
            keys.extend_from_slice(&v.to_ne_bytes());
        }
        for &v in &row[half..] {
            values.extend_from_slice(&v.to_ne_bytes());
        }
    }
    for &v in pooled {
        values.extend_from_slice(&v.to_ne_bytes());
    }
    ColdLayerBlock { keys, values }
}

/// The inverse of [`split_block`]: `(latent, pooled)` f16 bits of a block
/// of `g`; `None` when the blobs are not of its sizes.
pub fn join_block(layer: &ColdLayerBlock, g: &PageGeometry) -> Option<(Vec<u16>, Vec<u16>)> {
    let half = g.half as usize;
    let plane = BLOCK as usize * half * 2;
    let index = g.index_rows as usize * g.index_dim as usize * 2;
    if layer.keys.len() != plane || layer.values.len() != plane + index {
        return None;
    }
    let bits = |b: &[u8]| -> Vec<u16> {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_ne_bytes(*c))
            .collect()
    };
    let (k, v) = (bits(&layer.keys), bits(&layer.values[..plane]));
    let mut latent = Vec::with_capacity(BLOCK as usize * 2 * half);
    for (kr, vr) in k.chunks_exact(half).zip(v.chunks_exact(half)) {
        latent.extend_from_slice(kr);
        latent.extend_from_slice(vr);
    }
    Some((latent, bits(&layer.values[plane..])))
}

/// What one trunk layer of a state is.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerShape {
    Window { window: usize, width: i64 },
    Tail(IndexerGeometry),
}

/// The shape of a load's states: its trunk layers and, with the MTP layer,
/// the draft's indexer, K/V width and seed width.
#[derive(Debug, Clone, PartialEq)]
pub struct StateShape {
    pub layers: Vec<LayerShape>,
    pub draft: Option<DraftShape>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DraftShape {
    pub index: IndexerGeometry,
    pub width: i64,
    pub dim: i64,
}

impl StateShape {
    /// The states of `cfg`'s trunk, with the MTP layer's draft when `mtp`.
    pub fn of(cfg: &HyV4FlashConfig, mtp: bool) -> Result<Self> {
        let width = cfg.mla.kv_row() as i64;
        let layers = cfg
            .trunk()
            .iter()
            .enumerate()
            .map(|(l, lc)| match lc.attn {
                Attn::Sliding => Ok(LayerShape::Window {
                    window: cfg.window,
                    width,
                }),
                Attn::Indexed => IndexerGeometry::of(cfg, l).map(LayerShape::Tail),
            })
            .collect::<Result<Vec<_>>>()?;
        let draft = if mtp {
            Some(DraftShape {
                index: IndexerGeometry::of(cfg, cfg.n_layers)?,
                width,
                dim: cfg.dim as i64,
            })
        } else {
            None
        };
        Ok(Self { layers, draft })
    }

    /// The bytes one state record spans: room for the most rows any
    /// state of this shape holds.
    pub fn span_bytes(&self) -> u64 {
        let arrays = self.layers.len() + if self.draft.is_some() { 4 } else { 0 };
        let mut bytes = 64 + self.layers.len() * 9 + arrays * (3 + 2 * 8);
        for l in &self.layers {
            bytes += match l {
                LayerShape::Window { window, width } => {
                    (window + WINDOW_SLACK) * *width as usize * 2
                }
                LayerShape::Tail(g) => tail_keep(g) * g.head_dim * 4,
            };
        }
        if let Some(d) = &self.draft {
            let open = BLOCK_ROWS - 1;
            bytes += open * d.width as usize * 2
                + open / d.index.kpool.max(1) * d.index.head_dim * 2
                + tail_keep(&d.index) * d.index.head_dim * 4
                + d.dim as usize * 4;
        }
        bytes as u64
    }
}

/// Most `tok_k` rows a tail holds (`TokTail::keep`).
fn tail_keep(g: &IndexerGeometry) -> usize {
    g.kpool.saturating_sub(1) + WINDOW_SLACK
}

// ---------------------------------------------------------------------------
// States.
// ---------------------------------------------------------------------------

/// A state's arrays in their encoded order: each trunk layer's held rows,
/// then the draft's open K/V and pooled rows, its tail rows and its seed.
fn state_arrays(s: &PrefixState) -> Vec<Option<&MxArray>> {
    let mut arrays: Vec<Option<&MxArray>> = s
        .layers
        .iter()
        .map(|l| match l {
            StateLayer::Window(r) => r.rows(),
            StateLayer::Tail(t) => t.rows(),
        })
        .collect();
    if let Some(d) = &s.draft {
        arrays.push(d.open.as_ref().map(|o| &o.kv));
        arrays.push(d.open.as_ref().and_then(|o| o.pooled.as_ref()));
        arrays.push(d.tail.rows());
        arrays.push(Some(&d.seed));
    }
    arrays
}

fn encode_state_to<S: Sink>(out: &mut Writer<S>, s: &PrefixState) -> Result<bool> {
    out.put(STATE_MAGIC);
    out.u32(FORMAT);
    out.u64(s.row as u64);
    out.u32(s.layers.len() as u32);
    for l in &s.layers {
        let (kind, len) = match l {
            StateLayer::Window(r) => (0, r.len()),
            StateLayer::Tail(t) => (1, t.len()),
        };
        out.u8(kind);
        out.u64(len as u64);
    }
    match &s.draft {
        Some(d) => {
            out.u8(1);
            out.u64(d.segments.len() as u64);
            out.u64(d.tail.len() as u64);
        }
        None => out.u8(0),
    }
    out.arrays(&state_arrays(s), false)
}

/// `s`'s bytes written into `into`, zeros after them to its end. False when
/// they do not fit or a dtype has no code.
pub fn encode_state_into(s: &PrefixState, into: &mut [u8]) -> Result<bool> {
    let mut out = Writer(Span::new(into));
    if !encode_state_to(&mut out, s)? {
        return Ok(false);
    }
    Ok(out.into_inner().finish_zeroed())
}

/// The state of [`encode_state_into`] of a load of `shape`, its draft's
/// whole segments `segments`; `None` for bytes that are not one.
pub fn decode_state(
    bytes: &[u8],
    shape: &StateShape,
    segments: Vec<Arc<Segment>>,
) -> Option<PrefixState> {
    let mut r = Reader::new(bytes);
    if r.take(4)? != STATE_MAGIC || r.u32()? != FORMAT {
        return None;
    }
    let row = r.usize()?;
    if row == 0 || !row.is_multiple_of(BLOCK_ROWS) || r.u32()? as usize != shape.layers.len() {
        return None;
    }
    for _ in &shape.layers {
        let _kind = r.u8()?;
        if r.usize()? != row {
            return None;
        }
    }
    let draft = match r.u8()? {
        0 => None,
        1 => Some((r.usize()?, r.usize()?)),
        _ => return None,
    };
    if draft.is_some() != shape.draft.is_some() {
        return None;
    }
    let count = shape.layers.len() + if draft.is_some() { 4 } else { 0 };
    let mut arrays = r.arrays(count, false)?.into_iter();
    let mut layers = Vec::with_capacity(shape.layers.len());
    for l in &shape.layers {
        let rows = arrays.next()?;
        layers.push(match l {
            LayerShape::Window { window, width } => StateLayer::Window(
                WindowRingSnapshot::of_rows(*window, WINDOW_SLACK, *width, rows, row).ok()?,
            ),
            LayerShape::Tail(g) => StateLayer::Tail(TokTail::of_rows(g, rows, row).ok()?),
        });
    }
    let draft = match (draft, &shape.draft) {
        (Some((whole, m)), Some(ds)) => {
            let (kv, pooled, tail, seed) = (
                arrays.next()?,
                arrays.next()?,
                arrays.next()?,
                arrays.next()??,
            );
            if m + 1 != row || whole != segments.len() || whole != m / BLOCK_ROWS {
                return None;
            }
            let open_rows = m - whole * BLOCK_ROWS;
            let open = match kv {
                Some(kv) if open_rows > 0 && kv.shape_at(0).ok()? as usize == open_rows => {
                    Some(OpenRows { kv, pooled })
                }
                None if open_rows == 0 => None,
                _ => return None,
            };
            if seed.shape().ok()?.as_ref() != [1, ds.dim] {
                return None;
            }
            Some(DraftState {
                segments,
                open,
                tail: TokTail::of_rows(&ds.index, tail, m).ok()?,
                seed,
            })
        }
        (None, None) => None,
        _ => return None,
    };
    if arrays.next().is_some() {
        return None;
    }
    let state = PrefixState { row, layers, draft };
    let live = state_arrays(&state);
    let live: Vec<&MxArray> = live.into_iter().flatten().collect();
    if !live.is_empty() {
        MxArray::eval_arrays(&live).ok()?;
    }
    Some(state)
}

// ---------------------------------------------------------------------------
// MTP segments.
// ---------------------------------------------------------------------------

/// One segment's bytes.
pub fn encode_segment(s: &Segment) -> Result<Option<Vec<u8>>> {
    let mut out = Writer(Vec::new());
    out.put(SEGMENT_MAGIC);
    out.u32(FORMAT);
    out.u64(BLOCK_ROWS as u64);
    let ok = out.arrays(&[Some(&s.kv), Some(&s.pooled)], false)?;
    Ok(ok.then(|| out.into_inner()))
}

/// The inverse of [`encode_segment`] for the draft of `shape`.
pub fn decode_segment(bytes: &[u8], shape: &DraftShape) -> Option<Segment> {
    let mut r = Reader::new(bytes);
    if r.take(4)? != SEGMENT_MAGIC || r.u32()? != FORMAT || r.usize()? != BLOCK_ROWS {
        return None;
    }
    let mut arrays = r.arrays(2, false)?.into_iter();
    let (kv, pooled) = (arrays.next()??, arrays.next()??);
    if !r.at_end() {
        return None;
    }
    let per = (BLOCK_ROWS / shape.index.kpool.max(1)) as i64;
    let fits = kv.shape().ok()?.as_ref() == [BLOCK_ROWS as i64, shape.width]
        && pooled.shape().ok()?.as_ref() == [per, shape.index.head_dim as i64];
    if !fits {
        return None;
    }
    MxArray::eval_arrays(&[&kv, &pooled]).ok()?;
    Some(Segment { kv, pooled })
}
