//! A wave: one model step over the rows of one or more owners (sequences),
//! layer by layer. Each op runs as one of three groups:
//!
//! ```text
//! op          decode wave                                   prefill slice (one owner)
//! Engram      fused   table rows of every row, Q8 kv         fused
//! HcMix       fused   fused HC mix/reduce per row            fused
//! AttnQ8      fused   q_a, q_b, kv_a, k_b, v_b, gate, output fused
//! AttnCore    owner   cache write, indexer (its Q8 too),     owner
//!                     sparse attention on the owner's rows
//! Router      row     f32 router one row per call            fused
//! Experts     fused   one bank prepare over the union        fused
//! FfnQ8       fused   latent down / up, shared expert        fused
//! TrunkOut    fused   fused HC head + rms norm               fused
//! VocabHead   fused   native BF16 LM head / Q8 linear        fused
//! ```
//!
//! Fused Q8 calls see at least `MIN_Q8_ROWS` rows and a capped decode wave
//! at most `row_cap` rows, so every Q8 call of a decode row takes
//! `qmv_wide` and a row's bits do not depend on the other rows; the f32
//! matmuls round by row count, so they run one row per call. A decode wave
//! is one FFN chunk: the router scores of every row are read back once per
//! layer.

use napi::{Error, Result};

use super::attention::Projected;
use super::block::Block;
use super::model::Want;
use super::moe::{FFN_CHUNK_ROWS, RowKind};
use super::weights::LayerWeights;
use crate::array::MxArray;

/// Rows every Q8 (`Dense::Qmm`) call takes: MLX runs 1 row through `qmv`
/// and 2 or more through `qmv_wide` (`backend/metal/quantized.cpp`).
pub const MIN_Q8_ROWS: i64 = 2;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 wave: {msg}"))
}

/// How an op runs over a wave's rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group {
    /// One call over every row.
    Fused,
    /// One call per owner, on its rows and its state.
    Owner,
    /// One call per row.
    Row,
}

/// The ops of a layer step whose grouping the code follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Engram,
    HcMix,
    AttnQ8,
    AttnCore,
    Router,
    Experts,
    FfnQ8,
    TrunkOut,
    VocabHead,
}

/// Every op, in layer order.
pub const OPS: [Op; 9] = [
    Op::Engram,
    Op::HcMix,
    Op::AttnQ8,
    Op::AttnCore,
    Op::Router,
    Op::Experts,
    Op::FfnQ8,
    Op::TrunkOut,
    Op::VocabHead,
];

/// The group of `op` in a decode wave.
pub const fn decode_group(op: Op) -> Group {
    match op {
        Op::Engram
        | Op::HcMix
        | Op::AttnQ8
        | Op::Experts
        | Op::FfnQ8
        | Op::TrunkOut
        | Op::VocabHead => Group::Fused,
        Op::AttnCore => Group::Owner,
        Op::Router => Group::Row,
    }
}

/// The group of `op` for rows of `kind`: a prefill slice is one owner, so
/// only its attention core is the owner's.
pub const fn group(op: Op, kind: RowKind) -> Group {
    match (kind, op) {
        (RowKind::Decode, op) => decode_group(op),
        (RowKind::Prefill, Op::AttnCore) => Group::Owner,
        (RowKind::Prefill, _) => Group::Fused,
    }
}

/// One owner's rows of a wave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Position of the owner's first row.
    pub start: usize,
    pub rows: usize,
    /// Index of the owner's first row in the wave.
    pub lo: usize,
}

/// The rows of one step: what they are and whose they are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wave {
    kind: RowKind,
    spans: Vec<Span>,
    rows: usize,
}

impl Wave {
    /// A wave of `kind` over owners at `(start position, rows)`, in row
    /// order. A prefill slice takes exactly one owner; a decode wave holds
    /// at most `FFN_CHUNK_ROWS` rows, and at most `row_cap` when given.
    pub fn new(kind: RowKind, owners: &[(usize, usize)], row_cap: Option<usize>) -> Result<Self> {
        if owners.is_empty() {
            return Err(bad("a wave without owners"));
        }
        if kind == RowKind::Prefill && owners.len() != 1 {
            return Err(bad(format!(
                "a prefill slice runs alone, not with {} owners",
                owners.len()
            )));
        }
        if let Some(i) = owners.iter().position(|&(_, n)| n == 0) {
            return Err(bad(format!("owner {i} has no rows")));
        }
        let mut spans = Vec::with_capacity(owners.len());
        let mut rows = 0usize;
        for &(start, n) in owners {
            spans.push(Span {
                start,
                rows: n,
                lo: rows,
            });
            rows = rows
                .checked_add(n)
                .ok_or_else(|| bad("wave rows overflow usize"))?;
        }
        if kind == RowKind::Decode {
            if rows > FFN_CHUNK_ROWS {
                return Err(bad(format!(
                    "{rows} decode rows exceed one FFN chunk of {FFN_CHUNK_ROWS}"
                )));
            }
            if let Some(cap) = row_cap {
                if cap < MIN_Q8_ROWS as usize {
                    return Err(bad(format!("a row cap of {cap} holds no wave")));
                }
                if rows > cap {
                    return Err(bad(format!(
                        "{rows} decode rows exceed the row cap of {cap}"
                    )));
                }
            }
        }
        Ok(Self { kind, spans, rows })
    }

    pub fn kind(&self) -> RowKind {
        self.kind
    }

    /// Rows over every owner.
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn owners(&self) -> usize {
        self.spans.len()
    }

    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    pub fn group(&self, op: Op) -> Group {
        group(op, self.kind)
    }

    /// Every row's position, in row order.
    pub fn positions(&self) -> Vec<u64> {
        let mut pos = Vec::with_capacity(self.rows);
        for s in &self.spans {
            pos.extend((s.start..s.start + s.rows).map(|p| p as u64));
        }
        pos
    }

    /// Owner `owner`'s rows of `x [rows, ..]` along axis 0 (`x` itself for
    /// a one-owner wave).
    pub fn owner_rows(&self, x: &MxArray, owner: usize) -> Result<MxArray> {
        let s = self
            .spans
            .get(owner)
            .ok_or_else(|| bad(format!("owner {owner} of {}", self.spans.len())))?;
        let shape = x.shape()?.as_ref().to_vec();
        if shape.first().copied() != Some(self.rows as i64) {
            return Err(bad(format!(
                "array {shape:?} for a wave of {} rows",
                self.rows
            )));
        }
        if self.spans.len() == 1 {
            return Ok(x.clone());
        }
        let mut start = vec![0i64; shape.len()];
        let mut stop = shape;
        start[0] = s.lo as i64;
        stop[0] = (s.lo + s.rows) as i64;
        x.slice(&start, &stop)
    }
}

/// One owner's attention core of one layer.
pub struct Core<'a> {
    pub blk: &'a Block,
    pub w: &'a LayerWeights,
    /// The owner's rows of the layer's fused projections.
    pub p: &'a Projected,
    pub start: usize,
    pub kind: RowKind,
}

/// The owners of a wave, in row order: their calls and their states.
pub trait Owners {
    fn count(&self) -> usize;
    /// The ids the owner appends; empty for an owner out of range.
    fn ids(&self, owner: usize) -> &[u32];
    /// The owner's tokens before this call (as long as its start).
    fn history(&self, owner: usize) -> &[u32];
    fn want(&self, owner: usize) -> Want;
    /// Writes the owner's rows of `call.blk`'s layer to its state and
    /// returns `o_lat [rows, H, kv_lora]`. On error the caller puts the
    /// owner's state back.
    fn core(&mut self, owner: usize, call: Core<'_>) -> Result<MxArray>;
}
