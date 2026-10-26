//! Engram block (`model.py` `Engram`): MXFP8 table rows of a token's n-gram
//! hashes -> `wkv` -> one key per hc copy plus a shared value, written into
//! the residual stream through a per-copy gate.

use napi::{Error, Result};

use super::attention::linear;
use super::ops::engram_gate;
use crate::array::MxArray;
use crate::utils::gguf::decode_mxfp8_rows;

/// Bytes of one MXFP8 table row of `head_dim` values (33 B per 32).
pub fn row_bytes(head_dim: usize) -> usize {
    head_dim / 32 * 33
}

/// Where table rows come from: decoded rows `[ids.len(), head_dim]` (f32,
/// exact) for global row ids of the one GGUF table.
pub trait EngramRows {
    fn rows(&self, ids: &[u32]) -> Result<Vec<f32>>;
}

/// The whole table held in memory (tiny fixture). The real 203 GB table is
/// read by row (S11), never held or mapped.
pub struct ResidentTable {
    raw: Vec<u8>,
    head_dim: usize,
}

impl ResidentTable {
    pub fn new(raw: Vec<u8>, head_dim: usize) -> Result<Self> {
        if head_dim == 0
            || !head_dim.is_multiple_of(32)
            || !raw.len().is_multiple_of(row_bytes(head_dim))
        {
            return Err(Error::from_reason(format!(
                "engram table: {} bytes do not hold rows of {head_dim}",
                raw.len()
            )));
        }
        Ok(Self { raw, head_dim })
    }
}

impl EngramRows for ResidentTable {
    fn rows(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let rb = row_bytes(self.head_dim);
        let n_rows = self.raw.len() / rb;
        let mut out = vec![0f32; ids.len() * self.head_dim];
        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= n_rows {
                return Err(Error::from_reason(format!(
                    "engram row {id} outside the table ({n_rows} rows)"
                )));
            }
            decode_mxfp8_rows(
                &self.raw[id * rb..(id + 1) * rb],
                &mut out[i * self.head_dim..(i + 1) * self.head_dim],
            )?;
        }
        Ok(out)
    }
}

/// One engram layer's weights: `wkv` `[dim * (hc + 1), cols * head_dim]`,
/// `gate` `[hc, dim]` (the stored `q_weight * k_weight` product).
pub struct Engram {
    wkv: MxArray,
    gate: MxArray,
    dim: i64,
    hc: i64,
    eps: f32,
}

impl Engram {
    pub fn from_gguf(
        layer: usize,
        dim: usize,
        hc: usize,
        eps: f32,
        mut get: impl FnMut(&str) -> Result<MxArray>,
    ) -> Result<Self> {
        Ok(Self {
            wkv: get(&format!("blk.{layer}.engram_kv.weight"))?,
            gate: get(&format!("blk.{layer}.engram_gate.weight"))?,
            dim: dim as i64,
            hc: hc as i64,
            eps,
        })
    }

    /// `x` `[n, hc, dim]` with this layer's decoded table rows `[n * cols *
    /// head_dim]` (A3: rows enter in the activation dtype, exact) -> `x` with
    /// the gated value added, in `x`'s dtype.
    pub fn forward(&self, x: &MxArray, rows: &[f32]) -> Result<MxArray> {
        let n = x.shape_at(0)?;
        let width = self.wkv.shape_at(1)?;
        if rows.len() as i64 != n * width {
            return Err(Error::from_reason(format!(
                "engram: {} row values for {n} positions of width {width}",
                rows.len()
            )));
        }
        let emb = MxArray::from_float32(rows, &[n, width])?.astype(x.dtype()?)?;
        let kv = linear(&emb, &self.wkv)?;
        let hd = self.hc * self.dim;
        let key = kv.slice(&[0, 0], &[n, hd])?;
        let value = kv.slice(&[0, hd], &[n, hd + self.dim])?;
        engram_gate(x, &key, &value, &self.gate, self.eps)
    }
}
