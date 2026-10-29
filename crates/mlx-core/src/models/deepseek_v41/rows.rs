//! Engram table rows read by `pread` from the GGUF table's own shard (its
//! own `F_NOCACHE` + `F_RDAHEAD 0` descriptor; never mapped, never a Metal
//! buffer). A call reads each distinct row once, as a 16 KiB-aligned window,
//! on up to 8 threads (decode) or 32 (prefill).

use std::path::Path;
use std::sync::atomic::Ordering;

use napi::{Error, Result};

use super::config::EngramConfig;
use super::engram::{EngramRows, row_bytes};
use super::stats::{ENGRAM_BYTES, ENGRAM_NS};
use crate::moe_stream::io::{AlignedBuf, Extent, ShardSet};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType, decode_mxfp8_rows};

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("deepseek_v41 engram rows: {msg}"))
}

pub struct PreadTable {
    shard: ShardSet,
    /// File offset of row 0.
    base: u64,
    n_rows: u64,
    head_dim: usize,
    cols: usize,
    offsets: Vec<u64>,
    sizes: Vec<u64>,
}

impl PreadTable {
    /// `info` is `per_layer_token_embd.weight` (MXFP8 `[rows, head_dim]`) in
    /// the shard at `path`, whose data section starts at `data_offset`.
    pub fn open(
        path: &Path,
        data_offset: u64,
        info: &GgufTensorInfo,
        cfg: &EngramConfig,
    ) -> Result<Self> {
        let shape = info.mlx_shape();
        if info.tensor_type != GgufTensorType::MXFP8
            || shape.len() != 2
            || shape[1] as usize != cfg.head_dim
            || info.data_size()? != shape[0] as u64 * row_bytes(cfg.head_dim) as u64
        {
            return Err(bad(format!(
                "{}: {} {shape:?} is not an MXFP8 table of width {}",
                info.name,
                info.tensor_type.name(),
                cfg.head_dim
            )));
        }
        let cols = cfg.n_cols();
        let want = cols * cfg.layers.len();
        if cfg.head_offsets.len() < want || cfg.head_vocab_sizes.len() < want {
            return Err(bad("head offsets / sizes shorter than the columns"));
        }
        Ok(Self {
            shard: ShardSet::open(&[path])?,
            base: data_offset + info.offset,
            n_rows: shape[0] as u64,
            head_dim: cfg.head_dim,
            cols,
            offsets: cfg.head_offsets.clone(),
            sizes: cfg.head_vocab_sizes.clone(),
        })
    }
}

impl EngramRows for PreadTable {
    fn rows(&self, layer: usize, ids: &[u32]) -> Result<Vec<f32>> {
        if !ids.len().is_multiple_of(self.cols) {
            return Err(bad(format!("{} ids for {} columns", ids.len(), self.cols)));
        }
        for (i, &id) in ids.iter().enumerate() {
            let g = layer * self.cols + i % self.cols;
            let (lo, n) = match (self.offsets.get(g), self.sizes.get(g)) {
                (Some(&lo), Some(&n)) => (lo, n),
                _ => return Err(bad(format!("engram layer {layer} has no column {g}"))),
            };
            let id = u64::from(id);
            if id < lo || id >= lo + n || id >= self.n_rows {
                return Err(bad(format!(
                    "row {id} outside column {g}'s [{lo}, {}) (table {} rows)",
                    lo + n,
                    self.n_rows
                )));
            }
        }
        let mut uniq = ids.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        let (rb, hd) = (row_bytes(self.head_dim), self.head_dim);
        let mut dec = vec![0f32; uniq.len() * hd];
        let t0 = std::time::Instant::now();
        let bytes = std::sync::atomic::AtomicU64::new(0);
        let bytes = &bytes;
        let threads = if ids.len() <= 2 * self.cols { 8 } else { 32 };
        let per = uniq.len().div_ceil(threads).max(1);
        std::thread::scope(|sc| {
            let jobs: Vec<_> = uniq
                .chunks(per)
                .zip(dec.chunks_mut(per * hd))
                .map(|(rows, out)| {
                    sc.spawn(move || -> Result<()> {
                        let mut buf = AlignedBuf::new();
                        for (&r, o) in rows.iter().zip(out.chunks_mut(hd)) {
                            let e = Extent {
                                shard: 0,
                                offset: self.base + u64::from(r) * rb as u64,
                                len: rb as u64,
                            };
                            let w = self.shard.read_aligned(e, &mut buf)?;
                            decode_mxfp8_rows(w, o)?;
                            bytes.fetch_add(window_len(e), Ordering::Relaxed);
                        }
                        Ok(())
                    })
                })
                .collect();
            jobs.into_iter()
                .try_for_each(|j| j.join().unwrap_or_else(|_| Err(bad("reader panicked"))))
        })?;
        ENGRAM_BYTES.fetch_add(bytes.load(Ordering::Relaxed), Ordering::Relaxed);
        ENGRAM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let mut out = vec![0f32; ids.len() * hd];
        for (&id, o) in ids.iter().zip(out.chunks_mut(hd)) {
            let k = uniq.binary_search(&id).map_err(|_| bad("lost row"))?;
            o.copy_from_slice(&dec[k * hd..(k + 1) * hd]);
        }
        Ok(out)
    }
}

/// Bytes of the 16 KiB-aligned window `read_aligned` reads for `e` (the
/// file-end clip is ignored: table rows sit far from the shard end).
fn window_len(e: Extent) -> u64 {
    const ALIGN: u64 = 16 << 10;
    let start = e.offset - e.offset % ALIGN;
    (e.offset + e.len).next_multiple_of(ALIGN) - start
}
