//! Engram table rows (F8_E4M3, `head_dim` bytes per row) read by `pread`
//! through the load's one shard set (`F_NOCACHE` + `F_RDAHEAD 0`; never
//! mapped, never a Metal buffer). A call reads each distinct row once, as a 16
//! KiB-aligned window, on up to 8 threads (decode) or 32 (prefill).
//!
//! The file says type 43, which mlx-node parses as MXFP8. The table is read
//! as F8_E4M3 only when its bytes to the end of the shard are exactly
//! `rows · head_dim`; the MXFP8 size (`rows · head_dim / 32 · 33`) and any
//! other size are refused.

#[cfg(test)]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use napi::{Error, Result};

use super::f8::decode_f8_rows;
#[cfg(test)]
use super::scan::Scan;
use super::scan::{ENGRAM_TABLE, Located};
use crate::models::deepseek_v41::EngramConfig;
use crate::moe_stream::io::{AlignedBuf, Extent, ShardSet};
use crate::utils::gguf::GgufTensorType;

const ALIGN: u64 = 16 << 10;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 Engram rows: {msg}"))
}

/// Decoded rows `[ids.len(), head_dim]` (f32, exact) for global row ids of
/// the one table, laid out `[pos][col]` for Engram layer `layer` (its index
/// in the Engram layer list).
pub trait EngramRows {
    fn rows(&self, layer: usize, ids: &[u32]) -> Result<Vec<f32>>;
}

/// Accepts `extent` (bytes from the table start to the shard end) only as an
/// F8_E4M3 table of `n_rows × head_dim`.
pub fn prove_f8_extent(extent: u64, n_rows: u64, head_dim: usize) -> Result<()> {
    if n_rows == 0 || head_dim == 0 {
        return Err(bad(format!("empty table [{n_rows}, {head_dim}]")));
    }
    let f8 = n_rows
        .checked_mul(head_dim as u64)
        .ok_or_else(|| bad("table size overflows u64"))?;
    if extent == f8 {
        return Ok(());
    }
    let mxfp8 = head_dim
        .is_multiple_of(32)
        .then(|| n_rows.checked_mul(head_dim as u64 / 32 * 33))
        .flatten();
    if mxfp8 == Some(extent) {
        return Err(bad(format!(
            "{extent} bytes is an MXFP8 table [{n_rows}, {head_dim}], not F8_E4M3"
        )));
    }
    Err(bad(format!(
        "{extent} bytes to the end of the shard; F8_E4M3 [{n_rows}, {head_dim}] is {f8}"
    )))
}

/// Which global rows each (layer, column) may address.
struct Columns {
    cols: usize,
    n_rows: u64,
    offsets: Vec<u64>,
    sizes: Vec<u64>,
}

impl Columns {
    fn new(cfg: &EngramConfig, n_rows: u64) -> Result<Self> {
        let cols = cfg.n_cols();
        let want = cols * cfg.layers.len();
        if cols == 0 || cfg.head_offsets.len() < want || cfg.head_vocab_sizes.len() < want {
            return Err(bad("head offsets / sizes do not cover every column"));
        }
        if n_rows > u64::from(u32::MAX) + 1 {
            return Err(bad(format!("{n_rows} rows exceed u32 row ids")));
        }
        Ok(Self {
            cols,
            n_rows,
            offsets: cfg.head_offsets.clone(),
            sizes: cfg.head_vocab_sizes.clone(),
        })
    }

    fn check(&self, layer: usize, ids: &[u32]) -> Result<()> {
        if !ids.len().is_multiple_of(self.cols) {
            return Err(bad(format!("{} ids for {} columns", ids.len(), self.cols)));
        }
        for (i, &id) in ids.iter().enumerate() {
            let g = layer * self.cols + i % self.cols;
            let (lo, n) = match (self.offsets.get(g), self.sizes.get(g)) {
                (Some(&lo), Some(&n)) => (lo, n),
                _ => return Err(bad(format!("Engram layer {layer} has no column {g}"))),
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
        Ok(())
    }
}

/// The real table, read by row from its shard.
pub struct F8Table {
    shards: Arc<ShardSet>,
    shard: usize,
    /// File offset of row 0.
    base: u64,
    file_len: u64,
    head_dim: usize,
    columns: Columns,
    bytes: AtomicU64,
    ns: AtomicU64,
}

impl F8Table {
    #[cfg(test)]
    pub fn open(scan: &Scan) -> Result<Self> {
        Self::open_in(
            Arc::new(ShardSet::open(&scan.paths)?),
            &scan.engram_table,
            &scan.cfg.engram,
        )
    }

    /// `loc` is `per_layer_token_embd.weight` in the shard at `path`.
    #[cfg(test)]
    pub(crate) fn open_at(path: &Path, loc: &Located, cfg: &EngramConfig) -> Result<Self> {
        Self::on_shard(Arc::new(ShardSet::open(&[path])?), 0, loc, cfg)
    }

    /// `loc` is `per_layer_token_embd.weight` in shard `loc.shard` of
    /// `shards`, the descriptors every other reader of the load uses.
    pub fn open_in(shards: Arc<ShardSet>, loc: &Located, cfg: &EngramConfig) -> Result<Self> {
        Self::on_shard(shards, loc.shard, loc, cfg)
    }

    fn on_shard(
        shards: Arc<ShardSet>,
        shard: usize,
        loc: &Located,
        cfg: &EngramConfig,
    ) -> Result<Self> {
        let n_rows = table_rows(loc, cfg)?;
        let columns = Columns::new(cfg, n_rows)?;
        let (file, path) = shards
            .file(shard)
            .zip(shards.path(shard))
            .ok_or_else(|| bad(format!("no shard {shard} of {}", shards.len())))?;
        let file_len = file
            .metadata()
            .map_err(|e| bad(format!("stat '{}': {e}", path.display())))?
            .len();
        let base = loc.abs_offset()?;
        let extent = file_len.checked_sub(base).ok_or_else(|| {
            bad(format!(
                "'{}': table starts at {base}, past the end ({file_len})",
                path.display()
            ))
        })?;
        prove_f8_extent(extent, n_rows, cfg.head_dim)
            .map_err(|e| bad(format!("'{}': {}", path.display(), e.reason)))?;
        Ok(Self {
            shards,
            shard,
            base,
            file_len,
            head_dim: cfg.head_dim,
            columns,
            bytes: AtomicU64::new(0),
            ns: AtomicU64::new(0),
        })
    }

    pub fn n_rows(&self) -> u64 {
        self.columns.n_rows
    }

    /// Bytes read from disk and nanoseconds spent, since open.
    pub fn read_stats(&self) -> (u64, u64) {
        (
            self.bytes.load(Ordering::Relaxed),
            self.ns.load(Ordering::Relaxed),
        )
    }

    /// Decoded rows `[ids.len(), head_dim]` for any global row ids, in order.
    pub fn read_rows(&self, ids: &[u32]) -> Result<Vec<f32>> {
        if let Some(&id) = ids.iter().find(|&&id| u64::from(id) >= self.columns.n_rows) {
            return Err(bad(format!(
                "row {id} outside the table ({} rows)",
                self.columns.n_rows
            )));
        }
        let mut uniq = ids.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        let t0 = std::time::Instant::now();
        let threads = if ids.len() <= 2 * self.columns.cols {
            8
        } else {
            32
        };
        let dec = self.read_distinct(&uniq, threads)?;
        self.ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let hd = self.head_dim;
        let mut out = vec![0f32; ids.len() * hd];
        for (&id, o) in ids.iter().zip(out.chunks_mut(hd)) {
            let k = uniq.binary_search(&id).map_err(|_| bad("lost row"))?;
            o.copy_from_slice(&dec[k * hd..(k + 1) * hd]);
        }
        Ok(out)
    }

    /// `uniq` sorted, distinct, in range.
    fn read_distinct(&self, uniq: &[u32], threads: usize) -> Result<Vec<f32>> {
        let hd = self.head_dim;
        let mut dec = vec![0f32; uniq.len() * hd];
        let per = uniq.len().div_ceil(threads).max(1);
        std::thread::scope(|sc| {
            let jobs: Vec<_> = uniq
                .chunks(per)
                .zip(dec.chunks_mut(per * hd))
                .map(|(rows, out)| sc.spawn(move || self.read_chunk(rows, out)))
                .collect();
            jobs.into_iter()
                .try_for_each(|j| j.join().unwrap_or_else(|_| Err(bad("reader panicked"))))
        })?;
        Ok(dec)
    }

    fn read_chunk(&self, rows: &[u32], out: &mut [f32]) -> Result<()> {
        let hd = self.head_dim;
        let mut buf = AlignedBuf::new();
        let mut bytes = 0;
        for (&r, o) in rows.iter().zip(out.chunks_mut(hd)) {
            let e = Extent {
                shard: self.shard,
                offset: self.base + u64::from(r) * hd as u64,
                len: hd as u64,
            };
            let codes = self.shards.read_aligned(e, &mut buf)?;
            decode_f8_rows(codes, o).map_err(|err| bad(format!("row {r}: {}", err.reason)))?;
            bytes += window_len(e, self.file_len);
        }
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }
}

impl EngramRows for F8Table {
    fn rows(&self, layer: usize, ids: &[u32]) -> Result<Vec<f32>> {
        self.columns.check(layer, ids)?;
        self.read_rows(ids)
    }
}

/// The whole table held in memory (tiny fixtures; the real 63 GB table is
/// read by row).
pub struct ResidentF8Table {
    codes: Vec<u8>,
    head_dim: usize,
    columns: Columns,
}

impl ResidentF8Table {
    pub fn new(codes: Vec<u8>, cfg: &EngramConfig) -> Result<Self> {
        let n_rows = cfg.head_vocab_sizes.iter().sum::<u64>();
        prove_f8_extent(codes.len() as u64, n_rows, cfg.head_dim)?;
        Ok(Self {
            codes,
            head_dim: cfg.head_dim,
            columns: Columns::new(cfg, n_rows)?,
        })
    }
}

impl EngramRows for ResidentF8Table {
    fn rows(&self, layer: usize, ids: &[u32]) -> Result<Vec<f32>> {
        self.columns.check(layer, ids)?;
        let hd = self.head_dim;
        let mut out = vec![0f32; ids.len() * hd];
        for (&id, o) in ids.iter().zip(out.chunks_mut(hd)) {
            let at = id as usize * hd;
            decode_f8_rows(&self.codes[at..at + hd], o)
                .map_err(|err| bad(format!("row {id}: {}", err.reason)))?;
        }
        Ok(out)
    }
}

/// Name, type 43 and ggml dims `[head_dim, Σ head_vocab_sizes]`; the row
/// count.
fn table_rows(loc: &Located, cfg: &EngramConfig) -> Result<u64> {
    let info = &loc.info;
    if info.name != ENGRAM_TABLE {
        return Err(bad(format!("'{}' is not {ENGRAM_TABLE}", info.name)));
    }
    if info.tensor_type != GgufTensorType::MXFP8 {
        return Err(bad(format!(
            "{}: type {} is not GGUF type 43",
            info.name,
            info.tensor_type.name()
        )));
    }
    let n_rows = cfg.head_vocab_sizes.iter().sum::<u64>();
    if info.dims != [cfg.head_dim as u64, n_rows] {
        return Err(bad(format!(
            "{}: ggml dims {:?}, expected [{}, {n_rows}]",
            info.name, info.dims, cfg.head_dim
        )));
    }
    Ok(n_rows)
}

/// Bytes of the window `read_aligned` reads for `e`, clipped at the file end.
fn window_len(e: Extent, file_len: u64) -> u64 {
    let start = e.offset - e.offset % ALIGN;
    let end = (e.offset + e.len).next_multiple_of(ALIGN).min(file_len);
    end - start
}
