//! Per-session attention state (never global).

use napi::{Error, Result};

use crate::array::MxArray;

/// The sliding-window KV of one layer: the last `window` rows (post-RoPE,
/// after the fp8 round trip), kept in position order. The reference keeps
/// the same rows in a ring with position `p` in slot `p % window`.
#[derive(Clone)]
pub struct WindowCache {
    window: usize,
    rows: Option<MxArray>,
    len: usize,
}

impl WindowCache {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            rows: None,
            len: 0,
        }
    }

    /// Positions appended so far (the next chunk starts here).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The kept rows `[min(len, window), head_dim]`, oldest first.
    pub fn rows(&self) -> Option<&MxArray> {
        self.rows.as_ref()
    }

    /// Appends `kv` `[n, head_dim]` at `start` (== `len()`). Returns the keys
    /// a chunk query may read, kept rows then `kv`, and the first key's
    /// position; masking the window per query is the caller's job.
    pub fn append(&mut self, kv: &MxArray, start: usize) -> Result<(MxArray, usize)> {
        if start != self.len {
            return Err(Error::from_reason(format!(
                "window cache: append at {start}, but {} positions are cached",
                self.len
            )));
        }
        let shape = kv.shape()?.as_ref().to_vec();
        let (n, hd) = (shape[0] as usize, shape[1]);
        let (keys, first) = match &self.rows {
            Some(r) => {
                let kept = r.shape_at(0)? as usize;
                (MxArray::concatenate(r, kv, 0)?, start - kept)
            }
            None => (kv.clone(), start),
        };
        let total = keys.shape_at(0)? as usize;
        let keep = total.min(self.window);
        // an independent copy: a slice view would pin the whole chunk's buffer
        let rows = keys
            .slice(&[(total - keep) as i64, 0], &[total as i64, hd])?
            .deep_copy()?;
        MxArray::eval_arrays_with_context(&[&rows], "deepseek_v41 window cache")?;
        self.rows = Some(rows);
        self.len = start + n;
        Ok((keys, first))
    }
}

/// Selected compressed rows per query row `[rows * k]` (row ids, ascending,
/// `-1` = none), as an index source publishes them (`shared_attn.topk_idxs`
/// without the window offset).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopK {
    pub rows: usize,
    pub k: usize,
    pub ids: Vec<i32>,
}

/// Kept candidate blocks per query row `[rows * n_blocks]`
/// (`shared_attn.candidates`, at block granularity).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidates {
    pub rows: usize,
    pub n_blocks: usize,
    pub block_size: usize,
    pub keep: Vec<bool>,
}

/// What one forward hands down the layer stack (the reference's
/// `SharedAttentionRuntime` minus the caches): the latest index source's
/// top-k and the candidate source's blocks. New for every forward call;
/// tests may inject or export either.
#[derive(Debug, Default, Clone)]
pub struct ForwardShared {
    pub topk: Option<TopK>,
    pub candidates: Option<Candidates>,
}

/// The compressed state a KV source layer owns, read by every layer that
/// uses it: compressed KV rows (post RoPE, after the latent round trip),
/// index keys (when the source owns them) and the compressor's tail.
#[derive(Clone)]
pub struct CompressedCache {
    pub ratio: usize,
    /// Positions consumed so far.
    pub len: usize,
    pub kv: Option<MxArray>,
    pub index_k: Option<MxArray>,
    pub tail: crate::models::deepseek_v41::compressor::CompressorTail,
}

impl CompressedCache {
    pub fn new(ratio: usize) -> Self {
        Self {
            ratio,
            len: 0,
            kv: None,
            index_k: None,
            tail: Default::default(),
        }
    }

    /// Compressed rows so far (`len / ratio`).
    pub fn rows(&self) -> usize {
        self.len / self.ratio
    }

    /// `old` with `rows` appended, materialized (errors surface here).
    pub(crate) fn appended(old: &Option<MxArray>, rows: &MxArray) -> Result<MxArray> {
        let next = match old {
            Some(old) => MxArray::concatenate(old, rows, 0)?,
            None => rows.clone(),
        };
        MxArray::eval_arrays_with_context(&[&next], "deepseek_v41 compressed cache")?;
        Ok(next)
    }
}
