//! Per-session attention state (never global).

use napi::{Error, Result};

use crate::array::MxArray;

/// The sliding-window KV of one layer: the last `window` rows (post-RoPE,
/// after the fp8 round trip), kept in position order. The reference keeps
/// the same rows in a ring with position `p` in slot `p % window`.
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
        let rows = keys.slice(&[(total - keep) as i64, 0], &[total as i64, hd])?;
        rows.eval();
        self.rows = Some(rows);
        self.len = start + n;
        Ok((keys, first))
    }
}
