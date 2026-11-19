//! Per-sequence attention state, f16 latent rows `[kv_norm | k_pe]`:
//!
//! | layer | state |
//! |---|---|
//! | sliding | `WindowRing`: the last `window + slack` rows, a fresh array per append |
//! | indexed | `LatentBuf`: one `[cap, width]` array written in place, a logical `len` |
//!
//! A `LatentBuf` array must have exactly one handle (this struct): a second
//! handle turns every in-place write into a copy of the whole buffer. Save
//! and restore go through lengths (`len` / `truncate`), never array clones.

use napi::{Error, Result};

use super::config::Attn;
use crate::array::{DType, MxArray};

/// Rows a `WindowRing` keeps past the window, so a speculative append of up
/// to this many rows can be dropped exactly.
pub const WINDOW_SLACK: usize = 3;
/// Cache dtype of every latent row.
pub const ROW_DTYPE: DType = DType::Float16;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 attention cache: {msg}"))
}

fn rows_of(x: &MxArray, width: i64, what: &str) -> Result<usize> {
    let d = x.shape()?.as_ref().to_vec();
    match d.as_slice() {
        &[n, w] if w == width && n > 0 => Ok(n as usize),
        s => Err(bad(format!("{what}: rows {s:?}, need [n > 0, {width}]"))),
    }
}

/// The latent rows of one sliding layer: positions `len - kept .. len`,
/// oldest first. A clone is a snapshot (every append makes a new array).
#[derive(Clone)]
pub struct WindowRing {
    window: usize,
    keep: usize,
    width: i64,
    rows: Option<MxArray>,
    held: usize,
    len: usize,
}

impl WindowRing {
    pub fn new(window: usize, slack: usize, width: i64) -> Self {
        Self {
            window,
            keep: window + slack,
            width,
            rows: None,
            held: 0,
            len: 0,
        }
    }

    /// A ring of `len` positions holding `rows [held, width]` (f16), the
    /// rows of its last `held` positions. Refused past `window + slack`
    /// rows, past `len`, or below a window of rows (all of `len` when
    /// shorter).
    pub fn of_rows(
        window: usize,
        slack: usize,
        width: i64,
        rows: Option<MxArray>,
        len: usize,
    ) -> Result<Self> {
        let held = match &rows {
            None => 0,
            Some(r) => rows_of(r, width, "window rows")?,
        };
        if held > window + slack || held > len || held < len.min(window) {
            return Err(bad(format!(
                "a ring of {len} positions with {held} rows (window {window}, slack {slack})"
            )));
        }
        Ok(Self {
            window,
            keep: window + slack,
            width,
            rows: rows.map(|r| r.astype(ROW_DTYPE)).transpose()?,
            held,
            len,
        })
    }

    /// The held rows `[kept, width]` f16.
    pub fn rows(&self) -> Option<&MxArray> {
        self.rows.as_ref()
    }

    /// Positions appended so far.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn window(&self) -> usize {
        self.window
    }

    /// Rows held now.
    pub fn kept(&self) -> usize {
        self.held
    }

    /// Appends `kv [n, width]` at `start` (== `len()`), stored as f16.
    /// Returns the keys a query of this call may read (kept rows, then the
    /// new rows) and the position of the first one.
    pub fn append(&mut self, kv: &MxArray, start: usize) -> Result<(MxArray, usize)> {
        if start != self.len {
            return Err(bad(format!(
                "window append at {start}, {} positions held",
                self.len
            )));
        }
        let n = rows_of(kv, self.width, "window append")?;
        let kv = kv.astype(ROW_DTYPE)?;
        let kept = self.kept();
        let keys = match &self.rows {
            Some(r) => MxArray::concatenate(r, &kv, 0)?,
            None => kv,
        };
        let total = kept + n;
        let hold = total.min(self.keep);
        // an independent copy: a slice view would pin the whole chunk's buffer
        let rows = keys
            .slice(&[(total - hold) as i64, 0], &[total as i64, self.width])?
            .deep_copy()?;
        MxArray::eval_arrays_with_context(&[&rows], "hy_v4_flash window ring")?;
        self.rows = Some(rows);
        self.held = hold;
        self.len = start + n;
        Ok((keys, start - kept))
    }

    /// Drops the positions past `len`. Refused when fewer than a window of
    /// rows (or all of `len`) would remain held.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.len {
            return Err(bad(format!(
                "window truncate to {len}, {} positions held",
                self.len
            )));
        }
        if len == self.len {
            return Ok(());
        }
        if len == 0 {
            self.rows = None;
            self.held = 0;
            self.len = 0;
            return Ok(());
        }
        let drop = self.len - len;
        let kept = self.kept();
        let left = kept.saturating_sub(drop);
        if left < len.min(self.window) {
            return Err(bad(format!(
                "window truncate {} -> {len} drops {drop} of {kept} held rows",
                self.len
            )));
        }
        let rows = self
            .rows
            .as_ref()
            .ok_or_else(|| bad("window rows missing"))?
            .slice(&[0, 0], &[left as i64, self.width])?
            .deep_copy()?;
        MxArray::eval_arrays_with_context(&[&rows], "hy_v4_flash window ring")?;
        self.rows = Some(rows);
        self.held = left;
        self.len = len;
        Ok(())
    }

    pub fn nbytes(&self) -> usize {
        self.rows.as_ref().map_or(0, MxArray::nbytes)
    }
}

/// The latent rows of one indexed layer: one f16 `[cap, width]` array made
/// at the first write and kept until drop; rows `len..cap` are dead and are
/// overwritten by later writes.
pub struct LatentBuf {
    cap: usize,
    width: i64,
    buf: Option<MxArray>,
    len: usize,
}

impl LatentBuf {
    pub fn new(cap: usize, width: i64) -> Self {
        Self {
            cap,
            width,
            buf: None,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn width(&self) -> i64 {
        self.width
    }

    /// The whole buffer `[cap, width]`; only rows `0..len` are live. A
    /// caller must not keep the handle (or a view of it) past its call.
    pub fn rows(&self) -> Result<&MxArray> {
        self.buf
            .as_ref()
            .ok_or_else(|| bad("latent buffer read before its first write"))
    }

    /// Writes `kv [n, width]` (stored as f16) to rows `start..start + n`
    /// in place; `start` must equal `len()`.
    pub fn write(&mut self, kv: &MxArray, start: usize) -> Result<()> {
        if start != self.len {
            return Err(bad(format!(
                "latent write at {start}, {} positions held",
                self.len
            )));
        }
        let n = rows_of(kv, self.width, "latent write")?;
        let end = start + n;
        if end > self.cap {
            return Err(bad(format!(
                "latent write to {end} past the context of {}",
                self.cap
            )));
        }
        let rows = kv.astype(ROW_DTYPE)?;
        if self.buf.is_none() {
            let cap = i64::try_from(self.cap).map_err(|_| bad("context exceeds i64"))?;
            let b = MxArray::zeros(&[cap, self.width], Some(ROW_DTYPE))?;
            MxArray::eval_arrays_with_context(&[&b], "hy_v4_flash latent alloc")?;
            self.buf = Some(b);
        }
        let buf = self
            .buf
            .as_mut()
            .ok_or_else(|| bad("latent buffer missing"))?;
        buf.slice_assign_axis_inplace(0, start as i64, end as i64, &rows)?;
        MxArray::eval_arrays_with_context(&[&*buf], "hy_v4_flash latent write")?;
        self.len = end;
        Ok(())
    }

    /// Forgets the positions past `len`; the buffer stays.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.len {
            return Err(bad(format!(
                "latent truncate to {len}, {} positions held",
                self.len
            )));
        }
        self.len = len;
        Ok(())
    }

    pub fn nbytes(&self) -> usize {
        self.buf.as_ref().map_or(0, MxArray::nbytes)
    }
}

/// The attention state of one layer.
pub enum LayerCache {
    Window(WindowRing),
    Latent(LatentBuf),
}

impl LayerCache {
    /// Empty state for a layer of kind `attn`; `cap` = the admitted context.
    pub fn new(attn: Attn, window: usize, cap: usize, width: i64) -> Self {
        match attn {
            Attn::Sliding => Self::Window(WindowRing::new(window, WINDOW_SLACK, width)),
            Attn::Indexed => Self::Latent(LatentBuf::new(cap, width)),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Window(r) => r.len(),
            Self::Latent(b) => b.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn truncate(&mut self, len: usize) -> Result<()> {
        match self {
            Self::Window(r) => r.truncate(len),
            Self::Latent(b) => b.truncate(len),
        }
    }

    pub fn nbytes(&self) -> usize {
        match self {
            Self::Window(r) => r.nbytes(),
            Self::Latent(b) => b.nbytes(),
        }
    }
}
