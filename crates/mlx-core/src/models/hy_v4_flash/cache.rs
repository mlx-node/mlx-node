//! Per-sequence attention state, f16 latent rows `[kv_norm | k_pe]`:
//!
//! | layer | state |
//! |---|---|
//! | sliding | `WindowRing`: fixed `[window + slack, width]` circular ring written in place |
//! | indexed | `LatentBuf`: one `[cap, width]` array written in place, a logical `len` |
//!
//! Live `WindowRing` and `LatentBuf` buffers must have exactly one handle: a
//! second live reference prevents MLX buffer donation and forces a full buffer
//! copy on the next write. Snapshots go through `WindowRingSnapshot` (linear
//! oldest-first copy); short rollbacks go through lengths (`truncate`).

use napi::{Error, Result};

use super::config::Attn;
use crate::array::{DType, MxArray};

/// Rows a `WindowRing` keeps past the window, so a speculative append of up
/// to this many rows can be dropped exactly.
pub const WINDOW_SLACK: usize = 3;
/// Cache dtype of every latent row.
pub const ROW_DTYPE: DType = DType::Float16;

#[cfg(test)]
thread_local! {
    static TEST_ROW_DTYPE: std::cell::Cell<DType> = const { std::cell::Cell::new(ROW_DTYPE) };
}

#[cfg(not(test))]
fn row_dtype() -> DType {
    ROW_DTYPE
}

#[cfg(test)]
fn row_dtype() -> DType {
    TEST_ROW_DTYPE.with(std::cell::Cell::get)
}

/// Runs `f` with this thread's flat caches storing rows as `dtype`.
#[cfg(test)]
pub fn with_row_dtype<R>(dtype: DType, f: impl FnOnce() -> R) -> R {
    struct Reset(DType);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_ROW_DTYPE.with(|d| d.set(self.0));
        }
    }
    let _reset = Reset(TEST_ROW_DTYPE.with(|d| d.replace(dtype)));
    f()
}

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

/// Reads `count` logical rows starting at `first_pos` out of a circular
/// `[cap, width]` buffer in oldest-first linear order.
fn linear_range(
    buf: &MxArray,
    cap: usize,
    width: i64,
    first_pos: usize,
    count: usize,
    detach: bool,
) -> Result<MxArray> {
    let lo = first_pos % cap;
    if lo + count <= cap {
        let view = buf.slice(&[lo as i64, 0], &[(lo + count) as i64, width])?;
        if detach { view.deep_copy() } else { Ok(view) }
    } else {
        let head = buf.slice(&[lo as i64, 0], &[cap as i64, width])?;
        let tail = buf.slice(&[0, 0], &[(lo + count - cap) as i64, width])?;
        MxArray::concatenate(&head, &tail, 0)
    }
}

/// Keys returned by [`WindowRing::append`] for one attention call.
pub enum WindowKeys<'a> {
    /// Physical circular ring `[cap, width]`; position `kp` lives at slot `kp % cap`.
    Ring { keys: &'a MxArray, cap: usize },
    /// Linear buffer `[m, width]`; position `kp` lives at row `kp - first`.
    Linear { keys: MxArray, first: usize },
}

/// Linear oldest-first snapshot of a [`WindowRing`]: positions `len - kept .. len`
/// in `[kept, width]` f16. Safe to clone, retain in prefix states, and persist.
#[derive(Clone)]
pub struct WindowRingSnapshot {
    window: usize,
    width: i64,
    rows: Option<MxArray>,
    held: usize,
    len: usize,
}

impl WindowRingSnapshot {
    /// A snapshot of `len` positions holding `rows [held, width]` (f16), the
    /// rows of its last `held` positions. Refused past `window + slack` rows,
    /// past `len`, or below a window of rows (all of `len` when shorter).
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
        let rows = rows
            .map(|r| {
                if r.dtype()? == row_dtype() {
                    Ok(r)
                } else {
                    r.astype(row_dtype())
                }
            })
            .transpose()?;
        Ok(Self {
            window,
            width,
            rows,
            held,
            len,
        })
    }

    /// The held rows `[kept, width]` in oldest-first linear order.
    pub fn rows(&self) -> Option<&MxArray> {
        self.rows.as_ref()
    }

    /// Positions appended up to this snapshot.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn window(&self) -> usize {
        self.window
    }

    pub fn width(&self) -> i64 {
        self.width
    }

    /// Rows held in this snapshot.
    pub fn kept(&self) -> usize {
        self.held
    }

    pub fn nbytes(&self) -> usize {
        self.rows.as_ref().map_or(0, MxArray::nbytes)
    }
}

/// The latent rows of one sliding layer in a fixed `[cap, width]` circular
/// buffer (`cap = window + slack`). Position `p` in `(len - held)..len` lives
/// at physical row `p % cap`. Never `Clone` so `buf` keeps refcount 1 for
/// zero-copy MLX buffer donation.
pub struct WindowRing {
    window: usize,
    cap: usize,
    width: i64,
    buf: Option<MxArray>,
    held: usize,
    len: usize,
}

impl WindowRing {
    pub fn new(window: usize, slack: usize, width: i64) -> Self {
        Self {
            window,
            cap: window + slack,
            width,
            buf: None,
            held: 0,
            len: 0,
        }
    }

    /// Rebuilds a live circular ring from a linear [`WindowRingSnapshot`].
    pub fn from_snapshot(snap: &WindowRingSnapshot, slack: usize) -> Result<Self> {
        let mut ring = Self::new(snap.window, slack, snap.width);
        let Some(rows) = &snap.rows else {
            if snap.len != 0 {
                return Err(bad(format!(
                    "a ring of {} positions with 0 rows (window {}, slack {slack})",
                    snap.len, ring.window
                )));
            }
            return Ok(ring);
        };
        if ring.cap == 0 {
            return Err(bad("window ring capacity must be > 0"));
        }
        let held = snap.held;
        let len = snap.len;
        if held > ring.cap || held > len || held < len.min(ring.window) {
            return Err(bad(format!(
                "a ring of {len} positions with {held} rows (window {}, slack {slack})",
                ring.window
            )));
        }
        let rows = if rows.dtype()? == row_dtype() {
            rows.clone()
        } else {
            rows.astype(row_dtype())?
        };
        let cap_i64 = i64::try_from(ring.cap).map_err(|_| bad("window capacity exceeds i64"))?;
        let zero = MxArray::zeros(&[cap_i64, ring.width], Some(row_dtype()))?;
        let first_pos = len - held;
        let buf = super::ops::ring_write(&zero, &rows, first_pos % ring.cap, ring.cap)?;
        drop(zero);
        super::stats::defer(&buf, "hy_v4_flash window ring")?;
        ring.buf = Some(buf);
        ring.held = held;
        ring.len = len;
        Ok(ring)
    }

    /// Builds a live circular ring directly from oldest-first linear `rows`.
    pub fn of_rows(
        window: usize,
        slack: usize,
        width: i64,
        rows: Option<MxArray>,
        len: usize,
    ) -> Result<Self> {
        let snap = WindowRingSnapshot::of_rows(window, slack, width, rows, len)?;
        Self::from_snapshot(&snap, slack)
    }

    /// Builds an unevaluated linear oldest-first snapshot detached from `buf`.
    /// Callers taking snapshots across many layers can batch-evaluate the
    /// returned arrays once so live `buf` drops back to refcount 1.
    pub fn snapshot_lazy(&self) -> Result<WindowRingSnapshot> {
        if self.held == 0 {
            return Ok(WindowRingSnapshot {
                window: self.window,
                width: self.width,
                rows: None,
                held: 0,
                len: self.len,
            });
        }
        let buf = self
            .buf
            .as_ref()
            .ok_or_else(|| bad("window buffer missing"))?;
        let first_pos = self.len - self.held;
        let rows = linear_range(buf, self.cap, self.width, first_pos, self.held, true)?;
        Ok(WindowRingSnapshot {
            window: self.window,
            width: self.width,
            rows: Some(rows),
            held: self.held,
            len: self.len,
        })
    }

    /// Evaluates and returns a detached linear oldest-first snapshot of the
    /// held rows, leaving live `buf` with refcount 1 for donation.
    pub fn snapshot(&self) -> Result<WindowRingSnapshot> {
        let snap = self.snapshot_lazy()?;
        if let Some(rows) = snap.rows() {
            MxArray::eval_arrays_with_context(&[rows], "hy_v4_flash window snapshot")?;
        }
        Ok(snap)
    }

    /// The underlying `[cap, width]` circular buffer when allocated.
    pub fn buf(&self) -> Option<&MxArray> {
        self.buf.as_ref()
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

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn slack(&self) -> usize {
        self.cap - self.window
    }

    pub fn width(&self) -> i64 {
        self.width
    }

    /// Valid rows held now.
    pub fn kept(&self) -> usize {
        self.held
    }

    /// Whether `truncate(len)` would succeed.
    pub fn can_truncate(&self, len: usize) -> bool {
        if len > self.len {
            return false;
        }
        if len == self.len || len == 0 {
            return true;
        }
        let drop = self.len - len;
        self.held.saturating_sub(drop) >= len.min(self.window)
    }

    /// Appends `kv [n, width]` at `start` (== `len()`), stored as f16.
    /// When all keys needed by queries `start..start + n` fit in the ring
    /// without overwriting needed history, returns [`WindowKeys::Ring`] pointing
    /// directly at `buf` (O(1) zero-copy in-place write). Larger prefill chunks
    /// return [`WindowKeys::Linear`] covering the call's needed window and
    /// update `buf` with the trailing `min(held + n, cap)` rows.
    pub fn append(&mut self, kv: &MxArray, start: usize) -> Result<WindowKeys<'_>> {
        if start != self.len {
            return Err(bad(format!(
                "window append at {start}, {} positions held",
                self.len
            )));
        }
        if self.cap == 0 {
            return Err(bad("window ring capacity must be > 0"));
        }
        let n = rows_of(kv, self.width, "window append")?;
        let kv = if kv.dtype()? == row_dtype() {
            kv.clone()
        } else {
            kv.astype(row_dtype())?
        };
        let end = start + n;
        let kept = self.held;
        let first_needed = (start + 1).saturating_sub(self.window).max(start - kept);
        let cap_i64 = i64::try_from(self.cap).map_err(|_| bad("window capacity exceeds i64"))?;

        if end - first_needed <= self.cap {
            let prev = match self.buf.take() {
                Some(b) => b,
                None => MxArray::zeros(&[cap_i64, self.width], Some(row_dtype()))?,
            };
            let next = super::ops::ring_write(&prev, &kv, start % self.cap, self.cap)?;
            drop(prev);
            super::stats::defer(&next, "hy_v4_flash window ring")?;
            self.buf = Some(next);
            self.held = (kept + n).min(self.cap);
            self.len = end;
            let keys = self
                .buf
                .as_ref()
                .ok_or_else(|| bad("window buffer missing"))?;
            return Ok(WindowKeys::Ring {
                keys,
                cap: self.cap,
            });
        }

        let prefix_len = start - first_needed;
        let keys = if prefix_len > 0 {
            let buf = self
                .buf
                .as_ref()
                .ok_or_else(|| bad("window buffer missing"))?;
            let prefix = linear_range(buf, self.cap, self.width, first_needed, prefix_len, false)?;
            MxArray::concatenate(&prefix, &kv, 0)?
        } else {
            kv.clone()
        };

        let (src, write_start) = if n >= self.cap {
            (
                kv.slice(&[(n - self.cap) as i64, 0], &[n as i64, self.width])?,
                end - self.cap,
            )
        } else {
            (kv, start)
        };
        let prev = match self.buf.take() {
            Some(b) => b,
            None => MxArray::zeros(&[cap_i64, self.width], Some(row_dtype()))?,
        };
        let next = super::ops::ring_write(&prev, &src, write_start % self.cap, self.cap)?;
        drop(prev);
        super::stats::defer(&next, "hy_v4_flash window ring")?;
        self.buf = Some(next);
        self.held = (kept + n).min(self.cap);
        self.len = end;
        Ok(WindowKeys::Linear {
            keys,
            first: first_needed,
        })
    }

    /// Drops the positions past `len` in O(1) metadata-only. Refused when fewer
    /// than a window of rows (or all of `len`) would remain held.
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
            self.held = 0;
            self.len = 0;
            return Ok(());
        }
        let drop = self.len - len;
        let left = self.held.saturating_sub(drop);
        if left < len.min(self.window) {
            return Err(bad(format!(
                "window truncate {} -> {len} drops {drop} of {} held rows",
                self.len, self.held
            )));
        }
        self.held = left;
        self.len = len;
        Ok(())
    }

    pub fn nbytes(&self) -> usize {
        self.buf.as_ref().map_or(0, MxArray::nbytes)
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
        let rows = kv.astype(row_dtype())?;
        if self.buf.is_none() {
            let cap = i64::try_from(self.cap).map_err(|_| bad("context exceeds i64"))?;
            let b = MxArray::zeros(&[cap, self.width], Some(row_dtype()))?;
            MxArray::eval_arrays_with_context(&[&b], "hy_v4_flash latent alloc")?;
            self.buf = Some(b);
        }
        let buf = self
            .buf
            .as_mut()
            .ok_or_else(|| bad("latent buffer missing"))?;
        buf.slice_assign_axis_inplace(0, start as i64, end as i64, &rows)?;
        super::stats::defer(&*buf, "hy_v4_flash latent write")?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::hy_v4_flash::stats;

    fn pattern_f16(rows: std::ops::Range<usize>, width: usize) -> MxArray {
        let n = rows.len();
        let bits: Vec<u16> = rows
            .flat_map(|p| {
                (0..width).map(move |c| {
                    // Finite non-zero f16 bit pattern unique per (p, c)
                    0x3C00 | (((p * 37 + c * 13) & 0x03FF) as u16)
                })
            })
            .collect();
        MxArray::from_float16(&bits, &[n as i64, width as i64]).unwrap()
    }

    fn bits_of(x: &MxArray) -> Vec<u16> {
        x.to_uint16_native().unwrap().to_vec()
    }

    #[test]
    fn circular_window_ring_wrap_around_truncate_and_snapshot_round_trip() {
        let window = 12usize;
        let width = 576usize;
        let cap = window + WINDOW_SLACK;
        let mut ring = WindowRing::new(window, WINDOW_SLACK, width as i64);

        // Prefill chunk larger than cap (Linear path) followed by decode wrap-arounds (Ring path).
        let total = cap * 2 + 7;
        let all = pattern_f16(0..total, width);
        let all_bits = bits_of(&all);
        let row_bits = |p: usize| &all_bits[p * width..(p + 1) * width];

        // 1. Large prefill chunk: 0..20 (> cap = 15).
        match ring.append(&pattern_f16(0..20, width), 0).unwrap() {
            WindowKeys::Linear { keys, first } => {
                assert_eq!(first, 0);
                assert_eq!(keys.shape_at(0).unwrap(), 20);
            }
            WindowKeys::Ring { .. } => panic!("expected Linear for prefill chunk > cap"),
        }
        assert_eq!((ring.len(), ring.kept(), ring.cap()), (20, cap, cap));

        // 2. Decode single & multi-row appends crossing wrap boundary multiple times.
        let mut pos = 20;
        for step in [1usize, 3, 2, 1, 4, 1, 3, 2] {
            let end = (pos + step).min(total);
            if pos >= end {
                break;
            }
            let chunk = pattern_f16(pos..end, width);
            match ring.append(&chunk, pos).unwrap() {
                WindowKeys::Ring { keys, cap: c } => {
                    assert_eq!(c, cap);
                    let buf_bits = bits_of(keys);
                    let first_valid = end - cap;
                    for p in first_valid..end {
                        let slot = p % cap;
                        assert_eq!(
                            &buf_bits[slot * width..(slot + 1) * width],
                            row_bits(p),
                            "slot {slot} for pos {p}"
                        );
                    }
                }
                WindowKeys::Linear { .. } => panic!("expected Ring for decode step {step}"),
            }
            pos = end;
        }
        assert_eq!(pos, total);

        // 3. Snapshot + from_snapshot bit-exact round-trip.
        let snap = ring.snapshot().unwrap();
        assert_eq!((snap.len(), snap.kept()), (total, cap));
        let snap_bits = bits_of(snap.rows().unwrap());
        let expected_linear: Vec<u16> = (total - cap..total)
            .flat_map(|p| row_bits(p).iter().copied())
            .collect();
        assert_eq!(snap_bits, expected_linear);

        let rebuilt = WindowRing::from_snapshot(&snap, WINDOW_SLACK).unwrap();
        assert_eq!((rebuilt.len(), rebuilt.kept()), (total, cap));
        let rebuilt_buf = bits_of(rebuilt.buf().unwrap());
        for p in total - cap..total {
            let slot = p % cap;
            assert_eq!(
                &rebuilt_buf[slot * width..(slot + 1) * width],
                row_bits(p),
                "rebuilt slot {slot} for pos {p}"
            );
        }
        assert_eq!(
            bits_of(rebuilt.snapshot().unwrap().rows().unwrap()),
            snap_bits
        );

        // 4. O(1) metadata-only truncate within WINDOW_SLACK triggers zero evals and
        // preserves all remaining window rows intact.
        let evals_before = stats::thread_evals();
        let target = total - WINDOW_SLACK;
        ring.truncate(target).unwrap();
        assert_eq!(stats::thread_evals(), evals_before);
        assert_eq!((ring.len(), ring.kept()), (target, window));
        let after_trunc = bits_of(ring.snapshot().unwrap().rows().unwrap());
        let expected_after_trunc: Vec<u16> = (target - window..target)
            .flat_map(|p| row_bits(p).iter().copied())
            .collect();
        assert_eq!(after_trunc, expected_after_trunc);

        // Appending replacement rows overwrites discarded tail slots without touching snap.
        let repl = pattern_f16(1000..1002, width);
        let repl_bits = bits_of(&repl);
        ring.append(&repl, target).unwrap();
        let after_repl = bits_of(ring.snapshot().unwrap().rows().unwrap());
        assert_eq!(&after_repl[..window * width], &expected_after_trunc[..]);
        assert_eq!(&after_repl[window * width..], &repl_bits[..]);
        assert_eq!(bits_of(snap.rows().unwrap()), expected_linear);
    }

    #[test]
    fn ring_write_non_donatable_preserves_shared_buffer() {
        let cap = 7usize;
        let width = 12usize;
        let base = pattern_f16(0..cap, width);
        MxArray::eval_arrays_with_context(&[&base], "test ring base").unwrap();
        let base_bits = bits_of(&base);

        // Holding an alias makes `is_donatable(base, out)` false, forcing a safe copy.
        let alias = base.clone();
        let patch = pattern_f16(100..103, width);
        let patch_bits = bits_of(&patch);
        let out = super::super::ops::ring_write(&base, &patch, 5, cap).unwrap();
        MxArray::eval_arrays_with_context(&[&out], "test ring write").unwrap();

        assert_eq!(bits_of(&alias), base_bits);
        let out_bits = bits_of(&out);
        for slot in 0..cap {
            let got = &out_bits[slot * width..(slot + 1) * width];
            let want = match slot {
                5 => &patch_bits[0..width],
                6 => &patch_bits[width..2 * width],
                0 => &patch_bits[2 * width..3 * width],
                _ => &base_bits[slot * width..(slot + 1) * width],
            };
            assert_eq!(got, want, "slot {slot}");
        }
    }
}
