//! The DFlash2 draft context window (Splash `DFlashContextWindow`): the
//! draft's fused context rows of the newest `slots` positions in 4-bit
//! codes, from which a restore projects the draft K/V again.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};

/// Values per quantization group (Splash `SPLASH_DRAFT_CONTEXT_GROUP`).
pub(crate) const DRAFT_CONTEXT_GROUP: u32 = 64;

#[cfg(test)]
thread_local! {
    /// Context windows allocated on this thread.
    pub(crate) static WINDOW_ALLOCATIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Rows `[begin, end)` of `width` values in a ring of `slots` slots, row p
/// at slot p % slots: all slots' codes (`width / 2` bytes each, low nibble
/// first), then each slot's per-group fp16 `(scale, minimum)` (Splash
/// `draft_context_window_bytes`).
pub(crate) struct DraftContextWindow {
    buffer: MxArray,
    width: u32,
    slots: u32,
    begin: u32,
    end: u32,
}

impl DraftContextWindow {
    /// Bytes of a window (Splash `DraftAttention::contextWindowBytes`).
    pub(crate) fn bytes(width: u32, slots: u32) -> u64 {
        let groups = (width / DRAFT_CONTEXT_GROUP) as u64;
        slots as u64 * (width as u64 / 2) + slots as u64 * groups * 4
    }

    /// An empty window at row 0.
    pub(crate) fn new(width: u32, slots: u32) -> Result<Self> {
        if width == 0 || !width.is_multiple_of(DRAFT_CONTEXT_GROUP) || slots == 0 {
            return Err(Error::from_reason(format!(
                "a draft context window holds whole {DRAFT_CONTEXT_GROUP}-value groups in at least one slot (width {width}, slots {slots})"
            )));
        }
        let bytes = i64::try_from(Self::bytes(width, slots))
            .map_err(|_| Error::from_reason("draft context window size overflow"))?;
        #[cfg(test)]
        WINDOW_ALLOCATIONS.with(|count| count.set(count.get() + 1));
        Ok(Self {
            buffer: MxArray::zeros(&[bytes], Some(DType::Uint8))?,
            width,
            slots,
            begin: 0,
            end: 0,
        })
    }

    pub(crate) fn begin(&self) -> u32 {
        self.begin
    }

    pub(crate) fn end(&self) -> u32 {
        self.end
    }

    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    /// The rows the window holds at most.
    #[cfg(test)]
    pub(crate) fn slots(&self) -> u32 {
        self.slots
    }

    /// Empties the window at `row` (a capture span with `reset`).
    pub(crate) fn restart(&mut self, row: u32) {
        self.begin = row;
        self.end = row;
    }

    /// Stores bf16 `rows` (`[n, width]`, or `[1, n, width]`) as rows
    /// `first_row..`, which must continue the window. Only the newest
    /// `slots` rows stay.
    pub(crate) fn store_rows(&mut self, rows: &MxArray, first_row: u32) -> Result<()> {
        if first_row != self.end {
            return Err(Error::from_reason(format!(
                "draft context rows at {first_row} do not continue the window ending at {}",
                self.end
            )));
        }
        let ndim = rows.ndim()?;
        if ndim < 2 {
            return Err(Error::from_reason("draft context rows are [n, width]"));
        }
        let n = rows.shape_at(ndim - 2)?;
        let width = rows.shape_at(ndim - 1)?;
        if width != i64::from(self.width) || n <= 0 {
            return Err(Error::from_reason(format!(
                "draft context rows are [{n}, {width}], the window is {} wide",
                self.width
            )));
        }
        let mut rows = rows.reshape(&[n, width])?;
        let mut at = first_row;
        let n = u32::try_from(n).map_err(|_| Error::from_reason("draft context row overflow"))?;
        if n > self.slots {
            rows = rows.slice_axis(0, i64::from(n - self.slots), i64::from(n))?;
            at += n - self.slots;
        }
        if unsafe { sys::mlx_metal_is_available() } {
            // The kernel reads bf16 rows; an f16 or f32 draft's are cast.
            if rows.dtype()? != DType::BFloat16 {
                rows = rows.astype(DType::BFloat16)?;
            }
            let mut out = std::ptr::null_mut();
            // SAFETY: both handles are live for the call; `out` receives an
            // owned handle aliasing the window's buffer, or stays null.
            let ok = unsafe {
                sys::mlx_draft_context_store(
                    self.buffer.as_raw_ptr(),
                    rows.as_raw_ptr(),
                    at,
                    self.width,
                    self.slots,
                    &mut out,
                )
            };
            if !ok {
                return Err(Error::from_reason("draft context store contract failed"));
            }
            self.buffer = MxArray::from_handle(out, "draft_context_store")?;
        } else {
            let mut window = self.buffer.to_uint8()?;
            let values = rows.astype(DType::BFloat16)?.to_uint16_native()?;
            store_rows_host(&mut window, &values, self.width, at, self.slots);
            self.buffer = MxArray::from_uint8(&window, &[window.len() as i64])?;
        }
        self.end = first_row + n;
        self.begin = self.begin.max(self.end - self.end.min(self.slots));
        Ok(())
    }

    /// Rows `[first_row, first_row + n)` as bf16 `[n, width]`, dequantized
    /// (Splash `prefill_draft_context_load`).
    pub(crate) fn load_rows(&self, first_row: u32, n: u32) -> Result<MxArray> {
        if n == 0 || first_row < self.begin || first_row + n > self.end {
            return Err(Error::from_reason(format!(
                "draft context rows [{first_row}, {}) are not in the window [{}, {})",
                first_row + n,
                self.begin,
                self.end
            )));
        }
        if unsafe { sys::mlx_metal_is_available() } {
            let mut out = std::ptr::null_mut();
            // SAFETY: the window handle is live for the call; `out`
            // receives an owned handle or stays null.
            let ok = unsafe {
                sys::mlx_draft_context_load(
                    self.buffer.as_raw_ptr(),
                    first_row,
                    n,
                    self.width,
                    self.slots,
                    &mut out,
                )
            };
            if !ok {
                return Err(Error::from_reason("draft context load contract failed"));
            }
            return MxArray::from_handle(out, "draft_context_load");
        }
        let window = self.buffer.to_uint8()?;
        let values = load_rows_host(&window, self.width, first_row, n, self.slots);
        MxArray::from_bfloat16(&values, &[i64::from(n), i64::from(self.width)])
    }

    /// An evaluated copy: later stores to either never reach the other. A
    /// failed copy is an error, never a state (Splash
    /// `NativeRuntime::executionFailed` publishes nothing).
    pub(crate) fn snapshot(&self) -> Result<Self> {
        let buffer = self.buffer.deep_copy()?;
        MxArray::eval_arrays_with_context(&[&buffer], "draft_context_snapshot")?;
        Ok(Self {
            buffer,
            width: self.width,
            slots: self.slots,
            begin: self.begin,
            end: self.end,
        })
    }

    /// Takes `other`'s rows by an in-place copy into this window's buffer
    /// (Splash's restore writes into the lane's window). The bounds move
    /// only after the copy ran.
    pub(crate) fn copy_from(&mut self, other: &Self) -> Result<()> {
        if other.width != self.width || other.slots != self.slots {
            return Err(Error::from_reason(
                "draft context windows of different geometry",
            ));
        }
        let copied = if unsafe { sys::mlx_metal_is_available() } {
            let mut out = std::ptr::null_mut();
            // SAFETY: both handles are live for the call; `out` receives an
            // owned handle aliasing this window's buffer, or stays null.
            let ok = unsafe {
                sys::mlx_draft_context_copy(
                    self.buffer.as_raw_ptr(),
                    other.buffer.as_raw_ptr(),
                    &mut out,
                )
            };
            if !ok {
                return Err(Error::from_reason("draft context copy contract failed"));
            }
            MxArray::from_handle(out, "draft_context_copy")?
        } else {
            other.buffer.deep_copy()?
        };
        MxArray::eval_arrays_with_context(&[&copied], "draft_context_copy")?;
        self.buffer = copied;
        self.begin = other.begin;
        self.end = other.end;
        Ok(())
    }

    pub(crate) fn nbytes(&self) -> u64 {
        Self::bytes(self.width, self.slots)
    }

    /// Materializes the buffer, so its bytes are real storage.
    pub(crate) fn eval(&self) -> Result<()> {
        MxArray::eval_arrays_with_context(&[&self.buffer], "draft_context_window")
    }

    /// Copies the window's bytes into `out`; false unless `out` is exactly
    /// [`Self::nbytes`] long.
    pub(crate) fn write_bytes(&self, out: &mut [u8]) -> Result<bool> {
        if out.len() as u64 != self.nbytes() {
            return Ok(false);
        }
        let bytes = self.buffer.to_uint8()?;
        if bytes.len() != out.len() {
            return Ok(false);
        }
        out.copy_from_slice(&bytes);
        Ok(true)
    }

    /// Takes the rows `[begin, end)` [`Self::write_bytes`] wrote into `bytes`
    /// (Splash `beginRestore` reads a disk state's window into the lane's);
    /// false unless `bytes` is exactly [`Self::nbytes`] long and the rows
    /// fit the slots.
    pub(crate) fn read_bytes(&mut self, bytes: &[u8], begin: u32, end: u32) -> Result<bool> {
        if bytes.len() as u64 != self.nbytes() || begin > end || end - begin > self.slots {
            return Ok(false);
        }
        self.eval()?;
        let ptr = self.buffer.storage_ptr() as *mut u8;
        if ptr.is_null() {
            self.buffer = MxArray::zeros(&[0], Some(DType::Uint8))?;
            let buffer = MxArray::from_uint8(bytes, &[bytes.len() as i64])?;
            MxArray::eval_arrays_with_context(&[&buffer], "draft_context_read")?;
            self.buffer = buffer;
        } else {
            // SAFETY: Splash `stateSpans` reads into the lane's own window:
            // the settled buffer holds exactly `nbytes` unified-memory bytes
            // and no GPU work is in flight on it after the eval.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
        }
        self.begin = begin;
        self.end = end;
        Ok(true)
    }

    pub(crate) fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        out.push(&self.buffer);
    }

    #[cfg(test)]
    pub(crate) fn host_bytes(&self) -> Result<Vec<u8>> {
        self.buffer.to_uint8()
    }
}

fn codes_bytes(width: u32, slots: u32) -> usize {
    slots as usize * (width as usize / 2)
}

/// Splash `prefill_draft_context_store` on host data: bf16 `values`
/// (`[n, width]` bits) as rows `first_row..`.
pub(crate) fn store_rows_host(
    window: &mut [u8],
    values: &[u16],
    width: u32,
    first_row: u32,
    slots: u32,
) {
    let width = width as usize;
    let group = DRAFT_CONTEXT_GROUP as usize;
    let groups = width / group;
    let scales_at = codes_bytes(width as u32, slots);
    for (row, row_values) in values.chunks_exact(width).enumerate() {
        let slot = (first_row as usize + row) % slots as usize;
        for g in 0..groups {
            let x: Vec<f32> = row_values[g * group..(g + 1) * group]
                .iter()
                .map(|&bits| half::bf16::from_bits(bits).to_f32())
                .collect();
            let low = x.iter().copied().fold(f32::INFINITY, f32::min);
            let high = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = half::f16::from_f32((high - low) / 15.0);
            let minimum = half::f16::from_f32(low);
            let code = |value: f32| -> u8 {
                if scale > half::f16::ZERO {
                    ((value - minimum.to_f32()) / scale.to_f32())
                        .round_ties_even()
                        .clamp(0.0, 15.0) as u8
                } else {
                    0
                }
            };
            for lane in 0..group / 2 {
                window[slot * (width / 2) + g * (group / 2) + lane] =
                    code(x[2 * lane]) | (code(x[2 * lane + 1]) << 4);
            }
            let at = scales_at + (slot * groups + g) * 4;
            window[at..at + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
            window[at + 2..at + 4].copy_from_slice(&minimum.to_bits().to_le_bytes());
        }
    }
}

/// Splash `prefill_draft_context_load` on host data: rows `first_row..` as
/// bf16 `[n, width]` bits.
pub(crate) fn load_rows_host(
    window: &[u8],
    width: u32,
    first_row: u32,
    n: u32,
    slots: u32,
) -> Vec<u16> {
    let width = width as usize;
    let group = DRAFT_CONTEXT_GROUP as usize;
    let groups = width / group;
    let scales_at = codes_bytes(width as u32, slots);
    let mut out = Vec::with_capacity(n as usize * width);
    for row in 0..n as usize {
        let slot = (first_row as usize + row) % slots as usize;
        for g in 0..groups {
            let at = scales_at + (slot * groups + g) * 4;
            let scale = half::f16::from_bits(u16::from_le_bytes([window[at], window[at + 1]]));
            let minimum =
                half::f16::from_bits(u16::from_le_bytes([window[at + 2], window[at + 3]]));
            for lane in 0..group / 2 {
                let byte = window[slot * (width / 2) + g * (group / 2) + lane];
                for code in [byte & 15, byte >> 4] {
                    let value = f32::from(code).mul_add(scale.to_f32(), minimum.to_f32());
                    out.push(half::bf16::from_f32(value).to_bits());
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random bf16 rows `[37, 128]` with a constant group (scale 0), a group
    /// with one outlier and ±0, stored from row 2040 so they wrap the 2048
    /// slots: the Metal window bytes equal the host port of Splash
    /// `prefill_draft_context_store` byte for byte, `load_rows` equals the
    /// host `fma` port bit for bit, and row p sits in slot p % 2048.
    /// Mutation Q15 (the scale divides by 16) → red.
    #[test]
    fn context_window_quantizer_matches_the_kernel() -> Result<()> {
        if !unsafe { sys::mlx_metal_is_available() } {
            return Ok(());
        }
        const WIDTH: u32 = 128;
        const SLOTS: u32 = 2048;
        const ROWS: u32 = 37;
        const FIRST: u32 = 2040;
        let random = MxArray::random_normal(
            &[i64::from(ROWS), i64::from(WIDTH)],
            0.0,
            3.0,
            Some(DType::BFloat16),
        )?;
        let mut values = random.to_uint16_native()?;
        let bf = |v: f32| half::bf16::from_f32(v).to_bits();
        for value in &mut values[..64] {
            *value = bf(1.25);
        }
        values[WIDTH as usize + 64 + 5] = bf(4000.0);
        values[2 * WIDTH as usize] = bf(0.0);
        values[2 * WIDTH as usize + 1] = half::bf16::NEG_ZERO.to_bits();
        let rows = MxArray::from_bfloat16(&values, &[i64::from(ROWS), i64::from(WIDTH)])?;

        let mut window = DraftContextWindow::new(WIDTH, SLOTS)?;
        window.restart(FIRST);
        window.store_rows(&rows, FIRST)?;
        assert_eq!((window.begin(), window.end()), (FIRST, FIRST + ROWS));
        let got = window.host_bytes()?;
        let mut want = vec![0u8; DraftContextWindow::bytes(WIDTH, SLOTS) as usize];
        store_rows_host(&mut want, &values, WIDTH, FIRST, SLOTS);
        assert_eq!(got.len(), want.len());
        let first_diff = got.iter().zip(&want).position(|(a, b)| a != b);
        assert_eq!(first_diff, None, "window bytes differ");

        // Row 2047 sits in the last slot, row 2048 in slot 0.
        let codes = |slot: usize| &got[slot * 64..slot * 64 + 64];
        let mut alone = vec![0u8; want.len()];
        store_rows_host(
            &mut alone,
            &values[7 * WIDTH as usize..9 * WIDTH as usize],
            WIDTH,
            2047,
            SLOTS,
        );
        assert_eq!(codes(2047), &alone[2047 * 64..2048 * 64]);
        assert_eq!(codes(0), &alone[..64]);
        assert!(got[..64].iter().any(|&b| b != 0), "slot 0 holds row 2048");
        let constant = &got[2040 * 64..2040 * 64 + 32];
        assert!(constant.iter().all(|&b| b == 0), "a constant group codes 0");

        let loaded = window.load_rows(FIRST, ROWS)?;
        assert_eq!(
            (loaded.shape_at(0)?, loaded.shape_at(1)?),
            (i64::from(ROWS), i64::from(WIDTH))
        );
        assert_eq!(
            loaded.to_uint16_native()?,
            load_rows_host(&want, WIDTH, FIRST, ROWS, SLOTS),
            "dequantized rows differ"
        );
        Ok(())
    }

    /// A store past the slots keeps the newest rows; a copy takes the
    /// rows and bounds in place.
    #[test]
    fn window_keeps_the_newest_slots_and_copies_in_place() -> Result<()> {
        if !unsafe { sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let rows = MxArray::random_normal(&[1, 11, 64], 0.0, 1.0, Some(DType::BFloat16))?;
        let mut window = DraftContextWindow::new(64, 8)?;
        window.restart(5);
        window.store_rows(&rows, 5)?;
        assert_eq!((window.begin(), window.end()), (8, 16));
        let tail = window.load_rows(8, 8)?.to_uint16_native()?;
        let mut lane = DraftContextWindow::new(64, 8)?;
        lane.copy_from(&window)?;
        assert_eq!((lane.begin(), lane.end()), (8, 16));
        assert_eq!(lane.load_rows(8, 8)?.to_uint16_native()?, tail);
        assert!(window.store_rows(&rows, 3).is_err(), "rows must continue");
        Ok(())
    }

    /// Splash `beginRestore` reads a disk state into the lane's own window
    /// (`stateSpans` over `writableContents`): the restore allocates no
    /// second window beside the one the governor admitted.
    /// Mutation read_bytes_allocates (a fresh buffer per read) → red.
    #[test]
    fn a_disk_read_lands_in_the_window_it_has() -> Result<()> {
        if !unsafe { sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let mut window = DraftContextWindow::new(128, 4)?;
        window.eval()?;
        let storage = window.buffer.storage_ptr();
        assert!(!storage.is_null());
        let allocations = WINDOW_ALLOCATIONS.with(std::cell::Cell::get);
        let bytes: Vec<u8> = (0..window.nbytes() as usize)
            .map(|i| (i * 13 % 251) as u8)
            .collect();
        assert!(window.read_bytes(&bytes, 3, 5)?);
        assert_eq!(
            window.buffer.storage_ptr(),
            storage,
            "the read lands in the window's storage"
        );
        assert_eq!(window.host_bytes()?, bytes);
        assert_eq!((window.begin(), window.end()), (3, 5));
        assert_eq!(WINDOW_ALLOCATIONS.with(std::cell::Cell::get), allocations);

        let mut lazy = DraftContextWindow::new(128, 4)?;
        assert!(lazy.read_bytes(&bytes, 0, 4)?);
        assert!(!lazy.buffer.storage_ptr().is_null());
        assert_eq!(lazy.host_bytes()?, bytes);
        assert!(
            !window.read_bytes(&bytes[1..], 0, 1)?,
            "a short read is refused"
        );
        Ok(())
    }

    /// A window whose buffer reads a lazy array Metal cannot allocate
    /// (1 TiB, past its largest buffer), so the copy throws inside eval.
    fn unallocatable(width: u32, slots: u32) -> Result<DraftContextWindow> {
        let bytes = DraftContextWindow::bytes(width, slots) as i64;
        let huge = MxArray::zeros(&[1 << 20, 1 << 20], Some(DType::Uint8))?;
        Ok(DraftContextWindow {
            buffer: huge
                .slice_axis(1, 0, bytes)?
                .slice_axis(0, 0, 1)?
                .reshape(&[bytes])?,
            width,
            slots,
            begin: 0,
            end: 0,
        })
    }

    /// A copy that fails on the GPU is an error: the snapshot returns no
    /// window and the lane keeps its bounds; a later copy still runs.
    /// Mutations EV1 (the snapshot evaluates without the error) and EV2
    /// (the copy does) → red.
    #[test]
    fn a_failed_window_copy_is_an_error() -> Result<()> {
        if !unsafe { sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let mut source = unallocatable(64, 8)?;
        source.restart(5);
        assert!(source.snapshot().is_err(), "a failed snapshot is a window");
        let mut lane = DraftContextWindow::new(64, 8)?;
        assert!(
            lane.copy_from(&source).is_err(),
            "a failed copy is a window"
        );
        assert_eq!((lane.begin(), lane.end()), (0, 0));
        let mut small = DraftContextWindow::new(64, 8)?;
        small.store_rows(
            &MxArray::random_normal(&[3, 64], 0.0, 1.0, Some(DType::BFloat16))?,
            0,
        )?;
        assert_eq!(small.snapshot()?.host_bytes()?, small.host_bytes()?);
        Ok(())
    }

    /// A draft whose weights are f32 (or f16) fuses f32 rows: the window
    /// stores them as their bf16 values.
    /// Mutation CB1 (the Metal store passes the rows uncast) → red (the
    /// kernel's dtype contract fails).
    #[test]
    fn rows_of_a_non_bf16_draft_store_as_bf16() -> Result<()> {
        if !unsafe { sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let rows = MxArray::random_normal(&[5, 64], 0.0, 1.0, Some(DType::Float32))?;
        let mut wide = DraftContextWindow::new(64, 8)?;
        wide.store_rows(&rows, 0)?;
        let mut narrow = DraftContextWindow::new(64, 8)?;
        narrow.store_rows(&rows.astype(DType::BFloat16)?, 0)?;
        assert_eq!(wide.host_bytes()?, narrow.host_bytes()?);
        Ok(())
    }
}
