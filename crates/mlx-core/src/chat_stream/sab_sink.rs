//! WASM-only SharedArrayBuffer ring-buffer producer for streaming chat output.
//!
//! ## Protocol overview
//!
//! The producer ([`SabSink`]) and the JS consumer share a `SharedArrayBuffer`
//! whose first [`SAB_HEADER_BYTES`] bytes form a 32-byte header of `u32 LE`
//! atomic slots (seq, write_cur, read_cur, cancelled). The remainder is the
//! ring body where records are written sequentially, wrapping at body_len.
//!
//! ## Back-pressure design
//!
//! When the ring is full the producer calls `memory.atomic.wait32` on the
//! `read_cur` slot with a 5 ms timeout and re-checks on each wake. This is a
//! **cooperative poll** rather than a consumer-driven notify: the JS reader
//! advances `read_cur` and the WASM atomic wait wakes on timeout (or
//! immediately if `read_cur` already changed). The alternative — requiring the
//! JS side to call `Atomics.notify(read_cur_addr)` — would be more efficient
//! but adds a coordination requirement that is harder to bind correctly from
//! JS. The 5 ms timeout is short enough to be invisible to the user and long
//! enough to avoid a tight busy-spin.
//!
//! ## Ring full/empty disambiguation
//!
//! Classic single-producer single-consumer rings have a `write == read`
//! ambiguity (empty or full). We resolve it by **reserving one byte of
//! slack**: the ring is considered full when
//! `(write + 1) % cap == read`. Usable capacity is therefore `cap − 1`
//! rather than `cap`. The JS reader must use the same invariant.
//!
//! ## Wasm atomic operations (stable)
//!
//! `core::arch::wasm32::{memory_atomic_notify, memory_atomic_wait32}` are
//! behind the `stdarch_wasm_atomic_wait` unstable feature gate on the stable
//! toolchain. We instead declare the equivalent LLVM compiler-rt builtins
//! (`__wasm_i32_atomic_wait`, `__wasm_atomic_notify`) via `unsafe extern "C"`
//! — these symbols are always available in the `wasm32-wasip1-threads` target
//! because the atomics proposal is already enabled by the target itself.
//!
//! ## Safety of aliased SharedArrayBuffer access
//!
//! The SAB is shared memory that the JS thread may also read/write. We
//! never form a `&[u8]` or `&mut [u8]` reference into the SAB body;
//! instead, all writes go through `core::ptr::copy_nonoverlapping` (raw
//! pointer arithmetic) and all header reads/writes go through
//! `core::sync::atomic::AtomicU32`. The release store on `write_cur` after
//! copying bytes into the ring is the visibility fence for the consumer.

#[cfg(target_arch = "wasm32")]
use core::sync::atomic::{AtomicU32, Ordering};

use crate::models::qwen3_5::model::ChatStreamChunk;

use super::ChatStreamSink;
use super::wire::{
    EncodedPayload, RECORD_HEADER_BYTES, SAB_HEADER_BYTES, SAB_HEADER_OFFSET_CANCELLED,
    SAB_HEADER_OFFSET_READ_CUR, SAB_HEADER_OFFSET_SEQ, SAB_HEADER_OFFSET_WRITE_CUR,
    classify_payload, write_record,
};

// ---------------------------------------------------------------------------
// Stable wasm32 atomic-wait / atomic-notify via LLVM compiler-rt builtins.
//
// `core::arch::wasm32::{memory_atomic_notify, memory_atomic_wait32}` are
// unstable on the stable toolchain (feature `stdarch_wasm_atomic_wait`).
// The LLVM-level names below are always available on wasm32-wasip1-threads
// because the atomics proposal is baked into the target.
//
// Return values:
//   wasm_i32_atomic_wait: 0 = "not equal" (value changed), 1 = "ok"
//     (woken by notify), 2 = "timed out"
//   wasm_atomic_notify: number of waiters woken
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
unsafe extern "C" {
    /// `memory.atomic.wait32` — suspends the current thread until the i32 at
    /// `ptr` differs from `expected`, a notify fires, or `timeout_ns` elapses
    /// (pass -1 for infinite). Returns 0, 1, or 2.
    #[link_name = "__wasm_i32_atomic_wait"]
    fn wasm_atomic_wait32(ptr: *mut i32, expected: i32, timeout_ns: i64) -> i32;

    /// `memory.atomic.notify` — wakes up to `count` threads waiting on `ptr`.
    /// Returns the number of threads woken.
    #[link_name = "__wasm_atomic_notify"]
    fn wasm_atomic_notify(ptr: *mut i32, count: u32) -> u32;
}

// ---------------------------------------------------------------------------
// Thread-local scratch buffer — avoids a heap alloc on every token write.
// ---------------------------------------------------------------------------

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<u8>> =
        std::cell::RefCell::new(Vec::with_capacity(1024));
}

// ---------------------------------------------------------------------------
// Minimum usable SAB size: header + record header + 1 payload byte + 1 slack.
// ---------------------------------------------------------------------------
pub const MIN_SAB_LEN: usize = SAB_HEADER_BYTES + RECORD_HEADER_BYTES + 1 + 1;

// ---------------------------------------------------------------------------
// SabSink struct
// ---------------------------------------------------------------------------

/// A [`ChatStreamSink`] that writes [`ChatStreamChunk`] records into a
/// `SharedArrayBuffer` ring buffer using WASM atomic primitives.
///
/// Designed to run on the WASM worker thread that runs the model decode loop.
/// The JS consumer on the main thread reads from the same SAB and delivers
/// tokens to user callbacks.
///
/// See the module-level documentation for the protocol and safety details.
#[cfg(target_arch = "wasm32")]
pub struct SabSink {
    /// Raw pointer into the SharedArrayBuffer's backing memory. We store a
    /// pointer instead of a slice because the SAB is `SharedArrayBuffer` on
    /// the JS side — Rust cannot form a safe `&mut [u8]` into memory that
    /// another thread may mutate. All reads/writes go through atomic ops or
    /// `copy_nonoverlapping`.
    ptr: *mut u8,
    /// Body length = total_len - SAB_HEADER_BYTES. Cached to avoid recomputing.
    body_len: usize,
}

// SAFETY: The ptr is into a SharedArrayBuffer owned for the lifetime of this
// SabSink by the JS caller. All accesses go through atomic ops or
// `copy_nonoverlapping`, so we do not create aliasing references. The struct
// is therefore Send + Sync.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for SabSink {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for SabSink {}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
impl SabSink {
    /// Construct a `SabSink` from a raw SAB pointer and its byte length.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to the start of a `SharedArrayBuffer`'s backing
    ///   memory, aligned for `u32` access (Emscripten/WASI SABs always are).
    /// - `total_len` must be ≥ `MIN_SAB_LEN` (`SAB_HEADER_BYTES +
    ///   RECORD_HEADER_BYTES + 2`) so that at least one byte of payload can be
    ///   written with one byte of slack reserved.
    /// - The pointer must remain valid for the entire lifetime of the returned
    ///   `SabSink`. Task 5's NAPI export is the intended caller.
    pub unsafe fn from_raw(ptr: *mut u8, total_len: usize) -> napi::Result<Self> {
        if total_len < MIN_SAB_LEN {
            return Err(napi::Error::from_reason(format!(
                "SabSink: SAB too small ({total_len} < {MIN_SAB_LEN})"
            )));
        }
        // All header slots are AtomicU32; the ring body starts at SAB_HEADER_BYTES
        // which is itself 4-byte aligned, so alignment of the base pointer is the
        // sole requirement. Emscripten/WASI SABs are page-aligned but assert
        // anyway — a misaligned pointer would make every atomic op UB.
        debug_assert!(
            (ptr as usize) % 4 == 0,
            "SabSink: SAB pointer must be 4-byte aligned"
        );
        let body_len = total_len - SAB_HEADER_BYTES;
        Ok(Self { ptr, body_len })
    }
}

// ---------------------------------------------------------------------------
// Private atomic header helpers
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
impl SabSink {
    /// Return a reference to the `AtomicU32` at `ptr + offset` in the header.
    ///
    /// # Safety
    /// `offset` must be 4-byte-aligned and `offset + 4 <= SAB_HEADER_BYTES`.
    #[inline]
    unsafe fn header_atomic(&self, offset: usize) -> &AtomicU32 {
        debug_assert!(offset + 4 <= SAB_HEADER_BYTES);
        // SAFETY: caller guarantees offset is 4-byte-aligned and in-range.
        unsafe { &*(self.ptr.add(offset) as *const AtomicU32) }
    }

    #[inline]
    fn load_write_cur(&self) -> u32 {
        // SAFETY: offset 4, 4-byte aligned, within header.
        unsafe { self.header_atomic(SAB_HEADER_OFFSET_WRITE_CUR) }.load(Ordering::Acquire)
    }

    #[inline]
    fn store_write_cur(&self, v: u32) {
        // SAFETY: offset 4, 4-byte aligned, within header.
        unsafe { self.header_atomic(SAB_HEADER_OFFSET_WRITE_CUR) }.store(v, Ordering::Release);
    }

    #[inline]
    fn load_read_cur(&self) -> u32 {
        // SAFETY: offset 8, 4-byte aligned, within header.
        unsafe { self.header_atomic(SAB_HEADER_OFFSET_READ_CUR) }.load(Ordering::Acquire)
    }

    #[inline]
    fn load_cancelled(&self) -> bool {
        // SAFETY: offset 12, 4-byte aligned, within header.
        unsafe { self.header_atomic(SAB_HEADER_OFFSET_CANCELLED) }.load(Ordering::Acquire) != 0
    }
}

// ---------------------------------------------------------------------------
// Back-pressure helpers
// ---------------------------------------------------------------------------

/// Compute the number of free bytes in the ring.
///
/// The ring reserves one byte of slack so that `write == read` unambiguously
/// means "empty". Usable capacity is therefore `cap − 1`.
///
/// `write` and `read` are body offsets in `[0, cap)`.
fn free_bytes(write: u32, read: u32, cap: u32) -> u32 {
    // (read + cap - write - 1) % cap
    // When write == read this yields cap - 1 (fully empty).
    // When (write + 1) % cap == read this yields 0 (full).
    (read + cap - write - 1) % cap
}

#[cfg(target_arch = "wasm32")]
impl SabSink {
    /// Block until at least `needed` bytes are free in the ring body, or until
    /// the consumer signals cancellation.
    ///
    /// Returns `true` when space is available, `false` if cancelled.
    ///
    /// Uses `memory.atomic.wait32` on the `read_cur` slot with a 5 ms timeout
    /// so the consumer's advance of `read_cur` — or any spurious wake — is
    /// enough to re-check. See the module-level doc for the design rationale.
    fn wait_for_space(&self, needed: usize) -> bool {
        const WAIT_TIMEOUT_NS: i64 = 5_000_000; // 5 ms
        loop {
            if self.load_cancelled() {
                return false;
            }
            let write = self.load_write_cur();
            let read = self.load_read_cur();
            let free = free_bytes(write, read, self.body_len as u32);
            if free as usize >= needed {
                return true;
            }
            // Ring is full. Wait on read_cur with a short timeout. If read_cur
            // already differs from `read`, wait32 returns immediately (value
            // changed — return code 0 "not equal"). Either way we loop back.
            //
            // SAFETY: ptr + SAB_HEADER_OFFSET_READ_CUR is 4-byte aligned and
            // within the SAB allocation (validated in from_raw).
            let read_addr = unsafe { self.ptr.add(SAB_HEADER_OFFSET_READ_CUR) } as *mut i32;
            // SAFETY: read_addr is valid and 4-byte aligned.
            unsafe { wasm_atomic_wait32(read_addr, read as i32, WAIT_TIMEOUT_NS) };
        }
    }

    /// Bump `seq` by 1 (Release) and notify all waiters on that slot.
    fn bump_seq_and_notify(&self) {
        // SAFETY: SAB_HEADER_OFFSET_SEQ is 0, 4-byte aligned, within header.
        let seq = unsafe { self.header_atomic(SAB_HEADER_OFFSET_SEQ) };
        seq.fetch_add(1, Ordering::Release);
        // Wake all JS Atomics.wait() callers on the seq slot.
        // SAFETY: same address, valid and aligned.
        let seq_addr = unsafe { self.ptr.add(SAB_HEADER_OFFSET_SEQ) } as *mut i32;
        unsafe { wasm_atomic_notify(seq_addr, u32::MAX) };
    }
}

// ---------------------------------------------------------------------------
// Record writing (handles wrap-around)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
impl SabSink {
    /// Serialise `payload` into the ring body at body offset `start`,
    /// handling wrap-around with two `copy_nonoverlapping` calls if needed.
    ///
    /// The caller must have already verified that `total_len` bytes are free
    /// (via `wait_for_space`). The `store_write_cur` Release store that
    /// follows this call acts as the visibility fence for the consumer.
    fn write_record_at(&self, start: u32, payload: &EncodedPayload<'_>) {
        let start = start as usize;
        let payload_len = payload_byte_len(payload);
        let total_len = RECORD_HEADER_BYTES + payload_len;

        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.resize(total_len, 0u8);

            // Serialise into the scratch buffer. This cannot fail: the buffer
            // is exactly `total_len` bytes and `payload_byte_len` is accurate.
            let written = write_record(&mut scratch[..], payload)
                .expect("write_record into scratch must succeed");
            debug_assert_eq!(written, total_len);

            // body_base is the first byte of the ring body.
            // SAFETY: ptr + SAB_HEADER_BYTES is within the SAB allocation
            // (validated in from_raw). We never form a &mut [u8] reference;
            // only raw pointer arithmetic + copy_nonoverlapping.
            let body_base = unsafe { self.ptr.add(SAB_HEADER_BYTES) };

            if start + total_len <= self.body_len {
                // Case A — contiguous write: fits without wrapping.
                // SAFETY: src is scratch.as_ptr() (valid, total_len bytes),
                // dst is body_base + start (within body by the if-condition).
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        scratch.as_ptr(),
                        body_base.add(start),
                        total_len,
                    );
                }
            } else {
                // Case B — wrapped write: record spans the end of the body.
                let first_chunk = self.body_len - start; // bytes until body end
                let second_chunk = total_len - first_chunk; // bytes that wrap to offset 0

                // SAFETY: first_chunk <= total_len; body_base + start +
                // first_chunk == body_base + body_len (end of body), and
                // second_chunk == total_len - first_chunk, both in-range.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        scratch.as_ptr(),
                        body_base.add(start),
                        first_chunk,
                    );
                    core::ptr::copy_nonoverlapping(
                        scratch.as_ptr().add(first_chunk),
                        body_base,
                        second_chunk,
                    );
                }
            }
        });
    }

    /// Attempt to send a short error record describing an oversize payload.
    ///
    /// Silently drops if even the error record would exceed the ring capacity
    /// (cannot happen in practice — error strings are < 100 bytes).
    fn send_error_record(&self, msg: &str) {
        let payload = EncodedPayload::Error(msg.to_string());
        let payload_len = payload_byte_len(&payload);
        let total_len = RECORD_HEADER_BYTES + payload_len;

        if total_len > self.body_len - 1 {
            // Silently drop — ring is too small even for an error record.
            return;
        }
        if self.load_cancelled() {
            return;
        }
        if !self.wait_for_space(total_len) {
            return;
        }

        let write_cur_before = self.load_write_cur();
        self.write_record_at(write_cur_before, &payload);
        let write_cur_after = (write_cur_before + total_len as u32) % self.body_len as u32;
        self.store_write_cur(write_cur_after);
        self.bump_seq_and_notify();
    }
}

// ---------------------------------------------------------------------------
// payload_byte_len helper
// ---------------------------------------------------------------------------

/// Returns the byte length of the payload portion of the record (not including
/// the 4-byte record header).
fn payload_byte_len(payload: &EncodedPayload<'_>) -> usize {
    match payload {
        EncodedPayload::Text { bytes, .. } => bytes.len(),
        EncodedPayload::Json(v) => v.len(),
        EncodedPayload::Error(s) => s.len(),
    }
}

// ---------------------------------------------------------------------------
// ChatStreamSink impl
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
impl ChatStreamSink for SabSink {
    fn send(&self, result: napi::Result<ChatStreamChunk>) {
        // 1. Classify the payload.
        let payload = match classify_payload(&result) {
            Ok(p) => p,
            Err(e) => EncodedPayload::Error(format!("wire: {}", e.reason)),
        };

        // 2. Compute total_len = RECORD_HEADER_BYTES + payload_bytes.
        let payload_len = payload_byte_len(&payload);
        let total_len = RECORD_HEADER_BYTES + payload_len;

        // 3. Reject records larger than usable ring capacity (body_len - 1).
        //    Replace with a short error record so the consumer knows something
        //    went wrong. Even the error record is guaranteed small enough.
        if total_len > self.body_len - 1 {
            return self.send_error_record("wire: record exceeds ring capacity");
        }

        // 4. Early-out if already cancelled.
        if self.load_cancelled() {
            return;
        }

        // 5. Back-pressure: block until there is room (or cancelled).
        if !self.wait_for_space(total_len) {
            // Consumer cancelled while we were waiting — drop the record.
            return;
        }

        // 6. Write the record into the ring body, handling wrap-around.
        let write_cur_before = self.load_write_cur();
        self.write_record_at(write_cur_before, &payload);

        // 7. Advance write_cur (Release store — visibility fence for consumer).
        let write_cur_after = (write_cur_before + total_len as u32) % self.body_len as u32;
        self.store_write_cur(write_cur_after);

        // 8. Bump seq and wake the JS consumer.
        self.bump_seq_and_notify();
    }
}

// ---------------------------------------------------------------------------
// Unit tests — pure arithmetic, no wasm32 atomics, runs on native.
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // free_bytes arithmetic
    // -----------------------------------------------------------------------

    #[test]
    fn test_free_bytes_empty_ring() {
        // write == read == 0 → fully empty → free = cap - 1
        assert_eq!(free_bytes(0, 0, 100), 99);
    }

    #[test]
    fn test_free_bytes_half_full() {
        // write=50, read=0 → used=50, free=49
        assert_eq!(free_bytes(50, 0, 100), 49);
    }

    #[test]
    fn test_free_bytes_full() {
        // (write + 1) % cap == read → full → free = 0
        // write=99, read=0: (0 + 100 - 99 - 1) % 100 = 0
        assert_eq!(free_bytes(99, 0, 100), 0);
    }

    #[test]
    fn test_free_bytes_after_wrap() {
        // write=5, read=10 → free = (10 + 100 - 5 - 1) % 100 = 104 % 100 = 4
        assert_eq!(free_bytes(5, 10, 100), 4);
    }

    #[test]
    fn test_free_bytes_write_equals_read_unambiguous_empty() {
        // Whenever write == read the ring is considered empty (not full),
        // yielding cap - 1 free bytes. This invariant holds for all cap values.
        for cap in [8u32, 64, 128, 1024] {
            assert_eq!(
                free_bytes(0, 0, cap),
                cap - 1,
                "write==read must mean empty for cap={cap}"
            );
            // Also hold after wrap: write=cap-1, read=cap-1
            assert_eq!(
                free_bytes(cap - 1, cap - 1, cap),
                cap - 1,
                "write==read==cap-1 must mean empty for cap={cap}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Write-offset math (contiguous vs wrapped)
    // -----------------------------------------------------------------------

    /// Compute how many bytes fall in [start, body_len) and how many wrap.
    fn split_at_wrap(start: usize, total_len: usize, body_len: usize) -> (usize, usize) {
        if start + total_len <= body_len {
            (total_len, 0)
        } else {
            let first = body_len - start;
            (first, total_len - first)
        }
    }

    #[test]
    fn test_contiguous_write_no_wrap() {
        // Record of 10 bytes starting at offset 20 in a 100-byte body.
        let (first, second) = split_at_wrap(20, 10, 100);
        assert_eq!(first, 10);
        assert_eq!(second, 0);
    }

    #[test]
    fn test_write_exactly_at_end_no_wrap() {
        // Record of 10 bytes starting at offset 90 in a 100-byte body — fills
        // exactly to the end without wrapping.
        let (first, second) = split_at_wrap(90, 10, 100);
        assert_eq!(first, 10);
        assert_eq!(second, 0);
    }

    #[test]
    fn test_write_wraps_one_byte() {
        // Record of 10 bytes starting at offset 95 in a 100-byte body.
        // 5 bytes fit before the end, 5 bytes wrap.
        let (first, second) = split_at_wrap(95, 10, 100);
        assert_eq!(first, 5);
        assert_eq!(second, 5);
    }

    #[test]
    fn test_write_almost_full_wrap() {
        // Record of 10 bytes starting at offset 99 in a 100-byte body.
        // 1 byte before the end, 9 wrap.
        let (first, second) = split_at_wrap(99, 10, 100);
        assert_eq!(first, 1);
        assert_eq!(second, 9);
    }

    #[test]
    fn test_write_cur_advance_wraps() {
        // After writing a 10-byte record at write_cur=95 in a 100-byte body,
        // the new write_cur should be (95 + 10) % 100 = 5.
        let body_len = 100u32;
        let write_cur_before = 95u32;
        let total_len = 10u32;
        let write_cur_after = (write_cur_before + total_len) % body_len;
        assert_eq!(write_cur_after, 5);
    }

    #[test]
    fn test_write_cur_advance_no_wrap() {
        let body_len = 100u32;
        let write_cur_before = 20u32;
        let total_len = 10u32;
        let write_cur_after = (write_cur_before + total_len) % body_len;
        assert_eq!(write_cur_after, 30);
    }
}
