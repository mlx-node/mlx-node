//! Four-split D128 draft attention: independent F64 oracle, guarded views,
//! dynamic compiled replays, and a separately invoked operator benchmark.
use mlx_core::array::{DType, MxArray};
use std::sync::atomic::{AtomicUsize, Ordering};
const D: usize = 128;
const QH: usize = 32;
const KH: usize = 8;
const T: usize = 8;
const SCALE: f32 = 0.088388346;
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn numbers(n: usize, mut seed: u32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            half::bf16::from_f32(((seed >> 8) as f32 / 16777216.0 - 0.5) * scale).to_f32()
        })
        .collect()
}

// Padding is NaN on every side of the view, including the physical row tail.
// Accidentally taking a static full tensor slice at a partial context tile
// therefore fails the finite/F64 gates, even when the allocation is in bounds.
fn guarded_view(
    values: &[f32],
    heads: usize,
    rows: usize,
    offset: usize,
    padding: bool,
) -> MxArray {
    let capacity = rows + offset + 9;
    let width = if padding { 160 } else { D };
    let mut storage = vec![f32::NAN; heads * capacity * width];
    for h in 0..heads {
        for r in 0..rows {
            storage[(h * capacity + offset + r) * width..(h * capacity + offset + r) * width + D]
                .copy_from_slice(&values[(h * rows + r) * D..(h * rows + r + 1) * D]);
        }
    }
    MxArray::from_float32(&storage, &[1, heads as i64, capacity as i64, width as i64])
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap()
        .slice(
            &[0, 0, offset as i64, 0],
            &[1, heads as i64, (offset + rows) as i64, D as i64],
        )
        .unwrap()
}

struct Fixture {
    a: Vec<MxArray>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    mask: Vec<u8>,
    len: usize,
}
impl Fixture {
    fn new(len: usize, seed: u32, offset: usize, padding: bool, mask_mode: usize) -> Self {
        let q = numbers(QH * T * D, seed, 3.0);
        let k = numbers(KH * len * D, seed + 1, 3.0);
        let v = numbers(KH * len * D, seed + 2, 2.0);
        let mask: Vec<u8> = (0..T)
            .flat_map(|r| {
                (0..len).map(move |c| {
                    u8::from(match mask_mode {
                        0 => true,
                        // Sliding window plus holes, with current rows always visible.
                        1 => {
                            c + 2048 > len - T + r
                                && (c >= len - T || !(c + r + seed as usize).is_multiple_of(7))
                        }
                        // One visible key: the answer must exactly equal that value.
                        2 => {
                            c == [
                                0,
                                len - 1,
                                len / 2,
                                len.saturating_sub(9),
                                1,
                                len / 3,
                                len - 2,
                                len - 1,
                            ][r]
                        }
                        // Fully hidden row also checks the defined zero-output case.
                        _ => r != 3 && (c + r) % 3 != 0,
                    })
                })
            })
            .collect();
        // Query head rows must be tightly adjacent; only dimension padding is
        // allowed there. K/V deliberately have offset and head-capacity gaps.
        let qw = if padding { 160 } else { D };
        let mut qs = vec![f32::NAN; QH * T * qw];
        for r in 0..QH * T {
            qs[r * qw..r * qw + D].copy_from_slice(&q[r * D..(r + 1) * D]);
        }
        let qa = MxArray::from_float32(&qs, &[1, QH as i64, T as i64, qw as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
            .slice(&[0, 0, 0, 0], &[1, QH as i64, T as i64, D as i64])
            .unwrap();
        let ma = if padding {
            let transposed: Vec<_> = (0..len)
                .flat_map(|c| {
                    (0..T).map({
                        let m = &mask;
                        move |r| m[r * len + c]
                    })
                })
                .collect();
            MxArray::from_uint8(&transposed, &[len as i64, T as i64])
                .unwrap()
                .equal(&MxArray::from_uint8(&[1], &[1]).unwrap())
                .unwrap()
                .transpose(Some(&[1, 0]))
                .unwrap()
                .reshape(&[1, 1, T as i64, len as i64])
                .unwrap()
        } else {
            MxArray::from_uint8(&mask, &[1, 1, T as i64, len as i64])
                .unwrap()
                .equal(&MxArray::from_uint8(&[1], &[1]).unwrap())
                .unwrap()
        };
        let a = vec![
            qa,
            guarded_view(&k, KH, len, offset, padding),
            guarded_view(&v, KH, len, offset + 2, padding),
            ma,
        ];
        Self {
            a,
            q,
            k,
            v,
            mask,
            len,
        }
    }
    fn call(&self, masked: bool) -> Raw {
        let mut out = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                mlx_sys::mlx_dflash2_attention(
                    self.a[0].as_raw_ptr(),
                    self.a[1].as_raw_ptr(),
                    self.a[2].as_raw_ptr(),
                    if masked {
                        self.a[3].as_raw_ptr()
                    } else {
                        std::ptr::null_mut()
                    },
                    SCALE,
                    &mut out,
                )
            },
            1
        );
        Raw::new(out)
    }
    fn native(&self, masked: bool) -> Raw {
        Raw::new(unsafe {
            mlx_sys::mlx_fast_scaled_dot_product_attention(
                self.a[0].as_raw_ptr(),
                self.a[1].as_raw_ptr(),
                self.a[2].as_raw_ptr(),
                SCALE,
                c"".as_ptr(),
                if masked {
                    self.a[3].as_raw_ptr()
                } else {
                    std::ptr::null_mut()
                },
                masked,
            )
        })
    }
    fn oracle(&self, masked: bool) -> Vec<f64> {
        let mut out = vec![0.0; QH * T * D];
        for h in 0..QH {
            for r in 0..T {
                let query = &self.q[(h * T + r) * D..(h * T + r + 1) * D];
                let scores: Vec<f64> = (0..self.len)
                    .map(|c| {
                        if masked && self.mask[r * self.len + c] == 0 {
                            return f64::NEG_INFINITY;
                        }
                        let key =
                            &self.k[((h / 4) * self.len + c) * D..((h / 4) * self.len + c + 1) * D];
                        query
                            .iter()
                            .zip(key)
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum::<f64>()
                            * f64::from(SCALE)
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if max == f64::NEG_INFINITY {
                    continue;
                }
                let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = weights.iter().sum();
                for (c, w) in weights.into_iter().enumerate() {
                    for d in 0..D {
                        out[(h * T + r) * D + d] +=
                            w * f64::from(self.v[((h / 4) * self.len + c) * D + d]) / sum;
                    }
                }
            }
        }
        out
    }
    fn check(&self, out: &Raw, masked: bool) {
        let got = out.floats();
        let expected = self.oracle(masked);
        let peak = expected
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max)
            .max(1e-8);
        let mut max: f64 = 0.0;
        let mut squares = 0.0;
        for (g, e) in got.iter().zip(&expected) {
            assert!(g.is_finite(), "non-finite output len={}", self.len);
            let diff = (f64::from(*g) - e).abs();
            max = max.max(diff);
            squares += diff * diff;
        }
        let rms = (squares / got.len() as f64).sqrt();
        eprintln!(
            "draft_attention len={} max/peak={:.6} rms/peak={:.6}",
            self.len,
            max / peak,
            rms / peak
        );
        assert!(
            max <= peak * 0.004 + 2e-6,
            "len={} max={max} peak={peak}",
            self.len
        );
        assert!(
            rms <= peak * 0.0015 + 1e-6,
            "len={} rms={rms} peak={peak}",
            self.len
        );
    }
}
struct Raw(*mut mlx_sys::mlx_array);
impl Raw {
    fn new(h: *mut mlx_sys::mlx_array) -> Self {
        assert!(!h.is_null());
        Self(h)
    }
    fn floats(&self) -> Vec<f32> {
        let f = Raw::new(unsafe { mlx_sys::mlx_array_astype(self.0, DType::Float32 as i32) });
        assert!(unsafe { mlx_sys::mlx_eval([f.0].as_mut_ptr(), 1) });
        let mut out = vec![0.0; unsafe { mlx_sys::mlx_array_size(f.0) }];
        assert!(unsafe { mlx_sys::mlx_array_to_float32(f.0, out.as_mut_ptr(), out.len()) });
        out
    }
}
impl Drop for Raw {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_array_delete(self.0) }
    }
}

#[test]
fn draft_attention_oracle_masks_and_guarded_partial_tiles() {
    let _guard = SERIAL.lock().unwrap();
    if !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
        return;
    }
    unsafe { std::env::set_var("MLX_DFLASH2_ATTENTION_NAX", "1") };
    for (i, len) in [8, 9, 63, 127, 128, 129, 135, 136, 137, 1023, 2048, 2055]
        .into_iter()
        .enumerate()
    {
        let f = Fixture::new(len, 71 + i as u32, 1 + i * 11, i % 2 == 0, 1);
        let got = f.call(true);
        f.check(&got, true);
        let native = f.native(true).floats();
        let result = got.floats();
        let peak = native.iter().map(|v| v.abs()).fold(1e-8, f32::max);
        let max = native
            .iter()
            .zip(result)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max <= peak * 0.012 + 2e-6,
            "native reference drift len={len}: max={max} peak={peak}"
        );
    }
    for mode in [0, 2, 3] {
        let f = Fixture::new(2055, 101 + mode as u32, 127, true, mode);
        let got = f.call(true);
        f.check(&got, true);
        if mode == 2 {
            assert_eq!(
                got.floats(),
                f.oracle(true)
                    .into_iter()
                    .map(|x| x as f32)
                    .collect::<Vec<_>>()
            );
        }
    }
    let f = Fixture::new(137, 151, 63, true, 1);
    f.check(&f.call(false), false);
    unsafe { std::env::remove_var("MLX_DFLASH2_ATTENTION_NAX") };
}

unsafe extern "C" fn build(
    context: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    n_inputs: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    n_outputs: usize,
) -> bool {
    if context.is_null() || n_inputs != 4 || n_outputs != 1 {
        return false;
    }
    unsafe { &*context.cast::<AtomicUsize>() }.fetch_add(1, Ordering::Relaxed);
    let a: Vec<_> = unsafe { std::slice::from_raw_parts(inputs, n_inputs) }
        .iter()
        .map(|h| h.cast_mut())
        .collect();
    let result = unsafe { mlx_sys::mlx_dflash2_attention(a[0], a[1], a[2], a[3], SCALE, outputs) };
    for handle in a {
        unsafe { mlx_sys::mlx_array_delete(handle) }
    }
    result == 1
}
struct Compiled(u64);
impl Drop for Compiled {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, self.0) };
    }
}
#[test]
fn draft_attention_compiled_replay_reads_lengths_strides_offsets_and_masks() {
    let _guard = SERIAL.lock().unwrap();
    if !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
        return;
    }
    unsafe { std::env::set_var("MLX_DFLASH2_ATTENTION_NAX", "1") };
    let key = Compiled(0xDFA2_A770_0000_0001);
    let mut traces = AtomicUsize::new(0);
    for (i, len) in [137, 8, 2055, 9, 128, 2048, 137].into_iter().enumerate() {
        let f = Fixture::new(len, 181 + i as u32, i * 31, i % 2 == 1, i % 4);
        let inputs: Vec<_> = f.a.iter().map(|a| a.as_raw_ptr().cast_const()).collect();
        let mut out = std::ptr::null_mut();
        assert!(unsafe {
            mlx_sys::mlx_compiled_graph_invoke(
                key.0,
                Some(build),
                (&mut traces as *mut AtomicUsize).cast(),
                inputs.as_ptr(),
                inputs.len(),
                &mut out,
                1,
                true,
            )
        });
        let out = Raw::new(out);
        assert_eq!(out.floats(), f.call(true).floats(), "replay len={len}");
        f.check(&out, true);
        assert_eq!(traces.load(Ordering::Relaxed), 1);
    }
    unsafe { std::env::remove_var("MLX_DFLASH2_ATTENTION_NAX") };
}
#[test]
fn draft_attention_declines_unsupported_shapes_and_dtypes() {
    let _guard = SERIAL.lock().unwrap();
    if !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
        return;
    }
    unsafe { std::env::set_var("MLX_DFLASH2_ATTENTION_NAX", "1") };
    let f = Fixture::new(137, 211, 0, false, 0);
    let q32 = f.a[0].astype(DType::Float32).unwrap();
    let q7 = f.a[0].slice(&[0, 0, 0, 0], &[1, 32, 7, 128]).unwrap();
    let bad_mask = f.a[3].astype(DType::Float32).unwrap();
    for (q, mask) in [(&q32, &f.a[3]), (&q7, &f.a[3]), (&f.a[0], &bad_mask)] {
        let mut out = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                mlx_sys::mlx_dflash2_attention(
                    q.as_raw_ptr(),
                    f.a[1].as_raw_ptr(),
                    f.a[2].as_raw_ptr(),
                    mask.as_raw_ptr(),
                    SCALE,
                    &mut out,
                )
            },
            0
        );
        assert!(out.is_null());
    }
    for index in 0..3 {
        for offset in [1, 4] {
            let shape = f.a[index].shape().unwrap();
            let unaligned = MxArray::zeros(&[1, shape[1], shape[2], 136], Some(DType::BFloat16))
                .unwrap()
                .slice(&[0, 0, 0, offset], &[1, shape[1], shape[2], 128 + offset])
                .unwrap();
            // Lazy Slice finalizes its byte offset only when evaluated.
            unaligned.eval();
            assert_eq!(
                unsafe { mlx_sys::mlx_array_get_buffer_offset(unaligned.as_raw_ptr()) } % 16,
                (offset as usize * 2) % 16
            );
            let mut a: Vec<_> = f.a.iter().map(MxArray::as_raw_ptr).collect();
            a[index] = unaligned.as_raw_ptr();
            let mut out = std::ptr::null_mut();
            assert_eq!(
                unsafe { mlx_sys::mlx_dflash2_attention(a[0], a[1], a[2], a[3], SCALE, &mut out) },
                0,
                "unaligned element offset {offset} must fall back for input {index}"
            );
            assert!(out.is_null());
        }
    }
    unsafe { std::env::remove_var("MLX_DFLASH2_ATTENTION_NAX") };
}

#[test]
#[ignore = "operator timing requires the exclusive GPU slot"]
fn draft_attention_microbenchmark() {
    let _guard = SERIAL.lock().unwrap();
    if !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
        return;
    }
    unsafe { std::env::set_var("MLX_DFLASH2_ATTENTION_NAX", "1") };
    for len in [128, 2055] {
        let fixtures: Vec<_> = (0..16)
            .map(|i| Fixture::new(len, 251 + i, 31, false, 1))
            .collect();
        for nax in [false, true, true, false] {
            let mut samples = Vec::new();
            for iteration in 0..47 {
                let outputs: Vec<_> = fixtures
                    .iter()
                    .map(|f| if nax { f.call(true) } else { f.native(true) })
                    .collect();
                let mut roots: Vec<_> = outputs.iter().map(|x| x.0).collect();
                let start = std::time::Instant::now();
                assert!(unsafe { mlx_sys::mlx_eval(roots.as_mut_ptr(), roots.len()) });
                if iteration >= 16 {
                    samples.push(start.elapsed().as_secs_f64() * 1e6 / outputs.len() as f64);
                }
            }
            samples.sort_by(f64::total_cmp);
            eprintln!(
                "draft_attention len={len} nax={nax} median_us={} p10_us={}",
                samples[15], samples[3]
            );
        }
    }
    unsafe { std::env::remove_var("MLX_DFLASH2_ATTENTION_NAX") };
}

#[test]
fn draft_attention_output_flattens_without_copy() {
    let _guard = SERIAL.lock().unwrap();
    if !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
        return;
    }
    unsafe { std::env::set_var("MLX_DFLASH2_ATTENTION_NAX", "1") };
    for (i, len) in [8, 137, 2055].into_iter().enumerate() {
        let f = Fixture::new(len, 291 + i as u32, 19 + i, i % 2 == 0, i + 1);
        let out = f.call(true);
        let logical = out.floats();
        let transposed =
            Raw::new(unsafe { mlx_sys::mlx_array_transpose(out.0, [0, 2, 1, 3].as_ptr(), 4) });
        let flattened =
            Raw::new(unsafe { mlx_sys::mlx_array_reshape(transposed.0, [1, 8, 4096].as_ptr(), 3) });
        assert!(unsafe { mlx_sys::mlx_eval([out.0, flattened.0].as_mut_ptr(), 2) });
        let base = unsafe { mlx_sys::mlx_array_get_metal_buffer(out.0) };
        assert!(!base.is_null());
        assert_eq!(
            base,
            unsafe { mlx_sys::mlx_array_get_metal_buffer(flattened.0) },
            "flatten copied the attention result at key length {len}"
        );
        assert_eq!(
            unsafe { mlx_sys::mlx_array_get_buffer_offset(out.0) },
            unsafe { mlx_sys::mlx_array_get_buffer_offset(flattened.0) }
        );
        let flat = flattened.floats();
        for row in 0..T {
            for head in 0..QH {
                for dim in 0..D {
                    assert_eq!(
                        flat[(row * QH + head) * D + dim].to_bits(),
                        logical[(head * T + row) * D + dim].to_bits()
                    );
                }
            }
        }
        f.check(&out, true);
    }
}

#[test]
fn draft_attention_explicit_policy_preserves_fallbacks_and_ignores_environment() {
    let _guard = SERIAL.lock().unwrap();
    if !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
        return;
    }
    struct Restore(Option<std::ffi::OsString>);
    impl Drop for Restore {
        fn drop(&mut self) {
            unsafe {
                match self.0.take() {
                    Some(value) => std::env::set_var("MLX_DFLASH2_ATTENTION_NAX", value),
                    None => std::env::remove_var("MLX_DFLASH2_ATTENTION_NAX"),
                }
            }
        }
    }
    let _restore = Restore(std::env::var_os("MLX_DFLASH2_ATTENTION_NAX"));
    let f = Fixture::new(137, 419, 0, false, 1);
    let q32 = f.a[0].astype(DType::Float32).unwrap();
    let q7 = f.a[0].slice(&[0, 0, 0, 0], &[1, 32, 7, 128]).unwrap();
    let mut expected = None;
    for legacy in ["0", "1"] {
        unsafe { std::env::set_var("MLX_DFLASH2_ATTENTION_NAX", legacy) };
        for (q, enabled, want) in [
            (&f.a[0], false, 0),
            (&f.a[0], true, 1),
            (&q32, true, 0),
            (&q7, true, 0),
        ] {
            let mut output = std::ptr::null_mut();
            let status = unsafe {
                mlx_sys::mlx_dflash2_attention_with_policy(
                    q.as_raw_ptr(),
                    f.a[1].as_raw_ptr(),
                    f.a[2].as_raw_ptr(),
                    f.a[3].as_raw_ptr(),
                    SCALE,
                    enabled,
                    &mut output,
                )
            };
            assert_eq!(status, want);
            if want == 0 {
                assert!(output.is_null());
            } else {
                let output = Raw::new(output);
                f.check(&output, true);
                let bits = output.floats();
                if let Some(expected) = &expected {
                    assert_eq!(&bits, expected);
                } else {
                    expected = Some(bits);
                }
            }
        }
    }
}
