//! K-quant dispatch guards: what must throw, what must work, what must not
//! regress.
//!
//! The numeric gate against ggml lives in `kquant_ggml_parity.rs`; this file
//! guards the shape of the support matrix around it:
//!
//! * `quantize` throws for every K-quant mode on every device — these formats
//!   are consumed, never produced;
//! * `dequantize` / `quantized_matmul` / `gather_qmm` work on the CPU, and on
//!   Metal every kernel the dispatcher can reach agrees with the CPU
//!   reference — all 33 `float` kernels `kquant.metal` instantiates per mode,
//!   plus the float16 and bfloat16 variants of `qmv`, `qmm_t` and
//!   `qmm_t_splitk`. On a device with the NAX tensor-op kernels the `qmm_t`
//!   shapes reach `kquant_qmm_t_nax` (`kquant_nax.metal`) instead of the
//!   simdgroup `kquant_qmm_t`, covering all four of its instantiations plus its
//!   unaligned-M branch and a row spanning several super-blocks;
//! * the validators reject malformed K-quant tensors, so the throws above prove
//!   something about the backends rather than about a missing argument check;
//! * the pre-existing affine / mxfp4 modes still work through the widened
//!   validator.
//!
//! `qqmm` is in the C++ smoke but is not reachable from here: mlx-sys exposes no
//! `mlx_qqmm` binding, so no Rust caller can construct that op.
//!
//! Run:
//!   cargo test -p mlx-core --release --test kquant_mode_guards
//!
//! That run reaches whichever kernels the host dispatches to, which is not the
//! whole matrix on any single machine, so two more spellings are worth knowing:
//!
//!   MLX_METAL_GPU_ARCH=applegpu_g16s cargo test ...    # simdgroup fallback
//!
//! `Device::Device` parses the GPU generation out of that string rather than
//! off the hardware (`metal/device.cpp:589-601`), so a generation below 17
//! fails the check at :956 and every NAX branch turns off. No rebuild, and
//! nothing else moves with it: of the four places the Metal backend reads the
//! generation, the other three test `== 13 || == 14`, `>= 15` and `< 15`
//! (`quantized.cpp:209`, `:429`, `matmul.cpp:1265`) and so read 16 and 17
//! alike. This is the only way to exercise the float16/bfloat16 simdgroup
//! `qmm_t` on a NAX box, since `MLX_ENABLE_TF32=0` reroutes float32 only.
//! `applegpu_g14s` goes further and also turns off `use_qmv_wide`
//! (`quantized.cpp:429`), matching an M1/M2 host. Keep the trailing size
//! letter: it feeds `max_ops_per_buffer` and friends (`device.cpp:602`).
//!
//!   KQ_NAX_REQUIRE=1 cargo test ...                   # fail if NAX is absent
//!
//! turns a missing-NAX host from a silent skip into a failure, for a box that
//! is supposed to have the kernels. See `nax_required`.

use std::ffi::CString;

use mlx_core::array::{DType, MxArray};

/// K = 256, N = 128 — one super-block per row, and an output dim that is a
/// multiple of both 8 (`qmv_fast`'s row tile) and 32 (`qmm_t`'s `aligned_N`).
const K: i64 = 256;
const N: i64 = 128;

/// N % 32 != 0, which is what picks the `_alN_false` kernels.
const N_UNALIGNED: i64 = 136;

/// Four super-blocks per row rather than one, so a flat group index crosses a
/// super-block boundary while walking a single row.
const K_DEEP: i64 = 1024;

/// The packed axis of a non-transposed weight is the *output* dim, so it is the
/// one that has to be a whole number of 256-element super-blocks.
const N_PACKED: i64 = 256;

struct KQuant {
    mode: &'static str,
    bits: i32,
    group_size: i32,
    scales_signed: bool,
    weight_cols: i64,
    scales_cols: i64,
    biases_cols: i64,
}

const KQUANTS: [KQuant; 3] = [
    KQuant {
        mode: "q6k",
        bits: 6,
        group_size: 16,
        scales_signed: true,
        weight_cols: 48,
        scales_cols: 16,
        biases_cols: 1,
    },
    KQuant {
        mode: "q4k",
        bits: 4,
        group_size: 32,
        scales_signed: false,
        weight_cols: 32,
        scales_cols: 16,
        biases_cols: 2,
    },
    KQuant {
        mode: "q5k",
        bits: 5,
        group_size: 32,
        scales_signed: false,
        weight_cols: 40,
        scales_cols: 16,
        biases_cols: 2,
    },
];

// ---------------------------------------------------------------------------
// tolerances
// ---------------------------------------------------------------------------

// A K-quant scale is `d * sc`, a float16 super-block scale times a signed 7-bit
// (q6k) or unsigned 6-bit (q4k/q5k) sub-block scale, and the decoded value is
// `scale * code + bias`. Every one of those products is exactly representable
// in float32, so with float32 activations the CPU reference and Metal evaluate
// the *same* real numbers per element and can only differ in how they sum them:
//
// * `qmv`, `qmv_fast`, `qmv_wide`, `qvm`, `qvm_split_k` and the gather vector
//   kernels decode straight into a float accumulator (`typedef float U`,
//   kquant.h:807);
// * the `qmm` family rounds each decoded weight to the activation type on the
//   way into the threadgroup tile (`dequantize_to`, kquant.h:579-589) while the
//   CPU keeps it in float (`kq_qmm_t`, cpu/quantized.cpp:1212) — with float32
//   activations that rounding is a no-op.
//
// So float32 is the numeric gate, and it is held to a float32 summation-order
// tolerance. Worst case over the shapes below: 7.5e-7 relative (`qmv_fast
// batch_1 q5k`). That bound is per-shape, not per-kernel — the same `qmv`
// kernel measures 7.7e-7 at K = 768 — so the constant carries headroom for
// shapes the gate does not drive rather than sitting just above the maximum.
//
// The whole `qmm` / `gather_qmm` family in fact comes out bit for bit equal.
// That is not a degenerate comparison — `compare_devices` refuses a reference
// that peaks below 1 — it is that those kernels dequantize into a tile and then
// evaluate `Sum(x_i * w_i)` in K order, which is what the CPU does. The vector
// kernels are the ones that drift, because `qdot` evaluates the algebraically
// equal but numerically different `scale * Sum(w_i x_i) + Sum(x_i) * bias`.
const F32_TOL: f32 = 1e-5;

// `qvm_split_k` is the one shape where the two backends sum different *depths*:
// the CPU walks all K terms into one float accumulator (`kq_qmm`,
// cpu/quantized.cpp:1156) while Metal sums K/split_k terms per partition and
// then reduces. At K = 16384 the sum of magnitudes runs ~sqrt(K) = 128x the
// result, so a relative-to-peak tolerance has to carry that factor. Measured
// 1.2e-6 at K = 1024 and 3.7e-6 at K = 16384; 3e-5 leaves 8x.
const F32_DEEP_TOL: f32 = 3e-5;

// `qmm` with a transposed weight routes to the NAX tensor-op kernel whenever
// the device has one (`metal/quantized.cpp:968`). Its float32 gate is
// `enable_tf32() || dtype != float32` and `MLX_ENABLE_TF32` defaults on
// (mlx/utils.h:196), so on a NAX device a float32 `qmm_t` runs the tensor op at
// tf32 — 10 mantissa bits into a float accumulator — against a CPU reference
// that keeps all 24. That is a precision mode, not a decode difference: it
// applies to affine identically, and the affine control at the end of
// `gpu_matches_cpu_on_every_quantized_matmul_kernel` measures it on affine
// weights with no K-quant code in the path.
// Measured 2.28e-4 to 4.36e-4 over the six NAX `qmm_t` shapes x three K-quants,
// and 4.28e-4 on the affine control — the control lands inside that band rather
// than outside it, i.e. no K-quant shape is an outlier against a path with no
// K-quant code in it, which is what rules out a K-quant-specific defect. 3e-3
// leaves 6.9x on the widest, the headroom the tolerances above carry. Running
// the suite with `MLX_ENABLE_TF32=0` puts the same float32 shapes back on the
// simdgroup `kquant_qmm_t` kernel, where they hold `F32_TOL`; every gap here
// then collapses to zero and the run says so.
const NAX_TF32_TOL: f32 = 3e-3;

// A two-sided band, not a one-sided ceiling: the tf32 control must keep showing
// a gap of its own, or NAX_TF32_TOL is over-generous and wants re-measuring.
// The floor is asserted whenever `nax_tf32_active()` says the float32 shapes
// really are on the tensor op, matching the AFFINE_BF16_GAP_FLOOR assert below;
// off a NAX device, or under MLX_ENABLE_TF32=0, the gap is legitimately zero
// and the run says so instead of failing.
const AFFINE_TF32_GAP_FLOOR: f32 = 1e-6;
const AFFINE_TF32_GAP_CEILING: f32 = 1e-2;

// The same floor for the K-quant side, and the positive proof that
// `kquant_qmm_t_nax` executed. The simdgroup `qmm` family is bit-for-bit equal
// to the CPU (see F32_TOL above), so a float32 K-quant `qmm_t` landing inside
// F32_TOL means it ran there and the NAX kernel did not. Only the tensor op's
// 10-bit mantissa produces a gap of this size, so requiring the gap to clear
// this constant turns "the K-quants were within tolerance" into "the K-quant
// NAX kernel ran". Closing `nax_supports_mode` (metal/quantized.cpp:66)
// collapses every gap to exactly 0 and fires this, where NAX_TF32_TOL alone
// passes trivially.
// The aggregate per mode is the SMALLEST gap over the six NAX `qmm_t` shapes,
// not the largest: a `max` would stay pinned by whichever shape still reached
// the tensor op and so could not see a regression confined to one
// instantiation — an alignment-rule or `nax_supports_mode` change is exactly
// that shape of bug. Taking the minimum makes every shape carry the proof.
// Measured smallest-per-mode 2.276e-4 (q6k), 3.194e-4 (q4k), 3.348e-4 (q5k),
// so the constant sits 4.5x under the tightest of them and 5x over F32_TOL. It
// separates the two regimes by orders of magnitude rather than by a percentage:
// on the tensor op the gap is thousands of times the floor, on the simdgroup
// kernel it is exactly 0.
const NAX_TF32_GAP_FLOOR: f32 = 5e-5;

// float16 / bfloat16 activations are dtype-instantiation coverage, not the
// numeric gate, and they carry two extra roundings the CPU does not have.
//
// The vector kernels factor the group bias out of the dot product:
// `qdot` returns `scale * Sum(w_i x_i) + Sum(x_i) * bias` (kquant.h:298) where
// the CPU evaluates `Sum(x_i * (scale * w_i + bias))` (cpu/quantized.cpp:1244).
// `Sum(x_i)` is built four terms at a time in `load_vector` (kquant.h:48) and
// `bfloat + bfloat` is `bfloat` in MSL, so on bfloat16 that partial sum rounds.
// The activations below are 8-bit integers over 128, so four of them need up to
// 10 significant bits: exact in float16's 11, not in bfloat16's 8.
//
// q6k then amplifies the error, because its bias is `-32 * scale` — the two
// terms of `qdot` are ~32x the result they add up to. That is visible in the
// measurements: on bfloat16 q6k lands at 7.5e-3 relative where q4k and q5k,
// whose bias is an independent `-dmin * m`, sit at 6.1e-4 and 4.7e-4. On
// float16 all three are at or below 2.4e-6.
//
// None of this is K-quant specific: `load_vector` and `qdot` are byte-identical
// to MLX's affine versions in `quantized.h`, and
// `affine_shows_the_same_bfloat16_vector_gap` measures 2.6e-2 on affine 4-bit
// weights arranged the same way — a wider gap than any K-quant here.
const F16_STORE_TOL: f32 = 1e-3;
const BF16_STORE_TOL: f32 = 3e-2;

// The `qmm` family instead rounds every decoded weight to the activation type
// in the threadgroup tile (kquant.h:587), which the CPU does not do: one
// T-rounding per term accumulated over K = 256 terms, up to sqrt(256) * 2^-9 =
// 3.1e-2 of the sum of magnitudes. Measured 5.8e-4 on float16 and 4.6e-3 on
// bfloat16.
const F16_TILE_TOL: f32 = 4e-3;
const BF16_TILE_TOL: f32 = 3e-2;

// `qmm_t_splitk` adds a second rounding on top of that tile rounding: it writes
// each K-partition's partial sum into an intermediate array built with the
// *activation* dtype (`array intermediate(temp_shape, x.dtype(), ...)`,
// metal/quantized.cpp:1101) and only then reduces across partitions, so a
// bfloat16 activation costs 8 bits of mantissa per partial where the CPU keeps
// one float accumulator over the whole of K. This is the float16/bfloat16
// analogue of what F32_DEEP_TOL covers for `qvm_split_k`.
//
// Measured over the six shapes (three modes x two alignments) at split_k = 8:
// worst 1.1e-3 on float16 and 6.0e-3 on bfloat16, against 6.4e-4 and 5.1e-3 for
// the same modes without split-K. Separate constants rather than reusing the
// tile ones because folding the wider number in would quietly cut float16's
// headroom from 6.2x to 3.6x and hide a regression in the partial store.
const F16_SPLITK_TOL: f32 = 6e-3;
const BF16_SPLITK_TOL: f32 = 4e-2;

// The affine control below must keep showing a bfloat16 gap of its own,
// otherwise the two constants above are over-generous and want re-measuring.
// It is a measurement, not a gate, so its ceiling is only there to catch a
// blow-up: it measured 2.6e-2 and is not held to the K-quant tolerance.
const AFFINE_BF16_GAP_FLOOR: f32 = 1e-4;
const AFFINE_BF16_GAP_CEILING: f32 = 1e-1;

/// 0 = CPU, 1 = GPU, matching `to_device_helper` in
/// `crates/mlx-sys/src/mlx_stream.cpp:28`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Device {
    Cpu,
    Gpu,
}

impl Device {
    fn code(self) -> i32 {
        match self {
            Self::Cpu => 0,
            Self::Gpu => 1,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

/// Pin MLX's process-wide default device and stream. `.cargo/config.toml` pins
/// `RUST_TEST_THREADS=1`, so mutating the global here is safe.
/// Returns false when the device is unavailable (e.g. no Metal).
fn select(device: Device) -> bool {
    // SAFETY: plain global setters in the MLX FFI; both catch internally.
    unsafe {
        mlx_sys::mlx_set_default_device(device.code());
        if mlx_sys::mlx_default_device() != device.code() {
            return false;
        }
        let stream = mlx_sys::mlx_default_stream(device.code());
        mlx_sys::mlx_set_default_stream(stream);
    }
    true
}

/// Whether this host has the NAX tensor-op kernels at all.
///
/// `metal::is_nax_available()` folds the GPU generation (>= 17, or >= 18 when
/// the architecture string ends in 'p'), the macOS 26.2 requirement and the
/// `MLX_METAL_NO_NAX` compile-out into one cached bool
/// (`metal/device.cpp:944-962`), and is the first thing the dispatcher's own
/// gate reads (`metal/quantized.cpp:968`). `mlx_gpu_architecture_gen()` is *not*
/// a substitute: it is a separate parse in `mlx_gated_delta.cpp:407-428` that
/// knows none of the last three.
fn nax_available() -> bool {
    // SAFETY: a nullary FFI predicate that catches internally and reports false
    // when the Metal device is unavailable (`mlx-sys/src/mlx_stream.cpp:140`).
    unsafe { mlx_sys::mlx_metal_is_nax_available() }
}

/// Whether the caller has declared that this host is supposed to have the NAX
/// kernels, so their absence is a failure rather than a skip.
///
/// Every NAX-specific assertion in this file is conditioned on the host having
/// them, which is the right default — the simdgroup fallback is a supported
/// configuration that CI's runners exercise on every push, not a failure. The
/// cost is that a host which silently stops being a NAX host still reports ok,
/// with the whole band quietly skipped. Setting this on a box that is meant to
/// have the kernels closes that gap. `kquant_nax_bench.rs` reads the same
/// variable for the same reason, so one setting covers both gates.
fn nax_required() -> bool {
    std::env::var("KQ_NAX_REQUIRE").as_deref() == Ok("1")
}

/// Whether a float32 `qmm_t` on this host actually reaches the NAX tensor op.
///
/// Both halves of the dispatcher's gate (`metal/quantized.cpp:968-970`):
///
/// * `nax_available()`, above;
/// * `env::enable_tf32()` — `get_var("MLX_ENABLE_TF32", 1)` (`mlx/utils.h:196`),
///   which reads the variable with `atoi` (`mlx/utils.cpp:288`): unset is on,
///   "0" is off. Anything that is not a clean nonzero integer is read as off
///   here, which is the conservative direction — a value this disagrees with
///   `atoi` about can only make the band skip, never fire wrongly.
///
/// float16 and bfloat16 satisfy the gate's `dtype != float32` arm whatever tf32
/// says; this is only about the float32 comparisons.
fn nax_tf32_active() -> bool {
    let tf32 = match std::env::var("MLX_ENABLE_TF32") {
        Ok(v) => v.trim().parse::<i32>().is_ok_and(|n| n != 0),
        Err(_) => true,
    };
    nax_available() && tf32
}

/// The tolerance for the float32 `qmm_t` shapes, which is a property of the
/// kernel that will run rather than of the shape.
///
/// Only the tensor op needs `NAX_TF32_TOL`: its 10-bit mantissa is what opens
/// the ~3e-4 gap. Everywhere else — off a NAX device, or under
/// `MLX_ENABLE_TF32=0` — these same shapes take the simdgroup `kquant_qmm_t`,
/// which is bit-for-bit equal to the CPU (see `F32_TOL`). Holding those runs to
/// the tf32 tolerance would leave a 300x band in which a real simdgroup
/// regression passes unnoticed, so the tolerance follows the kernel.
fn qmm_t_f32_tol() -> f32 {
    if nax_tf32_active() {
        NAX_TF32_TOL
    } else {
        F32_TOL
    }
}

fn zeros(shape: &[i64], dtype: DType) -> MxArray {
    MxArray::zeros(shape, Some(dtype)).expect("zeros")
}

/// Force a lazy graph to run and report the error text if it throws.
fn eval(handle: *mut mlx_sys::mlx_array) -> Result<(), String> {
    let mut buf = [0i8; 512];
    let mut h = handle;
    // SAFETY: `handle` is a live array; `buf` is a 512-byte sink for the error.
    let ok = unsafe { mlx_sys::mlx_eval_with_error(&mut h, 1, buf.as_mut_ptr(), buf.len()) };
    if ok {
        return Ok(());
    }
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    Err(String::from_utf8_lossy(&bytes).into_owned())
}

fn drop_array(handle: *mut mlx_sys::mlx_array) {
    if !handle.is_null() {
        // SAFETY: the handle is owned here and not referenced afterwards.
        unsafe { mlx_sys::mlx_array_delete(handle) };
    }
}

/// Result of one op attempt: either it produced values, or it was rejected —
/// at construction (null handle) or at eval (error text).
enum Attempt {
    Ok(Vec<i64>),
    Rejected(String),
}

fn finish(handle: *mut mlx_sys::mlx_array) -> Attempt {
    if handle.is_null() {
        return Attempt::Rejected("rejected at construction".to_string());
    }
    if let Err(e) = eval(handle) {
        drop_array(handle);
        return Attempt::Rejected(e);
    }
    // SAFETY: `handle` is a live, evaluated array.
    let ndim = unsafe { mlx_sys::mlx_array_ndim(handle) };
    let mut shape = vec![0i64; ndim];
    // SAFETY: `shape` has room for `ndim` dimensions.
    unsafe { mlx_sys::mlx_array_shape(handle, shape.as_mut_ptr()) };
    drop_array(handle);
    Attempt::Ok(shape)
}

fn expect_rejected(what: &str, attempt: Attempt) {
    match attempt {
        Attempt::Rejected(why) => println!("  rejects  {what}  ->  {why}"),
        Attempt::Ok(shape) => panic!("{what} was accepted (shape {shape:?}); it must be rejected"),
    }
}

fn expect_shape(what: &str, want: &[i64], attempt: Attempt) {
    match attempt {
        Attempt::Ok(shape) => {
            assert_eq!(shape, want, "{what} produced the wrong shape");
            println!("  works    {what}");
        }
        Attempt::Rejected(why) => panic!("{what} was rejected: {why}"),
    }
}

/// Evaluate an op result and read it back element by element as float32.
///
/// Element-wise rather than `mlx_array_to_float32`, which materialises through
/// `add(arr, zeros)` — the same reason `kquant_ggml_parity.rs` reads this way.
fn expect_f32(what: &str, handle: *mut mlx_sys::mlx_array) -> Vec<f32> {
    assert!(!handle.is_null(), "{what} was rejected at construction");
    if let Err(why) = eval(handle) {
        drop_array(handle);
        panic!("{what} was rejected: {why}");
    }
    // SAFETY: `handle` is a live, evaluated array.
    let len = unsafe { mlx_sys::mlx_array_size(handle) };
    let mut out = vec![0f32; len];
    for (i, slot) in out.iter_mut().enumerate() {
        // SAFETY: i < len == array size.
        let ok = unsafe { mlx_sys::mlx_array_item_at_float32(handle, i, slot) };
        assert!(ok, "{what}: mlx_array_item_at_float32 failed at index {i}");
    }
    drop_array(handle);
    out
}

// ---------------------------------------------------------------------------
// operands
// ---------------------------------------------------------------------------

/// A weight and its two companions, in the order the ops take them.
type Weights = (MxArray, MxArray, MxArray);

fn kquant_weights(kq: &KQuant) -> Weights {
    let scales_dtype = if kq.scales_signed {
        DType::Int8
    } else {
        DType::Uint8
    };
    (
        zeros(&[N, kq.weight_cols], DType::Uint32),
        zeros(&[N, kq.scales_cols], scales_dtype),
        zeros(&[N, kq.biases_cols], DType::Float16),
    )
}

/// Deterministic pseudo-random fill. A plain LCG so both devices see the same
/// bytes and a failure reproduces.
fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// float16 bit patterns for the super-block scale `d` (and `dmin`): small
/// powers of two plus two odd mantissas, so `d * sc` is not trivially exact.
const HALF_SCALES: [u16; 6] = [0x3000, 0x2C00, 0x3400, 0x2E66, 0x3155, 0x2800];

/// K-quant tensors with non-trivial codes and both scale levels populated.
///
/// `leading` is everything before the packed dimension and `packed` is the
/// number of values that dimension expands to. For a transposed weight that is
/// `(&[N], K)`; for a non-transposed one the packed axis is the output dim, so
/// it is `(&[K], N)`. An expert stack prepends to `leading`.
///
/// All three arrays hold a fixed number of entries per 256-value super-block,
/// so the `KQUANTS` column counts — which are stated for one super-block —
/// scale linearly.
fn filled_kquant_weights(kq: &KQuant, leading: &[i64], packed: i64) -> Weights {
    assert_eq!(packed % K, 0, "packed dim must be whole super-blocks");
    let rows: i64 = leading.iter().product();
    let supers = packed / K;
    let (weight_cols, scales_cols, biases_cols) = (
        kq.weight_cols * supers,
        kq.scales_cols * supers,
        kq.biases_cols * supers,
    );
    let shape = |cols: i64| {
        let mut s = leading.to_vec();
        s.push(cols);
        s
    };

    let mut st = 0x5eed_1234u32;
    let weight: Vec<u32> = (0..rows * weight_cols).map(|_| lcg(&mut st)).collect();
    let scales_len = (rows * scales_cols) as usize;
    let scales = if kq.scales_signed {
        // q6k sub-scales are signed.
        let v: Vec<i8> = (0..scales_len)
            .map(|_| (lcg(&mut st) % 17) as i8 - 8)
            .collect();
        MxArray::from_int8(&v, &shape(scales_cols))
    } else {
        // q4k/q5k interleave (sc, m), both 6-bit unsigned.
        let v: Vec<u8> = (0..scales_len).map(|_| (lcg(&mut st) % 64) as u8).collect();
        MxArray::from_uint8(&v, &shape(scales_cols))
    }
    .expect("scales array");
    let biases: Vec<u16> = (0..(rows * biases_cols) as usize)
        .map(|i| HALF_SCALES[i % HALF_SCALES.len()])
        .collect();
    (
        MxArray::from_uint32(&weight, &shape(weight_cols)).expect("weight array"),
        scales,
        MxArray::from_float16(&biases, &shape(biases_cols)).expect("biases array"),
    )
}

/// Exact float32 -> float16 for the activations built below. Each is an 8-bit
/// integer over 128, so it needs 8 significant bits where float16 has 11 and
/// the conversion is a pure bit reshuffle with nothing to round.
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    if v == 0.0 {
        return sign;
    }
    assert_eq!(b & 0x1fff, 0, "activation {v} does not fit float16 exactly");
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!(
        (1..=30).contains(&exp),
        "activation {v} is out of float16 range"
    );
    sign | ((exp as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

/// Deterministic activations: 8-bit integers over 128, so every value lands in
/// [-1, 1) and is exact in float32, float16 *and* bfloat16. Switching the
/// activation dtype then changes what the kernel does internally, not what it
/// is fed.
fn activation(shape: &[i64], seed: u32, dtype: DType) -> MxArray {
    let n: i64 = shape.iter().product();
    let mut st = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let values: Vec<f32> = (0..n)
        .map(|_| f32::from((lcg(&mut st) >> 24) as i8) / 128.0)
        .collect();
    match dtype {
        DType::Float32 => MxArray::from_float32(&values, shape),
        DType::Float16 => {
            let bits: Vec<u16> = values.iter().copied().map(f16_bits).collect();
            MxArray::from_float16(&bits, shape)
        }
        DType::BFloat16 => {
            // The low 16 bits are all zero for these values, so truncating the
            // float32 pattern is exact.
            let bits: Vec<u16> = values.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            MxArray::from_bfloat16(&bits, shape)
        }
        other => panic!("unsupported activation dtype {other:?}"),
    }
    .expect("activation array")
}

/// Build the same op on the CPU and on the GPU and compare the results.
/// `tol` is relative to the largest CPU magnitude; 0 demands bit equality.
/// Returns that relative difference so a caller can assert on it directly.
fn compare_devices<F>(what: &str, tol: f32, build: F) -> f32
where
    F: Fn() -> *mut mlx_sys::mlx_array,
{
    assert!(select(Device::Cpu), "CPU device is always available");
    let cpu = expect_f32(&format!("{what} cpu"), build());
    assert!(select(Device::Gpu), "the GPU device went away mid-test");
    let gpu = expect_f32(&format!("{what} gpu"), build());

    assert_eq!(cpu.len(), gpu.len(), "{what}: output lengths differ");
    let scale = cpu.iter().fold(1f32, |m, v| m.max(v.abs()));
    // Several kernels below agree bit for bit. That only means something if the
    // reference is not degenerate, so refuse to pass on an all-but-zero output:
    // `scale` is floored at 1, so this asserts a real peak above 1.
    assert!(
        scale > 1.0,
        "{what}: the CPU reference peaks at {scale}, too small for the \
         comparison to prove anything"
    );
    let limit = tol * scale;
    let mut worst = 0f32;
    let mut worst_at = 0usize;
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let d = (a - b).abs();
        if d.is_nan() || d > worst {
            worst = d;
            worst_at = i;
        }
    }
    assert!(
        worst <= limit,
        "{what}: Metal and the CPU differ by {worst} at index {worst_at} \
         (cpu {}, gpu {}, limit {limit})",
        cpu[worst_at],
        gpu[worst_at],
    );
    let relative = worst / scale;
    println!(
        "  matches  {what:<44}  peak {scale:>11.3e}  max |cpu - gpu| {worst:>11.3e}  \
         rel {relative:>9.2e}  (tol {tol:.0e})"
    );
    relative
}

// ---------------------------------------------------------------------------
// op wrappers
// ---------------------------------------------------------------------------

fn quantize(x: &MxArray, group_size: i32, bits: i32, mode: &str) -> Attempt {
    let mode = CString::new(mode).expect("mode");
    let mut q = std::ptr::null_mut();
    let mut s = std::ptr::null_mut();
    let mut b = std::ptr::null_mut();
    // SAFETY: `x` outlives the call; the three out-params are written only on
    // success.
    let ok = unsafe {
        mlx_sys::mlx_quantize(
            x.as_raw_ptr(),
            group_size,
            bits,
            mode.as_ptr(),
            &mut q,
            &mut s,
            &mut b,
        )
    };
    if !ok {
        return Attempt::Rejected("rejected at construction".to_string());
    }
    let out = finish(q);
    drop_array(s);
    drop_array(b);
    out
}

fn dequantize_handle(
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> *mut mlx_sys::mlx_array {
    let mode = CString::new(mode).expect("mode");
    // SAFETY: every handle outlives the call; a null `biases` is the documented
    // "no biases" encoding.
    unsafe {
        mlx_sys::mlx_dequantize(
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr()),
            group_size,
            bits,
            0,
            mode.as_ptr(),
        )
    }
}

fn dequantize(
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> Attempt {
    finish(dequantize_handle(w, scales, biases, group_size, bits, mode))
}

#[allow(clippy::too_many_arguments)]
fn quantized_matmul_handle(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> *mut mlx_sys::mlx_array {
    let mode = CString::new(mode).expect("mode");
    // SAFETY: every handle outlives the call.
    unsafe {
        mlx_sys::mlx_quantized_matmul(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr()),
            transpose,
            group_size,
            bits,
            mode.as_ptr(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn quantized_matmul(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> Attempt {
    finish(quantized_matmul_handle(
        x, w, scales, biases, true, group_size, bits, mode,
    ))
}

#[allow(clippy::too_many_arguments)]
fn gather_qmm_handle(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    lhs: Option<&MxArray>,
    rhs: &MxArray,
    transpose: bool,
    sorted: bool,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> *mut mlx_sys::mlx_array {
    let mode = CString::new(mode).expect("mode");
    // SAFETY: every handle outlives the call. A null `lhs` is the documented
    // "derive the left indices from x" encoding, and it is also what turns
    // `sorted` into `right_sorted_` (mlx/ops.cpp:5715).
    unsafe {
        mlx_sys::mlx_gather_qmm(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr()),
            lhs.map_or(std::ptr::null_mut(), |l| l.as_raw_ptr()),
            rhs.as_raw_ptr(),
            transpose,
            group_size,
            bits,
            mode.as_ptr(),
            sorted,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn gather_qmm(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    lhs: &MxArray,
    rhs: &MxArray,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> Attempt {
    finish(gather_qmm_handle(
        x,
        w,
        scales,
        biases,
        Some(lhs),
        rhs,
        true,
        false,
        group_size,
        bits,
        mode,
    ))
}

/// `quantized_matmul` on a `(weight, scales, biases)` triple with the mode's
/// fixed group size and bit width.
fn qmm_of(kq: &KQuant, x: &MxArray, w: &Weights, transpose: bool) -> *mut mlx_sys::mlx_array {
    quantized_matmul_handle(
        x,
        &w.0,
        &w.1,
        Some(&w.2),
        transpose,
        kq.group_size,
        kq.bits,
        kq.mode,
    )
}

/// `gather_qmm` on a `(weight, scales, biases)` triple.
#[allow(clippy::too_many_arguments)]
fn gather_of(
    kq: &KQuant,
    x: &MxArray,
    w: &Weights,
    lhs: Option<&MxArray>,
    rhs: &MxArray,
    transpose: bool,
    sorted: bool,
) -> *mut mlx_sys::mlx_array {
    gather_qmm_handle(
        x,
        &w.0,
        &w.1,
        Some(&w.2),
        lhs,
        rhs,
        transpose,
        sorted,
        kq.group_size,
        kq.bits,
        kq.mode,
    )
}

fn indices(v: &[u32]) -> MxArray {
    MxArray::from_uint32(v, &[v.len() as i64]).expect("indices")
}

// ---------------------------------------------------------------------------
// guards
// ---------------------------------------------------------------------------

/// These formats are only ever imported. `quantize` must refuse to produce them
/// on every device, so nothing can silently write a K-quant tensor.
#[test]
fn quantize_rejects_every_kquant_mode_on_every_device() {
    for device in [Device::Cpu, Device::Gpu] {
        if !select(device) {
            eprintln!("skipping {}: device unavailable", device.label());
            continue;
        }
        for kq in &KQUANTS {
            let x = zeros(&[N, K], DType::BFloat16);
            expect_rejected(
                &format!("quantize {} {}", kq.mode, device.label()),
                quantize(&x, kq.group_size, kq.bits, kq.mode),
            );
        }
    }
}

/// The CPU implements the K-quant decode, matmul and gather. If any of these
/// starts throwing, the parity gate next door stops measuring anything.
#[test]
fn cpu_runs_dequantize_matmul_and_gather_for_every_kquant_mode() {
    assert!(select(Device::Cpu), "CPU device is always available");
    for kq in &KQUANTS {
        let (w, scales, biases) = kquant_weights(kq);
        expect_shape(
            &format!("dequantize {} cpu", kq.mode),
            &[N, K],
            dequantize(&w, &scales, Some(&biases), kq.group_size, kq.bits, kq.mode),
        );
        for m in [1i64, 8] {
            let x = zeros(&[m, K], DType::BFloat16);
            expect_shape(
                &format!("quantized_matmul M={m} {} cpu", kq.mode),
                &[m, N],
                quantized_matmul(
                    &x,
                    &w,
                    &scales,
                    Some(&biases),
                    kq.group_size,
                    kq.bits,
                    kq.mode,
                ),
            );
        }
        let scales_dtype = if kq.scales_signed {
            DType::Int8
        } else {
            DType::Uint8
        };
        let we = zeros(&[4, N, kq.weight_cols], DType::Uint32);
        let se = zeros(&[4, N, kq.scales_cols], scales_dtype);
        let be = zeros(&[4, N, kq.biases_cols], DType::Float16);
        let x = zeros(&[2, 8, K], DType::BFloat16);
        let lhs = zeros(&[2], DType::Uint32);
        let rhs = zeros(&[2], DType::Uint32);
        expect_shape(
            &format!("gather_qmm {} cpu", kq.mode),
            &[2, 8, N],
            gather_qmm(
                &x,
                &we,
                &se,
                Some(&be),
                &lhs,
                &rhs,
                kq.group_size,
                kq.bits,
                kq.mode,
            ),
        );
    }
}

/// Every `QuantizedMatmul` kernel `kquant.metal` instantiates, against the CPU
/// reference. The shapes come from the dispatch conditions in
/// `mlx/backend/metal/quantized.cpp`, one per kernel:
///
/// ```text
///   M = 1        -> dispatch_qmv:1728, and 1 < vector_limit (>= 6 on every
///                   arch branch of get_qmv_batch_limit:191-233)
///   M in [2, 5]  -> qmv_wide:1668, which needs GPU gen >= 15 (use_qmv_wide:411)
///   M = 64       -> the qmm branch:1701, since vector_limit <= 32 everywhere
///   fast         -> N % 8 == 0 && K % 512 == 0 (:368)
///   aligned      -> N % 32 == 0 (:956, :1070)
///   batched      -> B = out.size() / M / N > 1 (:957)
///   splitk       -> transpose && B == 1 (:1704), then split_k > 1 (:1046)
/// ```
///
/// There is deliberately no `qmv_quad` case: `dispatch_qmv` routes there only
/// for K of 64 or 128 (:1660) and `validate_quantized_input` requires the
/// packed dim to be whole 256-element super-blocks (mlx/ops.cpp:145-153), so no
/// K-quant tensor can reach it. `validators_reject_malformed_kquant_tensors`
/// pins that, and `kquant.metal` instantiates no `qmv_quad` to fall into.
///
/// The `qmm_t` cases are the exception to "one shape per `kquant.metal`
/// kernel". On a device with the NAX tensor-op kernels they route to
/// `kquant_qmm_t_nax` (`kquant_nax.metal`) instead, because the NAX gate at
/// :968 needs only `transpose && K % 64 == 0` and a K-quant's K is always a
/// multiple of 256. Its alignment threshold is `N % 64` (qmm_nax:763) rather
/// than the simdgroup `N % 32`, so the four cases below cover all four NAX
/// instantiations — and the unaligned widths are chosen to be unaligned under
/// *both* rules, so the same four shapes cover all four simdgroup
/// instantiations on a device without NAX or under `MLX_ENABLE_TF32=0`:
///
/// ```text
///   N = 1024, B = 1  -> alN_true  batch_0   (16 x 32 = 512 tiles, split_k = 1)
///   N = 548,  B = 1  -> alN_false batch_0   (548 % 64 = 36, 548 % 32 = 4)
///   N = 128,  B = 3  -> alN_true  batch_1
///   N = 136,  B = 3  -> alN_false batch_1   (136 % 64 = 8, 136 % 32 = 8)
/// ```
///
/// Two more shapes follow them, for the parts of the kernel an aligned M and a
/// one-super-block row leave untouched: M = 70 takes the `kAlignedM == false`
/// branch, and K = 1024 makes a thread walk across super-block boundaries
/// inside a single row.
///
/// The simdgroup `kquant_qmm_t` is then unreachable here, and is covered by the
/// same shapes on a device without NAX or under `MLX_ENABLE_TF32=0`, which
/// keeps float32 off the tensor op. `gpu_matches_cpu_for_every_activation_dtype`
/// runs the NAX kernels again on bfloat16 and float16, where there is no tf32
/// and the tolerance is the ordinary tile-rounding one.
#[test]
fn gpu_matches_cpu_on_every_quantized_matmul_kernel() {
    if !select(Device::Gpu) {
        eprintln!("skipping gpu_matches_cpu_on_every_quantized_matmul_kernel: no GPU device");
        return;
    }
    println!("  nax={} tf32_gate={}", nax_available(), nax_tf32_active());
    assert!(
        nax_available() || !nax_required(),
        "KQ_NAX_REQUIRE=1 but metal::is_nax_available() is false, so every NAX \
         assertion in this file would be skipped and the run would still report \
         ok. Per metal/device.cpp:944-962 that means the GPU generation is below \
         17 (below 18 when the architecture string ends in 'p'), macOS is older \
         than 26.2, or the metallib was built without the NAX kernels \
         (MLX_METAL_NO_NAX). Note that MLX_METAL_GPU_ARCH overrides the \
         generation this check parses (metal/device.cpp:589-601), so an override \
         left in the environment reaches here."
    );

    // Worst tf32 gap per mode over the six NAX `qmm_t` shapes, asserted against
    // NAX_TF32_GAP_FLOOR after the affine control so one run reports all three
    // rather than dying on the first.
    let mut nax_gap: [(&str, f32); KQUANTS.len()] = [("", 0.0); KQUANTS.len()];

    for (ki, kq) in KQUANTS.iter().enumerate() {
        let m = kq.mode;
        let mut least_nax = f32::INFINITY;

        // dequantize is a pure per-element decode: no accumulation can reorder,
        // so Metal has to reproduce the CPU bits exactly.
        let w = filled_kquant_weights(kq, &[N], K);
        compare_devices(&format!("dequantize {m}"), 0.0, || {
            dequantize_handle(&w.0, &w.1, Some(&w.2), kq.group_size, kq.bits, m)
        });

        // qmv: K = 256 is not a multiple of 512, so `fast` is false.
        let x1 = activation(&[1, K], 7, DType::Float32);
        compare_devices(&format!("qmv batch_0 {m}"), F32_TOL, || {
            qmm_of(kq, &x1, &w, true)
        });

        // qmv_fast: N % 8 == 0 and K % 512 == 0.
        let wf = filled_kquant_weights(kq, &[N], 2 * K);
        let xf = activation(&[1, 2 * K], 23, DType::Float32);
        compare_devices(&format!("qmv_fast batch_0 {m}"), F32_TOL, || {
            qmm_of(kq, &xf, &wf, true)
        });

        // The same two with a leading batch, which flips the `_batch_1`
        // template parameter and turns on the stride/shape arguments (:403).
        let wb = filled_kquant_weights(kq, &[3, N], K);
        let xb1 = activation(&[3, 1, K], 31, DType::Float32);
        compare_devices(&format!("qmv batch_1 {m}"), F32_TOL, || {
            qmm_of(kq, &xb1, &wb, true)
        });
        let wbf = filled_kquant_weights(kq, &[3, N], 2 * K);
        let xbf = activation(&[3, 1, 2 * K], 37, DType::Float32);
        compare_devices(&format!("qmv_fast batch_1 {m}"), F32_TOL, || {
            qmm_of(kq, &xbf, &wbf, true)
        });

        // qmv_wide, one case per instantiated input-vector tile: n_tiles is
        // ceil(M / 5) and vecs_per_tg is ceil(M / n_tiles) (:438-439), so
        // M = 2..5 picks _nv_2 .. _nv_5. k_lanes is pinned to 8 for K-quants
        // (:446).
        for mm in 2i64..=5 {
            let xw = activation(&[mm, K], 41 + mm as u32, DType::Float32);
            compare_devices(&format!("qmv_wide nv_{mm} batch_0 {m}"), F32_TOL, || {
                qmm_of(kq, &xw, &w, true)
            });
        }
        for mm in 2i64..=5 {
            let xwb = activation(&[3, mm, K], 53 + mm as u32, DType::Float32);
            compare_devices(&format!("qmv_wide nv_{mm} batch_1 {m}"), F32_TOL, || {
                qmm_of(kq, &xwb, &wb, true)
            });
        }

        // qmm_t_splitk: M = 64 clears vector_limit, B == 1 picks the split-K
        // path, and split_k lands on 8 (512 / 8 tiles, capped by K / 32).
        let x64 = activation(&[64, K], 59, DType::Float32);
        compare_devices(&format!("qmm_t_splitk alN_true {m}"), F32_TOL, || {
            qmm_of(kq, &x64, &w, true)
        });
        let wu = filled_kquant_weights(kq, &[N_UNALIGNED], K);
        compare_devices(&format!("qmm_t_splitk alN_false {m}"), F32_TOL, || {
            qmm_of(kq, &x64, &wu, true)
        });

        // qmm_t with B == 1: reached only when qmm_splitk gives up, which needs
        // 512 / (n_tiles * m_tiles) to floor to 1. 512 rows x 1024 cols is
        // 16 x 32 = 512 tiles exactly, and 1024 is a multiple of 64, so this is
        // the NAX alN_true case. 548 cols is 18 tiles wide, so 18 x 16 = 288
        // threadgroups still floor split_k to 1 (quantized.cpp:1076), and 548
        // is unaligned under the simdgroup `N % 32` rule (:999) as well as the
        // NAX `N % 64` one (:763) — so this one shape is the alN_false case for
        // both kernel families, and stays covered when float32 leaves the
        // tensor op.
        let xbig = activation(&[512, K], 61, DType::Float32);
        let wbig = filled_kquant_weights(kq, &[1024], K);
        least_nax = least_nax.min(compare_devices(
            &format!("qmm_t alN_true batch_0 {m}"),
            qmm_t_f32_tol(),
            || qmm_of(kq, &xbig, &wbig, true),
        ));
        let wbigu = filled_kquant_weights(kq, &[548], K);
        least_nax = least_nax.min(compare_devices(
            &format!("qmm_t alN_false batch_0 {m}"),
            qmm_t_f32_tol(),
            || qmm_of(kq, &xbig, &wbigu, true),
        ));

        // qmm_t with a batch: B > 1 skips the split-K path outright (:1704).
        let xb64 = activation(&[3, 64, K], 67, DType::Float32);
        least_nax = least_nax.min(compare_devices(
            &format!("qmm_t alN_true batch_1 {m}"),
            qmm_t_f32_tol(),
            || qmm_of(kq, &xb64, &wb, true),
        ));
        let wbu = filled_kquant_weights(kq, &[3, N_UNALIGNED], K);
        least_nax = least_nax.min(compare_devices(
            &format!("qmm_t alN_false batch_1 {m}"),
            qmm_t_f32_tol(),
            || qmm_of(kq, &xb64, &wbu, true),
        ));

        // M that is not a multiple of BM = 64. The NAX kernel then takes its
        // `kAlignedM == false` branch: `Atile.load_safe` for the activation and
        // `Dtile.store_safe` for the result (kquant_nax.h:471, :498). A prompt
        // whose length is not a multiple of 64 hits this in production, and the
        // batch keeps it off the split-K path without needing a large N.
        let xm = activation(&[3, 70, K], 79, DType::Float32);
        least_nax = least_nax.min(compare_devices(
            &format!("qmm_t unaligned_M batch_1 {m}"),
            qmm_t_f32_tol(),
            || qmm_of(kq, &xm, &wb, true),
        ));

        // A row spanning four super-blocks instead of one. At K = 256 a row is
        // exactly one super-block, so the super-scale index `gi / super_ratio`
        // (kquant_nax.h:182) only ever changes between rows and a thread never
        // walks across a super-block boundary. Here each thread steps through
        // 16 tiles per row and crosses three, which is the only place a
        // two-level scale-stride error can show up.
        let wdeep = filled_kquant_weights(kq, &[1024], K_DEEP);
        let xdeep = activation(&[512, K_DEEP], 83, DType::Float32);
        least_nax = least_nax.min(compare_devices(
            &format!("qmm_t deep_K batch_0 {m}"),
            qmm_t_f32_tol(),
            || qmm_of(kq, &xdeep, &wdeep, true),
        ));
        nax_gap[ki] = (m, least_nax);

        // Non-transposed weights. The packed axis is now the output dim, so it
        // is N that has to be whole super-blocks; K is just the row count.
        // vector_limit is a flat 4 here (:1698).
        let wn = filled_kquant_weights(kq, &[K], N_PACKED);
        let xn8 = activation(&[8, K], 71, DType::Float32);
        compare_devices(&format!("qmm_n batch_0 {m}"), F32_TOL, || {
            qmm_of(kq, &xn8, &wn, false)
        });
        let wnb = filled_kquant_weights(kq, &[3, K], N_PACKED);
        let xnb8 = activation(&[3, 8, K], 73, DType::Float32);
        compare_devices(&format!("qmm_n batch_1 {m}"), F32_TOL, || {
            qmm_of(kq, &xnb8, &wnb, false)
        });

        // qvm: non-transposed, M < 4 and K < 1024 (:1747).
        let xn1 = activation(&[1, K], 79, DType::Float32);
        compare_devices(&format!("qvm batch_0 {m}"), F32_TOL, || {
            qmm_of(kq, &xn1, &wn, false)
        });
        let xnb1 = activation(&[3, 1, K], 83, DType::Float32);
        compare_devices(&format!("qvm batch_1 {m}"), F32_TOL, || {
            qmm_of(kq, &xnb1, &wnb, false)
        });

        // qvm_split_k: K >= 1024 (:1753), and split_k is 8 below 8192 rows and
        // 32 above (:523).
        let wk8 = filled_kquant_weights(kq, &[1024], N_PACKED);
        let xk8 = activation(&[1, 1024], 89, DType::Float32);
        compare_devices(&format!("qvm_split_k spk_8 {m}"), F32_DEEP_TOL, || {
            qmm_of(kq, &xk8, &wk8, false)
        });
        let wk32 = filled_kquant_weights(kq, &[16384], N_PACKED);
        let xk32 = activation(&[1, 16384], 97, DType::Float32);
        compare_devices(&format!("qvm_split_k spk_32 {m}"), F32_DEEP_TOL, || {
            qmm_of(kq, &xk32, &wk32, false)
        });
    }

    // The control for NAX_TF32_TOL: an affine 4-bit qmm_t at N = 544, the same
    // NAX alN_false batch_0 branch. Affine takes the identical NAX branch, so a
    // gap of the
    // same order with no K-quant code in the path is what says the widened
    // tolerance on the four qmm_t cases is the tensor op's precision mode and
    // not the two-level scale decode. Under MLX_ENABLE_TF32=0 both this and the
    // K-quant cases drop back to the simdgroup kernel and this gap collapses,
    // which is why the floor is a measurement rather than a gate.
    //
    // The scales carry a full float32 mantissa on purpose. tf32 keeps 10
    // mantissa bits, so a control built from short scales (a power of two, a
    // handful of significant bits) decodes to tile values that survive the
    // truncation unchanged and measures a gap of exactly zero — it would look
    // like the K-quants were at fault. A K-quant tile value is an fp16 `d`
    // times an integer sub-scale, which needs more than 10 bits, so the control
    // has to as well.
    let groups = K / 32;
    let mut st = 0x51ce_9e77u32;
    let codes: Vec<u32> = (0..544 * K * 4 / 32).map(|_| lcg(&mut st)).collect();
    let aw = MxArray::from_uint32(&codes, &[544, K * 4 / 32]).expect("affine weight");
    let asc: Vec<f32> = (0..544 * groups)
        .map(|_| 0.25 + 0.25 * (lcg(&mut st) & 0x007f_ffff) as f32 / 8_388_608.0)
        .collect();
    let abi: Vec<f32> = asc.iter().map(|s| -7.5 * s).collect();
    let a_s = MxArray::from_float32(&asc, &[544, groups]).expect("affine scales");
    let a_b = MxArray::from_float32(&abi, &[544, groups]).expect("affine biases");
    let xbig = activation(&[512, K], 61, DType::Float32);
    let gap = compare_devices(
        "affine qmm_t alN_false batch_0 tf32",
        qmm_t_f32_tol(),
        || quantized_matmul_handle(&xbig, &aw, &a_s, Some(&a_b), true, 32, 4, "affine"),
    );
    assert!(
        gap <= AFFINE_TF32_GAP_CEILING,
        "affine tf32 control blew up at {gap}: NAX_TF32_TOL wants re-measuring"
    );
    println!("  affine tf32 control gap {gap:.3e} (K-quant tolerance {NAX_TF32_TOL:.0e})");
    for (mode, smallest) in &nax_gap {
        println!("  smallest NAX qmm_t tf32 gap {mode} {smallest:.3e}");
    }

    if nax_tf32_active() {
        assert!(
            gap > AFFINE_TF32_GAP_FLOOR,
            "affine tf32 control gap {gap:e} is below {AFFINE_TF32_GAP_FLOOR:e} on a NAX \
             device with tf32 on -- the tensor op stopped truncating, so the qmm_t cases \
             above are no longer measuring what NAX_TF32_TOL was set for"
        );
        assert!(
            nax_gap.iter().all(|(_, w)| *w > NAX_TF32_GAP_FLOOR),
            "a float32 K-quant qmm_t agreed with the CPU to better than \
             {NAX_TF32_GAP_FLOOR:e}, so it ran on the simdgroup kquant_qmm_t rather than \
             kquant_qmm_t_nax: check nax_supports_mode in \
             mlx/backend/metal/quantized.cpp:66. Smallest gap per mode {} {:e}, {} {:e}, \
             {} {:e}; affine control {gap:e}",
            nax_gap[0].0,
            nax_gap[0].1,
            nax_gap[1].0,
            nax_gap[1].1,
            nax_gap[2].0,
            nax_gap[2].1,
        );
    } else {
        println!(
            "  note: not a NAX device, or MLX_ENABLE_TF32=0 -- the float32 qmm_t shapes ran \
             on the simdgroup kernel, so the tf32 gaps above are legitimately zero and the \
             NAX_TF32_GAP_FLOOR band does not apply"
        );
    }

    // leave the process on the CPU for whatever runs next
    select(Device::Cpu);
}

/// Every `GatherQMM` kernel, same method. `GatherQMM::eval_gpu`
/// (`mlx/backend/metal/quantized.cpp:1758`) picks:
///
/// ```text
///   M == 1 && B >= 16 && right_sorted && B / E >= 4 -> gather_qmm_rhs (:1787)
///   M >= vector_limit                               -> gather_qmm    (:1808)
///   transpose                                       -> gather_qmv    (:1829)
///   otherwise                                       -> gather_qvm    (:1849)
/// ```
///
/// `right_sorted_` is `sorted_indices && !lhs_indices` (`mlx/ops.cpp:5715`), so
/// the two `gather_qmm_rhs` cases pass a null left index and let MLX derive it.
#[test]
fn gpu_matches_cpu_on_every_gather_kernel() {
    if !select(Device::Gpu) {
        eprintln!("skipping gpu_matches_cpu_on_every_gather_kernel: no GPU device");
        return;
    }
    let lhs2 = indices(&[0, 1]);
    let rhs2 = indices(&[3, 1]);
    // 16 rows over 4 experts: B >= 16 and B / E >= 4, ascending as the sorted
    // path assumes.
    let rhs16 = indices(&[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);

    for kq in &KQUANTS {
        let m = kq.mode;
        let we = filled_kquant_weights(kq, &[4, N], K);

        // gather_qmv: M = 1, and K = 256 keeps `fast` false (:1253).
        let x1 = activation(&[2, 1, K], 101, DType::Float32);
        compare_devices(&format!("gather_qmv {m}"), F32_TOL, || {
            gather_of(kq, &x1, &we, Some(&lhs2), &rhs2, true, false)
        });

        // gather_qmv_fast: N % 8 == 0 and K % 512 == 0.
        let wef = filled_kquant_weights(kq, &[4, N], 2 * K);
        let xf = activation(&[2, 1, 2 * K], 103, DType::Float32);
        compare_devices(&format!("gather_qmv_fast {m}"), F32_TOL, || {
            gather_of(kq, &xf, &wef, Some(&lhs2), &rhs2, true, false)
        });

        // gather_qmm_t, both alignments.
        let x64 = activation(&[2, 64, K], 107, DType::Float32);
        compare_devices(&format!("gather_qmm_t alN_true {m}"), F32_TOL, || {
            gather_of(kq, &x64, &we, Some(&lhs2), &rhs2, true, false)
        });
        let weu = filled_kquant_weights(kq, &[4, N_UNALIGNED], K);
        compare_devices(&format!("gather_qmm_t alN_false {m}"), F32_TOL, || {
            gather_of(kq, &x64, &weu, Some(&lhs2), &rhs2, true, false)
        });

        // Non-transposed: M = 8 clears the flat vector_limit of 4 for
        // gather_qmm_n, M = 1 falls through to gather_qvm.
        let wen = filled_kquant_weights(kq, &[4, K], N_PACKED);
        let xn8 = activation(&[2, 8, K], 109, DType::Float32);
        compare_devices(&format!("gather_qmm_n {m}"), F32_TOL, || {
            gather_of(kq, &xn8, &wen, Some(&lhs2), &rhs2, false, false)
        });
        let xn1 = activation(&[2, 1, K], 113, DType::Float32);
        compare_devices(&format!("gather_qvm {m}"), F32_TOL, || {
            gather_of(kq, &xn1, &wen, Some(&lhs2), &rhs2, false, false)
        });

        // gather_qmm_rhs, the sorted-MoE path: one row per token, a null left
        // index, and 16 tokens over the 4 experts.
        let xr = activation(&[16, 1, K], 127, DType::Float32);
        compare_devices(&format!("gather_qmm_rhs_nt {m}"), F32_TOL, || {
            gather_of(kq, &xr, &we, None, &rhs16, true, true)
        });
        compare_devices(&format!("gather_qmm_rhs_nn {m}"), F32_TOL, || {
            gather_of(kq, &xr, &wen, None, &rhs16, false, true)
        });
    }
    select(Device::Cpu);
}

/// `kquant.metal` instantiates every kernel for float, float16 and bfloat16.
/// The float32 coverage above is the numeric gate; this pins that the other two
/// instantiations exist and stay within their own rounding.
///
/// It is also where `kquant_qmm_t_nax` is held to a tight tolerance. `qmm`
/// routes to the NAX kernels when the GPU has them, the weight is transposed
/// and K % 64 == 0 (`metal/quantized.cpp:968`); float16 and bfloat16 take that
/// branch whatever `MLX_ENABLE_TF32` says, and unlike float32 they carry no
/// tf32 truncation — the tensor op consumes them exactly and accumulates in
/// float, the same as the simdgroup kernel. So these comparisons hold the
/// ordinary tile-rounding tolerance and are the real numeric gate on the new
/// kernel; the float32 ones above are the tf32 gate.
///
/// Both alignments and both batch settings run, so all four instantiations
/// `kquant_nax.metal` emits per mode and type are covered:
///
/// ```text
///   N = 1024, B = 1  -> alN_true  batch_0
///   N = 548,  B = 1  -> alN_false batch_0   (548 % 64 = 36, 548 % 32 = 4)
///   N = 128,  B = 3  -> alN_true  batch_1
///   N = 136,  B = 3  -> alN_false batch_1   (136 % 64 = 8, 136 % 32 = 8)
/// ```
///
/// The unaligned widths are unaligned under the simdgroup `N % 32` rule as well
/// as the NAX `N % 64` one, so the same shapes cover both kernel families.
///
/// Which family those four actually run is a property of the host, and nothing
/// in the process can change it: the gate's other float32-only escape,
/// `env::enable_tf32()`, does not apply to these dtypes, and its `K % 64 == 0`
/// arm is free — a K-quant's packed dimension must already be a whole number of
/// 256-element super-blocks or the validator rejects it. So on a NAX host these
/// are `kquant_qmm_t_nax` and on every other host they are the simdgroup
/// `kquant_qmm_t`, and the only way to see the other family is to run the suite
/// somewhere else. `MLX_METAL_GPU_ARCH=applegpu_g16s` does that without a
/// rebuild: `Device::Device` parses the generation straight out of that string
/// (`metal/device.cpp:589-601`), so 16 fails the `gen >= 17` check at :956 and
/// every NAX branch turns off. Measured, the two families agree exactly here —
/// same peak and same worst gap on all 24 comparisons — which is expected,
/// since neither truncates float16 or bfloat16 and both accumulate in float.
///
/// The two `qmm_t_splitk` shapes are the exception that needs no such trick.
/// `qmm_splitk` has no NAX branch at all (`metal/quantized.cpp:1057`), so they
/// run the simdgroup kernel on every host including this one, and they are the
/// only float16/bfloat16 `qmm_t` numbers in the suite that a NAX box actually
/// exercises against the fallback code. They also cover a rounding step none of
/// the shapes above reach — see `F16_SPLITK_TOL`. Production hits this path on
/// an ordinary single-sequence bfloat16 prefill, where B == 1 and M clears
/// `vector_limit`.
#[test]
fn gpu_matches_cpu_for_every_activation_dtype() {
    if !select(Device::Gpu) {
        eprintln!("skipping gpu_matches_cpu_for_every_activation_dtype: no GPU device");
        return;
    }
    for kq in &KQUANTS {
        let m = kq.mode;
        let w = filled_kquant_weights(kq, &[N], K);
        let wu = filled_kquant_weights(kq, &[N_UNALIGNED], K);
        let wb = filled_kquant_weights(kq, &[3, N], K);
        let wbu = filled_kquant_weights(kq, &[3, N_UNALIGNED], K);
        let wbig = filled_kquant_weights(kq, &[1024], K);
        let wbigu = filled_kquant_weights(kq, &[548], K);

        for (dtype, name, store_tol, tile_tol, splitk_tol) in [
            (
                DType::Float16,
                "f16",
                F16_STORE_TOL,
                F16_TILE_TOL,
                F16_SPLITK_TOL,
            ),
            (
                DType::BFloat16,
                "bf16",
                BF16_STORE_TOL,
                BF16_TILE_TOL,
                BF16_SPLITK_TOL,
            ),
        ] {
            // qmv accumulates in float and rounds once on the store
            // (kquant.h:807, :843).
            let x1 = activation(&[1, K], 131, dtype);
            compare_devices(&format!("qmv {name} {m}"), store_tol, || {
                qmm_of(kq, &x1, &w, true)
            });

            // qmm_t additionally rounds each decoded weight into the
            // threadgroup tile (kquant.h:587 / kquant_nax.h:140), which the CPU
            // does not do. The batch is what keeps these two off the split-K
            // path.
            let x64 = activation(&[3, 64, K], 137, dtype);
            compare_devices(
                &format!("qmm_t alN_true batch_1 {name} {m}"),
                tile_tol,
                || qmm_of(kq, &x64, &wb, true),
            );
            compare_devices(
                &format!("qmm_t alN_false batch_1 {name} {m}"),
                tile_tol,
                || qmm_of(kq, &x64, &wbu, true),
            );

            // B == 1 with enough tiles that split-K gives up, which is what
            // leaves the two batch_0 instantiations reachable.
            let xbig = activation(&[512, K], 139, dtype);
            compare_devices(
                &format!("qmm_t alN_true batch_0 {name} {m}"),
                tile_tol,
                || qmm_of(kq, &xbig, &wbig, true),
            );
            compare_devices(
                &format!("qmm_t alN_false batch_0 {name} {m}"),
                tile_tol,
                || qmm_of(kq, &xbig, &wbigu, true),
            );

            // M = 64 clears vector_limit and B == 1, so this is the split-K
            // path: split_k lands on 8 and each partition's partial sum is
            // stored in the *activation* dtype (`intermediate` is built with
            // `x.dtype()`, metal/quantized.cpp:1101) before a separate reduce
            // adds them. `qmm_splitk` has no NAX branch at all, so these two
            // run the simdgroup kernel on every host.
            let xsk = activation(&[64, K], 149, dtype);
            compare_devices(
                &format!("qmm_t_splitk alN_true {name} {m}"),
                splitk_tol,
                || qmm_of(kq, &xsk, &w, true),
            );
            compare_devices(
                &format!("qmm_t_splitk alN_false {name} {m}"),
                splitk_tol,
                || qmm_of(kq, &xsk, &wu, true),
            );
        }
    }
    select(Device::Cpu);
}

/// The control for the bfloat16 tolerances above.
///
/// `load_vector` and `qdot` in `kquant.h` are byte-identical to the affine
/// versions in `quantized.h`, so the bfloat16 rounding of `Sum(x_i)` that
/// widens the K-quant vector kernels is a property of MLX's affine kernel
/// shape, not of the K-quant decode. This runs the same `qmv` comparison on
/// affine 4-bit weights biased the way q6k's are — `bias = -7.5 * scale`
/// against 4-bit codes centred on 7.5, the analogue of q6k's
/// `bias = -32 * scale` against 6-bit codes centred on 32 — and asserts the
/// same order of gap appears without any K-quant code in the path.
///
/// Read this as presence-of-mechanism, not as a magnitude comparison. A single
/// constant `(scale, bias)` makes the whole row cancel, so this control peaks
/// ~10x lower than the q6k case and its *relative* gap is inflated by the
/// smaller denominator; on absolute error it is the smaller of the two. A
/// shape-matched control (affine 6-bit / group 32, per-group scales,
/// `bias = -32 * scale`) peaks alongside q6k and still measures ~2.4x wider,
/// which is what actually rules out a K-quant-specific defect. It is not in
/// the suite because affine instantiates no group-16 variant, so the closest
/// match still differs in group size.
///
/// If MLX ever makes that partial sum exact, this fails and the two
/// `*_STORE_TOL` constants want re-measuring rather than keeping.
#[test]
fn affine_shows_the_same_bfloat16_vector_gap() {
    if !select(Device::Gpu) {
        eprintln!("skipping affine_shows_the_same_bfloat16_vector_gap: no GPU device");
        return;
    }
    // 0.25 and -1.875 = -7.5 * 0.25, both exact in bfloat16.
    let scale_bits = (0.25f32.to_bits() >> 16) as u16;
    let bias_bits = ((-1.875f32).to_bits() >> 16) as u16;
    let groups = K / 64;

    let mut st = 0x5eed_1234u32;
    let codes: Vec<u32> = (0..N * K * 4 / 32).map(|_| lcg(&mut st)).collect();
    let w = MxArray::from_uint32(&codes, &[N, K * 4 / 32]).expect("affine weight");
    let s = MxArray::from_bfloat16(&vec![scale_bits; (N * groups) as usize], &[N, groups])
        .expect("affine scales");
    let b = MxArray::from_bfloat16(&vec![bias_bits; (N * groups) as usize], &[N, groups])
        .expect("affine biases");
    let x = activation(&[1, K], 131, DType::BFloat16);

    let gap = compare_devices(
        "qmv bf16 affine 4/64 (control)",
        AFFINE_BF16_GAP_CEILING,
        || quantized_matmul_handle(&x, &w, &s, Some(&b), true, 64, 4, "affine"),
    );
    assert!(
        gap > AFFINE_BF16_GAP_FLOOR,
        "affine's own bfloat16 qmv gap fell to {gap:e}; the K-quant bfloat16 \
         tolerances were sized against it and need re-measuring"
    );
    select(Device::Cpu);
}

/// Without these, "the GPU throws" would prove nothing — a validator that
/// accepts anything would let the backend throw for the wrong reason.
#[test]
fn validators_reject_malformed_kquant_tensors() {
    assert!(select(Device::Cpu), "CPU device is always available");
    let x = zeros(&[1, K], DType::BFloat16);

    expect_rejected(
        "q6k with uint8 scales",
        quantized_matmul(
            &x,
            &zeros(&[N, 48], DType::Uint32),
            &zeros(&[N, 16], DType::Uint8),
            Some(&zeros(&[N, 1], DType::Float16)),
            16,
            6,
            "q6k",
        ),
    );
    expect_rejected(
        "q4k with missing biases",
        quantized_matmul(
            &x,
            &zeros(&[N, 32], DType::Uint32),
            &zeros(&[N, 16], DType::Uint8),
            None,
            32,
            4,
            "q4k",
        ),
    );
    expect_rejected(
        "q4k with un-interleaved scales",
        quantized_matmul(
            &x,
            &zeros(&[N, 32], DType::Uint32),
            &zeros(&[N, 8], DType::Uint8),
            Some(&zeros(&[N, 2], DType::Float16)),
            32,
            4,
            "q4k",
        ),
    );
    expect_rejected(
        "q6k with non-default group size",
        quantized_matmul(
            &x,
            &zeros(&[N, 48], DType::Uint32),
            &zeros(&[N, 16], DType::Int8),
            Some(&zeros(&[N, 1], DType::Float16)),
            32,
            6,
            "q6k",
        ),
    );

    // K = 128 is one of the two reduction sizes `dispatch_qmv` hands to
    // `qmv_quad` (metal/quantized.cpp:1660). The super-block check must reject
    // it for every K-quant, because `kquant.metal` instantiates no `qmv_quad`
    // and a dispatch that reached it would abort on a missing pipeline. q4k is
    // the case that matters: the quad route also wants `is_power_of_2(bits)`,
    // which only 4-bit satisfies.
    let x128 = zeros(&[1, 128], DType::BFloat16);
    expect_rejected(
        "q4k with a half super-block row (K = 128, the qmv_quad size)",
        quantized_matmul(
            &x128,
            &zeros(&[N, 16], DType::Uint32),
            &zeros(&[N, 8], DType::Uint8),
            Some(&zeros(&[N, 1], DType::Float16)),
            32,
            4,
            "q4k",
        ),
    );
    expect_rejected(
        "q6k with a half super-block row (K = 128, the qmv_quad size)",
        quantized_matmul(
            &x128,
            &zeros(&[N, 24], DType::Uint32),
            &zeros(&[N, 8], DType::Int8),
            Some(&zeros(&[N, 1], DType::Float16)),
            16,
            6,
            "q6k",
        ),
    );
}

/// The K-quant modes widened a validator every other mode goes through. These
/// are the shapes that were working before and must keep working.
#[test]
fn preexisting_quantization_modes_still_work() {
    for device in [Device::Cpu, Device::Gpu] {
        if !select(device) {
            eprintln!("skipping {}: device unavailable", device.label());
            continue;
        }
        let tail = device.label();
        let w = zeros(&[N, 32], DType::Uint32);
        let s = zeros(&[N, 4], DType::BFloat16);
        let b = zeros(&[N, 4], DType::BFloat16);
        for m in [1i64, 8] {
            let x = zeros(&[m, K], DType::BFloat16);
            expect_shape(
                &format!("affine 4/64 M={m} {tail}"),
                &[m, N],
                quantized_matmul(&x, &w, &s, Some(&b), 64, 4, "affine"),
            );
        }
        expect_shape(
            &format!("mxfp4 qmv {tail}"),
            &[1, N],
            quantized_matmul(
                &zeros(&[1, K], DType::BFloat16),
                &w,
                &zeros(&[N, 8], DType::Uint8),
                None,
                32,
                4,
                "mxfp4",
            ),
        );
        expect_shape(
            &format!("affine gather_qmm {tail}"),
            &[2, 8, N],
            gather_qmm(
                &zeros(&[2, 8, K], DType::BFloat16),
                &zeros(&[4, N, 32], DType::Uint32),
                &zeros(&[4, N, 4], DType::BFloat16),
                Some(&zeros(&[4, N, 4], DType::BFloat16)),
                &zeros(&[2], DType::Uint32),
                &zeros(&[2], DType::Uint32),
                64,
                4,
                "affine",
            ),
        );
        expect_shape(
            &format!("affine batched qmm 3d {tail}"),
            &[3, 8, N],
            quantized_matmul(
                &zeros(&[3, 8, K], DType::BFloat16),
                &zeros(&[3, N, 32], DType::Uint32),
                &zeros(&[3, N, 4], DType::BFloat16),
                Some(&zeros(&[3, N, 4], DType::BFloat16)),
                64,
                4,
                "affine",
            ),
        );
    }
    assert!(select(Device::Cpu), "CPU device is always available");
    let x = zeros(&[N, K], DType::BFloat16);
    expect_shape(
        "affine quantize round-trip cpu",
        &[N, K / 8],
        quantize(&x, 64, 4, "affine"),
    );
}
