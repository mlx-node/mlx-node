//! Shared K-quant fixtures for the bridge K-quant op tests. Every op goes
//! through a `mlx_test_kquant_*` entry with an explicit device, so no test
//! depends on MLX's process-wide default device.

#![allow(dead_code)]

use std::ffi::CString;

use mlx_core::array::{DType, MxArray};

pub const CPU: i32 = 0;
pub const GPU: i32 = 1;

pub struct KQuant {
    pub mode: &'static str,
    pub bits: i32,
    pub group_size: i32,
    pub scales_signed: bool,
    /// Columns per 256 decoded values.
    pub weight_cols: i64,
    pub scales_cols: i64,
    pub biases_cols: i64,
}

impl KQuant {
    /// A grid format (IQ1 / IQ2 / IQ3_XXS): `.scales` holds native companion
    /// bytes (sign indices, qh, scale nibbles), so a fixture fills them with
    /// whole random bytes rather than small sub-scales.
    pub fn is_grid(&self) -> bool {
        matches!(
            self.mode,
            "iq2xxs" | "iq2xs" | "iq2s" | "iq3xxs" | "iq1s" | "iq1m"
        )
    }

    /// Groups per super-block (IQ4_NL: one 32-value block).
    pub fn super_ratio(&self) -> i64 {
        match self.mode {
            "q6k" | "q3k" | "q2k" => 16,
            "iq4nl" => 1,
            _ => 8,
        }
    }

    /// `.scales` bytes per group (the Tiled64 companion unit is
    /// `super_ratio * scale_bytes_per_group`).
    pub fn scale_bytes_per_group(&self) -> i64 {
        self.scales_cols * i64::from(self.group_size) / 256
    }

    /// `.biases` entries per super-block (the Tiled64 companion unit).
    pub fn bias_entries_per_super_block(&self) -> i64 {
        if self.mode == "iq4nl" {
            1
        } else {
            self.biases_cols
        }
    }
}

pub const KQUANTS: [KQuant; 14] = [
    KQuant {
        mode: "q2k",
        bits: 2,
        group_size: 16,
        scales_signed: false,
        weight_cols: 16,
        scales_cols: 32,
        biases_cols: 2,
    },
    KQuant {
        mode: "q3k",
        bits: 3,
        group_size: 16,
        scales_signed: true,
        weight_cols: 24,
        scales_cols: 16,
        biases_cols: 1,
    },
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
    KQuant {
        mode: "iq4nl",
        bits: 4,
        group_size: 32,
        scales_signed: true,
        weight_cols: 32,
        scales_cols: 8,
        biases_cols: 8,
    },
    KQuant {
        mode: "iq4xs",
        bits: 4,
        group_size: 32,
        scales_signed: true,
        weight_cols: 32,
        scales_cols: 8,
        biases_cols: 1,
    },
    KQuant {
        mode: "iq3s",
        bits: 8,
        group_size: 32,
        scales_signed: true,
        weight_cols: 64,
        scales_cols: 8,
        biases_cols: 1,
    },
    // The grid formats (gguf_kquant.rs): `bits` words per 32-value unit of
    // native grid indices, `scales_cols` companion bytes per 8 units.
    KQuant {
        mode: "iq2xxs",
        bits: 1,
        group_size: 32,
        scales_signed: false,
        weight_cols: 8,
        scales_cols: 32,
        biases_cols: 1,
    },
    KQuant {
        mode: "iq2xs",
        bits: 2,
        group_size: 32,
        scales_signed: false,
        weight_cols: 16,
        scales_cols: 8,
        biases_cols: 1,
    },
    KQuant {
        mode: "iq2s",
        bits: 2,
        group_size: 32,
        scales_signed: false,
        weight_cols: 16,
        scales_cols: 16,
        biases_cols: 1,
    },
    KQuant {
        mode: "iq3xxs",
        bits: 2,
        group_size: 32,
        scales_signed: false,
        weight_cols: 16,
        scales_cols: 32,
        biases_cols: 1,
    },
    KQuant {
        mode: "iq1s",
        bits: 1,
        group_size: 32,
        scales_signed: false,
        weight_cols: 8,
        scales_cols: 16,
        biases_cols: 1,
    },
    KQuant {
        mode: "iq1m",
        bits: 1,
        group_size: 32,
        scales_signed: false,
        weight_cols: 8,
        scales_cols: 24,
        biases_cols: 1,
    },
];

pub fn kquant(mode: &str) -> &'static KQuant {
    KQUANTS.iter().find(|k| k.mode == mode).expect("mode")
}

pub const DTYPES: [DType; 3] = [DType::Float32, DType::Float16, DType::BFloat16];

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

const HALF_SCALES: [u16; 6] = [0x3000, 0x2C00, 0x3400, 0x2E66, 0x3155, 0x2800];

pub struct Weights {
    pub w: MxArray,
    pub scales: MxArray,
    pub biases: MxArray,
}

/// `leading` precedes the packed axis, which expands to `packed` values: a
/// multiple of 256, or of 32 for IQ4_NL.
pub fn weights(kq: &KQuant, leading: &[i64], packed: i64, seed: u32) -> Weights {
    let cols = |per_256: i64| {
        assert_eq!(
            (packed * per_256) % 256,
            0,
            "{} cannot pack {packed} values",
            kq.mode
        );
        packed * per_256 / 256
    };
    let rows: i64 = leading.iter().product();
    let shape = |cols: i64| {
        let mut s = leading.to_vec();
        s.push(cols);
        s
    };
    let (wc, sc, bc) = (
        cols(kq.weight_cols),
        cols(kq.scales_cols),
        cols(kq.biases_cols),
    );
    let mut st = seed;
    let w: Vec<u32> = (0..rows * wc).map(|_| lcg(&mut st)).collect();
    let scales_len = (rows * sc) as usize;
    let scales = if kq.scales_signed {
        let v: Vec<i8> = (0..scales_len)
            .map(|_| (lcg(&mut st) % 17) as i8 - 8)
            .collect();
        MxArray::from_int8(&v, &shape(sc))
    } else if kq.is_grid() {
        // Native companion bytes: every bit pattern is a valid index / sign /
        // scale field.
        let v: Vec<u8> = (0..scales_len)
            .map(|_| (lcg(&mut st) >> 24) as u8)
            .collect();
        MxArray::from_uint8(&v, &shape(sc))
    } else {
        let v: Vec<u8> = (0..scales_len).map(|_| (lcg(&mut st) % 64) as u8).collect();
        MxArray::from_uint8(&v, &shape(sc))
    }
    .expect("scales");
    let biases: Vec<u16> = (0..(rows * bc) as usize)
        .map(|i| HALF_SCALES[(i + seed as usize) % HALF_SCALES.len()])
        .collect();
    Weights {
        w: MxArray::from_uint32(&w, &shape(wc)).expect("w"),
        scales,
        biases: MxArray::from_float16(&biases, &shape(bc)).expect("biases"),
    }
}

/// 8-bit integers over 128: exact in all three activation dtypes.
pub fn activation(shape: &[i64], seed: u32, dtype: DType) -> MxArray {
    let n: i64 = shape.iter().product();
    let mut st = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let values: Vec<f32> = (0..n)
        .map(|_| f32::from((lcg(&mut st) >> 24) as i8) / 128.0)
        .collect();
    match dtype {
        DType::Float32 => MxArray::from_float32(&values, shape),
        DType::Float16 => {
            let bits: Vec<u16> = values
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            MxArray::from_float16(&bits, shape)
        }
        DType::BFloat16 => {
            let bits: Vec<u16> = values.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            MxArray::from_bfloat16(&bits, shape)
        }
        other => panic!("unsupported activation dtype {other:?}"),
    }
    .expect("activation")
}

pub fn indices(values: &[u32]) -> MxArray {
    MxArray::from_uint32(values, &[values.len() as i64]).expect("indices")
}

pub fn random_ids(seed: u32, n: usize, below: u32) -> Vec<u32> {
    let mut st = seed;
    (0..n).map(|_| lcg(&mut st) % below).collect()
}

pub fn ptr(a: Option<&MxArray>) -> *mut mlx_sys::mlx_array {
    a.map_or(std::ptr::null_mut(), |a| a.as_raw_ptr())
}

pub fn mode_cstr(kq: &KQuant) -> CString {
    CString::new(kq.mode).expect("mode")
}

fn eval(handle: *mut mlx_sys::mlx_array) -> Result<(), String> {
    let mut buf = [0i8; 512];
    let mut h = handle;
    // SAFETY: `handle` is live; `buf` is the error sink.
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

/// Raw output shape and bits; consumes the handle.
pub fn read_bits(what: &str, h: *mut mlx_sys::mlx_array) -> (Vec<i64>, DType, Vec<u32>) {
    assert!(!h.is_null(), "{what}: rejected at construction");
    if let Err(e) = eval(h) {
        // SAFETY: owned handle.
        unsafe { mlx_sys::mlx_array_delete(h) };
        panic!("{what}: {e}");
    }
    // SAFETY: `h` is live and evaluated.
    let (len, code, ndim) = unsafe {
        (
            mlx_sys::mlx_array_size(h),
            mlx_sys::mlx_array_dtype(h),
            mlx_sys::mlx_array_ndim(h),
        )
    };
    let mut shape = vec![0i64; ndim];
    // SAFETY: `shape` holds `ndim` entries.
    unsafe { mlx_sys::mlx_array_shape(h, shape.as_mut_ptr()) };
    let (dtype, bits) = if code == DType::Float32 as i32 {
        let mut v = vec![0f32; len];
        // SAFETY: `v` holds `len` elements.
        assert!(unsafe { mlx_sys::mlx_array_to_float32(h, v.as_mut_ptr(), len) });
        (DType::Float32, v.iter().map(|x| x.to_bits()).collect())
    } else {
        let mut v = vec![0u16; len];
        // SAFETY: `v` holds `len` elements.
        assert!(unsafe { mlx_sys::mlx_array_to_uint16(h, v.as_mut_ptr(), len) });
        let dtype = if code == DType::Float16 as i32 {
            DType::Float16
        } else {
            DType::BFloat16
        };
        (dtype, v.into_iter().map(u32::from).collect())
    };
    // SAFETY: owned handle, not used afterwards.
    unsafe { mlx_sys::mlx_array_delete(h) };
    (shape, dtype, bits)
}

/// Both handles must evaluate to the same shape, dtype and bits, not all zero.
pub fn assert_identical(
    what: &str,
    ours: *mut mlx_sys::mlx_array,
    reference: *mut mlx_sys::mlx_array,
) {
    let (ours_shape, ours_dtype, ours) = read_bits(&format!("{what} ours"), ours);
    let (ref_shape, ref_dtype, reference) = read_bits(&format!("{what} reference"), reference);
    assert_eq!(ours_shape, ref_shape, "{what}: shape differs");
    assert_eq!(ours_dtype, ref_dtype, "{what}: dtype differs");
    if let Some(i) = (0..ours.len()).find(|&i| ours[i] != reference[i]) {
        panic!(
            "{what}: first difference at {i}: ours {:#x}, reference {:#x} ({} of {} differ)",
            ours[i],
            reference[i],
            (0..ours.len()).filter(|&j| ours[j] != reference[j]).count(),
            ours.len()
        );
    }
    assert!(
        ours.iter().any(|&b| b & 0x7fff_ffff != 0),
        "{what}: an all-zero output proves nothing"
    );
}

/// Raw output shape and bits, not all zero; consumes the handle.
pub fn read_output(what: &str, h: *mut mlx_sys::mlx_array) -> (Vec<i64>, DType, Vec<u32>) {
    let out = read_bits(what, h);
    assert!(
        out.2.iter().any(|&b| b & 0x7fff_ffff != 0),
        "{what}: an all-zero output proves nothing"
    );
    out
}

pub fn quantized_matmul(
    x: &MxArray,
    w: &Weights,
    transpose: bool,
    kq: &KQuant,
    device: i32,
) -> *mut mlx_sys::mlx_array {
    let mode = mode_cstr(kq);
    // SAFETY: every handle outlives the call.
    unsafe {
        mlx_sys::mlx_test_kquant_quantized_matmul(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            transpose,
            kq.group_size,
            kq.bits,
            mode.as_ptr(),
            device,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(
    x: &MxArray,
    w: &Weights,
    lhs: Option<&MxArray>,
    rhs: &MxArray,
    transpose: bool,
    sorted: bool,
    kq: &KQuant,
    device: i32,
) -> *mut mlx_sys::mlx_array {
    let mode = mode_cstr(kq);
    // SAFETY: every handle outlives the call; null lhs means "derive from x".
    unsafe {
        mlx_sys::mlx_test_kquant_gather_qmm(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            ptr(lhs),
            rhs.as_raw_ptr(),
            transpose,
            kq.group_size,
            kq.bits,
            mode.as_ptr(),
            sorted,
            device,
        )
    }
}

pub fn dequantize(w: &Weights, dtype: DType, kq: &KQuant, device: i32) -> *mut mlx_sys::mlx_array {
    let mode = mode_cstr(kq);
    // SAFETY: every handle outlives the call.
    unsafe {
        mlx_sys::mlx_test_kquant_dequantize(
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            kq.group_size,
            kq.bits,
            dtype as i32,
            mode.as_ptr(),
            device,
        )
    }
}

/// Turns this thread's kernel-family counters on (and resets them).
pub fn start_counting() {
    // SAFETY: thread-local test hook.
    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
}

pub fn stop_counting() {
    // SAFETY: thread-local test hook.
    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
}

pub fn family_count(family: &str) -> u64 {
    let name = CString::new(family).expect("family");
    // SAFETY: thread-local test hook reading a NUL-terminated name.
    unsafe { mlx_sys::mlx_test_kquant_family_count(name.as_ptr()) }
}

/// The generation the Metal dispatcher sees (honours `MLX_METAL_GPU_ARCH`).
pub fn gpu_gen() -> i32 {
    // SAFETY: nullary test hook.
    unsafe { mlx_sys::mlx_test_kquant_gpu_gen() }
}

pub fn nax_available() -> bool {
    // SAFETY: nullary predicate that catches internally.
    unsafe { mlx_sys::mlx_metal_is_nax_available() }
}

/// float32 qmm reaches NAX only when `MLX_ENABLE_TF32` is unset or nonzero.
pub fn tf32_enabled() -> bool {
    match std::env::var("MLX_ENABLE_TF32") {
        Ok(v) => v.trim().parse::<i32>().is_ok_and(|n| n != 0),
        Err(_) => true,
    }
}
