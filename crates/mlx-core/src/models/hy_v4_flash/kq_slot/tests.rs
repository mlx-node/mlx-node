use std::time::Instant;

use super::*;
use crate::array::alloc::{OutLayout, SendPtr};
use crate::utils::gguf::GgufTensorType;
use crate::utils::gguf_kquant::KQuantFormat;

fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

fn info(ty: GgufTensorType, k: usize, rows: usize) -> GgufTensorInfo {
    GgufTensorInfo {
        name: "blk.1.ffn_down_exps.weight".to_string(),
        n_dims: 2,
        dims: vec![k as u64, rows as u64],
        tensor_type: ty,
        offset: 0,
    }
}

/// Row-major destination arrays of `rows` rows, every byte `fill`.
struct Host {
    w: Vec<u32>,
    s: Vec<u8>,
    b: Vec<u16>,
    rows: usize,
    k: usize,
}

impl Host {
    fn new(format: KQuantFormat, rows: usize, k: usize, fill: u8) -> Self {
        let f16 = u16::from_ne_bytes([fill, fill]);
        Self {
            w: vec![u32::from_ne_bytes([fill; 4]); rows * format.weight_cols(k)],
            s: vec![fill; rows * format.scales_cols(k)],
            b: vec![f16; rows * format.biases_cols(k)],
            rows,
            k,
        }
    }

    fn dst(&mut self) -> DstPtrs {
        DstPtrs {
            weight: SendPtr(self.w.as_mut_ptr().cast()),
            scales: Some(SendPtr(self.s.as_mut_ptr())),
            biases: Some(SendPtr(self.b.as_mut_ptr().cast())),
            rows: self.rows,
            k: self.k,
            layout: OutLayout::RowMajor,
        }
    }
}

fn check(ty: GgufTensorType, k: usize, rows: usize, row0: usize, seed: u64) {
    let format = ty.k_quant_format().unwrap();
    let total = row0 + rows + 3;
    let t = info(ty, k, total);
    let blocks = random_bytes(rows * format.row_bytes(k), seed);
    let mut want = Host::new(format, total, k, 0xA5);
    let mut got = Host::new(format, total, k, 0xA5);
    cpu_kquant_band_into(&t, &blocks, row0, rows, &want.dst()).unwrap();
    rows_into(&t, &blocks, row0, rows, &got.dst()).unwrap();
    let what = format!("{ty:?} k {k} rows {row0}+{rows}");
    assert!(got.w == want.w, "{what}: weight");
    assert!(got.s == want.s, "{what}: scales");
    assert!(got.b == want.b, "{what}: biases");
}

#[test]
fn rows_equal_the_cpu_band_repack() {
    let mut seed = 1;
    for ty in [GgufTensorType::Q4K, GgufTensorType::Q6K] {
        for k in [256, 1536] {
            for rows in [1, 7, 384, 1536] {
                for row0 in [0, 1, 5, 383] {
                    check(ty, k, rows, row0, seed);
                    seed += 1;
                }
            }
        }
    }
}

/// Every Q4_K scale byte pattern (both `get_scale_min_k4` branches see all
/// top-two-bit combinations) and all-ones / all-zero code bytes.
#[test]
fn edge_bytes_equal_the_cpu_band_repack() {
    for ty in [GgufTensorType::Q4K, GgufTensorType::Q6K] {
        let format = ty.k_quant_format().unwrap();
        for fill in [0x00u8, 0xFF, 0xC3, 0x3C] {
            let t = info(ty, 512, 4);
            let mut blocks = vec![fill; 4 * format.row_bytes(512)];
            for (i, b) in blocks.iter_mut().enumerate().step_by(7) {
                *b = i as u8;
            }
            let mut want = Host::new(format, 4, 512, 0);
            let mut got = Host::new(format, 4, 512, 0);
            cpu_kquant_band_into(&t, &blocks, 0, 4, &want.dst()).unwrap();
            rows_into(&t, &blocks, 0, 4, &got.dst()).unwrap();
            assert!(
                got.w == want.w && got.s == want.s && got.b == want.b,
                "{ty:?} fill {fill:#x}"
            );
        }
    }
}

#[test]
fn other_formats_take_the_cpu_band_repack() {
    let (k, rows) = (512, 5);
    let format = KQuantFormat::Q5K;
    let t = info(GgufTensorType::Q5K, k, rows + 2);
    let blocks = random_bytes(rows * format.row_bytes(k), 3);
    let mut want = Host::new(format, rows + 2, k, 0x11);
    let mut got = Host::new(format, rows + 2, k, 0x11);
    cpu_kquant_band_into(&t, &blocks, 2, rows, &want.dst()).unwrap();
    rows_into(&t, &blocks, 2, rows, &got.dst()).unwrap();
    assert!(got.w == want.w && got.s == want.s && got.b == want.b);
}

#[test]
fn bad_bands_are_refused() {
    for ty in [GgufTensorType::Q4K, GgufTensorType::Q6K] {
        let format = ty.k_quant_format().unwrap();
        let t = info(ty, 256, 4);
        let mut h = Host::new(format, 4, 256, 0);
        let d = h.dst();
        let row = format.row_bytes(256);
        assert!(rows_into(&t, &vec![0; 2 * row - 1], 0, 2, &d).is_err());
        assert!(rows_into(&t, &vec![0; 2 * row], 3, 2, &d).is_err());
        let mut no_scales = d;
        no_scales.scales = None;
        assert!(rows_into(&t, &vec![0; row], 0, 1, &no_scales).is_err());
        rows_into(&t, &vec![0; 2 * row], 2, 2, &d).unwrap();
    }
}

/// GGUF bytes per second of one thread, slot repack vs the CPU band repack,
/// one 1536 x 1536 expert projection per call.
#[test]
#[ignore = "CPU rate probe"]
fn slot_repack_rate() {
    let (k, rows, reps) = (1536usize, 1536usize, 200usize);
    for ty in [GgufTensorType::Q4K, GgufTensorType::Q6K] {
        let format = ty.k_quant_format().unwrap();
        let t = info(ty, k, rows);
        let blocks = random_bytes(rows * format.row_bytes(k), 7);
        let mut h = Host::new(format, rows, k, 0);
        let d = h.dst();
        let rate = |f: &dyn Fn() -> Result<()>| {
            f().unwrap();
            let t0 = Instant::now();
            for _ in 0..reps {
                f().unwrap();
            }
            (blocks.len() * reps) as f64 / t0.elapsed().as_secs_f64() / 1e9
        };
        let slot = rate(&|| rows_into(&t, &blocks, 0, rows, &d));
        let band = rate(&|| cpu_kquant_band_into(&t, &blocks, 0, rows, &d));
        println!(
            "{{\"format\":\"{}\",\"bytes\":{},\"reps\":{reps},\"slot_gbps\":{slot:.3},\"band_gbps\":{band:.3},\"speedup\":{:.2}}}",
            format.mlx_mode(),
            blocks.len(),
            slot / band
        );
    }
}
