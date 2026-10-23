//! Source-style selector edge arithmetic is a separate, opt-in experiment.
//! The oracle independently sums exact BF16 inputs in F64; compiled replays
//! vary anchors, candidates and codebooks while preserving the fixed shape.
use mlx_core::array::{DType, MxArray};
use std::sync::atomic::{AtomicUsize, Ordering};

const K: usize = 16;
const RANK: usize = 256;
const VOCAB: usize = 97;

fn values(count: usize, mut seed: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            half::bf16::from_f32(((seed >> 8) as f32 / 16777216.0 - 0.5) * 0.75).to_f32()
        })
        .collect()
}

fn matrix(values: &[f32], rows: usize, columns: usize, strided: bool) -> MxArray {
    if !strided {
        return MxArray::from_float32(values, &[rows as i64, columns as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
    }
    let transposed: Vec<_> = (0..columns)
        .flat_map(|c| (0..rows).map(move |r| values[r * columns + c]))
        .collect();
    MxArray::from_float32(&transposed, &[columns as i64, rows as i64])
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap()
        .transpose(Some(&[1, 0]))
        .unwrap()
}

struct Fixture {
    arrays: Vec<MxArray>,
    candidates: Vec<i32>,
    unary: Vec<f32>,
    hidden: Vec<f32>,
    predecessor: Vec<f32>,
    successor: Vec<f32>,
    anchor: i32,
    rows: usize,
}

impl Fixture {
    fn new(rows: usize, seed: u32, strided: bool, anchor: i32) -> Self {
        let mut candidates: Vec<_> = (0..rows * K)
            .map(|i| ((i * 17 + seed as usize * 3) % VOCAB) as i32)
            .collect();
        // The production ids are always valid. Exercise the source clamp too,
        // so a damaged id cannot read outside either codebook.
        if seed == 29 {
            candidates[1] = -1;
            candidates[7] = VOCAB as i32 + 3;
        }
        let unary = values(rows * K, seed + 1);
        let hidden = values(rows * RANK, seed + 2);
        let predecessor = values(VOCAB * RANK, seed + 3);
        let successor = values(VOCAB * RANK, seed + 4);
        let arrays = vec![
            MxArray::from_int32(&candidates, &[1, rows as i64, K as i64]).unwrap(),
            MxArray::from_float32(&unary, &[1, rows as i64, K as i64]).unwrap(),
            matrix(&hidden, rows, RANK, strided)
                .reshape(&[rows as i64, 1, RANK as i64])
                .unwrap(),
            matrix(&predecessor, VOCAB, RANK, strided),
            matrix(&successor, VOCAB, RANK, strided),
            MxArray::from_int32(&[anchor], &[1]).unwrap(),
        ];
        Self {
            arrays,
            candidates,
            unary,
            hidden,
            predecessor,
            successor,
            anchor,
            rows,
        }
    }

    fn oracle(&self) -> Vec<f64> {
        let safe = |id: i32| (id as u32 as usize).min(VOCAB - 1);
        let mut scores = vec![0.0; self.rows * K * K];
        for position in 0..self.rows {
            for previous in 0..K {
                let p = safe(if position == 0 {
                    self.anchor
                } else {
                    self.candidates[(position - 1) * K + previous]
                });
                for next in 0..K {
                    let s = safe(self.candidates[position * K + next]);
                    let edge = (0..RANK)
                        .map(|d| {
                            f64::from(self.predecessor[p * RANK + d])
                                * f64::from(self.hidden[position * RANK + d])
                                * f64::from(self.successor[s * RANK + d])
                        })
                        .sum::<f64>();
                    scores[(position * K + previous) * K + next] =
                        edge + f64::from(self.unary[position * K + next]);
                }
            }
        }
        scores
    }

    fn call(&self) -> Raw {
        let a: Vec<_> = self.arrays.iter().map(MxArray::as_raw_ptr).collect();
        Raw::new(unsafe { mlx_sys::mlx_dflash2_selector_edges(a[0], a[1], a[2], a[3], a[4], a[5]) })
    }

    fn check(&self, scores: &Raw) {
        let actual = scores.floats();
        let expected = self.oracle();
        assert_eq!(actual.len(), expected.len());
        let peak = expected
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max)
            .max(1e-8);
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, e)| (f64::from(*a) - e).abs())
            .fold(0.0, f64::max);
        assert!(
            max_error <= peak * 2e-6 + 1e-7,
            "rows={} max_error={max_error} peak={peak}",
            self.rows
        );
        assert!(
            actual.iter().any(|v| v.to_bits() & 0xffff != 0),
            "FP32 scores must not round through BF16"
        );
        let mut path = std::ptr::null_mut();
        let mut confidence = std::ptr::null_mut();
        assert!(unsafe {
            mlx_sys::mlx_dflash2_greedy_path_confidence(
                self.arrays[0].as_raw_ptr(),
                scores.0,
                &mut path,
                &mut confidence,
            )
        });
        let path = Raw::new(path);
        let confidence = Raw::new(confidence).floats();
        let mut ids = vec![0i32; self.rows];
        let mut roots = [path.0];
        assert!(unsafe { mlx_sys::mlx_eval(roots.as_mut_ptr(), roots.len()) });
        assert!(unsafe { mlx_sys::mlx_array_to_int32(path.0, ids.as_mut_ptr(), ids.len()) });
        let mut previous = 0;
        for position in 0..self.rows {
            let row = &expected[(position * K + previous) * K..(position * K + previous + 1) * K];
            let selected = (0..K).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            assert_eq!(ids[position], self.candidates[position * K + selected]);
            let mass: f64 = row.iter().map(|score| (score - row[selected]).exp()).sum();
            assert!((f64::from(confidence[position]) - 1.0 / mass).abs() < 2e-6);
            previous = selected;
        }
    }
}

struct Raw(*mut mlx_sys::mlx_array);
impl Raw {
    fn new(handle: *mut mlx_sys::mlx_array) -> Self {
        assert!(
            !handle.is_null(),
            "selector edge contract or construction failed"
        );
        Self(handle)
    }
    fn floats(&self) -> Vec<f32> {
        let mut roots = [self.0];
        assert!(unsafe { mlx_sys::mlx_eval(roots.as_mut_ptr(), 1) });
        assert_eq!(
            unsafe { mlx_sys::mlx_array_dtype(self.0) },
            DType::Float32 as i32
        );
        let mut values = vec![0.0f32; unsafe { mlx_sys::mlx_array_size(self.0) }];
        assert!(unsafe {
            mlx_sys::mlx_array_to_float32(self.0, values.as_mut_ptr(), values.len())
        });
        values
    }
}
impl Drop for Raw {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_array_delete(self.0) };
    }
}

#[test]
fn selector_edges_match_f64_oracle_and_preserve_fp32_confidence() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    for (rows, seed, strided, anchor) in [
        (1, 11, false, 7),
        (2, 19, true, 42),
        (7, 29, false, -1),
        (7, 37, true, 96),
    ] {
        let f = Fixture::new(rows, seed, strided, anchor);
        f.check(&f.call());
    }
}

unsafe extern "C" fn build(
    context: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    n_inputs: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    n_outputs: usize,
) -> bool {
    if context.is_null() || n_inputs != 6 || n_outputs != 1 {
        return false;
    }
    unsafe { &*context.cast::<AtomicUsize>() }.fetch_add(1, Ordering::Relaxed);
    let a: Vec<_> = unsafe { std::slice::from_raw_parts(inputs, n_inputs) }
        .iter()
        .map(|h| h.cast_mut())
        .collect();
    let out = unsafe { mlx_sys::mlx_dflash2_selector_edges(a[0], a[1], a[2], a[3], a[4], a[5]) };
    for handle in a {
        unsafe { mlx_sys::mlx_array_delete(handle) };
    }
    unsafe {
        *outputs = out;
    }
    !out.is_null()
}

struct Compiled(u64);
impl Drop for Compiled {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, self.0) };
    }
}

#[test]
fn selector_edges_compiled_replay_reads_new_anchor_and_codebooks() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let compiled = Compiled(0xED6E_3500_0000_0001);
    let mut traces = AtomicUsize::new(0);
    for (seed, anchor) in [(41, 0), (41, 96), (53, 96), (67, 9)] {
        let f = Fixture::new(7, seed, false, anchor);
        let a: Vec<_> = f
            .arrays
            .iter()
            .map(|a| a.as_raw_ptr().cast_const())
            .collect();
        let mut out = std::ptr::null_mut();
        assert!(unsafe {
            mlx_sys::mlx_compiled_graph_invoke(
                compiled.0,
                Some(build),
                (&mut traces as *mut AtomicUsize).cast(),
                a.as_ptr(),
                a.len(),
                &mut out,
                1,
                true,
            )
        });
        let out = Raw::new(out);
        f.check(&out);
        assert_eq!(
            out.floats(),
            f.call().floats(),
            "compiled replay changed FP32 score bits"
        );
        assert_eq!(traces.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn selector_edges_reject_bad_dtype_and_shape() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let f = Fixture::new(7, 73, false, 5);
    for replacement in [
        f.arrays[2].astype(DType::Float32).unwrap(),
        MxArray::zeros(&[7, 1, 128], Some(DType::BFloat16)).unwrap(),
    ] {
        let mut a: Vec<_> = f.arrays.iter().map(MxArray::as_raw_ptr).collect();
        a[2] = replacement.as_raw_ptr();
        assert!(
            unsafe { mlx_sys::mlx_dflash2_selector_edges(a[0], a[1], a[2], a[3], a[4], a[5]) }
                .is_null()
        );
    }
}
