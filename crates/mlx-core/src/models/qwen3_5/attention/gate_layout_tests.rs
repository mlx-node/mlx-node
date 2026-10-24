//! Regression coverage for moving the existing gate across a layout-only reshape.
use super::*;
use crate::compiled_graph::{erase_compiled_graphs_matching, invoke_compiled_graph};

const GEOMETRIES: [(i64, i64, i64, i64); 6] = [
    (1, 1, 16, 256),
    (1, 8, 16, 256),
    (2, 8, 4, 64),
    (3, 1, 8, 128),
    (2, 7, 6, 96),
    (1, 8, 1, 64),
];

fn metal_available() -> bool {
    unsafe { mlx_sys::mlx_metal_is_available() && mlx_sys::mlx_default_device() == 1 }
}

fn storage(shape: &[i64], dtype: DType, seed: usize, gate: bool) -> Result<MxArray> {
    let n = shape.iter().product::<i64>() as usize;
    let values: Vec<f32> = (0..n)
        .map(|i| {
            if gate {
                let boundary = [-100.0, -20.0, -2.0, -0.1, -0.0, 0.0, 0.1, 2.0, 20.0, 100.0];
                boundary[(i + seed) % boundary.len()] + ((i * 7 + seed) % 19) as f32 / 64.0
            } else {
                (((i * 13 + seed * 31) % 257) as f32 - 128.0) / 23.0
            }
        })
        .collect();
    let array = MxArray::from_float32(&values, shape)?.astype(dtype)?;
    MxArray::eval_arrays(&[&array])?;
    Ok(array)
}

fn metal_location(x: &MxArray) -> (*mut std::ffi::c_void, usize) {
    unsafe {
        (
            mlx_sys::mlx_array_get_metal_buffer(x.as_raw_ptr()),
            mlx_sys::mlx_array_get_buffer_offset(x.as_raw_ptr()),
        )
    }
}

/// Query a materialized view's effective stride without adding a test FFI.
/// Single-element slices must share storage and expose the exact byte delta.
fn assert_axis_stride(x: &MxArray, axis: usize, stride: i64) -> Result<()> {
    let shape = x.shape()?.to_vec();
    if shape[axis] <= 1 {
        return Ok(());
    }
    let zero = vec![0; shape.len()];
    let one = vec![1; shape.len()];
    let first = x.slice(&zero, &one)?;
    let mut starts = zero;
    let mut ends = one;
    starts[axis] = 1;
    ends[axis] = 2;
    let second = x.slice(&starts, &ends)?;
    MxArray::eval_arrays(&[x, &first, &second])?;
    let a = metal_location(&first);
    let b = metal_location(&second);
    assert!(!a.0.is_null());
    assert_eq!(a.0, metal_location(x).0, "slice unexpectedly copied");
    assert_eq!(a.0, b.0, "slice unexpectedly copied");
    let item = unsafe { mlx_sys::mlx_array_get_itemsize(x.as_raw_ptr()) };
    assert_eq!(b.1 - a.1, stride as usize * item, "axis={axis}");
    Ok(())
}

fn inputs(
    geometry: (i64, i64, i64, i64),
    gate_dtype: DType,
    value_dtype: DType,
    layout: usize,
    seed: usize,
) -> Result<(MxArray, MxArray)> {
    let (b, t, h, d) = geometry;
    let hd = h * d;
    let gate = match layout {
        0 => storage(&[b, t, hd], gate_dtype, seed, true)?,
        1 => {
            // Actual packed QGateBlock production view: second half of
            // [B,T,2*H*D], with 2*H*D between token rows.
            let packed = storage(&[b, t, 2 * hd], gate_dtype, seed, true)?;
            let gate = packed.split_sections(&[hd], 2)?[1].clone();
            MxArray::eval_arrays(&[&gate])?;
            assert_eq!(metal_location(&packed).0, metal_location(&gate).0);
            assert_axis_stride(&gate, 0, t * 2 * hd)?;
            assert_axis_stride(&gate, 1, 2 * hd)?;
            assert_axis_stride(&gate, 2, 1)?;
            gate
        }
        _ => {
            // Unfinalized checkpoint fallback: [Q_h0,G_h0,Q_h1,G_h1,...].
            // project_q_gate already flattens this view before the gate call.
            let packed = storage(&[b, t, h, 2 * d], gate_dtype, seed, true)?;
            let per_head = packed.slice(&[0, 0, 0, d], &[b, t, h, 2 * d])?;
            assert_axis_stride(&per_head, 0, t * h * 2 * d)?;
            assert_axis_stride(&per_head, 1, h * 2 * d)?;
            assert_axis_stride(&per_head, 2, 2 * d)?;
            assert_axis_stride(&per_head, 3, 1)?;
            per_head.reshape(&[b, t, hd])?
        }
    };
    let value = if layout == 2 {
        // Replay a noncontiguous head-major view as well as ordinary SDPA output.
        storage(&[b, h, t + 1, d], value_dtype, seed + 97, false)?
            .slice(&[0, 0, 1, 0], &[b, h, t + 1, d])?
    } else {
        storage(&[b, h, t, d], value_dtype, seed + 97, false)?
    };
    MxArray::eval_arrays(&[&gate, &value])?;
    Ok((gate, value))
}

fn original(gate: &MxArray, value: &MxArray, geometry: (i64, i64, i64, i64)) -> Result<MxArray> {
    let (b, t, h, d) = geometry;
    let flat = value
        .transpose(Some(&[0, 2, 1, 3]))?
        .reshape(&[b, t, h * d])?;
    Activations::sigmoid_mul_compiled(gate, &flat)
}

fn exact_bits(x: &MxArray) -> Result<Vec<u32>> {
    Ok(x.astype(DType::Float32)?
        .to_float32()?
        .iter()
        .map(|x| x.to_bits())
        .collect())
}

#[test]
#[ignore = "requires coordinated Metal GPU validation"]
fn verify_gate_layout_preserves_bits_dtype_and_view_reshape() -> Result<()> {
    if !metal_available() {
        return Ok(());
    }
    for geometry in GEOMETRIES {
        let (b, t, h, d) = geometry;
        for (gate_dtype, value_dtype) in [
            (DType::BFloat16, DType::BFloat16),
            (DType::Float16, DType::Float16),
            (DType::Float32, DType::Float32),
            (DType::Float32, DType::BFloat16),
            (DType::BFloat16, DType::Float32),
        ] {
            for layout in 0..3 {
                let (gate, value) = inputs(geometry, gate_dtype, value_dtype, layout, 37)?;
                let old = original(&gate, &value, geometry)?;
                let new = gate_verify_output(&gate, &value, b, t, h, d)?;
                assert_eq!(new.shape()?.as_ref(), &[b, t, h * d]);
                assert_eq!(new.dtype()?, old.dtype()?);
                assert_eq!(
                    exact_bits(&new)?,
                    exact_bits(&old)?,
                    "{geometry:?} {gate_dtype:?}/{value_dtype:?} layout={layout}"
                );
                // Retain the exact intermediate to prove the final flatten is
                // a shared-buffer view, not a postponed transpose/copy.
                let gated = Activations::sigmoid_mul_compiled(
                    &gate.reshape(&[b, t, h, d])?,
                    &value.transpose(Some(&[0, 2, 1, 3]))?,
                )?;
                MxArray::eval_arrays(&[&gated])?;
                let flat = gated.reshape(&[b, t, h * d])?;
                MxArray::eval_arrays(&[&flat])?;
                assert_eq!(metal_location(&gated), metal_location(&flat));
                assert_axis_stride(&gated, 0, t * h * d)?;
                assert_axis_stride(&gated, 1, h * d)?;
                assert_axis_stride(&gated, 2, d)?;
                assert_axis_stride(&gated, 3, 1)?;
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires coordinated Metal GPU validation"]
fn verify_gate_layout_compiled_replay_tracks_data_and_strides() -> Result<()> {
    if !metal_available() {
        return Ok(());
    }
    for (case, geometry) in GEOMETRIES.into_iter().enumerate() {
        let (b, t, h, d) = geometry;
        for (kind, dtype) in [DType::BFloat16, DType::Float16, DType::Float32]
            .into_iter()
            .enumerate()
        {
            let id = 0xDFCA_7100_0000_0000 | ((case as u64) << 8) | kind as u64;
            erase_compiled_graphs_matching(u64::MAX, id);
            let builds = std::cell::Cell::new(0);
            let mut builder = |args: &[MxArray]| {
                builds.set(builds.get() + 1);
                Ok(vec![gate_verify_output(&args[0], &args[1], b, t, h, d)?])
            };
            let mut previous = None;
            for (step, layout) in [0, 1, 2, 1].into_iter().enumerate() {
                let (gate, value) = inputs(geometry, dtype, dtype, layout, 17 + step * 23)?;
                let out = invoke_compiled_graph(id, &[&gate, &value], 1, true, &mut builder)?
                    .ok_or_else(|| Error::from_reason("gate layout compile failed"))?;
                assert_eq!(out[0].shape()?.as_ref(), &[b, t, h * d]);
                assert_eq!(out[0].dtype()?, dtype);
                let bits = exact_bits(&out[0])?;
                assert_eq!(
                    bits,
                    exact_bits(&original(&gate, &value, geometry)?)?,
                    "{geometry:?} {dtype:?} replay={step}"
                );
                if let Some(prior) = previous {
                    assert_ne!(bits, prior, "replay reused old data");
                }
                previous = Some(bits);
            }
            assert_eq!(builds.get(), 1, "layout replay retraced");
            erase_compiled_graphs_matching(u64::MAX, id);
        }
    }
    Ok(())
}

#[test]
#[ignore = "exclusive Metal slot; producer-packed attention SG8 exactness"]
fn attention_prepared_sg8_preserves_gate_packing_sums_and_replay() -> Result<()> {
    if !metal_available() || !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
        return Ok(());
    }
    const FLAG: &str = "MLX_ATTENTION_PREPARED_SG8";
    struct Restore(Option<std::ffi::OsString>);
    impl Drop for Restore {
        fn drop(&mut self) {
            unsafe {
                match &self.0 {
                    Some(v) => std::env::set_var(FLAG, v),
                    None => std::env::remove_var(FLAG),
                }
            }
        }
    }
    let _restore = Restore(std::env::var_os(FLAG));
    unsafe { std::env::set_var(FLAG, "1") };
    let id = 0xDFCA_7500_0000_0000;
    erase_compiled_graphs_matching(u64::MAX, id);
    let builds = std::cell::Cell::new(0);
    let mut builder = |a: &[MxArray]| {
        builds.set(builds.get() + 1);
        let (values, p) = gate_verify_output_prepared(&a[0], &a[1])?
            .ok_or_else(|| Error::from_reason("attention SG8 unexpectedly declined"))?;
        Ok(vec![values, p.packed, p.sums])
    };
    for layout in 0..4 {
        for seed in [1, 17, 53] {
            let (gate, output) = if layout < 3 {
                inputs(
                    (1, 8, 16, 256),
                    DType::BFloat16,
                    DType::BFloat16,
                    layout,
                    seed,
                )?
            } else {
                // Scalar source reads support arbitrary BF16 offsets and row
                // strides; only the freshly written packed table is vector read.
                let gate = storage(&[1, 8, 4097], DType::BFloat16, seed, true)?
                    .slice(&[0, 0, 1], &[1, 8, 4097])?;
                let output = storage(&[1, 16, 8, 257], DType::BFloat16, seed + 97, false)?
                    .slice(&[0, 0, 0, 1], &[1, 16, 8, 257])?;
                (gate, output)
            };
            let expected = gate_verify_output(&gate, &output, 1, 8, 16, 256)?;
            let expected_bits = exact_bits(&expected)?;
            unsafe { mlx_sys::mlx_test_kquant_counting(true) };
            let (values, p) = gate_verify_output_prepared(&gate, &output)?
                .ok_or_else(|| Error::from_reason("attention SG8 declined supported layout"))?;
            MxArray::eval_arrays(&[&values, &p.packed, &p.sums])?;
            let count = unsafe {
                mlx_sys::mlx_test_kquant_family_count(c"attention_gate_sg8_prepare".as_ptr())
            };
            unsafe { mlx_sys::mlx_test_kquant_counting(false) };
            assert_eq!(count, 1);
            assert_eq!(
                exact_bits(&values)?,
                expected_bits,
                "gate layout{layout} seed{seed}"
            );
            let x = expected.astype(DType::Float32)?.to_float32()?.to_vec();
            let mut packed = vec![0u32; 16384];
            let mut sums = vec![0f32; 1024];
            for unit in 0..128usize {
                for lane in 0..32usize {
                    let qid = lane >> 2;
                    let fm = (qid & 4) | ((lane >> 1) & 3);
                    let row = ((qid & 2) << 1) | ((lane & 1) << 1);
                    for j in 0..4 {
                        let col = unit * 32 + fm * 4 + j;
                        packed[(unit * 32 + lane) * 4 + j] = (x[row * 4096 + col].to_bits() >> 16)
                            | (x[(row + 1) * 4096 + col].to_bits() & 0xffff0000);
                    }
                }
                for row in 0..8 {
                    let mut partial = [0f32; 8];
                    for (fm, v) in partial.iter_mut().enumerate() {
                        let at = row * 4096 + unit * 32 + fm * 4;
                        *v = (x[at] + x[at + 1]) + (x[at + 2] + x[at + 3]);
                    }
                    for xor in [1, 2, 4] {
                        let old = partial;
                        for fm in 0..8 {
                            partial[fm] = old[fm] + old[fm ^ xor];
                        }
                    }
                    sums[unit * 8 + row] = partial[0];
                }
            }
            assert_eq!(
                p.packed.to_uint32()?.as_ref(),
                packed.as_slice(),
                "packed layout{layout}"
            );
            let got = p.sums.to_float32()?;
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                sums.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "sums layout{layout}"
            );
            let replay = invoke_compiled_graph(id, &[&gate, &output], 3, true, &mut builder)?
                .ok_or_else(|| Error::from_reason("attention SG8 compile failed"))?;
            assert_eq!(exact_bits(&replay[0])?, expected_bits);
            assert_eq!(replay[1].to_uint32()?.as_ref(), packed.as_slice());
            assert_eq!(
                replay[2]
                    .to_float32()?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                sums.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    assert_eq!(builds.get(), 1, "layout replay retraced");
    erase_compiled_graphs_matching(u64::MAX, id);
    for (b, t, h, d) in [(1, 7, 16, 256), (2, 8, 16, 256), (1, 8, 8, 256)] {
        let (gate, output) = inputs((b, t, h, d), DType::BFloat16, DType::BFloat16, 0, 1)?;
        assert!(gate_verify_output_prepared(&gate, &output)?.is_none());
    }
    let (gate, output) = inputs((1, 8, 16, 256), DType::Float32, DType::Float32, 0, 1)?;
    assert!(gate_verify_output_prepared(&gate, &output)?.is_none());
    Ok(())
}
