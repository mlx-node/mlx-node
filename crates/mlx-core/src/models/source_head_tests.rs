//! Loaded-projection policy tests. The native bridge has a separate original
//! Splash bit oracle and dynamic-weight/stride/recovery test suite.
use super::*;
use crate::array::DType;
use std::path::PathBuf;

fn root() -> PathBuf {
    std::env::var_os("SPLASH_SOURCE_HEAD_FIXTURES")
        .map(PathBuf::from)
        .expect("set SPLASH_SOURCE_HEAD_FIXTURES to the hash-bound original-source fixtures")
}

fn fixture_projection() -> Result<LinearProj> {
    let root = root();
    let weights = std::fs::read(root.join("weight.u32")).unwrap();
    let scales = std::fs::read(root.join("scales.i8")).unwrap();
    let bases = std::fs::read(root.join("bases.f16")).unwrap();
    Ok(LinearProj::Quantized(QuantizedLinear::new(
        MxArray::from_uint32(
            &weights
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| u32::from_le_bytes(*v))
                .collect::<Vec<_>>(),
            &[248320, 384],
        )?,
        MxArray::from_int8(
            &scales.iter().map(|v| *v as i8).collect::<Vec<_>>(),
            &[248320, 128],
        )?,
        Some(MxArray::from_float16(
            &bases
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| u16::from_le_bytes(*v))
                .collect::<Vec<_>>(),
            &[248320, 8],
        )?),
        None,
        16,
        6,
        "q6k@t64".into(),
    )))
}

fn input(case: usize) -> Result<MxArray> {
    let bytes = std::fs::read(root().join(format!("case{case}/x.bf16"))).unwrap();
    MxArray::from_bfloat16(
        &bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_le_bytes(*v))
            .collect::<Vec<_>>(),
        &[8, 2048],
    )
}

fn expected(case: usize) -> Vec<u32> {
    std::fs::read(root().join(format!("case{case}/source.f32")))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| u32::from_le_bytes(*v))
        .collect()
}

fn bits(array: &MxArray) -> Result<Vec<u32>> {
    assert_eq!(array.dtype()?, DType::Float32);
    Ok(array
        .to_float32()?
        .iter()
        .map(|value| value.to_bits())
        .collect())
}

#[test]
#[ignore = "requires source-linked Metal build and original source-head fixture"]
fn source_head_loaded_policy_shared_projection_and_single_trace() -> Result<()> {
    assert!(unsafe { mlx_sys::mlx_source_head_q6_available() });
    let mut projection = fixture_projection()?;
    let LinearProj::Quantized(q) = &projection else {
        unreachable!()
    };
    assert!(!q.source_head);
    assert!(q.try_source_head(&input(0)?)?.is_none());
    projection.enable_source_head()?;
    // The target and draft call this same projection. Both leading-rank
    // contracts must preserve source FP32 outputs and remain replayable.
    for rank in [2, 3] {
        let key = 0x5150_5348_4541_4400 + rank;
        crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
        let mut traces = 0;
        let mut build = |args: &[MxArray]| -> Result<Vec<MxArray>> {
            traces += 1;
            Ok(vec![projection.forward_logits(&args[0])?])
        };
        for case in [0, 2, 0] {
            let x = input(case)?;
            let x = if rank == 3 {
                x.reshape(&[1, 8, 2048])?
            } else {
                x
            };
            let before = x.to_uint16_native()?;
            let eager = projection.forward_logits(&x)?;
            assert_eq!(bits(&eager)?, expected(case));
            assert_eq!(bits(&projection.forward(&x)?)?, expected(case));
            let compiled =
                crate::compiled_graph::invoke_compiled_graph(key, &[&x], 1, true, &mut build)?
                    .unwrap();
            assert_eq!(bits(&compiled[0])?, bits(&eager)?);
            assert_eq!(x.to_uint16_native()?, before);
            assert_eq!(eager.shape()?.len(), rank as usize);
        }
        assert_eq!(traces, 1);
        crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
    }
    // Native tails must stay identical under the immutable policy. In
    // particular, no rounded source output is fed back into another forward.
    let x = input(0)?;
    for rows in [1, 2, 7] {
        let tail = x.slice_axis(0, 0, rows)?;
        let LinearProj::Quantized(q) = &projection else {
            unreachable!()
        };
        assert!(q.try_source_head(&tail)?.is_none());
        assert_eq!(
            projection.forward_logits(&tail)?.to_uint16_native()?,
            q.forward(&tail)?.to_uint16_native()?
        );
    }
    let LinearProj::Quantized(q) = &projection else {
        unreachable!()
    };
    assert!(q.source_head);
    assert!(q.concat_rows(q)?.is_none());
    assert!(q.slice_rows(0, 64).is_err());
    Ok(())
}

#[test]
#[ignore = "requires source-linked Metal build and original source-head fixture"]
fn source_head_weight_contract_failure_does_not_change_loaded_policy() -> Result<()> {
    assert!(unsafe { mlx_sys::mlx_source_head_q6_available() });
    let mut projection = fixture_projection()?;
    projection.enable_source_head()?;
    let x = input(0)?;
    let LinearProj::Quantized(q) = &mut projection else {
        unreachable!()
    };
    let original = q.scales.clone();
    q.scales = original.astype(DType::Uint8)?;
    assert!(q.validate_source_head().is_err());
    assert!(q.try_source_head(&x).is_err());
    assert!(q.source_head);
    q.scales = original;
    let original = q.biases.take().unwrap();
    assert!(q.validate_source_head().is_err());
    assert!(q.try_source_head(&x).is_err());
    q.biases = Some(original);
    q.input_amax = Some(1.0);
    assert!(q.validate_source_head().is_err());
    assert!(q.try_source_head(&x).is_err());
    q.input_amax = None;
    assert_eq!(bits(&q.try_source_head(&x)?.unwrap())?, expected(0));
    let small = MxArray::zeros(&[16, 8], Some(DType::BFloat16))?;
    let mut standard = LinearProj::Standard(Linear::from_weights(&small, None)?);
    assert!(standard.enable_source_head().is_err());
    Ok(())
}
