// Included in normalization.rs's test module so the diagnostic setter remains internal.
fn source_norm_fixture(rows: usize, seed: usize, strided: bool) -> (RMSNorm, MxArray, Vec<f32>) {
    let weights: Vec<f32> = (0..2048)
        .map(|i| 1.0 + ((i * 13 + seed) % 31) as f32 * 0.00019)
        .collect();
    let mut norm = RMSNorm::from_weight(
        &MxArray::from_float32(&weights, &[2048]).unwrap(),
        Some(1e-6),
    )
    .unwrap();
    norm.enable_source_f32_reference().unwrap();
    let width = if strided { 4096 } else { 2048 };
    let data: Vec<f32> = (0..rows * width)
        .map(|i| (((i * 43 + seed * 97) % 509) as f32 - 254.0) * 0.013)
        .collect();
    let input = MxArray::from_float32(&data, &[rows as i64, width as i64])
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
    let input = if strided {
        input.slice_axis(1, 1, 2049).unwrap()
    } else {
        input
    };
    (norm, input, weights)
}
fn source_norm_check_f64(input: &MxArray, weights: &[f32], output: &MxArray) {
    assert_eq!(output.dtype().unwrap(), DType::BFloat16);
    assert_eq!(
        input.shape().unwrap().as_ref(),
        output.shape().unwrap().as_ref()
    );
    let input = input.to_float32().unwrap();
    let got = output.to_float32().unwrap();
    for (x, y) in input.chunks_exact(2048).zip(got.chunks_exact(2048)) {
        let inverse = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 2048.0 + 1e-6)
            .sqrt()
            .recip();
        for (i, (&x, &y)) in x.iter().zip(y).enumerate() {
            let exact = x as f64 * inverse * weights[i] as f64;
            // BF16 output has at most one half-ULP of final rounding plus F32
            // reduction error. This is independent of the implementation graph.
            assert!(
                (y as f64 - exact).abs() <= exact.abs() * 0.0041 + 2e-6,
                "column{i}: got{y} exact{exact}"
            );
        }
    }
}
#[test]
fn source_norm_reference_preserves_bf16_residual_and_matches_f64() {
    for (rows, seed, strided) in [(1, 3, false), (7, 11, true), (8, 29, false), (33, 43, true)] {
        let (norm, input, weights) = source_norm_fixture(rows, seed, strided);
        source_norm_check_f64(&input, &weights, &norm.forward(&input).unwrap());
        let residual = input
            .mul_scalar(-0.03125)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        assert!(norm.forward_residual_add(&input, &residual).is_none());
        let (h, normalized) =
            crate::models::qwen3_5::decoder_layer::DecoderLayer::add_residual_norm(
                &norm, &input, &residual,
            )
            .unwrap();
        assert_eq!(h.dtype().unwrap(), DType::BFloat16);
        let expected: Vec<_> = input
            .to_float32()
            .unwrap()
            .iter()
            .zip(residual.to_float32().unwrap().iter())
            .map(|(&a, &b)| half::bf16::from_f32(a + b).to_bits())
            .collect();
        assert_eq!(h.to_uint16_native().unwrap(), expected);
        source_norm_check_f64(&h, &weights, &normalized);
        assert_eq!(
            norm.clone()
                .forward(&input)
                .unwrap()
                .to_uint16_native()
                .unwrap(),
            norm.forward(&input).unwrap().to_uint16_native().unwrap()
        );
    }
}
#[test]
fn source_norm_policy_does_not_promote_other_norms_or_allow_mutation() {
    let (mut norm, input, weights) = source_norm_fixture(8, 59, false);
    assert!(
        norm.set_weight(&MxArray::from_float32(&weights, &[2048]).unwrap())
            .is_err()
    );
    assert!(
        norm.forward(&input.astype(DType::Float32).unwrap())
            .is_err()
    );
    let weight = MxArray::from_float32(&weights, &[2048])
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
    let mut native = RMSNorm::from_weight(&weight, Some(1e-6)).unwrap();
    assert!(native.enable_source_f32_reference().is_err());
    let raw = unsafe { mlx_sys::mlx_fast_rms_norm(input.as_raw_ptr(), weight.as_raw_ptr(), 1e-6) };
    let baseline = MxArray::from_handle(raw, "native source-norm control").unwrap();
    assert_eq!(
        native.forward(&input).unwrap().to_uint16_native().unwrap(),
        baseline.to_uint16_native().unwrap()
    );
}
#[test]
fn source_norm_reference_replays_changing_rows_and_offsets() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let (norm, _, weights) = source_norm_fixture(8, 73, false);
    let key = 0x5150_4e4f_524d_0001;
    crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
    let mut traces = 0;
    for (rows, seed, strided) in [
        (8, 73, false),
        (7, 79, true),
        (33, 83, false),
        (1, 89, true),
        (8, 97, false),
    ] {
        let (_, input, _) = source_norm_fixture(rows, seed, strided);
        let mut build = |inputs: &[MxArray]| -> Result<Vec<MxArray>> {
            traces += 1;
            Ok(vec![norm.forward(&inputs[0])?])
        };
        let result =
            crate::compiled_graph::invoke_compiled_graph(key, &[&input], 1, true, &mut build)
                .unwrap()
                .unwrap();
        source_norm_check_f64(&input, &weights, &result[0]);
        assert_eq!(
            result[0].to_uint16_native().unwrap(),
            norm.forward(&input).unwrap().to_uint16_native().unwrap()
        );
    }
    assert_eq!(traces, 1);
    crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
}
