use super::*;
use crate::array::DType;
fn bf(values: &[f32], shape: &[i64]) -> MxArray {
    MxArray::from_float32(values, shape)
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap()
}
fn bits(x: &MxArray) -> Vec<u16> {
    x.to_uint16_native().unwrap()
}

#[test]
fn source_residual_hidden_and_pending_preserve_raw_taps_and_one_norm() {
    let norm = RMSNorm::from_weight(&bf(&[1.125; 64], &[64]), Some(1e-6)).unwrap();
    let h = bf(
        &(0..512)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.125)
            .collect::<Vec<_>>(),
        &[1, 8, 64],
    );
    let delta = bf(&[0.03125; 512], &[1, 8, 64]);
    let want = h.add(&delta).unwrap();
    for output in [
        VerifyLayerOutput::Hidden(want.clone()),
        VerifyLayerOutput::Pending {
            h: h.clone(),
            delta: delta.clone(),
        },
    ] {
        let (tap, normalized) = output.normalize(&norm).unwrap();
        assert_eq!(bits(&tap), bits(&want));
        assert_eq!(bits(&normalized), bits(&norm.forward(&want).unwrap()));
        match output.as_input() {
            VerifyResidual::Hidden(x) => assert_eq!(bits(x), bits(&want)),
            VerifyResidual::Pending { h, delta } => {
                assert_eq!(bits(&h.add(delta).unwrap()), bits(&want))
            }
        }
    }
}

#[test]
fn source_residual_hidden_tap_and_final_norm_replay_changing_inputs() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let norm = RMSNorm::from_weight(&bf(&[0.875; 64], &[64]), Some(1e-6)).unwrap();
    norm.get_weight().eval();
    let key = 0x5150_5352_4553_0001;
    crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
    let mut traces = 0;
    for (rows, seed) in [(8, 3), (7, 11), (33, 23), (1, 41), (8, 59)] {
        let input = bf(
            &(0..rows * 64)
                .map(|i| ((i * 7 + seed) % 61) as f32 * 0.125 - 3.0)
                .collect::<Vec<_>>(),
            &[1, rows as i64, 64],
        );
        let mut build = |inputs: &[MxArray]| -> Result<Vec<MxArray>> {
            traces += 1;
            let output = VerifyLayerOutput::Hidden(inputs[0].clone());
            let (tap, normalized) = output.normalize(&norm)?;
            Ok(vec![tap, normalized])
        };
        let got = crate::compiled_graph::invoke_compiled_graph(key, &[&input], 2, true, &mut build)
            .unwrap()
            .unwrap();
        assert_eq!(bits(&got[0]), bits(&input));
        assert_eq!(bits(&got[1]), bits(&norm.forward(&input).unwrap()));
    }
    assert_eq!(traces, 1);
    crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
}

fn combine_fixture(rows: usize, live: usize, seed: usize) -> Vec<MxArray> {
    let mut values = vec![0.0; rows * 9 * 2048];
    for row in 0..rows {
        for col in 0..2048 {
            values[(row * 9) * 2048 + col] = 128.0;
            values[(row * 9 + 1) * 2048 + col] = 0.25;
            values[(row * 9 + 8) * 2048 + col] = (seed % 3) as f32 * 0.125;
        }
    }
    vec![
        bf(&values, &[(rows * 9) as i64, 2048]),
        MxArray::from_uint32(
            &(0..rows * 9).map(|i| i as u32).collect::<Vec<_>>(),
            &[rows as i64, 9],
        )
        .unwrap(),
        MxArray::from_float32(&vec![1.0; rows * 9], &[rows as i64, 9]).unwrap(),
        bf(&vec![-128.0; rows * 2048], &[rows as i64, 2048]),
        MxArray::from_int32(&[live as i32], &[]).unwrap(),
    ]
}
fn combine(a: &[MxArray]) -> Result<MxArray> {
    let p = unsafe {
        mlx_sys::mlx_test_splash_moe_source_residual_combine(
            a[0].as_raw_ptr(),
            a[1].as_raw_ptr(),
            a[2].as_raw_ptr(),
            a[3].as_raw_ptr(),
            a[4].as_raw_ptr(),
        )
    };
    MxArray::from_handle(p, "source residual production combine probe")
}
#[test]
fn source_residual_combine_replay_keeps_hidden_meaning_and_finite_dead_rows() {
    if !unsafe { mlx_sys::mlx_splash_moe_available() } {
        return;
    }
    let key = 0x5150_5352_4553_0002;
    crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
    let mut traces = 0;
    for (rows, live, seed) in [(8, 8, 0), (1, 1, 1), (7, 3, 2), (33, 17, 0), (8, 3, 1)] {
        let a = combine_fixture(rows, live, seed);
        let input_bits = bits(&a[3]);
        let mut build = |a: &[MxArray]| -> Result<Vec<MxArray>> {
            traces += 1;
            Ok(vec![combine(a)?])
        };
        let args = a.iter().collect::<Vec<_>>();
        let result = crate::compiled_graph::invoke_compiled_graph(key, &args, 1, true, &mut build)
            .unwrap()
            .unwrap();
        assert_eq!(bits(&result[0]), bits(&combine(&a).unwrap()));
        let source = 0.25 + (seed % 3) as f32 * 0.125;
        let native = half::bf16::from_f32(128.0 + (seed % 3) as f32 * 0.125).to_f32() - 128.0;
        let values = result[0].to_float32().unwrap();
        for (row, got) in values.as_chunks::<2048>().0.iter().enumerate() {
            let expected = if row >= live {
                0.0
            } else if rows == 8 {
                source
            } else {
                native
            };
            assert!(got.iter().all(|&v| v == expected), "rows={rows} row={row}");
        }
        assert_eq!(bits(&a[3]), input_bits, "residual must remain read-only");
    }
    assert_eq!(traces, 1);
    crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
}

#[test]
fn source_residual_invalid_replay_fails_without_mutating_residual() {
    if !unsafe { mlx_sys::mlx_splash_moe_available() } {
        return;
    }
    let key = 0x5150_5352_4553_0003;
    crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
    let mut traces = 0;
    let mut build = |a: &[MxArray]| -> Result<Vec<MxArray>> {
        traces += 1;
        Ok(vec![combine(a)?])
    };
    let mut a = combine_fixture(8, 8, 0);
    let before = bits(&a[3]);
    let first = crate::compiled_graph::invoke_compiled_graph(
        key,
        &a.iter().collect::<Vec<_>>(),
        1,
        true,
        &mut build,
    )
    .unwrap()
    .unwrap();
    assert_eq!(first[0].to_float32().unwrap()[0], 0.25);
    a[1] = MxArray::zeros(&[8, 8], Some(DType::Uint32)).unwrap();
    let out = crate::compiled_graph::invoke_compiled_graph(
        key,
        &a.iter().collect::<Vec<_>>(),
        1,
        true,
        &mut build,
    )
    .unwrap()
    .unwrap();
    assert!(MxArray::eval_arrays(&[&out[0]]).is_err());
    assert_eq!(bits(&a[3]), before);
    a = combine_fixture(8, 8, 0);
    let parent = MxArray::zeros(&[8, 2049], Some(DType::BFloat16)).unwrap();
    parent.eval();
    a[3] = parent.slice_axis(1, 0, 2048).unwrap();
    let strided = crate::compiled_graph::invoke_compiled_graph(
        key,
        &a.iter().collect::<Vec<_>>(),
        1,
        true,
        &mut build,
    )
    .unwrap()
    .unwrap();
    assert!(
        MxArray::eval_arrays(&[&strided[0]]).is_err(),
        "test probe must decline noncontiguous oracle views"
    );
    let valid = combine_fixture(8, 8, 0);
    let recovered = crate::compiled_graph::invoke_compiled_graph(
        key,
        &valid.iter().collect::<Vec<_>>(),
        1,
        true,
        &mut build,
    )
    .unwrap()
    .unwrap();
    assert_eq!(recovered[0].to_float32().unwrap()[0], 0.25);
    assert_eq!(traces, 1);
    crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
}

#[test]
fn source_residual_lazy_output_retains_read_only_alias_operands() {
    if !unsafe { mlx_sys::mlx_splash_moe_available() } {
        return;
    }
    let mut a = combine_fixture(8, 8, 0);
    a[3] = a[0].slice_axis(0, 0, 8).unwrap();
    let original = bits(&a[0]);
    let expected = a[3]
        .to_float32()
        .unwrap()
        .iter()
        .map(|&r| half::bf16::from_f32((r + 128.0) + 0.25).to_bits())
        .collect::<Vec<_>>();
    let output = combine(&a).unwrap();
    assert_eq!(bits(&output), expected);
    assert_eq!(bits(&a[0]), original);
    let retained = combine(&a).unwrap();
    drop(a);
    assert_eq!(bits(&retained), expected);
}

#[test]
fn source_residual_profile_requires_explicit_measured_geometry() {
    let config: super::super::config::Qwen3_5MoeConfig =
        serde_json::from_value(serde_json::json!({
            "vocab_size":1024,"hidden_size":2048,"num_layers":40,"num_heads":16,"num_kv_heads":2,
            "intermediate_size":512,"rms_norm_eps":1e-6,"head_dim":256,"tie_word_embeddings":false,
            "max_position_embeddings":32768,"pad_token_id":0,"eos_token_id":2,"bos_token_id":1,
            "num_experts":256,"num_experts_per_tok":8
        }))
        .unwrap();
    use super::super::sparse_moe::source_residual_profile as eligible;
    assert!(eligible(&config));
    for field in 0..6 {
        let mut changed = config.clone();
        match field {
            0 => changed.hidden_size = 1024,
            1 => changed.num_layers = 39,
            2 => changed.num_experts = 255,
            3 => changed.num_experts_per_tok = 7,
            4 => changed.moe_intermediate_size = Some(1024),
            _ => changed.shared_expert_intermediate_size = Some(1024),
        }
        assert!(!eligible(&changed));
    }
}
