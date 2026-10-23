// Included in normalization.rs::tests; fixture gates run in an exclusive process.
fn exact_source_raw(
    input: &MxArray,
    weight: &MxArray,
    residual: Option<&MxArray>,
    staged: bool,
) -> Result<Option<Vec<MxArray>>> {
    let mut normalized = std::ptr::null_mut();
    let mut hidden = std::ptr::null_mut();
    let status = unsafe {
        sys::mlx_source_norm_2048(
            input.as_raw_ptr(),
            weight.as_raw_ptr(),
            residual.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            staged,
            &mut normalized,
            if residual.is_some() {
                &mut hidden
            } else {
                std::ptr::null_mut()
            },
        )
    };
    match status {
        0 => {
            assert!(normalized.is_null() && hidden.is_null());
            Ok(None)
        }
        1 => {
            let normalized = MxArray::from_handle(normalized, "exact source norm test")?;
            let mut outputs = vec![normalized];
            if residual.is_some() {
                outputs.push(MxArray::from_handle(hidden, "exact source hidden test")?);
            }
            Ok(Some(outputs))
        }
        _ => {
            assert!(normalized.is_null() && hidden.is_null());
            Err(Error::from_reason("exact source norm factory failed"))
        }
    }
}
fn exact_source_bf_view(values: &[u16], rows: usize, offset: usize, strided: bool) -> MxArray {
    let width = if strided { 2051 } else { 2048 };
    let mut storage = vec![f32::NAN; offset + rows * width + 1];
    for row in 0..rows {
        for col in 0..2048 {
            storage[offset + row * width + col] =
                half::bf16::from_bits(values[row * 2048 + col]).to_f32();
        }
    }
    let base = MxArray::from_float32(&storage, &[storage.len() as i64])
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
    base.eval(); // Preserve an actual element-offset view rather than a lazy slice-copy.
    let view = base
        .slice_axis(0, offset as i64, (offset + rows * width) as i64)
        .unwrap()
        .reshape(&[rows as i64, width as i64])
        .unwrap();
    if strided {
        view.slice_axis(1, 0, 2048).unwrap()
    } else {
        view
    }
}
fn exact_source_weight(values: &[f32], offset: bool) -> MxArray {
    if !offset {
        return MxArray::from_float32(values, &[2048]).unwrap();
    }
    let mut guarded = vec![f32::NAN];
    guarded.extend_from_slice(values);
    guarded.push(f32::NAN);
    let base = MxArray::from_float32(&guarded, &[2050]).unwrap();
    base.eval();
    base.slice_axis(0, 1, 2049).unwrap()
}
fn exact_source_fixture_bytes(
    root: &std::path::Path,
    manifest: &serde_json::Value,
    name: &str,
) -> Vec<u8> {
    use sha2::Digest;
    let bytes = std::fs::read(root.join(name)).unwrap();
    let actual: String = sha2::Sha256::digest(&bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect();
    assert_eq!(
        manifest["files"][name].as_str(),
        Some(actual.as_str()),
        "fixture identity {name}"
    );
    bytes
}
fn exact_source_words(bytes: &[u8]) -> Vec<u16> {
    assert_eq!(bytes.len() % 2, 0);
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|x| u16::from_le_bytes(*x))
        .collect()
}

#[test]
#[ignore = "exclusive M5 GPU, independent immutable Splash raw norm fixtures via SOURCE_NORM_ORACLE_DIR"]
fn source_norm_splash_jit_matches_independent_source_bits() {
    use sha2::Digest;
    let root = std::path::PathBuf::from(
        std::env::var_os("SOURCE_NORM_ORACLE_DIR").expect("set SOURCE_NORM_ORACLE_DIR"),
    );
    let bytes = std::fs::read(root.join("manifest.json")).unwrap();
    let digest: String = sha2::Sha256::digest(&bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect();
    assert_eq!(
        digest,
        "6e83478bb5f3c91c6811bf668589f55779bfc1f49993dafae4c5d087f4be1c0f"
    );
    let manifest: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        manifest["sourceRevision"],
        "1482f6bbc2abf12375332725302019594150b6cb"
    );
    let w = exact_source_fixture_bytes(&root, &manifest, "weight.f32");
    assert_eq!(w.len(), 8192);
    let weights: Vec<f32> = w
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 26);
    let mut compared = 0usize;
    let keys = [
        0x5150_4e4f_524d_2000,
        0x5150_4e4f_524d_2001,
        0x5150_4e4f_524d_2002,
        0x5150_4e4f_524d_2003,
    ];
    for key in keys {
        crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
    }
    let mut traces = [0usize; 4];
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let rows = case["rows"].as_u64().unwrap() as usize;
        let offset = case["offsetBytes"].as_u64().unwrap() as usize / 2;
        let add = case["residual"].as_bool().unwrap();
        let x = exact_source_words(&exact_source_fixture_bytes(
            &root,
            &manifest,
            &format!("{id}-input.bf16"),
        ));
        let r = exact_source_words(&exact_source_fixture_bytes(
            &root,
            &manifest,
            &format!("{id}-residual.bf16"),
        ));
        let expected = exact_source_words(&exact_source_fixture_bytes(
            &root,
            &manifest,
            &format!("{id}-expected.bf16"),
        ));
        let hidden = exact_source_words(&exact_source_fixture_bytes(
            &root,
            &manifest,
            &format!("{id}-hidden.bf16"),
        ));
        assert_eq!(x.len(), rows * 2048);
        assert_eq!(r.len(), x.len());
        assert_eq!(expected.len(), x.len());
        for strided in [false, true] {
            let input = exact_source_bf_view(&x, rows, offset, strided);
            let residual = exact_source_bf_view(&r, rows, offset, strided);
            let weight = exact_source_weight(&weights, offset != 0);
            for staged in [false, true] {
                let outputs = exact_source_raw(&input, &weight, add.then_some(&residual), staged)
                    .unwrap()
                    .expect("source-capable M5 required");
                assert_eq!(
                    outputs[0].to_uint16_native().unwrap(),
                    expected,
                    "{id} strided={strided} staged={staged}"
                );
                if add {
                    assert_eq!(
                        outputs[1].to_uint16_native().unwrap(),
                        hidden,
                        "raw hidden {id}"
                    );
                    compared += hidden.len();
                }
                compared += expected.len();
                let index = usize::from(add) * 2 + usize::from(staged);
                let args = if add {
                    vec![&input, &weight, &residual]
                } else {
                    vec![&input, &weight]
                };
                let mut build = |a: &[MxArray]| -> Result<Vec<MxArray>> {
                    traces[index] += 1;
                    exact_source_raw(&a[0], &a[1], if add { Some(&a[2]) } else { None }, staged)?
                        .ok_or_else(|| Error::from_reason("source fixture replay declined"))
                };
                let replay = crate::compiled_graph::invoke_compiled_graph(
                    keys[index],
                    &args,
                    if add { 2 } else { 1 },
                    true,
                    &mut build,
                )
                .unwrap()
                .unwrap();
                assert_eq!(
                    replay[0].to_uint16_native().unwrap(),
                    expected,
                    "compiled source fixture {id}"
                );
                if add {
                    assert_eq!(
                        replay[1].to_uint16_native().unwrap(),
                        hidden,
                        "compiled hidden fixture {id}"
                    );
                }
                compared += expected.len() + if add { hidden.len() } else { 0 };
            }
            let mut norm = RMSNorm::from_weight(&weight, Some(1e-6)).unwrap();
            norm.enable_source_f32_splash().unwrap();
            if add {
                let (h, y) = norm
                    .forward_residual_add(&input, &residual)
                    .expect("production fused source path");
                assert_eq!(h.to_uint16_native().unwrap(), hidden);
                assert_eq!(y.to_uint16_native().unwrap(), expected);
                // Separate BF16 Add followed by the same source norm must match.
                assert_eq!(
                    norm.forward(&input.add(&residual).unwrap())
                        .unwrap()
                        .to_uint16_native()
                        .unwrap(),
                    expected
                );
            } else {
                assert_eq!(
                    norm.forward(&input).unwrap().to_uint16_native().unwrap(),
                    expected
                );
            }
            assert_eq!(input.to_uint16_native().unwrap(), x);
            assert_eq!(residual.to_uint16_native().unwrap(), r);
        }
    }
    assert_eq!(
        traces, [1; 4],
        "one trace per fixed output/stage policy across every shape/layout"
    );
    for key in keys {
        crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
    }
    eprintln!(
        "SOURCE_NORM_JIT fixtures={} bf16_comparisons={compared}",
        cases.len()
    );
}

#[test]
fn source_norm_splash_replays_rows_offsets_weights_and_residual_meaning() {
    if !unsafe { sys::mlx_metal_is_available() } {
        return;
    }
    let (mut probe, input, _) = source_norm_fixture(8, 101, false);
    probe.enable_source_f32_splash().unwrap();
    if exact_source_raw(&input, &probe.get_weight(), None, true)
        .unwrap()
        .is_none()
    {
        return;
    }
    for add in [false, true] {
        let key = 0x5150_4e4f_524d_1000 + u64::from(add);
        crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
        let mut traces = 0;
        for (rows, seed, strided) in [
            (8, 103, false),
            (1, 107, true),
            (64, 109, false),
            (65, 113, true),
            (7, 127, true),
            (33, 131, false),
            (8, 137, true),
        ] {
            let (_, input, weights) = source_norm_fixture(rows, seed, strided);
            let weight = exact_source_weight(&weights, strided);
            let residual = input
                .mul_scalar(-0.0625)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let inputs = if add {
                vec![&input, &weight, &residual]
            } else {
                vec![&input, &weight]
            };
            let mut build = |a: &[MxArray]| -> Result<Vec<MxArray>> {
                traces += 1;
                exact_source_raw(&a[0], &a[1], if add { Some(&a[2]) } else { None }, true)?
                    .ok_or_else(|| Error::from_reason("compiled source norm unexpectedly declined"))
            };
            let outputs = crate::compiled_graph::invoke_compiled_graph(
                key,
                &inputs,
                if add { 2 } else { 1 },
                true,
                &mut build,
            )
            .unwrap()
            .unwrap();
            let eager = exact_source_raw(&input, &weight, add.then_some(&residual), true)
                .unwrap()
                .unwrap();
            assert_eq!(
                outputs[0].to_uint16_native().unwrap(),
                eager[0].to_uint16_native().unwrap()
            );
            let h = if add {
                assert_eq!(
                    outputs[1].to_uint16_native().unwrap(),
                    input.add(&residual).unwrap().to_uint16_native().unwrap()
                );
                &outputs[1]
            } else {
                &input
            };
            source_norm_check_f64(h, &weights, &outputs[0]);
        }
        assert_eq!(
            traces, 1,
            "one trace must cover stage/stream row transition"
        );
        crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, key);
    }
}

#[test]
fn source_norm_splash_policy_epsilon_dtype_and_output_slots_fail_closed() {
    let (mut norm, input, weights) = source_norm_fixture(8, 149, false);
    norm.enable_source_f32_splash().unwrap();
    assert!(
        norm.set_weight(&MxArray::from_float32(&weights, &[2048]).unwrap())
            .is_err()
    );
    assert!(
        norm.forward(&input.astype(DType::Float32).unwrap())
            .is_err()
    );
    let mut wrong_eps = RMSNorm::from_weight(&norm.get_weight(), Some(1e-5)).unwrap();
    assert!(wrong_eps.enable_source_f32_splash().is_err());
    assert!(matches!(
        wrong_eps.output_policy,
        RmsNormOutputPolicy::Native
    ));
    let promoted = f64::from(1e-6_f32);
    assert_ne!(promoted.to_bits(), 1e-6_f64.to_bits());
    let mut imported = RMSNorm::from_weight(&norm.get_weight(), Some(promoted)).unwrap();
    imported.enable_source_f32_splash().unwrap();
    for bits in [1e-6_f32.to_bits() - 1, 1e-6_f32.to_bits() + 1] {
        let mut different =
            RMSNorm::from_weight(&norm.get_weight(), Some(f64::from(f32::from_bits(bits))))
                .unwrap();
        assert!(different.enable_source_f32_splash().is_err());
        assert!(matches!(
            different.output_policy,
            RmsNormOutputPolicy::Native
        ));
    }
    let wrong_weight = MxArray::zeros(&[64], Some(DType::Float32)).unwrap();
    let mut wrong_width = RMSNorm::from_weight(&wrong_weight, Some(1e-6)).unwrap();
    assert!(wrong_width.enable_source_f32_splash().is_err());
    let mut shared_slot = std::ptr::dangling_mut::<sys::mlx_array>();
    let slot = &mut shared_slot as *mut *mut sys::mlx_array;
    assert_eq!(
        unsafe {
            sys::mlx_source_norm_2048(
                input.as_raw_ptr(),
                norm.get_weight().as_raw_ptr(),
                std::ptr::null_mut(),
                true,
                slot,
                slot,
            )
        },
        -1
    );
    assert!(shared_slot.is_null());
    assert!(matches!(
        norm.clone().output_policy,
        RmsNormOutputPolicy::SourceF32SplashBf16
    ));
}

#[test]
#[ignore = "exclusive serial process: temporarily selects the CPU default device"]
fn source_norm_splash_cpu_uses_reference_without_changing_output_meaning() {
    struct Restore(i32);
    impl Drop for Restore {
        fn drop(&mut self) {
            unsafe { sys::mlx_set_default_device(self.0) }
        }
    }
    let _restore = Restore(unsafe { sys::mlx_default_device() });
    unsafe { sys::mlx_set_default_device(0) };
    let (mut norm, input, weights) = source_norm_fixture(7, 151, true);
    norm.enable_source_f32_splash().unwrap();
    let residual = input
        .mul_scalar(0.125)
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
    assert!(
        exact_source_raw(&input, &norm.get_weight(), None, true)
            .unwrap()
            .is_none()
    );
    assert!(norm.forward_residual_add(&input, &residual).is_none());
    let (h, y) = crate::models::qwen3_5::decoder_layer::DecoderLayer::add_residual_norm(
        &norm, &input, &residual,
    )
    .unwrap();
    assert_eq!(
        h.to_uint16_native().unwrap(),
        input.add(&residual).unwrap().to_uint16_native().unwrap()
    );
    source_norm_check_f64(&h, &weights, &y);
}
