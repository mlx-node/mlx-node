use super::*;

#[test]
fn empty_resident_weight_concatenation_returns_an_error() {
    assert!(Weight::concatenate_rows(&[]).is_err());
}

#[test]
fn affine_direct_experts_preserve_promoted_arithmetic_and_tail_rows() {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }
    for bits in [5, 8] {
        for (k, n, rows) in [(32, 17, 1), (640, 2560, 10), (672, 19, 80)] {
            let experts = 3;
            let codes = (0..experts * n * k * bits / 32)
                .map(|i| (i as u32).wrapping_mul(2654435761))
                .collect::<Vec<_>>();
            let sides = (0..experts * n * k / 32)
                .map(|i| ((i % 29) as f32 + 1.) / 1024.)
                .collect::<Vec<_>>();
            let scales = MxArray::from_float32(&sides, &[experts * n, k / 32])
                .unwrap()
                .astype(DType::Float16)
                .unwrap();
            let biases = scales.mul_scalar(-7.).unwrap();
            let bank = Weight {
                dense_bf16: None,
                values: MxArray::from_uint32(&codes, &[experts * n, k * bits / 32]).unwrap(),
                scales: Some(scales),
                biases: Some(biases),
                group: 32,
                bits: bits as i32,
                mode: "affine".into(),
                q8_symmetric_128: false,
                qlinear: None,
            };
            let ids = MxArray::from_uint32(
                &(0..rows).map(|i| (i % experts) as u32).collect::<Vec<_>>(),
                &[rows],
            )
            .unwrap();
            let raw = (0..rows * k)
                .map(|i| (i as f32 * 0.13).sin())
                .collect::<Vec<_>>();
            for dtype in [DType::BFloat16, DType::Float16, DType::Float32] {
                let x = MxArray::from_float32(&raw, &[rows, 1, k])
                    .unwrap()
                    .astype(dtype)
                    .unwrap();
                let want = bank
                    .gather_rows(&x, &ids, experts as usize, false)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let got = bank
                    .expert_rows(&x, &ids, experts as usize, false)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let error = got
                    .iter()
                    .zip(want.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert_eq!(
                    error, 0.,
                    "bits={bits} K={k} N={n} rows={rows} dtype={dtype:?}"
                );
            }
        }
    }
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp")
}

#[test]
fn dense_compute_cache_preserves_source_dtypes_and_row_views() {
    let values = MxArray::from_float32(
        &(0..19 * 32)
            .map(|i| (i as f32 * 0.131).sin() * 0.27)
            .collect::<Vec<_>>(),
        &[19, 32],
    )
    .unwrap();
    let original = Weight {
        values,
        dense_bf16: None,
        scales: None,
        biases: None,
        group: 0,
        bits: 0,
        mode: String::new(),
        q8_symmetric_128: false,
        qlinear: None,
    };
    let mut cached = original.clone();
    cached.dense_bf16 = Some(cached.values.astype(DType::BFloat16).unwrap());
    MxArray::eval_arrays_with_context(
        &[cached.dense_bf16.as_ref().unwrap()],
        "test cached projection",
    )
    .unwrap();
    assert_eq!(cached.bytes().unwrap(), 19 * 32 * 6);
    assert_eq!(
        cached.dense().unwrap().to_float32().unwrap().to_vec(),
        original.values.to_float32().unwrap().to_vec()
    );
    let parts = [
        Arc::new(cached.slice_rows(0, 7).unwrap()),
        Arc::new(cached.slice_rows(7, 12).unwrap()),
    ];
    let joined = Weight::concatenate_rows(&parts).unwrap();
    for dtype in [DType::BFloat16, DType::Float16, DType::Float32] {
        for rows in [1, 8] {
            let x = MxArray::from_float32(
                &(0..rows * 32)
                    .map(|i| (i as f32 * 0.19).cos())
                    .collect::<Vec<_>>(),
                &[1, rows, 32],
            )
            .unwrap()
            .astype(dtype)
            .unwrap();
            let want = original
                .linear(&x)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec();
            for bank in [&cached, &joined] {
                let got = bank.linear(&x).unwrap();
                assert_eq!(got.dtype().unwrap(), dtype);
                assert_eq!(
                    got.astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .to_vec(),
                    want
                );
            }
        }
    }
    // Mixed cached/uncached concatenations must never inherit a stale
    // derived representation of only the first matrix.
    assert!(
        Weight::concatenate_rows(&[Arc::new(cached), Arc::new(original)])
            .unwrap()
            .dense_bf16
            .is_none()
    );
}

#[test]
fn dense_gate_pair_preserves_projection_rows_and_cache_bytes() {
    for phase in (0..16).map(|i| 0.3 + i as f32 * 0.7) {
        let make = |shift: f32| {
            let values = MxArray::from_float32(
                &(0..48 * 2560)
                    .map(|i| (i as f32 * 0.131 + shift).sin() * 0.27)
                    .collect::<Vec<_>>(),
                &[48, 2560],
            )
            .unwrap();
            Weight {
                dense_bf16: Some(values.astype(DType::BFloat16).unwrap()),
                values,
                scales: None,
                biases: None,
                group: 0,
                bits: 0,
                mode: String::new(),
                q8_symmetric_128: false,
                qlinear: None,
            }
        };
        let a = Arc::new(make(phase));
        let b = Arc::new(make(phase + 0.7));
        let joined = Weight::concatenate_rows(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(
            joined.bytes().unwrap(),
            a.bytes().unwrap() + b.bytes().unwrap()
        );
        for rows in [1] {
            let x = MxArray::from_float32(
                &(0..rows * 2560)
                    .map(|i| (i as f32 * 0.017 + phase).cos())
                    .collect::<Vec<_>>(),
                &[1, rows, 2560],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let want = MxArray::concatenate_many(
                vec![&a.linear(&x).unwrap(), &b.linear(&x).unwrap()],
                Some(-1),
            )
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
            let got = joined.linear(&x).unwrap();
            assert_eq!(got.dtype().unwrap(), DType::BFloat16);
            assert_eq!(
                got.to_float32().unwrap().to_vec(),
                want,
                "phase={phase} rows={rows}"
            );
            for (start, original) in [(0, &a), (48, &b)] {
                let view = joined.slice_rows(start, 48).unwrap();
                assert_eq!(
                    view.linear(&x).unwrap().to_float32().unwrap().to_vec(),
                    original.linear(&x).unwrap().to_float32().unwrap().to_vec()
                );
                assert_eq!(
                    view.values.to_float32().unwrap().to_vec(),
                    original.values.to_float32().unwrap().to_vec()
                );
            }
        }
        // Reproduce the actual cache state after a singleton projection.
        // Short verification windows must use the two original N=48
        // matmuls even though the joined N=96 bank is already cached.
        let mut store = Store::open_metadata(
            &fixture().join("model.gguf"),
            Some(&Store::fixture_pack_root()),
        )
        .unwrap();
        let first = "blk.0.ssm_alpha.weight";
        let second = "blk.0.ssm_beta.weight";
        for (name, start) in [(first, 0), (second, 48)] {
            store
                .banks
                .insert(name.into(), Arc::new(joined.slice_rows(start, 48).unwrap()));
            store.tensors.insert(
                name.into(),
                Tensor {
                    // Cached banks satisfy every read; no tensor payload is opened.
                    path: PathBuf::new(),
                    shape: vec![48, 2560],
                    offset: 0,
                    bytes: 48 * 2560 * 4,
                    encoding: Encoding::Gguf(GgufTensorType::F32),
                },
            );
        }
        store
            .paired_banks
            .insert((first.into(), second.into()), Arc::new(joined));
        for rows in [2, 8, 16] {
            let x = MxArray::from_float32(
                &(0..rows * 2560)
                    .map(|i| (i as f32 * 0.017 + phase).cos())
                    .collect::<Vec<_>>(),
                &[1, rows, 2560],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let (ga, gb) = store.linear_pair(&x, first, second).unwrap();
            for (want, got) in [(&a, &ga), (&b, &gb)] {
                let want = want.linear(&x).unwrap().to_float32().unwrap().to_vec();
                let got = got.to_float32().unwrap().to_vec();
                let differences = want.iter().zip(&got).filter(|(a, b)| a != b).count();
                assert_eq!(
                    differences, 0,
                    "cached pair fallback phase={phase} rows={rows}"
                );
            }
        }
    }
}

#[test]
fn failed_bank_load_releases_banks_before_the_retry() {
    let mut store = Store::open_metadata(
        &fixture().join("model.gguf"),
        Some(&Store::fixture_pack_root()),
    )
    .unwrap();
    {
        store.plan.hot_bytes = store.hot_bytes().unwrap();
        let mut refused = false;
        let result = store.prepare_hot_with_admission(&mut |bytes| {
            // Simulate headroom loss only after the bank set is
            // complete. No large allocation or OS settings change needed.
            if bytes == super::super::super::memory::WORKING_BYTES && !refused {
                refused = true;
                Err(err("simulated post-load headroom loss"))
            } else {
                Ok(())
            }
        });
        assert!(refused);
        result.unwrap();
    }
    assert!(!store.banks.is_empty());
    {
        {
            assert!(
                store
                    .banks
                    .keys()
                    .all(|name| is_fixed_projection(name, &store.tensors[name]))
            );
            assert_eq!(store.cache_bytes, store.bank_bytes);
            assert_eq!(
                store.plan.resident_bank_bytes.load(Ordering::Relaxed),
                store.bank_bytes
            );
            assert!(store.bank_bytes < store.plan.hot_bytes);
            assert!(
                store.cache.is_empty() && store.recency.is_empty() && store.protected.is_empty()
            );
        }
    }
}

#[test]
fn mtp_banks_decline_when_the_cache_cannot_stage_both() {
    // 3 x MAX_READ_BYTES lets the first bank's eviction target saturate
    // to zero and pass; the second bank's cannot once the first is pinned.
    let mut s = Store::open_fixture(&fixture().join("bf16/paged"), None).unwrap();
    s.cache_limit = MAX_READ_BYTES * 3;
    let before = s.cache_bytes;
    let mut declined = |_: u64| -> Result<()> { panic!("a declined build reached the host") };
    assert_eq!(s.prepare_mtp_expert_banks_with(&mut declined).unwrap(), 0);
    assert_eq!(s.mtp_bank_bytes().unwrap(), 0);
    assert!(MTP_EXPERT_BANKS.iter().all(|n| !s.banks.contains_key(*n)));
    assert_eq!(s.cache_bytes, before);
    assert_eq!(s.bank_bytes, 0);
    assert_eq!(s.plan.resident_bank_bytes.load(Ordering::Relaxed), 0);
    assert!(s.mtp_banks_declined, "a declined build must not retry");
    s.cache_limit = CACHE_BYTES;
    assert_eq!(s.prepare_mtp_expert_banks_with(&mut declined).unwrap(), 0);
    s.mtp_banks_declined = false;
    let bytes = s.prepare_fixture_mtp_expert_banks().unwrap();
    assert!(bytes > 0);
    assert_eq!(s.cache_bytes, before + bytes);
}

#[test]
fn failed_second_mtp_bank_leaves_no_partial_bank() {
    let mut s = Store::open_fixture(&fixture().join("bf16/paged"), None).unwrap();
    let before = (s.cache_bytes, s.bank_bytes);
    let mut admissions = 0;
    let error = s
        .prepare_mtp_expert_banks_with(&mut |_| {
            admissions += 1;
            if admissions == 2 {
                Err(err("simulated headroom loss before the second bank"))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert!(error.reason.contains("second bank"), "{error}");
    assert_eq!(admissions, 2);
    assert!(MTP_EXPERT_BANKS.iter().all(|n| !s.banks.contains_key(*n)));
    assert_eq!((s.cache_bytes, s.bank_bytes), before);
    assert_eq!(s.mtp_bank_bytes().unwrap(), 0);
    assert_eq!(s.plan.resident_bank_bytes.load(Ordering::Relaxed), 0);
    assert!(s.mtp_banks_declined);
    assert_eq!(s.prepare_mtp_expert_banks().unwrap(), 0);
    s.mtp_banks_declined = false;
    let bytes = s.prepare_mtp_expert_banks_with(&mut |_| Ok(())).unwrap();
    assert!(bytes > 0);
    assert!(MTP_EXPERT_BANKS.iter().all(|n| s.banks.contains_key(*n)));
    assert_eq!(s.bank_bytes, before.1 + bytes);
    assert_eq!(
        s.plan.resident_bank_bytes.load(Ordering::Relaxed),
        s.bank_bytes
    );
}

struct Cleanup(PathBuf);
impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const MTP_TEST_EXPERTS: usize = 4;

/// The MTP head's routed-expert banks at their real expert shapes (four
/// experts, BF16 in a temporary file), with each bank's source rows.
fn real_shape_mtp_experts() -> (Vec<(String, Tensor)>, Vec<MxArray>, Cleanup) {
    let dir = std::env::temp_dir().join(format!("qwen4-mtp-sides-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let cleanup = Cleanup(dir.clone());
    let path = dir.join("experts.bin");
    let banks = [
        (MTP_EXPERT_BANKS[0], 1280, 2560),
        (MTP_EXPERT_BANKS[1], 2560, 640),
    ];
    let (mut payload, mut tensors, mut sources) = (Vec::new(), Vec::new(), Vec::new());
    for (k, (name, n, width)) in banks.into_iter().enumerate() {
        let rows = MTP_TEST_EXPERTS * n;
        let bits: Vec<u16> = (0..rows * width)
            .map(|i| {
                let hash = (i as u32 ^ k as u32).wrapping_mul(2654435761) >> 8;
                let row = 0.05 + (i / width % 7) as f32 * 0.04;
                let v = (hash as f32 / (1u32 << 24) as f32 - 0.5) * row;
                (v.to_bits() >> 16) as u16
            })
            .collect();
        tensors.push((
            name.into(),
            Tensor {
                path: path.clone(),
                shape: vec![MTP_TEST_EXPERTS, n, width],
                offset: payload.len() as u64,
                bytes: bits.len() as u64 * 2,
                encoding: Encoding::Safe("BF16".into()),
            },
        ));
        payload.extend(bits.iter().flat_map(|b| b.to_le_bytes()));
        sources.push(MxArray::from_bfloat16(&bits, &[rows as i64, width as i64]).unwrap());
    }
    std::fs::write(&path, &payload).unwrap();
    (tensors, sources, cleanup)
}

/// The draft gate_up bank at its real expert shape (1280 x 2560, a K the
/// affine expert GEMV refuses) is built with f32 sides once, and its
/// gather keeps the bits of the f16-sided bank it replaces; the down bank
/// keeps f16 sides for the GEMV, and the estimate is the built bytes.
/// MUTATIONS: `quantize_q8` keeps f16 sides -> the gate_up dtype
/// assertion; 9/8 bytes for f32 sides -> the bytes; `quantize_q8`
/// quantizes at the side dtype -> `rows=10`.
#[test]
fn mtp_gate_up_bank_holds_f32_sides_with_the_f16_gather_bits() {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }
    let (experts, sources, _cleanup) = real_shape_mtp_experts();
    let mut s = Store::open_fixture(&fixture().join("bf16/paged"), None).unwrap();
    s.tensors.extend(experts);
    let estimate = s.mtp_bank_estimate().unwrap();
    let installed = s.prepare_mtp_expert_banks_with(&mut |_| Ok(())).unwrap();
    let sides = |name: &str| {
        let bank = s.resident_bank(name).unwrap();
        [&bank.scales, &bank.biases].map(|a| a.as_ref().unwrap().dtype().unwrap())
    };
    assert_eq!(
        sides(MTP_EXPERT_BANKS[0]),
        [DType::Float32; 2],
        "the gather casts f16 gate_up sides on every draft"
    );
    assert_eq!(
        sides(MTP_EXPERT_BANKS[1]),
        [DType::Float16; 2],
        "the affine expert GEMV reads f16 down sides"
    );
    assert_eq!(installed, s.mtp_bank_bytes().unwrap());
    assert_eq!(
        installed,
        (MTP_TEST_EXPERTS * 1280 * 2560 * 10 / 8 + MTP_TEST_EXPERTS * 2560 * 640 * 9 / 8) as u64
    );
    assert_eq!(estimate, installed, "the estimate charges the built banks");
    let gate_up = s.resident_bank(MTP_EXPERT_BANKS[0]).unwrap();
    let before = Weight::quantize_q8(&sources[0], DType::Float16).unwrap();
    for rows in [10usize, 40, 120] {
        let ids = MxArray::from_uint32(
            &(0..rows)
                .map(|i| ((i * 7 + i / 3) % MTP_TEST_EXPERTS) as u32)
                .collect::<Vec<_>>(),
            &[rows as i64],
        )
        .unwrap();
        let x = MxArray::from_float32(
            &(0..rows * 2560)
                .map(|i| (i as f32 * 0.013).cos())
                .collect::<Vec<_>>(),
            &[rows as i64, 1, 2560],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
        let bits = |bank: &Weight| {
            bank.expert_rows(&x, &ids, MTP_TEST_EXPERTS, false)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        };
        let want = bits(&before);
        assert!(want.iter().any(|&b| b != 0), "rows={rows}");
        assert_eq!(bits(&gate_up), want, "rows={rows}");
    }
}

/// The f32 sides need more live headroom than the f16 layout: a refused
/// f32 build builds the f16 one once instead of declining the draft banks,
/// and declines them only when that is refused too.
/// MUTATIONS: no f16 retry -> "the f32 gate_up staging"; the retry for
/// any layout -> `failed_second_mtp_bank_leaves_no_partial_bank`.
#[test]
fn refused_f32_draft_banks_fall_back_to_the_f16_layout() {
    let (experts, _, _cleanup) = real_shape_mtp_experts();
    let bytes = |n: usize, width: usize, eighths: usize| {
        (MTP_TEST_EXPERTS * n * width * eighths / 8) as u64
    };
    let (wide, narrow, down) = (
        bytes(1280, 2560, 10),
        bytes(1280, 2560, 9),
        bytes(2560, 640, 9),
    );
    let [wide_up, narrow_up, down_up] =
        [wide, narrow, down].map(|bank| 2 * bank + super::super::super::memory::WORKING_BYTES);
    let cases: [(&str, &[usize], &[u64], bool); 3] = [
        (
            "the f32 gate_up staging",
            &[0],
            &[wide_up, narrow_up, down_up],
            true,
        ),
        (
            "the down staging beside the f32 gate_up bank",
            &[1],
            &[wide_up, down_up, narrow_up, down_up],
            true,
        ),
        (
            "both gate_up stagings",
            &[0, 1],
            &[wide_up, narrow_up],
            false,
        ),
    ];
    for (why, refused, want, built) in cases {
        let mut s = Store::open_fixture(&fixture().join("bf16/paged"), None).unwrap();
        s.tensors.extend(experts.iter().cloned());
        assert_eq!(s.mtp_bank_estimate().unwrap(), wide + down, "{why}");
        let mut seen = Vec::new();
        let result = s.prepare_mtp_expert_banks_with(&mut |bytes| {
            seen.push(bytes);
            if refused.contains(&(seen.len() - 1)) {
                Err(err("simulated headroom loss"))
            } else {
                Ok(())
            }
        });
        if built {
            assert_eq!(result.ok(), Some(narrow + down), "{why}");
            let bank = s.resident_bank(MTP_EXPERT_BANKS[0]).unwrap();
            assert_eq!(
                [&bank.scales, &bank.biases].map(|a| a.as_ref().unwrap().dtype().unwrap()),
                [DType::Float16; 2],
                "{why}"
            );
            assert!(!s.mtp_banks_declined, "{why}");
        } else {
            assert!(result.is_err(), "{why}");
            assert!(s.mtp_banks_declined, "{why}");
            assert_eq!(s.mtp_bank_bytes().unwrap(), 0, "{why}");
        }
        assert_eq!(seen, want, "{why}");
    }
}

/// The load keeps the f16 retry: a host that refuses the f32
/// gate_up staging gets the f16 layout, one that refuses both loads
/// without draft banks.
#[test]
fn the_load_retries_the_f16_layout_when_f32_is_refused() {
    let (experts, _, _cleanup) = real_shape_mtp_experts();
    let bytes = |n: usize, width: usize, eighths: usize| {
        (MTP_TEST_EXPERTS * n * width * eighths / 8) as u64
    };
    let (wide, narrow, down) = (
        bytes(1280, 2560, 10),
        bytes(1280, 2560, 9),
        bytes(2560, 640, 9),
    );
    let staging = |bank: u64| 2 * bank + super::super::super::memory::WORKING_BYTES;
    for (refused, built) in [
        (vec![staging(wide)], true),
        (vec![staging(wide), staging(narrow)], false),
    ] {
        let mut s = Store::open_fixture(&fixture().join("bf16/paged"), None).unwrap();
        s.tensors.extend(experts.iter().cloned());
        let mut seen = Vec::new();
        s.finish_load_with(&mut |bytes| {
            seen.push(bytes);
            if refused.contains(&bytes) {
                Err(err("simulated headroom loss at load"))
            } else {
                Ok(())
            }
        })
        .unwrap();
        assert_eq!(seen[..2], [staging(wide), staging(narrow)], "built={built}");
        if built {
            assert_eq!(s.mtp_bank_bytes().unwrap(), narrow + down);
            let bank = s.resident_bank(MTP_EXPERT_BANKS[0]).unwrap();
            assert_eq!(
                [&bank.scales, &bank.biases].map(|a| a.as_ref().unwrap().dtype().unwrap()),
                [DType::Float16; 2]
            );
            assert!(!s.mtp_banks_declined);
        } else {
            assert_eq!(s.mtp_bank_bytes().unwrap(), 0);
            assert!(s.mtp_banks_declined);
        }
    }
}

#[test]
fn partial_projection_banks_use_actual_sizes_and_preserve_expert_lru() {
    for path in [fixture(), fixture().join("model.gguf")] {
        let mut s = Store::open_metadata(&path, Some(&Store::fixture_pack_root())).unwrap();
        let bytes: u64 = s
            .tensors
            .iter()
            .filter(|(name, t)| is_fixed_projection(name, t))
            .map(|(name, t)| t.bank_bytes(name).unwrap())
            .sum();
        assert!(bytes > 0);
        s.cache_limit = bytes * 4 - 1;
        assert!(s.projection_bank_names().unwrap().is_empty());
        s.cache_limit = bytes * 4;
        let names = s.projection_bank_names().unwrap();
        assert!(!names.is_empty());
        assert_eq!(
            s.bytes_read, 0,
            "budget selection must read only descriptors"
        );

        // A metadata-only fixture can exercise the bank preparation without
        // requesting tens of GiB from the machine running these tests.
        s.cache_limit = CACHE_BYTES;
        s.prepare_fixture_hot().unwrap();
        assert_eq!(s.bank_bytes, bytes);
        assert_eq!(s.cache_bytes, bytes);
        assert_eq!(s.plan.resident_bank_bytes.load(Ordering::Relaxed), bytes);
        assert!(s.cache.is_empty());
        assert_eq!(s.banks.len(), names.len());
        assert!(
            s.banks
                .keys()
                .all(|n| is_fixed_projection(n, &s.tensors[n]))
        );
        let loaded = s.bytes_read;
        let mut control = Store::open_metadata(&path, Some(&Store::fixture_pack_root())).unwrap();
        for name in names {
            let rows = s.descriptor(&name).unwrap().rows().unwrap();
            for start in [0, rows - 1] {
                let got = s
                    .read(&name, start, 1)
                    .unwrap()
                    .dense()
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let want = control
                    .read(&name, start, 1)
                    .unwrap()
                    .dense()
                    .unwrap()
                    .to_float32()
                    .unwrap();
                assert_eq!(&*got, &*want, "partial bank changed {name}, row {start}");
            }
        }
        assert_eq!(s.bytes_read, loaded);
        let expert = s
            .tensors
            .keys()
            .find(|n| n.contains(".experts.") || n.contains("_exps."))
            .unwrap()
            .clone();
        assert!(!s.banks.contains_key(&expert));
        s.read(&expert, 0, 1).unwrap();
        assert!(s.cache_bytes > bytes);
        s.evict_to(bytes).unwrap();
        assert!(s.cache.is_empty(), "expert rows must remain evictable");
        assert_eq!(s.cache_bytes, s.bank_bytes);
    }
}

#[test]
fn projection_banks_preserve_rows_and_leave_lookup_and_auxiliary_weights_cold() {
    for path in [fixture(), fixture().join("bf16/paged/model.gguf")] {
        let mut s = Store::open_metadata(&path, Some(&Store::fixture_pack_root())).unwrap();
        assert_eq!(s.bytes_read, 0);
        s.prepare_fixture_hot().unwrap();
        assert_eq!(s.cache_bytes, s.bank_bytes);
        assert!(s.cache.is_empty());
        assert!(s.banks.keys().all(|name| is_hot(name)));
        assert!(
            s.banks
                .keys()
                .all(|name| is_fixed_projection(name, &s.tensors[name]))
        );
        assert!(!s.banks.is_empty());
        let reads = s.bytes_read;
        let mut reference = Store::open_metadata(&path, Some(&Store::fixture_pack_root())).unwrap();
        for name in s.banks.keys().cloned().collect::<Vec<_>>() {
            let rows = s.descriptor(&name).unwrap().rows().unwrap();
            for start in [0, rows - 1] {
                let a = s
                    .read(&name, start, 1)
                    .unwrap()
                    .dense()
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let b = reference
                    .read(&name, start, 1)
                    .unwrap()
                    .dense()
                    .unwrap()
                    .to_float32()
                    .unwrap();
                assert_eq!(&*a, &*b, "resident row changed: {name}, {start}");
            }
            assert!(s.read(&name, rows, 1).is_err());
        }
        assert_eq!(s.bytes_read, reads);
        for name in [
            "token_embd.weight",
            "embed_tokens.weight",
            "per_layer_token_embd.weight",
            "layers.0.ple_embedding.ngram_embedding.weight",
            "mtp.fc_embedding.weight",
            "visual.patch_embed.proj.weight",
        ] {
            assert!(!is_hot(name), "cold tensor was selected: {name}");
        }
    }
}

#[test]
fn oversized_bank_is_rejected_before_any_payload_read() {
    let mut s = Store::open_metadata(&fixture(), None).unwrap();
    let d = s.tensors.get_mut("lm_head.weight").unwrap();
    d.shape = vec![1 << 30, 1];
    d.bytes = 4 << 30;
    // Budget selection leaves such a checkpoint on its chunk path...
    assert!(s.projection_bank_names().unwrap().is_empty());
    s.prepare_fixture_hot().unwrap();
    assert!(s.banks.is_empty());
    // ...and an explicit bank request is refused before any read.
    assert!(
        s.prepare_banks(vec!["lm_head.weight".into()], &mut |_| Ok(()))
            .unwrap_err()
            .to_string()
            .contains("1 GiB bank budget")
    );
    assert_eq!(s.bytes_read, 0);
    assert!(s.banks.is_empty());
}
