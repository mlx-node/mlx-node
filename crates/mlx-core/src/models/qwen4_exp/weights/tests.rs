mod cache_tests {
    use super::super::*;

    /// Splash `storedRowBytes`: a BF16 / F16 PLE table row is `columns * 2`
    /// bytes at any width, 160 included (not a multiple of 256).
    #[test]
    fn ple_table_rows_bf16_and_f16_at_width_160() {
        use crate::utils::gguf_test_fixtures::build_minimal_gguf;
        const WIDTH: usize = 160;
        const ROWS: usize = 7;
        let value = |r: usize, c: usize| ((r * ROWS + c) % 128) as f32 - 64.0;
        let dir = std::env::temp_dir().join(format!("qwen4-ple-rows-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(dir.clone());
        for ty in [GgufTensorType::BF16, GgufTensorType::F16] {
            let payload: Vec<u8> = (0..ROWS)
                .flat_map(|r| (0..WIDTH).map(move |c| (r, c)))
                .flat_map(|(r, c)| match ty {
                    GgufTensorType::BF16 => half::bf16::from_f32(value(r, c)).to_le_bytes(),
                    _ => half::f16::from_f32(value(r, c)).to_le_bytes(),
                })
                .collect();
            let path = dir.join(format!("ple-{ty:?}.gguf"));
            fs::write(
                &path,
                build_minimal_gguf(
                    &[(
                        "general.architecture",
                        GgufMetaValue::String("qwen4exp".into()),
                    )],
                    &[(
                        "per_layer_token_embd.weight",
                        &[WIDTH as u64, ROWS as u64],
                        ty,
                        &payload,
                    )],
                ),
            )
            .unwrap();
            let store = Store::open_descriptors(&path).unwrap();
            let d = store.descriptor("per_layer_token_embd.weight").unwrap();
            assert_eq!(d.rows().unwrap(), ROWS, "{ty:?}");
            assert_eq!(d.width().unwrap(), WIDTH, "{ty:?}");
            let check = |start: usize, rows: usize| {
                let weight = d
                    .import_rows(start, rows)
                    .unwrap_or_else(|e| panic!("{ty:?} rows {start}+{rows}: {}", e.reason));
                let got = weight.values.astype(DType::Float32).unwrap();
                let got = got.to_float32().unwrap();
                let want: Vec<f32> = (start..start + rows)
                    .flat_map(|r| (0..WIDTH).map(move |c| value(r, c)))
                    .collect();
                assert_eq!(&*got, want.as_slice(), "{ty:?} rows {start}+{rows}");
            };
            for r in 0..ROWS {
                check(r, 1);
            }
            check(2, 3);
        }
    }

    #[test]
    fn pq2_gguf_is_rejected_during_store_indexing() {
        let dir = std::env::temp_dir().join(format!("qwen4-pq2-reject-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(dir.clone());
        let string = |bytes: &mut Vec<u8>, value: &str| {
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.bytes());
        };
        let write = |path: &Path, name: &str, ty: GgufTensorType, index: u32, count: u32| {
            let mut bytes = b"GGUF".to_vec();
            bytes.extend(3u32.to_le_bytes());
            bytes.extend(1u64.to_le_bytes());
            bytes.extend(3u64.to_le_bytes());
            string(&mut bytes, "general.architecture");
            bytes.extend(8u32.to_le_bytes());
            string(&mut bytes, "qwen4exp");
            for (key, value) in [("split.no", index), ("split.count", count)] {
                string(&mut bytes, key);
                bytes.extend(4u32.to_le_bytes());
                bytes.extend(value.to_le_bytes());
            }
            string(&mut bytes, name);
            bytes.extend(2u32.to_le_bytes());
            bytes.extend(128u64.to_le_bytes());
            bytes.extend(1u64.to_le_bytes());
            bytes.extend((ty as u32).to_le_bytes());
            bytes.extend(0u64.to_le_bytes());
            let payload_bytes = 128 / ty.block_size() * ty.type_size();
            bytes.resize(bytes.len().div_ceil(32) * 32 + payload_bytes, 0);
            fs::write(path, bytes).unwrap();
        };
        let supported = dir.join("supported.gguf");
        write(&supported, "supported.weight", GgufTensorType::Q4_0, 0, 1);
        let store = Store::open_metadata(&supported, Some(&dir.join("supported-cache"))).unwrap();
        assert_eq!(
            store
                .descriptor("supported.weight")
                .unwrap()
                .width()
                .unwrap(),
            128
        );
        assert_eq!(store.bytes_read, 0);
        let single = dir.join("single.gguf");
        write(&single, "test.pq2.weight", GgufTensorType::PQ2_0, 0, 1);
        let first = dir.join("split-00001-of-00002.gguf");
        let second = dir.join("split-00002-of-00002.gguf");
        write(&first, "supported.weight", GgufTensorType::Q4_0, 0, 2);
        write(&second, "test.pq2.weight", GgufTensorType::PQ2_0, 1, 2);
        for (label, path, pq2_path) in
            [("single", single.clone(), single), ("split", first, second)]
        {
            let parsed = parse_gguf(&pq2_path).unwrap();
            assert_eq!(parsed.tensors[0].tensor_type, GgufTensorType::PQ2_0);
            let cache = dir.join(format!("{label}-cache"));
            let error = match Store::open_metadata(&path, Some(&cache)) {
                Ok(_) => panic!("{label}: Qwen4 must reject PQ2 before loading payloads"),
                Err(error) => error,
            };
            assert!(error.reason.contains("PQ2_0"), "{}", error.reason);
            assert!(error.reason.contains("Qwen4"), "{}", error.reason);
            assert!(error.reason.contains("test.pq2.weight"), "{}", error.reason);
            assert!(
                !cache.exists(),
                "{label}: rejection must precede packed-cache setup"
            );
        }
    }

    #[test]
    fn pq2_runtime_residency_is_rejected() {
        let mut tensor = Tensor {
            path: PathBuf::new(),
            shape: vec![1, 128],
            offset: 0,
            bytes: 34,
            encoding: Encoding::Gguf(GgufTensorType::PQ2_0),
        };
        let error = tensor
            .runtime_bytes()
            .expect_err("Qwen4 must not budget unsupported PQ2");
        assert!(error.reason.contains("PQ2_0"), "{}", error.reason);
        tensor.encoding = Encoding::Gguf(GgufTensorType::Q4_0);
        tensor.bytes = 72;
        assert_eq!(tensor.runtime_bytes().unwrap(), 80);
    }

    #[test]
    fn descriptor_dimensions_reject_empty_zero_and_overflowing_shapes() {
        let mut tensor = Tensor {
            path: PathBuf::from("unused.safetensors"),
            shape: vec![2, 3],
            offset: 0,
            bytes: 24,
            encoding: Encoding::Safe("F32".into()),
        };
        assert_eq!(tensor.width().unwrap(), 3);
        assert_eq!(tensor.rows().unwrap(), 2);
        assert_eq!(tensor.range(1, 1).unwrap(), (12, 12));
        for shape in [vec![], vec![0], vec![0, 3], vec![2, 0], vec![usize::MAX, 2]] {
            tensor.shape = shape;
            assert!(tensor.rows().is_err());
            assert!(tensor.range(0, 1).is_err());
        }
    }

    #[test]
    fn fixture_forward_headroom_isolated_from_production_guard() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp");
        let fixture = Store::open_fixture(&path, None).unwrap();
        let production = Store::open_metadata(&path, None).unwrap();
        let checked = std::cell::Cell::new(0);
        let unavailable = || {
            checked.set(checked.get() + 1);
            Err(err("simulated exhausted system headroom"))
        };
        fixture.check_forward_headroom_with(unavailable).unwrap();
        assert_eq!(checked.get(), 0, "tiny fixture queried production reserves");
        assert!(production.check_forward_headroom_with(unavailable).is_err());
        assert_eq!(checked.get(), 1, "production skipped its headroom check");
    }

    #[test]
    fn direct_store_keeps_the_forward_headroom_guard() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp");
        let mut production = Store::open_metadata(&path, None).unwrap();
        production.plan.direct = true;
        let checked = std::cell::Cell::new(0);
        let unavailable = || {
            checked.set(checked.get() + 1);
            Err(err("simulated exhausted system headroom"))
        };
        assert!(production.check_forward_headroom_with(unavailable).is_err());
        assert_eq!(
            checked.get(),
            1,
            "direct store skipped the forward headroom check"
        );
    }

    /// Full direct experts bypass the row importer; every other
    /// read is anonymous growth and keeps the 256 MiB system-headroom recheck.
    #[test]
    fn direct_store_rechecks_growth_headroom_for_anonymous_rows() {
        if !cfg!(target_os = "macos") {
            return;
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp");
        let mut store = Store::open_metadata(&path, None).unwrap();
        store.plan.direct = true;
        let checked = std::cell::Cell::new(0);
        let unavailable = || {
            checked.set(checked.get() + 1);
            Err(err("simulated exhausted system headroom"))
        };
        let row = store
            .read_impl_with("embed_tokens.weight", 0, 1, false, unavailable)
            .unwrap();
        assert_eq!(
            checked.get(),
            0,
            "a read under 256 MiB of growth queried headroom"
        );
        assert_eq!(
            store.growth_bytes,
            row.bytes().unwrap(),
            "direct mode must charge anonymous rows as growth"
        );
        store.growth_bytes = 256 << 20;
        let error = match store.read_impl_with("embed_tokens.weight", 1, 1, false, unavailable) {
            Ok(_) => panic!("direct store ignored exhausted growth headroom"),
            Err(error) => error,
        };
        assert!(error.reason.contains("simulated"), "{}", error.reason);
        assert_eq!(checked.get(), 1);
        assert_eq!(
            store.growth_bytes,
            256 << 20,
            "a refused read must leave the growth pending"
        );
        assert!(
            !store
                .cache
                .contains_key(&("embed_tokens.weight".to_string(), 1, 1))
        );
        store
            .read_impl_with("embed_tokens.weight", 1, 1, false, || Ok(()))
            .unwrap();
        assert_eq!(
            store.growth_bytes,
            row.bytes().unwrap(),
            "a passing check restarts the growth count"
        );
    }

    #[test]
    fn reference_shared_prefill_pair_preserves_projections_and_storage() {
        if !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
            return;
        }
        let mut store = Store::open_metadata(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        let (n, k) = (640i64, 2560i64);
        let make = |offset: u32| {
            Arc::new(Weight {
                dense_bf16: None,
                values: MxArray::from_uint32(
                    &(0..n * k / 4)
                        .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(offset))
                        .collect::<Vec<_>>(),
                    &[n, k / 4],
                )
                .unwrap(),
                scales: Some(
                    MxArray::from_float32(
                        &(0..n * k / 32)
                            .map(|i| ((i % 19) as f32 + 1.) / 4096.)
                            .collect::<Vec<_>>(),
                        &[n, k / 32],
                    )
                    .unwrap()
                    .astype(DType::Float16)
                    .unwrap(),
                ),
                biases: Some(
                    MxArray::from_float32(
                        &(0..n * k / 32)
                            .map(|i| -((i % 19) as f32 + 1.) / 32.)
                            .collect::<Vec<_>>(),
                        &[n, k / 32],
                    )
                    .unwrap()
                    .astype(DType::Float16)
                    .unwrap(),
                ),
                group: 32,
                bits: 8,
                mode: "affine".into(),
                q8_symmetric_128: true,
                qlinear: None,
            })
        };
        let banks = [make(13), make(97)];
        let names = ["blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight"];
        for (name, bank) in names.iter().zip(&banks) {
            store.tensors.insert(
                (*name).into(),
                Tensor {
                    path: PathBuf::new(),
                    shape: vec![n as usize, k as usize],
                    offset: 0,
                    bytes: (n * k) as u64,
                    encoding: Encoding::Gguf(GgufTensorType::Q8_0),
                },
            );
            store.banks.insert((*name).into(), bank.clone());
        }
        let bytes = banks[0].bytes().unwrap() + banks[1].bytes().unwrap();
        for rows in [511i64, 512, 1024] {
            let x = MxArray::from_float32(
                &(0..rows * k)
                    .map(|i| ((i * 17 % 251) as f32 - 125.) / 128.)
                    .collect::<Vec<_>>(),
                &[1, rows, k],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let expected = [banks[0].linear(&x).unwrap(), banks[1].linear(&x).unwrap()];
            let (a, b) = store.linear_pair(&x, names[0], names[1]).unwrap();
            for (actual, want) in [&a, &b].into_iter().zip(&expected) {
                assert_eq!(
                    &*actual.to_float32().unwrap(),
                    &*want.to_float32().unwrap(),
                    "rows={rows}"
                );
            }
            let key = (names[0].to_owned(), names[1].to_owned());
            if rows < 512 {
                assert!(!store.paired_banks.contains_key(&key));
            } else if super::super::super::tf32_enabled() {
                assert_eq!(store.paired_banks[&key].bytes().unwrap(), bytes);
            }
        }
    }

    #[test]
    fn quantized_ple_windows_preserve_singleton_projection_rounding() {
        let mut store = Store::open_metadata(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        let (n, k) = (64i64, 2560i64);
        let scales = MxArray::from_float32(
            &(0..n * k / 32)
                .map(|i| ((i % 19) as f32 + 1.) / 4096.)
                .collect::<Vec<_>>(),
            &[n, k / 32],
        )
        .unwrap()
        .astype(DType::Float16)
        .unwrap();
        let bank = Arc::new(Weight {
            dense_bf16: None,
            values: MxArray::from_uint32(
                &(0..n * k / 4)
                    .map(|i| (i as u32).wrapping_mul(2654435761))
                    .collect::<Vec<_>>(),
                &[n, k / 4],
            )
            .unwrap(),
            biases: Some(scales.mul_scalar(-128.).unwrap()),
            scales: Some(scales),
            group: 32,
            bits: 8,
            mode: "affine".into(),
            q8_symmetric_128: true,
            qlinear: None,
        });
        let name = "test.ple_key.weight";
        store.tensors.insert(
            name.into(),
            Tensor {
                path: PathBuf::new(),
                shape: vec![n as usize, k as usize],
                offset: 0,
                bytes: (n * k) as u64,
                encoding: Encoding::Gguf(GgufTensorType::Q8_0),
            },
        );
        store.banks.insert(name.into(), bank.clone());
        for dtype in [DType::BFloat16, DType::Float16, DType::Float32] {
            let x = MxArray::from_float32(
                &(0..17 * k)
                    .map(|i| (i as f32 * 0.13).sin())
                    .collect::<Vec<_>>(),
                &[1, 17, k],
            )
            .unwrap()
            .astype(dtype)
            .unwrap();
            let parts = (0..17)
                .map(|t| bank.linear(&x.slice_axis(1, t, t + 1).unwrap()).unwrap())
                .collect::<Vec<_>>();
            let expected = MxArray::concatenate_many(parts.iter().collect(), Some(1))
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            let actual = store
                .linear_vector_window(name, &x)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(&*actual, &*expected, "dtype {dtype:?}");
        }
    }
    struct Cleanup(PathBuf);
    impl Cleanup {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("qwen4-lookup-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A GGUF of one Q5_1 `token_embd.weight` [8, 32]: 24 bytes per row.
    fn rows_gguf(dir: &Path) -> (PathBuf, Vec<u8>) {
        let rows = 8u32;
        let path = dir.join("rows.gguf");
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        let string = |b: &mut Vec<u8>, s: &str| {
            b.extend((s.len() as u64).to_le_bytes());
            b.extend(s.bytes());
        };
        string(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        string(&mut b, "qwen4exp");
        string(&mut b, "token_embd.weight");
        b.extend(2u32.to_le_bytes());
        b.extend(32u64.to_le_bytes());
        b.extend(u64::from(rows).to_le_bytes());
        b.extend(7u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        for row in 0..rows {
            b.extend(half::f16::from_f32(0.015625).to_bits().to_le_bytes());
            b.extend(half::f16::from_f32(-0.25).to_bits().to_le_bytes());
            b.extend((row * 7919).to_le_bytes());
            b.extend([row as u8; 16]);
        }
        fs::write(&path, &b).unwrap();
        (path, b)
    }

    #[test]
    fn batched_lookup_reuses_decoded_rows_and_survives_eviction() {
        let cleanup = Cleanup::new();
        let dir = cleanup.0.clone();
        let (path, _) = rows_gguf(&dir);
        let mut store = Store::open_fixture(&path, Some(&dir)).unwrap();
        let mut oracle = Store::open_fixture(&path, Some(&dir)).unwrap();
        let ids = [3usize, 1, 3, 7];
        let expected = ids
            .iter()
            .map(|&id| {
                oracle
                    .read("token_embd.weight", id, 1)
                    .unwrap()
                    .dense()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let expected = MxArray::concatenate_many(expected.iter().collect(), Some(0))
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        let cold = store.lookup_rows("token_embd.weight", &ids).unwrap();
        assert!(
            !unsafe { mlx_sys::mlx_array_is_available(cold.as_raw_ptr()) },
            "a cold lookup window must stay lazy for its consumer's graph"
        );
        assert_eq!(store.bytes_read, 3 * 24);
        assert_eq!((store.lookup.lookups(), store.lookup.misses()), (3, 3));
        assert_eq!(store.cache_bytes, 3 * 32 * 2);
        let warm = store.lookup_rows("token_embd.weight", &ids).unwrap();
        assert_eq!((store.lookup.lookups(), store.lookup.misses()), (6, 3));
        assert_eq!(store.bytes_read, 3 * 24);
        assert!(store.lookup_rows("token_embd.weight", &[0, 8]).is_err());
        assert_eq!(
            store.bytes_read,
            3 * 24,
            "validate the whole window before IO"
        );
        store.evict_to(0).unwrap();
        assert_eq!(store.cache_bytes, 0);
        // Reader threads: the same rows, requested ahead and consumed by the
        // import, decode to the same bits through the in-memory GGUF path.
        let keys: Vec<_> = ids
            .iter()
            .map(|&id| ("token_embd.weight".to_string(), id))
            .collect();
        store.prefetch_rows(&keys).unwrap();
        let fetched = store.lookup_rows("token_embd.weight", &ids).unwrap();
        assert_eq!((store.lookup.lookups(), store.lookup.misses()), (9, 6));
        assert_eq!(store.lookup.prefetched() + store.lookup.late(), 3);
        store.prefetch_rows(&keys).unwrap();
        let hits = store.lookup_rows("token_embd.weight", &ids).unwrap();
        assert_eq!(store.lookup.misses(), 6, "LRU hits request nothing");
        assert_eq!(store.lookup.prefetched() + store.lookup.late(), 3);
        for output in [cold, warm, fetched, hits] {
            assert_eq!(
                &*output.astype(DType::Float32).unwrap().to_float32().unwrap(),
                &expected
            );
        }
    }

    /// A lookup window leaves one Metal buffer behind for the rows it
    /// imported; rows it hit stay where they are; the LRU charges a window's
    /// buffer whole until its last row view is evicted and reports the
    /// bytes it released, so the charge is the memory that is allocated.
    #[test]
    fn decoded_rows_share_their_windows_buffer_and_stay_under_the_lru_bound() {
        let cleanup = Cleanup::new();
        let dir = cleanup.0.clone();
        let (path, _) = rows_gguf(&dir);
        let mut store = Store::open_fixture(&path, Some(&dir)).unwrap();
        let mut oracle = Store::open_fixture(&path, Some(&dir)).unwrap();
        let name = "token_embd.weight";
        let rb: usize = 32 * 2;
        let key = |id: usize| (name.to_string(), id, 1);
        let cached = |store: &Store, id: usize| store.cache[&key(id)].values.clone();
        let f32s = |a: &MxArray| a.astype(DType::Float32).unwrap().to_float32().unwrap();
        let same_bits = |store: &Store, oracle: &mut Store, id: usize| {
            let expected = oracle.read(name, id, 1).unwrap().dense().unwrap();
            assert_eq!(&*f32s(&cached(store, id)), &*f32s(&expected), "row {id}");
        };
        // What is allocated: every live window buffer whole, plus every row
        // that has its own buffer.
        let allocated = |store: &Store| -> u64 {
            store.window_banks.values().map(|b| b.bytes).sum::<u64>()
                + store
                    .cache
                    .iter()
                    .filter(|(k, _)| !store.row_bank.contains_key(*k))
                    .map(|(_, w)| w.bytes().unwrap())
                    .sum::<u64>()
        };
        let held =
            |store: &Store, ids: &[usize]| ids.iter().all(|&id| store.cache.contains_key(&key(id)));
        let gone = |store: &Store, ids: &[usize]| {
            ids.iter().all(|&id| !store.cache.contains_key(&key(id)))
        };

        let window = store.lookup_rows(name, &[5, 1, 6, 1, 2]).unwrap();
        MxArray::eval_arrays(&[&window]).unwrap();
        let views: Vec<_> = [5usize, 1, 6, 2]
            .iter()
            .map(|&id| cached(&store, id))
            .collect();
        MxArray::eval_arrays(&views.iter().collect::<Vec<_>>()).unwrap();
        let base = views[0].storage_ptr() as usize;
        assert_ne!(base, 0);
        for (slot, (view, id)) in views.iter().zip([5usize, 1, 6, 2]).enumerate() {
            assert_eq!(
                view.storage_ptr() as usize,
                base + slot * rb,
                "row {id} is slot {slot} of the window's buffer"
            );
            same_bits(&store, &mut oracle, id);
        }
        assert_eq!(store.cache_bytes, 4 * rb as u64);
        assert_eq!(store.cache_bytes, allocated(&store));
        assert_eq!(store.window_banks.len(), 1);

        // A hit stays in its buffer; a window's single miss keeps its own.
        let next = store.lookup_rows(name, &[1, 7]).unwrap();
        MxArray::eval_arrays(&[&next]).unwrap();
        let (kept, own) = (cached(&store, 1), cached(&store, 7));
        MxArray::eval_arrays(&[&kept, &own]).unwrap();
        assert_eq!(
            kept.storage_ptr() as usize,
            base + rb,
            "a hit is not copied"
        );
        assert!(!store.row_bank.contains_key(&key(7)));
        assert_eq!(store.cache_bytes, 5 * rb as u64);
        assert_eq!(store.cache_bytes, allocated(&store));
        assert_eq!(
            &*f32s(&next),
            &*f32s(&MxArray::concatenate_many(vec![&kept, &own], Some(0)).unwrap())
        );

        // Two misses around a hit: a second buffer holds the two.
        let third = store.lookup_rows(name, &[3, 1, 4]).unwrap();
        MxArray::eval_arrays(&[&third]).unwrap();
        let (a, b) = (cached(&store, 3), cached(&store, 4));
        MxArray::eval_arrays(&[&a, &b]).unwrap();
        assert_eq!(a.storage_ptr() as usize + rb, b.storage_ptr() as usize);
        assert_eq!(cached(&store, 1).storage_ptr() as usize, base + rb);
        assert_eq!(store.cache_bytes, 7 * rb as u64);
        assert_eq!(store.cache_bytes, allocated(&store));
        assert_eq!(store.window_banks.len(), 2);
        for id in [3, 1, 4] {
            same_bits(&store, &mut oracle, id);
        }

        // LRU order is 5, 6, 2, 7, 3, 1, 4. Evicting rows of a buffer that
        // still has a view releases nothing; the last view releases it all,
        // and the counters say what was released.
        store.evict_to(7 * rb as u64 - 1).unwrap();
        assert!(gone(&store, &[5, 6, 2, 7]) && held(&store, &[3, 1, 4]));
        assert_eq!(
            store.cache_bytes,
            6 * rb as u64,
            "the first buffer is charged whole for its one live row"
        );
        assert_eq!(store.cache_bytes, allocated(&store));
        assert_eq!(
            (
                store.lookup.evicted_rows(),
                store.lookup.evicted_row_bytes()
            ),
            (4, rb as u64)
        );
        assert_eq!(cached(&store, 1).storage_ptr() as usize, base + rb);
        same_bits(&store, &mut oracle, 1);
        store.evict_to(5 * rb as u64).unwrap();
        assert!(gone(&store, &[3, 1]) && held(&store, &[4]));
        assert_eq!(
            store.cache_bytes,
            2 * rb as u64,
            "the first buffer went with row 1"
        );
        assert_eq!(store.cache_bytes, allocated(&store));
        assert_eq!(store.lookup.evicted_row_bytes(), 5 * rb as u64);
        same_bits(&store, &mut oracle, 4);
        store.evict_to(0).unwrap();
        assert_eq!(store.cache_bytes, 0);
        assert!(
            store.cache.is_empty() && store.window_banks.is_empty() && store.row_bank.is_empty()
        );
        assert_eq!(
            (
                store.lookup.evicted_rows(),
                store.lookup.evicted_row_bytes()
            ),
            (7, 7 * rb as u64)
        );

        // Four rows fit beside the staging reserve; a window of eight evicts
        // in LRU order while it imports and folds the rows that survived.
        store.cache_limit = MAX_READ_BYTES * 3 + 4 * rb as u64;
        let wide = store.lookup_rows(name, &[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        MxArray::eval_arrays(&[&wide]).unwrap();
        assert_eq!(store.cache_bytes, 5 * rb as u64, "LRU bound");
        assert_eq!(store.cache_bytes, allocated(&store));
        assert_eq!(store.window_banks.len(), 1);
        assert!(gone(&store, &[0, 1, 2]) && held(&store, &[3, 4, 5, 6, 7]));
        assert_eq!(
            (
                store.lookup.evicted_rows(),
                store.lookup.evicted_row_bytes()
            ),
            (10, 10 * rb as u64)
        );
        let expected: Vec<_> = (0..8)
            .map(|id| oracle.read(name, id, 1).unwrap().dense().unwrap())
            .collect();
        assert_eq!(
            &*f32s(&wide),
            &*f32s(&MxArray::concatenate_many(expected.iter().collect(), Some(0)).unwrap())
        );
        same_bits(&store, &mut oracle, 7);
    }

    /// A reader's failed read is consumed with its request and never
    /// returned: the import reads the row itself, so the file as it is now
    /// decides, and nothing for the key stays behind.
    #[test]
    fn a_row_readers_error_is_dropped_and_the_import_reads_the_row_itself() {
        let cleanup = Cleanup::new();
        let dir = cleanup.0.clone();
        let (path, bytes) = rows_gguf(&dir);
        let mut store = Store::open_fixture(&path, Some(&dir)).unwrap();
        let mut oracle = Store::open_fixture(&path, Some(&dir)).unwrap();
        let name = "token_embd.weight";
        let keys = [(name.to_string(), 7usize)];
        let (offset, _) = store.descriptor(name).unwrap().range(7, 1).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(offset + 10)
            .unwrap();
        store.prefetch_rows(&keys).unwrap();
        store.rows.as_ref().unwrap().wait_idle();
        fs::write(&path, &bytes).unwrap();
        store.prefetch_rows(&keys).unwrap();
        let row = store.lookup_rows(name, &[7]).unwrap();
        let expected = oracle.read(name, 7, 1).unwrap().dense().unwrap();
        assert_eq!(
            &*row.astype(DType::Float32).unwrap().to_float32().unwrap(),
            &*expected
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
        );
        assert_eq!((store.lookup.prefetched(), store.lookup.late()), (0, 1));
        store.rows.as_ref().unwrap().wait_idle();
        assert!(
            store.rows.as_ref().unwrap().take(&keys[0]).is_none(),
            "nothing for the key may stay behind"
        );
    }

    /// PLE rows of the GGUF fixture through the reader threads against the
    /// direct importer, bit for bit, with every requested row consumed.
    #[test]
    fn prefetched_ple_rows_match_the_direct_importer_and_are_consumed() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp/model.gguf");
        let mut store = Store::open_fixture(&path, None).unwrap();
        let name = "per_layer_token_embd.weight";
        let d = store.descriptor(name).unwrap().clone();
        let total = d.rows().unwrap();
        let ids: Vec<usize> = (0..16usize)
            .map(|h| (h * 7919 + 13) % total)
            .chain([13 % total])
            .collect();
        let distinct = ids.iter().collect::<HashSet<_>>().len() as u64;
        let expected = ids
            .iter()
            .map(|&id| d.import_rows(id, 1).unwrap().dense().unwrap())
            .collect::<Vec<_>>();
        let expected = MxArray::concatenate_many(expected.iter().collect(), Some(0))
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        let keys: Vec<_> = ids.iter().map(|&id| (name.to_string(), id)).collect();
        store.prefetch_rows(&keys).unwrap();
        let fetched = store.lookup_rows(name, &ids).unwrap();
        assert!(!unsafe { mlx_sys::mlx_array_is_available(fetched.as_raw_ptr()) });
        assert_eq!(
            &*fetched
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap(),
            &expected
        );
        assert_eq!(
            (store.lookup.lookups(), store.lookup.misses()),
            (distinct, distinct)
        );
        assert_eq!(
            store.lookup.prefetched() + store.lookup.late(),
            distinct,
            "every imported row must come from the request"
        );
    }

    #[test]
    fn partial_banks_share_the_protected_allowance() {
        let mut store = Store::open_metadata(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        store.bank_bytes = store.cache_limit * 3 / 4 - 768;
        let descriptor = store
            .descriptor("layers.0.attn_hyper_connection.hc_norm.weight")
            .unwrap()
            .clone();
        for i in 0..32 {
            let name = format!("mtp.test_norm_{i}.weight");
            store.tensors.insert(name.clone(), descriptor.clone());
            store.read(&name, 0, descriptor.rows().unwrap()).unwrap();
        }
        assert!(store.protected_bytes > 0);
        assert!(store.protected_bytes <= 768);
        assert!(!store.order.is_empty());
    }

    #[test]
    fn auxiliary_protection_reserves_space_after_pinned_main_weights() {
        let mut store = Store::open_metadata(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        // Pinned banks leave 768 bytes of the protected allowance.
        store.bank_bytes = CACHE_BYTES * 3 / 4 - 768;
        let descriptor = store
            .descriptor("layers.0.attn_hyper_connection.hc_norm.weight")
            .unwrap()
            .clone();
        for i in 0..32 {
            let name = format!("mtp.test_norm_{i}.weight");
            store.tensors.insert(name.clone(), descriptor.clone());
            store.read(&name, 0, descriptor.rows().unwrap()).unwrap();
        }
        assert!(store.protected_bytes > 0);
        assert!(store.protected_bytes <= 768);
        assert!(
            !store.order.is_empty(),
            "auxiliary weights must remain evictable"
        );
        store.evict_to(store.protected_bytes).unwrap();
        assert_eq!(store.cache_bytes, store.protected_bytes);
    }

    #[test]
    fn expert_lru_preserves_recency_and_bounds_hot_hit_metadata() {
        let mut store = Store::open_fixture(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        let key = |row| ("embed_tokens.weight".to_string(), row, 1);
        store.read("embed_tokens.weight", 0, 1).unwrap();
        let row_bytes = store.cache_bytes;
        store.read("embed_tokens.weight", 1, 1).unwrap();
        for _ in 0..1000 {
            store.read("embed_tokens.weight", 0, 1).unwrap();
        }
        assert!(store.order.len() <= 64);
        store.evict_to(row_bytes).unwrap();
        assert!(store.cache.contains_key(&key(0)));
        assert!(!store.cache.contains_key(&key(1)));
        store.read("embed_tokens.weight", 2, 1).unwrap();
        store.evict_to(row_bytes).unwrap();
        assert!(store.cache.contains_key(&key(2)));
        assert!(!store.cache.contains_key(&key(0)));
        store.evict_to(0).unwrap();
        assert!(store.cache.is_empty());
    }
}

mod direct_tests {
    use super::super::*;
    use crate::models::qwen4_exp::memory;

    /// A fixture directory whose test plans against [`memory::fixture_host`],
    /// never the runner's free memory.
    struct Temp(PathBuf, #[expect(dead_code)] memory::TestHostGuard);
    impl Temp {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("qwen4-direct-store-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&path).unwrap();
            Self(path, memory::scoped_fixture_host())
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    const EXPERTS: usize = 5;
    type Part = (&'static str, usize, usize, u32, usize);
    const PARTS: [Part; 3] = [
        // (name, rows per expert, width, GGUF type, block bytes per row)
        ("blk.0.ffn_gate_exps.weight", 32, 256, 12, 144),
        ("blk.0.ffn_up_exps.weight", 32, 256, 12, 144),
        ("blk.0.ffn_down_exps.weight", 128, 64, 7, 48),
    ];
    const Q8_PARTS: [Part; 3] = [
        PARTS[0],
        PARTS[1],
        ("blk.0.ffn_down_exps.weight", 128, 64, 8, 68),
    ];
    const Q5K_PARTS: [Part; 3] = [
        ("blk.0.ffn_gate_exps.weight", 32, 256, 13, 176),
        ("blk.0.ffn_up_exps.weight", 32, 256, 13, 176),
        PARTS[2],
    ];
    const FLASH_PARTS: [Part; 3] = [
        ("blk.0.ffn_gate_exps.weight", 640, 2560, 12, 1440),
        ("blk.0.ffn_up_exps.weight", 640, 2560, 12, 1440),
        ("blk.0.ffn_down_exps.weight", 2560, 640, 7, 480),
    ];

    /// Three routed expert tensors of one layer: Q4_K gate/up and Q5_1 down,
    /// filled with deterministic pseudo-random block bytes.
    fn write_fixture(path: &Path) {
        write_fixture_layers(path, 1)
    }

    /// The same three tensors for layers `0..layers`.
    fn write_fixture_layers(path: &Path, layers: usize) {
        write_fixture_layout(path, layers, EXPERTS, &PARTS, 0, 0);
    }

    fn write_fixture_layout(
        path: &Path,
        layers: usize,
        experts: usize,
        parts: &[Part; 3],
        header_pad: usize,
        trailing: usize,
    ) {
        write_fixture_with_fixed(path, layers, experts, parts, &[], header_pad, trailing)
    }

    /// The routed parts of `write_fixture_layout`, then `fixed` F32
    /// projections of `(name, rows, width)`.
    fn write_fixture_with_fixed(
        path: &Path,
        layers: usize,
        experts: usize,
        parts: &[Part; 3],
        fixed: &[(&str, usize, usize)],
        header_pad: usize,
        trailing: usize,
    ) {
        let string = |bytes: &mut Vec<u8>, value: &str| {
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.bytes());
        };
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(((parts.len() * layers + fixed.len()) as u64).to_le_bytes());
        b.extend(1u64.to_le_bytes());
        string(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        string(&mut b, "qwen4exp");
        let mut offset = 0u64;
        for layer in 0..layers {
            for (name, rows, width, ty, row_bytes) in parts {
                string(&mut b, &name.replace("blk.0.", &format!("blk.{layer}.")));
                b.extend(3u32.to_le_bytes());
                for d in [*width as u64, *rows as u64, experts as u64] {
                    b.extend(d.to_le_bytes());
                }
                b.extend(ty.to_le_bytes());
                b.extend(offset.to_le_bytes());
                offset += (experts * *rows * *row_bytes) as u64;
            }
        }
        for (name, rows, width) in fixed {
            string(&mut b, name);
            b.extend(2u32.to_le_bytes());
            for d in [*width as u64, *rows as u64] {
                b.extend(d.to_le_bytes());
            }
            b.extend(0u32.to_le_bytes());
            b.extend(offset.to_le_bytes());
            offset += (*rows * *width * 4) as u64;
        }
        b.resize(b.len() + header_pad, 0);
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        let mut state = 0x2545_f491u32;
        for _ in 0..offset {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            b.push((state >> 24) as u8);
        }
        b.resize(b.len() + trailing, 0x5a);
        fs::write(path, b).unwrap();
    }

    fn write_split_fixture_layout(
        dir: &Path,
        label: &str,
        shards: usize,
        layers: usize,
        experts: usize,
        parts: &[Part; 3],
    ) -> Vec<PathBuf> {
        write_split_fixture_layout_with_empty_first(
            dir, label, shards, layers, experts, parts, false,
        )
    }

    fn write_split_fixture_layout_with_empty_first(
        dir: &Path,
        label: &str,
        shards: usize,
        layers: usize,
        experts: usize,
        parts: &[Part; 3],
        empty_first: bool,
    ) -> Vec<PathBuf> {
        assert!(shards > 1);
        let paths: Vec<PathBuf> = (1..=shards)
            .map(|ordinal| dir.join(format!("{label}-{ordinal:05}-of-{shards:05}.gguf")))
            .collect();
        let mut assignments = vec![Vec::new(); shards];
        for (index, (layer, part)) in (0..layers)
            .flat_map(|layer| (0..parts.len()).map(move |part| (layer, part)))
            .enumerate()
        {
            let target = if empty_first {
                (index % (shards - 1)) + 1
            } else {
                index % shards
            };
            assignments[target].push((layer, part));
        }
        let string = |bytes: &mut Vec<u8>, value: &str| {
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.bytes());
        };
        let mut state = 0x2545_f491u32;
        for (shard, assigned) in assignments.into_iter().enumerate() {
            let mut b = b"GGUF".to_vec();
            b.extend(3u32.to_le_bytes());
            b.extend((assigned.len() as u64).to_le_bytes());
            b.extend(7u64.to_le_bytes());
            string(&mut b, "general.architecture");
            b.extend(8u32.to_le_bytes());
            string(&mut b, "qwen4exp");
            for (key, value) in [
                ("split.no", shard as u64),
                ("split.count", shards as u64),
                ("split.tensors.count", (layers * parts.len()) as u64),
                ("qwen4exp.expert_count", experts as u64),
                ("qwen4exp.expert_used_count", 2),
            ] {
                string(&mut b, key);
                b.extend(10u32.to_le_bytes());
                b.extend(value.to_le_bytes());
            }
            string(&mut b, "general.alignment");
            b.extend(4u32.to_le_bytes());
            b.extend(32u32.to_le_bytes());
            let mut offset = 0u64;
            for &(layer, part) in &assigned {
                let (name, rows, width, ty, row_bytes) = parts[part];
                string(&mut b, &name.replace("blk.0.", &format!("blk.{layer}.")));
                b.extend(3u32.to_le_bytes());
                for d in [width as u64, rows as u64, experts as u64] {
                    b.extend(d.to_le_bytes());
                }
                b.extend(ty.to_le_bytes());
                b.extend(offset.to_le_bytes());
                offset += (experts * rows * row_bytes) as u64;
            }
            while !b.len().is_multiple_of(32) {
                b.push(0);
            }
            for _ in 0..offset {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                b.push((state >> 24) as u8);
            }
            fs::write(&paths[shard], b).unwrap();
        }
        paths
    }

    fn open_split_slots(
        temp: &Temp,
        label: &str,
        parts: &[Part; 3],
        layers: usize,
        shards: usize,
    ) -> (Store, Vec<PathBuf>) {
        let paths = write_split_fixture_layout(&temp.0, label, shards, layers, EXPERTS, parts);
        let mut store =
            Store::open_fixture(&paths[0], Some(&temp.0.join(format!("{label}-cache")))).unwrap();
        store.slot_capacity = Some(EXPERTS);
        (store, paths)
    }

    /// Reads are aligned in every shard they touch. The whole-store fill's
    /// 8 MiB windows cross a shard boundary (`must_cross`); a slot fill reads
    /// its experts' pages only, which need not.
    fn assert_split_read_windows(store: &Store, must_cross: bool) {
        let source = store.gguf_source.as_ref().unwrap();
        assert!(source.shards.len() > 1);
        assert!(!store.direct_read_windows.is_empty());
        let mut crossed = false;
        for &(off, len) in &store.direct_read_windows {
            let len = len as u64;
            assert!(len <= 64 << 20 && off % (16 << 10) == 0);
            for shard in &source.shards {
                let actual_start = off.max(shard.base);
                let actual_end = (off + len).min(shard.base + shard.len);
                if actual_start < actual_end {
                    assert_eq!((actual_start - shard.base) % (16 << 10), 0);
                }
                if off < shard.base && off + len > shard.base {
                    crossed = true;
                }
            }
        }
        assert!(
            crossed || !must_cross,
            "no direct read window crossed a shard boundary"
        );
    }

    /// Every slot-fill read of a single-file store is one 64 MiB batch or
    /// less, 16 KiB aligned, and covers only pages of the experts its
    /// partial leases wanted: no gap between two of them is read.
    fn assert_reads_hold_only_leased_experts(store: &Store) {
        assert!(!store.direct_read_windows.is_empty());
        assert!(!store.layer_leases.is_empty());
        let align = 16u64 << 10;
        let mut pages: Vec<(u64, u64)> = Vec::new();
        for (&layer, (leased, _)) in &store.layer_leases {
            for part in ["gate", "up", "down"] {
                let tensor = store
                    .descriptor(&format!("blk.{layer}.ffn_{part}_exps.weight"))
                    .unwrap();
                let rows = tensor.shape[1];
                for &expert in leased {
                    let (offset, bytes) = tensor.range(expert as usize * rows, rows).unwrap();
                    let end = (offset + bytes).next_multiple_of(align);
                    pages.push((offset - offset % align, end));
                }
            }
        }
        for &(offset, len) in &store.direct_read_windows {
            assert!(
                len <= 64 << 20 && offset % align == 0,
                "read {offset} + {len}"
            );
            let mut at = offset;
            while at < offset + len as u64 {
                let page = pages
                    .iter()
                    .find(|&&(lo, hi)| lo <= at && at < hi)
                    .unwrap_or_else(|| {
                        panic!("read {offset} + {len} covers byte {at} of no leased expert")
                    });
                at = page.1.min(offset + len as u64);
            }
        }
    }

    fn encoded(weight: &Weight) -> (Vec<u8>, i32, i32, String) {
        let mut data = Vec::new();
        for array in [
            Some(&weight.values),
            weight.scales.as_ref(),
            weight.biases.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let dtype = array.dtype().unwrap();
            data.extend(format!("{dtype:?}:{:?};", array.shape().unwrap().to_vec()).as_bytes());
            match dtype {
                DType::Uint32 => data.extend(
                    array
                        .to_uint32()
                        .unwrap()
                        .iter()
                        .flat_map(|value| value.to_le_bytes()),
                ),
                DType::Float32 => data.extend(
                    array
                        .to_float32()
                        .unwrap()
                        .iter()
                        .flat_map(|value| value.to_le_bytes()),
                ),
                DType::Float16 | DType::BFloat16 => data.extend(
                    array
                        .to_uint16_native()
                        .unwrap()
                        .iter()
                        .flat_map(|value| value.to_le_bytes()),
                ),
                DType::Uint8 => data.extend(array.to_uint8().unwrap()),
                DType::Int8 => data.extend(array.to_int8().unwrap().into_iter().map(|v| v as u8)),
                other => panic!("unsupported test oracle dtype {other:?}"),
            }
        }
        (data, weight.group, weight.bits, weight.mode.clone())
    }

    #[test]
    fn routed_expert_names_are_the_only_direct_bank_sources() {
        assert_eq!(
            super::super::routed_expert_tensor("blk.47.ffn_down_exps.weight"),
            Some((47, 2))
        );
        assert_eq!(
            super::super::routed_expert_tensor("blk.3.ffn_down_shexp.weight"),
            None
        );
        assert_eq!(
            super::super::routed_expert_tensor("nextn.0.ffn_gate_exps.weight"),
            None
        );
    }

    fn open_slots(temp: &Temp, gguf: &Path, label: &str, capacity: usize, direct: bool) -> Store {
        let mut store =
            Store::open_fixture(gguf, Some(&temp.0.join(format!("{label}-cache")))).unwrap();
        store.slot_capacity = Some(capacity);
        store.direct_gguf = direct;
        assert!(!store.plan.partial_direct_gate);
        store
    }

    fn open_full(temp: &Temp, gguf: &Path, label: &str, layers: usize, experts: usize) -> Store {
        let mut store =
            Store::open_fixture(gguf, Some(&temp.0.join(format!("{label}-cache")))).unwrap();
        store.plan_fixture_store().unwrap();
        let handled = store
            .select_direct_residency_with(experts, 0, &mut memory::direct_gate, &mut |_| Ok(()))
            .unwrap();
        assert!(
            handled,
            "the single-file fixture must take direct selection"
        );
        assert!(store.plan.direct);
        assert!(!store.plan.partial_direct_gate);
        assert_eq!(store.plan.direct_layers, layers);
        assert_eq!(store.plan.partial_expert_slots, experts);
        assert_eq!(store.plan.direct_bytes, store.slot_bytes);
        assert_eq!(store.cache_bytes, store.slot_bytes);
        assert_eq!(store.slots.len(), layers);
        store
    }

    fn raw_fill_bytes(store: &Store, parts: &[Part; 3], ids: &[u32]) -> u64 {
        ids.iter()
            .flat_map(|&expert| {
                parts.iter().map(move |&(name, rows, ..)| {
                    store
                        .descriptor(name)
                        .unwrap()
                        .range(expert as usize * rows, rows)
                        .unwrap()
                        .1
                })
            })
            .sum()
    }

    /// Compare every value/scale/bias byte of every requested filled slot
    /// with the ordinary CPU checkpoint importer for that expert.
    fn assert_banks_match_checkpoint(
        store: &Store,
        parts: &[Part; 3],
        banks: &[Arc<Weight>; 3],
        ids: &[u32],
        local: &[u32],
    ) -> Vec<u8> {
        assert_banks_match_checkpoint_layer(store, 0, parts, banks, ids, local)
    }

    fn assert_banks_match_checkpoint_layer(
        store: &Store,
        layer: usize,
        parts: &[Part; 3],
        banks: &[Arc<Weight>; 3],
        ids: &[u32],
        local: &[u32],
    ) -> Vec<u8> {
        let mut all = Vec::new();
        for (position, &expert) in ids.iter().enumerate() {
            for (part, &(name, rows, ..)) in parts.iter().enumerate() {
                let name = name.replace("blk.0.", &format!("blk.{layer}."));
                let expected = store
                    .descriptor(&name)
                    .unwrap()
                    .import_rows(expert as usize * rows, rows)
                    .unwrap();
                let actual = banks[part]
                    .slice_rows(local[position] as usize * rows, rows)
                    .unwrap();
                let actual = encoded(&actual);
                assert_eq!(actual, encoded(&expected), "{name} expert {expert}");
                all.extend(actual.0);
            }
        }
        all
    }

    #[test]
    fn direct_gguf_split_reader_preserves_bytes_gaps_and_bounds() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let paths = write_split_fixture_layout_with_empty_first(
            &temp.0,
            "split-reader",
            4,
            2,
            EXPERTS,
            &PARTS,
            true,
        );
        assert!(parse_gguf(&paths[0]).unwrap().tensors.is_empty());
        let source = DirectGgufSource::new(&paths).unwrap();
        let files = source.open_files().unwrap();
        let refs: Vec<&File> = files.iter().collect();
        let reader = DirectGgufReader::new(&source, &refs).unwrap();
        let mut actual = vec![0u8; source.len as usize];
        crate::utils::gguf_read::ReadAt::read_exact_at(&reader, &mut actual, 0).unwrap();
        for (index, shard) in source.shards.iter().enumerate() {
            let begin = shard.base as usize;
            let end = begin + shard.len as usize;
            assert_eq!(&actual[begin..end], fs::read(&paths[index]).unwrap());
            if let Some(next) = source.shards.get(index + 1) {
                assert!(
                    actual[end..next.base as usize]
                        .iter()
                        .all(|&byte| byte == 0)
                );
            }
        }
        let error =
            crate::utils::gguf_read::ReadAt::read_exact_at(&reader, &mut [0u8; 2], source.len - 1)
                .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn direct_gguf_split_formats_match_checkpoint_and_cross_shards() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        for (label, parts) in [
            ("split-q4k-q5-1", &PARTS),
            ("split-q5k", &Q5K_PARTS),
            ("split-q8-bias", &Q8_PARTS),
        ] {
            let (mut store, _) = open_split_slots(&temp, label, parts, 2, 3);
            let ids: Vec<u32> = (0..EXPERTS as u32).collect();
            for layer in 0..2 {
                let (banks, local, _) = store
                    .expert_slots(layer, EXPERTS, 2, &ids)
                    .unwrap()
                    .unwrap();
                assert_banks_match_checkpoint_layer(&store, layer, parts, &banks, &ids, &local);
            }
            assert_split_read_windows(&store, false);
        }
    }

    #[test]
    fn direct_gguf_split_full_capacity_matches_checkpoint() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let paths = write_split_fixture_layout_with_empty_first(
            &temp.0,
            "split-full",
            4,
            2,
            EXPERTS,
            &PARTS,
            true,
        );
        let mut store = open_full(&temp, &paths[0], "split-full-direct", 2, EXPERTS);
        let ids: Vec<u32> = (0..EXPERTS as u32).collect();
        store.begin_turn().unwrap();
        for layer in 0..2 {
            let (banks, local, capacity) = store
                .expert_slots(layer, EXPERTS, 2, &ids)
                .unwrap()
                .unwrap();
            assert_eq!(capacity, EXPERTS);
            assert_eq!(local, ids);
            assert_banks_match_checkpoint_layer(&store, layer, &PARTS, &banks, &ids, &local);
        }
        assert!(store.device_slots_ready(2));
        assert_eq!(store.slot_misses, 0);
        assert!(store.plan.direct);
        assert_eq!(store.plan.direct_layers, 2);
        assert_eq!(store.plan.direct_bytes, store.slot_bytes);
        assert_eq!(store.slot_upload_bytes, store.slot_bytes);
        assert_split_read_windows(&store, true);
    }

    #[test]
    fn direct_gguf_split_source_failures_name_the_failed_shard() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let (mut flipped, paths) = open_split_slots(&temp, "split-flip", &PARTS, 1, 3);
        let parsed = parse_gguf(&paths[1]).unwrap();
        let tensor = parsed.tensors.first().unwrap();
        let at = (parsed.data_offset + tensor.offset) as usize + 8;
        let mut bytes = fs::read(&paths[1]).unwrap();
        bytes[at] ^= 0x20;
        fs::write(&paths[1], bytes).unwrap();
        let error = flipped.expert_slots(0, EXPERTS, 2, &[0]).err().unwrap();
        assert!(error.reason.contains("changed since open"), "{error}");
        assert!(
            error.reason.contains("split-flip-00002-of-00003.gguf"),
            "{error}"
        );
        assert!(!flipped.device_slots_ready(1));

        let (mut missing, paths) = open_split_slots(&temp, "split-missing", &PARTS, 1, 3);
        fs::remove_file(&paths[2]).unwrap();
        let error = missing.expert_slots(0, EXPERTS, 2, &[0]).err().unwrap();
        assert!(error.reason.contains("Failed to stat GGUF"), "{error}");
        assert!(
            error.reason.contains("split-missing-00003-of-00003.gguf"),
            "{error}"
        );
        assert!(!missing.device_slots_ready(1));

        let (mut truncated, paths) = open_split_slots(&temp, "split-truncated", &PARTS, 1, 3);
        let len = fs::metadata(&paths[1]).unwrap().len();
        fs::OpenOptions::new()
            .write(true)
            .open(&paths[1])
            .unwrap()
            .set_len(len - 1)
            .unwrap();
        let error = truncated.expert_slots(0, EXPERTS, 2, &[0]).err().unwrap();
        assert!(error.reason.contains("truncated"), "{error}");
        assert!(
            error.reason.contains("split-truncated-00002-of-00003.gguf"),
            "{error}"
        );
        assert!(!truncated.device_slots_ready(1));
    }

    #[test]
    fn direct_gguf_incomplete_or_reordered_splits_fail_before_residency() {
        let temp = Temp::new();
        let paths = write_split_fixture_layout(&temp.0, "split-incomplete", 3, 1, EXPERTS, &PARTS);
        fs::remove_file(&paths[1]).unwrap();
        let error = match Store::open_fixture(&paths[0], Some(&temp.0.join("incomplete-cache"))) {
            Ok(_) => panic!("an incomplete split set was accepted"),
            Err(error) => error,
        };
        assert!(error.reason.contains("Expected one GGUF split"), "{error}");
        assert!(
            error.reason.contains("split-incomplete-00002-of-00003"),
            "{error}"
        );

        let paths = write_split_fixture_layout(&temp.0, "split-reordered", 3, 1, EXPERTS, &PARTS);
        let held = temp.0.join("split-reordered.held");
        fs::rename(&paths[0], &held).unwrap();
        fs::rename(&paths[1], &paths[0]).unwrap();
        fs::rename(&held, &paths[1]).unwrap();
        let error = match Store::open_fixture(&paths[0], Some(&temp.0.join("reordered-cache"))) {
            Ok(_) => panic!("a reordered split set was accepted"),
            Err(error) => error,
        };
        assert!(
            error.reason.contains("Inconsistent GGUF split metadata")
                || error.reason.contains("Open the first GGUF split"),
            "{error}"
        );
    }

    /// A host that refuses the whole bank set the plan chose lowers the
    /// allocation to partial slots of what it admits; the plan keeps the store
    /// at its cap, so the bytes the slots leave are held back and the window
    /// does not move. Mutation: lower the weight budget to the admitted banks
    /// on a refusal in `select_direct_residency_with` → red (cache limit
    /// 2,147,514,880).
    #[test]
    fn host_refusal_lowers_the_allocation_not_the_plan() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("full-gate-refusal.gguf");
        write_fixture(&gguf);
        let mut store = Store::open_fixture(&gguf, Some(&temp.0.join("cache"))).unwrap();
        store.plan_fixture_store().unwrap();
        let planned_limit = store.cache_limit;
        let handled = store
            .select_direct_residency_with(
                EXPERTS,
                0,
                &mut |plan, _, bank_bytes| {
                    plan.bank_budget = bank_bytes * 2 / EXPERTS as u64;
                    plan.live_non_reclaimable = 40 << 30;
                    plan.available_bytes = Some(88 << 30);
                    plan.physical_bytes = Some(128 << 30);
                    Err(err(
                        "simulated full-bank refusal beside a 40 GiB live-pressure row",
                    ))
                },
                &mut |_| panic!("a refused gate must not allocate"),
            )
            .unwrap();
        assert!(handled);
        assert!(!store.plan.direct);
        assert!(store.plan.partial_direct_gate);
        assert!(store.slots.is_empty());
        assert_eq!(store.slot_bytes, 0);
        assert_eq!(store.cache_bytes, 0);
        assert_eq!(store.expert_slot_capacity(EXPERTS, 2).unwrap(), 2);
        let bank_budget = store.plan.bank_budget;
        assert_eq!(store.slot_bank_budget, Some(bank_budget));
        assert_eq!(store.cache_limit, planned_limit);
        assert_eq!(store.plan.budget, planned_limit);
        assert!(store.held_back_bytes() > 0);
        assert!(store.planned_slot_bytes().unwrap() <= bank_budget);
        assert_eq!(store.plan.live_non_reclaimable, 40 << 30);
        assert_eq!(store.plan.direct_bytes, 0);
        assert_eq!(store.plan.direct_layers, 0);
        assert!(matches!(
            store.live.fallback(),
            Some(Fallback::LoadGate { budget, .. }) if budget == bank_budget
        ));
    }

    /// A host that refuses the whole store the plan chose prepares the partial
    /// store under what it admits now: the fixed banks pass the host, and the
    /// plan keeps the store at its cap with the rest held back. Mutation: admit
    /// the store's cap in `prepare_hot_with_admission` → red ("requested 6.0
    /// GiB, the host leaves 5.0 GiB").
    #[test]
    fn host_refusal_partial_store_prepares_under_the_host() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("refusal-prepare.gguf");
        write_fixture_with_fixed(
            &gguf,
            1,
            EXPERTS,
            &PARTS,
            &[("blk.0.attn_qkv.weight", 16, 64)],
            0,
            0,
        );
        let host = memory::HostMemory {
            physical: 16 << 30,
            free: 4 << 30,
            file: 0,
            anonymous: 5 << 30,
            purgeable: 0,
            wired: 1 << 30,
            compressor: 1 << 30,
        };
        let _host = memory::scoped_test_host(host);
        let mut store = Store::open_fixture(&gguf, Some(&temp.0.join("cache"))).unwrap();
        store.plan.physical_bytes = Some(host.physical);
        store.plan_fixture_store().unwrap();
        let whole = store.store_plan.unwrap();
        assert!(whole.whole);
        let fixed = store.cache_limit - whole.expert_cap_bytes - whole.row_cache_bytes;
        assert!(fixed > 0, "the fixture must hold a fixed projection bank");
        let rows = 6u64 << 30;
        store.set_store_plan(
            crate::engine::memory_plan::StorePlan {
                row_cache_bytes: rows,
                ..whole
            },
            fixed,
        );
        let planned = store.cache_limit;
        assert!(
            planned > 5 << 30,
            "the planned store must exceed what the host admits, or this test proves nothing"
        );
        assert!(
            store
                .select_direct_residency_with(
                    EXPERTS,
                    fixed + rows,
                    &mut memory::direct_gate,
                    &mut |_| panic!("a refused gate must not allocate"),
                )
                .unwrap()
        );
        assert!(!store.plan.direct);
        assert!(store.plan.partial_direct_gate);
        store.prepare_hot().unwrap();
        assert!(store.banks.contains_key("blk.0.attn_qkv.weight"));
        assert_eq!(store.cache_limit, planned);
        assert!(store.held_back_bytes() > 0);
        assert!(matches!(
            store.live.fallback(),
            Some(Fallback::LoadGate { .. })
        ));
    }

    /// A streamed store is sized by its plan whatever the host holds: a quiet
    /// and a busy host give the same weight budget, slot budget and slots.
    /// Mutation: cap the streamed slot budget with the host's bank gate in
    /// `select_direct_residency_with` → red (a zero slot budget and no slots
    /// on the busy host).
    #[test]
    fn store_cap_ignores_the_host() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("store-cap.gguf");
        write_fixture(&gguf);
        let quiet = memory::HostMemory {
            physical: 16 << 30,
            free: 8 << 30,
            file: 4 << 30,
            anonymous: 1 << 30,
            purgeable: 0,
            wired: 2 << 30,
            compressor: 1 << 30,
        };
        let busy = memory::HostMemory {
            physical: 8 << 30,
            free: 4 << 30,
            file: 0,
            ..quiet
        };
        let sized = |host: memory::HostMemory, label: &str| {
            let _host = memory::scoped_test_host(host);
            let mut store =
                Store::open_fixture(&gguf, Some(&temp.0.join(format!("{label}-cache")))).unwrap();
            store.plan.physical_bytes = Some(host.physical);
            store.set_store_plan(
                crate::engine::memory_plan::StorePlan {
                    whole: false,
                    expert_cap_bytes: 1 << 40,
                    row_cache_bytes: 2 << 30,
                },
                1 << 30,
            );
            assert!(
                store
                    .select_direct_residency_with(
                        EXPERTS,
                        1 << 30,
                        &mut memory::direct_gate,
                        &mut |_| panic!("a streamed store allocates no full banks"),
                    )
                    .unwrap()
            );
            (
                store.cache_limit,
                store.slot_bank_budget,
                store.plan.partial_expert_slots,
            )
        };
        let at_quiet = sized(quiet, "quiet");
        assert_eq!(at_quiet.0, (1 << 40) + (3 << 30));
        assert_eq!(at_quiet.2, EXPERTS);
        assert_eq!(sized(busy, "busy"), at_quiet);
    }

    #[test]
    fn direct_gguf_zero_bank_gate_refusal_marks_partial_gate() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("zero-bank-gate-refusal.gguf");
        write_fixture(&gguf);
        let mut store = Store::open_fixture(&gguf, Some(&temp.0.join("cache"))).unwrap();
        store.plan_fixture_store().unwrap();
        let planned_limit = store.cache_limit;
        let handled = store
            .select_direct_residency_with(
                EXPERTS,
                0,
                &mut |plan, _, _| {
                    plan.bank_budget = 0;
                    Err(err("injected zero-bank full-gate refusal"))
                },
                &mut |_| panic!("a refused gate must not allocate"),
            )
            .unwrap();
        assert!(handled);
        assert!(!store.plan.direct);
        assert!(store.plan.partial_direct_gate);
        assert!(store.plan.use_direct_turn_floor());
        assert_eq!(store.plan.budget, planned_limit);
        assert_eq!(store.plan.bank_budget, 0);
        assert_eq!(store.slot_bank_budget, Some(0));
        assert_eq!(store.plan.partial_expert_slots, 0);
    }

    #[test]
    fn direct_gguf_allocation_rollback_restores_ordinary_gate_marker() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("full-allocation-rollback.gguf");
        write_fixture(&gguf);
        let mut store = Store::open_fixture(&gguf, Some(&temp.0.join("cache"))).unwrap();
        store.plan_fixture_store().unwrap();
        let original_budget = store.plan.budget;
        let original_cache_limit = store.cache_limit;
        let error = store
            .select_direct_residency_with(EXPERTS, 0, &mut memory::direct_gate, &mut |_| {
                Err(err("injected full-bank allocation refusal"))
            })
            .unwrap_err();
        assert!(error.reason.contains("full-bank allocation refusal"));
        assert!(!store.plan.direct);
        assert!(!store.plan.partial_direct_gate);
        assert!(!store.plan.use_direct_turn_floor());
        assert_eq!(store.plan.budget, original_budget);
        assert_eq!(store.cache_limit, original_cache_limit);
        assert!(store.slot_bank_budget.is_none());
        assert!(store.slots.is_empty());
    }

    /// Real fixture inference through the partial direct fill path. The gate
    /// rows are injected so the two capacities and the 40/20 GiB pressure
    /// cases are deterministic; they are not real-model throughput rows.
    #[test]
    fn direct_partial_slot_text_matches_cpu_import_at_two_capacities() {
        use super::super::super::decoder;
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let path = temp.0.join("quantized-fixture.gguf");
        let base_config = super::super::super::tests::write_direct_gguf_fixture(&path);
        let experts = base_config.num_experts;
        let make_store = || Store::open_fixture(&path, Some(&temp.0.join("unused-cache"))).unwrap();
        let prompt = [3u32, 9, 4, 7, 1, 5, 8, 2, 3, 6, 9, 4];
        // Oracle: the same checkpoint through the bounded per-row CPU
        // importer, with no direct slots or derived pack involved.
        let store = make_store();
        let mut oracle = decoder::Decoder::new(base_config.clone(), store).unwrap();
        oracle.register_cache_limit();
        oracle.weights.prepare_fixture_hot().unwrap();
        oracle.weights.slot_capacity = Some(0);
        oracle.weights.publish_live();
        oracle.residency_changed();
        oracle.weights.begin_turn().unwrap();
        let mut logits = oracle.prefill_prompt(&prompt).unwrap();
        let mut expected = Vec::new();
        for step in 0..6 {
            let id = super::super::super::tests::argmax(&logits);
            expected.push(id);
            if step < 5 {
                logits = oracle.step(id).unwrap();
            }
        }

        for (capacity, live_gib) in [(2usize, 40u64), (3, 20)] {
            let store = make_store();
            let mut decoder = decoder::Decoder::new(base_config.clone(), store).unwrap();
            decoder.weights.plan_fixture_store().unwrap();
            let planned_limit = decoder.weights.cache_limit;
            decoder.register_cache_limit();
            let handled = decoder
                .weights
                .select_direct_residency_with(
                    experts,
                    0,
                    &mut |plan, _, bank_bytes| {
                        plan.bank_budget = bank_bytes * capacity as u64 / experts as u64;
                        plan.live_non_reclaimable = live_gib << 30;
                        plan.available_bytes = Some((128 - live_gib) << 30);
                        plan.physical_bytes = Some(128 << 30);
                        Err(err("injected partial-capacity gate row"))
                    },
                    &mut |_| panic!("partial selection must not allocate full banks"),
                )
                .unwrap();
            assert!(handled);
            let bank_budget = decoder.weights.plan.bank_budget;
            assert!(bank_budget > 0);
            assert!(!decoder.weights.plan.direct);
            assert_eq!(decoder.weights.plan.partial_expert_slots, capacity);
            assert_eq!(decoder.weights.slot_capacity, Some(capacity));
            assert_eq!(decoder.weights.slot_bank_budget, Some(bank_budget));
            assert_eq!(decoder.weights.plan.live_non_reclaimable, live_gib << 30);
            assert!(decoder.weights.planned_slot_bytes().unwrap() <= bank_budget);
            decoder.weights.prepare_fixture_hot().unwrap();
            decoder.weights.publish_live();
            decoder.residency_changed();
            assert_eq!(decoder.weights.cache_budget(), planned_limit);

            decoder.weights.begin_turn().unwrap();
            let mut logits = decoder.prefill_prompt(&prompt).unwrap();
            // MUTATION: lease a window's assignments in chunks of `capacity`
            // in router order → an expert is read twice → red.
            let leases = &decoder.weights.layer_leases;
            assert_eq!(leases.len(), decoder.config.num_hidden_layers);
            for (layer, (leased, misses)) in leases {
                assert!(!leased.is_empty());
                assert_eq!(
                    *misses,
                    leased.len() as u64,
                    "layer {layer}: last-assignment expert groups re-read an expert during prefill"
                );
            }
            let unique = leases
                .values()
                .map(|(leased, _)| leased.len() as u64)
                .sum::<u64>();
            assert_eq!(
                decoder.weights.live.slot_misses.load(Ordering::Relaxed),
                unique,
                "slotMisses counts each prefill expert once"
            );
            assert!(decoder.weights.slot_bytes <= bank_budget);
            assert_reads_hold_only_leased_experts(&decoder.weights);
            let mut actual = Vec::new();
            for step in 0..6 {
                let id = super::super::super::tests::argmax(&logits);
                actual.push(id);
                if step < 5 {
                    logits = decoder.step(id).unwrap();
                }
            }
            assert_eq!(actual, expected, "capacity {capacity}, live {live_gib} GiB");
            assert_eq!(
                decoder.weights.slots.len(),
                decoder.config.num_hidden_layers
            );
            assert!(
                decoder
                    .weights
                    .device_slots_ready(decoder.config.num_hidden_layers)
            );
            assert!(decoder.weights.slot_bytes <= bank_budget);
            assert!(decoder.weights.cache_bytes <= decoder.weights.cache_limit);
            assert!(decoder.weights.peak_cache_bytes <= decoder.weights.cache_limit);
            // MUTATION: a slot miss left out of the live counters → red.
            let live = &decoder.weights.live;
            assert!(decoder.weights.slot_misses > 0 && decoder.weights.bytes_read > 0);
            assert_eq!(
                [
                    live.slot_hits.load(Ordering::Relaxed),
                    live.slot_misses.load(Ordering::Relaxed),
                    live.bytes_read.load(Ordering::Relaxed),
                ],
                [
                    decoder.weights.slot_hits,
                    decoder.weights.slot_misses,
                    decoder.weights.bytes_read,
                ],
                "residencyInfo reads the store's own counters"
            );
            eprintln!(
                "synthetic pressure row (not a real-model benchmark): live={live_gib} GiB, capacity={capacity}, prefill unique misses={unique}, slot bytes={}, text={actual:?}",
                decoder.weights.slot_bytes,
            );
        }
    }

    #[test]
    fn direct_gguf_full_capacity_fills_layers_with_identity_and_keeps_them() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        const LAYERS: usize = 3;
        let temp = Temp::new();
        let gguf = temp.0.join("full-capacity.gguf");
        write_fixture_layers(&gguf, LAYERS);
        let mut store = open_full(&temp, &gguf, "full-capacity-direct", LAYERS, EXPERTS);
        let source_bytes: u64 = store
            .tensors
            .iter()
            .filter(|(name, _)| name.contains("_exps.weight"))
            .map(|(_, tensor)| tensor.bytes)
            .sum();
        let ids: Vec<u32> = (0..EXPERTS as u32).collect();
        assert!(!store.device_slots_ready(LAYERS));
        assert!(store.prefill_device_slots_ready(LAYERS));
        store.begin_turn().unwrap();
        assert!(store.direct_fill_pass());
        for layer in [1, 2, 0] {
            let (banks, local, capacity) = store
                .expert_slots(layer, EXPERTS, 2, &ids)
                .unwrap()
                .unwrap();
            assert_eq!(capacity, EXPERTS);
            assert_eq!(local, ids);
            assert_banks_match_checkpoint_layer(&store, layer, &PARTS, &banks, &ids, &local);
        }
        assert!(store.device_slots_ready(LAYERS));
        assert_eq!(store.slot_misses, 0);
        assert_eq!(store.bytes_read, source_bytes);
        assert_eq!(store.slot_upload_bytes, store.slot_bytes);
        assert_eq!(store.cache_bytes, store.slot_bytes);
        assert_eq!(
            store.live.direct_filled_layers.load(Ordering::Acquire),
            LAYERS as u64
        );
        assert!(!store.direct_read_windows.is_empty());
        assert!(
            store
                .direct_read_windows
                .iter()
                .all(|&(off, len)| len <= 8 << 20 && off % (16 << 10) == 0)
        );
        assert!(
            store
                .commit_device_routes(&[0, 1, 2], &[0, 1, 2], 1)
                .unwrap()
        );

        let allocated = store.slot_bytes;
        let first_upload = store.slot_upload_bytes;
        store.begin_turn().unwrap();
        assert!(!store.direct_fill_pass());
        assert!(store.device_slots_ready(LAYERS));
        for layer in 0..LAYERS {
            let (banks, local, _) = store
                .expert_slots(layer, EXPERTS, 2, &ids)
                .unwrap()
                .unwrap();
            assert_eq!(local, ids);
            assert_banks_match_checkpoint_layer(&store, layer, &PARTS, &banks, &ids, &local);
        }
        assert_eq!(store.bytes_read, source_bytes);
        assert_eq!(store.slot_upload_bytes, first_upload);
        assert_eq!(store.slot_bytes, allocated);
        assert_eq!(store.cache_bytes, allocated);
        assert_eq!(store.slot_misses, 0);
        assert_eq!(
            store.live.slot_hits.load(Ordering::Relaxed),
            store.slot_hits,
            "device-route hits reach the live counters"
        );
    }

    #[test]
    fn direct_gguf_full_fill_error_stays_unpublished_and_rejects_changed_source() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("full-source-error.gguf");
        write_fixture(&gguf);
        let mut store = open_full(&temp, &gguf, "full-source-error-direct", 1, EXPERTS);
        let held = temp.0.join("full-source-error.held");
        fs::rename(&gguf, &held).unwrap();
        store.start_direct_fill().unwrap();
        let error = store.expert_slots(0, EXPERTS, 2, &[0]).err().unwrap();
        assert!(error.reason.contains("Failed to open GGUF"), "{error}");
        assert!(!store.device_slots_ready(1));
        fs::rename(&held, &gguf).unwrap();
        store.begin_turn().unwrap();
        let error = store.expert_slots(0, EXPERTS, 2, &[0]).err().unwrap();
        assert!(error.reason.contains("changed since open"), "{error}");
        assert_eq!(store.slot_misses, 0);
        assert!(!store.device_slots_ready(1));
    }

    #[test]
    fn direct_gguf_q5_1_slots_match_cpu_import_and_eviction() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("q5_1.gguf");
        write_fixture_layout(&gguf, 1, EXPERTS, &PARTS, 32, 17);
        let mut store = open_slots(&temp, &gguf, "q5_1-direct", 2, true);
        let ids = [0u32, 1];
        let (banks, local, _) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        let mut compared =
            assert_banks_match_checkpoint(&store, &PARTS, &banks, &ids, &local).len();
        let first_fill = raw_fill_bytes(&store, &PARTS, &ids);
        assert_eq!(store.bytes_read, first_fill);

        // One miss evicts expert 0 and reuses its bank; expert 1 stays warm.
        let next = [2u32, 1];
        let one_fill = raw_fill_bytes(&store, &PARTS, &[2]);
        let (banks, local, _) = store.expert_slots(0, EXPERTS, 2, &next).unwrap().unwrap();
        assert_eq!(local, [0, 1], "the LRU bank was reused");
        compared += assert_banks_match_checkpoint(&store, &PARTS, &banks, &next, &local).len();
        assert_eq!(store.bytes_read, first_fill + one_fill);

        assert!(store.device_slots_ready(1));
        eprintln!("P3.0-d Q5_1 parity: {compared} bank bytes, eviction");
    }

    #[test]
    fn direct_gguf_partial_source_error_invalidates_changed_source() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("partial-source-error.gguf");
        write_fixture(&gguf);
        let mut store = open_slots(&temp, &gguf, "partial-source-error-direct", 2, true);
        store.expert_slots(0, EXPERTS, 2, &[0, 1]).unwrap();
        let held = temp.0.join("partial-source-error.held");
        fs::rename(&gguf, &held).unwrap();
        let error = store.expert_slots(0, EXPERTS, 2, &[2, 3]).err().unwrap();
        assert!(error.reason.contains("Failed to stat GGUF"), "{error}");
        assert!(!store.device_slots_ready(1));

        fs::rename(&held, &gguf).unwrap();
        let before = store.bytes_read;
        let error = store.expert_slots(0, EXPERTS, 2, &[2, 3]).err().unwrap();
        assert!(error.reason.contains("changed since open"), "{error}");
        assert_eq!(store.bytes_read, before);
        assert!(!store.device_slots_ready(1));
    }

    #[test]
    /// A turn after any idle gap keeps the filled banks, non-volatile, and
    /// reads nothing again (Splash `ExpertStore`: no purgeable state, no
    /// timed purge).
    ///
    /// Mutation M9: the next turn empties the banks first (the old 30 s
    /// purge).
    fn idle_gap_keeps_the_expert_banks() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("idle-gap.gguf");
        write_fixture(&gguf);
        let mut store = open_slots(&temp, &gguf, "idle-gap-direct", EXPERTS, true);
        let ids = [0u32, 1];
        store.begin_turn().unwrap();
        store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        let filled = store.bytes_read;
        let reclaimable = crate::array::purgeable::reclaimable_purgeable_bytes();
        store.begin_turn().unwrap();
        let (banks, local, _) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        assert_banks_match_checkpoint(&store, &PARTS, &banks, &ids, &local);
        assert_eq!(store.bytes_read, filled);
        assert_eq!(
            crate::array::purgeable::reclaimable_purgeable_bytes(),
            reclaimable
        );
    }

    #[test]
    fn direct_gguf_source_flips_match_cpu_import_for_q5_1_and_q8_0_bias() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        for (label, parts, flip) in [
            ("q5_1", &PARTS, 8usize),
            ("q5_k", &Q5K_PARTS, 8),
            ("q8_0", &Q8_PARTS, 1),
        ] {
            let gguf = temp.0.join(format!("{label}-flip.gguf"));
            write_fixture_layout(&gguf, 1, EXPERTS, parts, 64, 13);
            let mut before = open_slots(&temp, &gguf, &format!("{label}-before"), EXPERTS, true);
            let ids = [0u32];
            let (banks, local, _) = before.expert_slots(0, EXPERTS, 1, &ids).unwrap().unwrap();
            let before = assert_banks_match_checkpoint(&before, parts, &banks, &ids, &local);

            let parsed = parse_gguf(&gguf).unwrap();
            let tensor = parsed
                .tensors
                .iter()
                .find(|t| t.name == "blk.0.ffn_down_exps.weight")
                .unwrap();
            let at = (parsed.data_offset + tensor.offset) as usize + flip;
            let mut bytes = fs::read(&gguf).unwrap();
            bytes[at] ^= 0x20;
            fs::write(&gguf, &bytes).unwrap();
            let mut after = open_slots(&temp, &gguf, &format!("{label}-after"), EXPERTS, true);
            let (banks, local, _) = after.expert_slots(0, EXPERTS, 1, &ids).unwrap().unwrap();
            let after_bytes = assert_banks_match_checkpoint(&after, parts, &banks, &ids, &local);
            assert_ne!(before, after_bytes, "{label}: source flip had no effect");

            // A source that changes after open is never mixed with the old
            // identity across repeated direct fills.
            let mut bytes = fs::read(&gguf).unwrap();
            bytes[at + 1] ^= 0x04;
            fs::write(&gguf, &bytes).unwrap();
            let error = after.expert_slots(0, EXPERTS, 1, &[1]).err().unwrap();
            assert!(error.reason.contains("changed since open"), "{error}");
            assert!(!after.device_slots_ready(1));
        }
    }

    #[test]
    fn direct_gguf_missing_and_truncated_sources_fail_without_stale_slots() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        for truncated in [false, true] {
            let temp = Temp::new();
            let gguf = temp.0.join("source-error.gguf");
            write_fixture(&gguf);
            let mut store = open_slots(&temp, &gguf, "source-error-direct", EXPERTS, true);
            if truncated {
                let len = fs::metadata(&gguf).unwrap().len();
                fs::OpenOptions::new()
                    .write(true)
                    .open(&gguf)
                    .unwrap()
                    .set_len(len - 1)
                    .unwrap();
            } else {
                fs::remove_file(&gguf).unwrap();
            }
            let error = store.expert_slots(0, EXPERTS, 2, &[0, 1]).err().unwrap();
            if truncated {
                assert!(error.reason.contains("truncated"), "{error}");
            } else {
                assert!(error.reason.contains("Failed to stat GGUF"), "{error}");
            }
            assert!(error.reason.contains("source-error.gguf"), "{error}");
            assert!(!store.device_slots_ready(1));
            assert!(store.direct_read_windows.is_empty());
        }
    }

    #[test]
    fn direct_gguf_kernel_unaligned_offsets_use_cpu_import() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("unaligned.gguf");
        write_fixture(&gguf);
        let mut store = open_slots(&temp, &gguf, "unaligned-direct", EXPERTS, true);
        let names = ["gate", "up", "down"].map(|part| format!("blk.0.ffn_{part}_exps.weight"));
        store.tensors.get_mut(&names[0]).unwrap().offset += 1;
        assert!(!store.direct_layer(&names));
        assert!(store.expert_slots(0, EXPERTS, 2, &[0]).unwrap().is_none());
        store.select_residency(0).unwrap();
        assert_eq!(store.plan.partial_expert_slots, 0);
        assert_eq!(store.slot_capacity, Some(0));
    }

    #[test]
    fn direct_gguf_flash_geometry_matches_checkpoint_and_reads_exact_ranges() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        const EXPERTS: usize = 8;
        let temp = Temp::new();
        let gguf = temp.0.join("flash-geometry.gguf");
        write_fixture_layout(&gguf, 1, EXPERTS, &FLASH_PARTS, 0, 0);
        let mut store = open_slots(&temp, &gguf, "flash-geometry-direct", EXPERTS, true);
        let ids: Vec<u32> = (0..EXPERTS as u32).collect();
        let (banks, local, _) = store.expert_slots(0, EXPERTS, 8, &ids).unwrap().unwrap();
        let compared =
            assert_banks_match_checkpoint(&store, &FLASH_PARTS, &banks, &ids, &local).len();
        assert_eq!(store.bytes_read, 24_576_000);
        assert_eq!(store.slot_upload_bytes, store.slot_bytes);
        // The three tensors' eight experts lie back to back: one read.
        assert_eq!(store.direct_read_windows.len(), 1);
        assert_reads_hold_only_leased_experts(&store);
        eprintln!(
            "P3.0-d Flash geometry parity: {compared} bank bytes in {} aligned reads",
            store.direct_read_windows.len()
        );
    }

    #[test]
    fn direct_full_banks_preserve_identity_routing_and_kernel_results() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("identity.gguf");
        write_fixture(&gguf);
        let mut store = open_full(&temp, &gguf, "identity-direct", 1, EXPERTS);
        store.begin_turn().unwrap();
        let ids = [4u32, 0, 2, 2, 1, 3];
        let (banks, local, capacity) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        assert_eq!(local, ids);
        assert_eq!(capacity, EXPERTS);
        let device = MxArray::from_uint32(&ids, &[2, 3]).unwrap();
        let (_, device_local) = store.device_expert_slots(0, &device).unwrap().unwrap();
        assert_eq!(&*device_local.to_uint32().unwrap(), &ids);
        let (_, safe, count) = store.prefill_device_expert_slots(0, &device).unwrap();
        assert_eq!(&*safe.to_uint32().unwrap(), &ids);
        assert_eq!(count, EXPERTS);
        for _ in 0..3 {
            assert!(store.commit_device_routes(&[0, 0, 0], &ids, 2).unwrap());
        }
        assert_eq!(store.slot_misses, 0);
        assert_eq!(store.slot_hits, 5 + 3 * 5);
        assert_eq!(store.slot_upload_bytes, store.slot_bytes);

        let x = MxArray::from_float32(
            &(0..6 * 256)
                .map(|n| ((n % 97) as f32 - 48.) / 64.)
                .collect::<Vec<_>>(),
            &[6, 1, 256],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
        let selected = MxArray::from_uint32(&ids, &[6]).unwrap();
        let bits = |array: &MxArray| {
            array
                .to_float32()
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        for (part, &(name, rows, ..)) in PARTS.iter().enumerate() {
            let chunks: Vec<_> = (0..EXPERTS)
                .map(|expert| {
                    store
                        .descriptor(name)
                        .unwrap()
                        .import_rows(expert * rows, rows)
                        .map(Arc::new)
                })
                .collect::<Result<_>>()
                .unwrap();
            let oracle = Weight::concatenate_rows(&chunks).unwrap();
            assert_eq!(encoded(&banks[part]), encoded(&oracle), "{name}");
            if part == 0 {
                let actual = banks[0].gather_rows(&x, &selected, EXPERTS, false).unwrap();
                let expected = oracle.gather_rows(&x, &selected, EXPERTS, false).unwrap();
                assert_eq!(bits(&actual), bits(&expected));
                store.finish_expert_slots(0, &actual).unwrap();
                store.complete_expert_window(0, &expected).unwrap();
            }
        }
        assert!(
            store
                .expert_slots(0, EXPERTS, 2, &[EXPERTS as u32])
                .is_err()
        );
        assert!(
            store
                .commit_device_routes(&[0], &[EXPERTS as u32, 0], 2)
                .is_err()
        );
    }

    #[test]
    fn direct_full_load_writes_no_derived_weight_pack() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("no-pack.gguf");
        write_fixture(&gguf);
        let cache = temp.0.join("fresh-research-cache");
        let mut store = Store::open_fixture(&gguf, Some(&cache)).unwrap();
        store.plan_fixture_store().unwrap();
        assert!(
            store
                .select_direct_residency_with(EXPERTS, 0, &mut memory::direct_gate, &mut |_| Ok(()))
                .unwrap()
        );
        store.begin_turn().unwrap();
        let ids: Vec<u32> = (0..EXPERTS as u32).collect();
        let (banks, local, _) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        assert_banks_match_checkpoint(&store, &PARTS, &banks, &ids, &local);
        assert!(
            !cache.exists(),
            "the direct load created a weight-pack root"
        );
        let entries: Vec<_> = fs::read_dir(&temp.0)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(entries.len(), 1, "unexpected derived files: {entries:?}");
        assert!(entries[0].to_string_lossy().ends_with(".gguf"));
    }

    #[test]
    fn direct_load_admits_draft_banks_before_starting_expert_fill() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("draft-order.gguf");
        write_fixture(&gguf);
        let mut store = Store::open_fixture(&gguf, Some(&temp.0.join("draft-cache"))).unwrap();
        let aux =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp/bf16/paged");
        store
            .attach_auxiliary(Store::open_fixture(&aux, None).unwrap())
            .unwrap();
        let estimate = store.mtp_bank_estimate().unwrap();
        assert!(estimate > 0);
        store.plan_fixture_store().unwrap();
        let mut charged = None;
        assert!(
            store
                .select_direct_residency_with(
                    EXPERTS,
                    1 << 30,
                    &mut |plan, need, banks| {
                        charged = Some((need, banks));
                        memory::direct_gate(plan, need, banks)
                    },
                    &mut |_| Ok(()),
                )
                .unwrap()
        );
        assert_eq!(charged, Some((1 << 30, store.plan.direct_bytes)));
        assert_eq!(store.draft_bank_budget, estimate);
        store.prepare_fixture_hot().unwrap();
        let live = Arc::clone(&store.live);
        store
            .finish_load_with(&mut |_| {
                assert!(!live.direct_fill_in_progress.load(Ordering::Acquire));
                Ok(())
            })
            .unwrap();
        assert_eq!(store.mtp_bank_bytes().unwrap(), estimate);
        assert!(
            store.direct_fill.is_some(),
            "the direct fill starts after the draft banks"
        );
    }

    #[test]
    fn cuda_phase0_weight_linear_persistent_qlinear_and_small_m_paths() {
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        let make_q8 = |n: usize, k: usize, next: &mut dyn FnMut() -> u32| -> Weight {
            let codes: Vec<u32> = (0..n * k / 4).map(|_| next()).collect();
            let scales: Vec<f32> = (0..n * k / 32)
                .map(|_| ((next() % 97) as f32 + 1.) / 16384.)
                .collect();
            let biases: Vec<f32> = scales
                .iter()
                .map(|s| -s * (64 + next() % 129) as f32)
                .collect();
            Weight {
                values: MxArray::from_uint32(&codes, &[n as i64, k as i64 / 4]).unwrap(),
                dense_bf16: None,
                scales: Some(
                    MxArray::from_float32(&scales, &[n as i64, k as i64 / 32])
                        .unwrap()
                        .astype(DType::Float16)
                        .unwrap(),
                ),
                biases: Some(
                    MxArray::from_float32(&biases, &[n as i64, k as i64 / 32])
                        .unwrap()
                        .astype(DType::Float16)
                        .unwrap(),
                ),
                group: 32,
                bits: 8,
                mode: "affine".into(),
                q8_symmetric_128: false,
                qlinear: None,
            }
            .with_cached_qlinear()
        };

        // 1. Persistent `qlinear` on CUDA, plus safe rebuild on slice_rows & concatenate_rows.
        let (n, k) = (64usize, 256usize);
        let q8_a = make_q8(n, k, &mut next);
        let q8_b = make_q8(n, k, &mut next);
        if crate::accelerator::host_has_cuda() {
            assert!(q8_a.qlinear.is_some());
        } else {
            assert!(q8_a.qlinear.is_none());
        }
        let sliced = q8_a.slice_rows(16, 32).unwrap();
        let joined =
            Weight::concatenate_rows(&[Arc::new(q8_a.clone()), Arc::new(q8_b.clone())]).unwrap();
        if crate::accelerator::host_has_cuda() {
            assert!(sliced.qlinear.is_some());
            assert!(joined.qlinear.is_some());
            assert!(!Arc::ptr_eq(
                sliced.qlinear.as_ref().unwrap(),
                q8_a.qlinear.as_ref().unwrap()
            ));
        }

        // Multi-row prefill (M=20) followed by M=1 decode matches dequantized reference.
        let x20 = MxArray::from_float32(
            &(0..20 * k)
                .map(|i| ((i * 13 % 199) as f32 - 99.0) / 128.0)
                .collect::<Vec<_>>(),
            &[1, 20, k as i64],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
        let x1 = x20.slice_axis(1, 0, 1).unwrap();
        let ref_sliced_w = sliced.dense().unwrap().transpose(None).unwrap();
        let y20 = sliced.linear(&x20).unwrap().to_float32().unwrap();
        let ref20 = x20.matmul(&ref_sliced_w).unwrap().to_float32().unwrap();
        let err20 = y20
            .iter()
            .zip(ref20.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err20 < 2e-2, "sliced Q8 M=20 prefill mismatch: {err20}");
        let y1 = sliced.linear(&x1).unwrap().to_float32().unwrap();
        let ref1 = x1.matmul(&ref_sliced_w).unwrap().to_float32().unwrap();
        let err1 = y1
            .iter()
            .zip(ref1.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err1 < 2e-2, "sliced Q8 M=1 decode mismatch: {err1}");

        // 2. `linear_rows` and `linear_pair_rows` for M=4 (Q8_0 BF16) match M=1 bit-for-bit.
        let mut store = Store::open_metadata(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        for (name, bank) in [("test.pair_a.weight", &q8_a), ("test.pair_b.weight", &q8_b)] {
            store.tensors.insert(
                name.into(),
                Tensor {
                    path: PathBuf::new(),
                    shape: vec![n, k],
                    offset: 0,
                    bytes: (n * k) as u64,
                    encoding: Encoding::Gguf(GgufTensorType::Q8_0),
                },
            );
            store.banks.insert(name.into(), Arc::new(bank.clone()));
        }
        let x4 = x20.slice_axis(1, 0, 4).unwrap();
        let got_a = store
            .linear_rows("test.pair_a.weight", &x4)
            .unwrap()
            .to_float32()
            .unwrap();
        let want_a = MxArray::concatenate_many(
            (0..4)
                .map(|r| q8_a.linear(&x4.slice_axis(1, r, r + 1).unwrap()).unwrap())
                .collect::<Vec<_>>()
                .iter()
                .collect(),
            Some(1),
        )
        .unwrap()
        .to_float32()
        .unwrap();
        assert_eq!(
            &*got_a, &*want_a,
            "linear_rows M=4 must match M=1 bit-for-bit"
        );

        let (pair_a, pair_b) = store
            .linear_pair_rows(&x4, "test.pair_a.weight", "test.pair_b.weight")
            .unwrap();
        assert_eq!(&*pair_a.to_float32().unwrap(), &*want_a);
        let want_b = MxArray::concatenate_many(
            (0..4)
                .map(|r| q8_b.linear(&x4.slice_axis(1, r, r + 1).unwrap()).unwrap())
                .collect::<Vec<_>>()
                .iter()
                .collect(),
            Some(1),
        )
        .unwrap()
        .to_float32()
        .unwrap();
        assert_eq!(&*pair_b.to_float32().unwrap(), &*want_b);

        // 3. Unquantized F32 M=4 runs cleanly in a single matmul on non-Metal.
        let f32_w = Weight {
            values: MxArray::from_float32(
                &(0..32 * 64)
                    .map(|i| ((i * 7 % 101) as f32 - 50.0) / 64.0)
                    .collect::<Vec<_>>(),
                &[32, 64],
            )
            .unwrap(),
            dense_bf16: None,
            scales: None,
            biases: None,
            group: 0,
            bits: 0,
            mode: String::new(),
            q8_symmetric_128: false,
            qlinear: None,
        };
        let xf32 = MxArray::from_float32(
            &(0..4 * 64)
                .map(|i| ((i * 11 % 89) as f32 - 44.0) / 64.0)
                .collect::<Vec<_>>(),
            &[1, 4, 64],
        )
        .unwrap();
        let yf32 = f32_w.linear(&xf32).unwrap().to_float32().unwrap();
        let ref_f32 = xf32
            .matmul(&f32_w.values.transpose(None).unwrap())
            .unwrap()
            .to_float32()
            .unwrap();
        let f32_diff = yf32
            .iter()
            .zip(ref_f32.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(f32_diff < 1e-4, "F32 M=4 linear mismatch: {f32_diff}");

        // 4. `dense_bf16` router shape `[512, 2560]` at M=1 and M=4 via `bf16_matmul_few`.
        let router_vals = MxArray::from_float32(
            &(0..512 * 2560)
                .map(|i| ((i * 17 % 251) as f32 - 125.0) / 4096.0)
                .collect::<Vec<_>>(),
            &[512, 2560],
        )
        .unwrap();
        let router_bf16 = router_vals.astype(DType::BFloat16).unwrap();
        let router = Weight {
            values: router_vals,
            dense_bf16: Some(router_bf16.clone()),
            scales: None,
            biases: None,
            group: 0,
            bits: 0,
            mode: String::new(),
            q8_symmetric_128: false,
            qlinear: None,
        };
        for m in [1i64, 4] {
            let xr = MxArray::from_float32(
                &(0..m as usize * 2560)
                    .map(|i| ((i * 23 % 211) as f32 - 105.0) / 256.0)
                    .collect::<Vec<_>>(),
                &[1, m, 2560],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let got = router.linear(&xr).unwrap().to_float32().unwrap();
            let want = xr
                .matmul(&router_bf16.transpose(None).unwrap())
                .unwrap()
                .to_float32()
                .unwrap();
            let diff = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                diff < 5e-2,
                "dense_bf16 router [512, 2560] M={m} mismatch: {diff}"
            );
        }
    }
}
