mod input_validation_tests {
    use super::super::*;

    #[test]
    fn scalar_inputs_return_shape_errors_without_changing_convolution_history() {
        let scalar = MxArray::from_float32(&[1.], &[]).unwrap();
        let rotary_error = mrope(&scalar, [0; 3], 2, 10_000., [0; 3], true, None)
            .err()
            .expect("scalar rotary input must be rejected");
        assert_eq!(
            rotary_error.reason,
            "Qwen4 rotary input must have a feature dimension"
        );
        let recurrent_error = recurrent_step(&scalar, &scalar, &scalar, &scalar, &scalar, &scalar)
            .err()
            .expect("scalar recurrent input must be rejected");
        assert_eq!(
            recurrent_error.reason,
            "Qwen4 recurrent query must have shape [1, heads, 1, key_dim]"
        );

        let mut history = Some(MxArray::from_float32(&[2., 3.], &[2, 1]).unwrap());
        let convolution_error = conv_with_completion(&scalar, &scalar, &mut history, 3, 1, false)
            .err()
            .expect("scalar convolution input must be rejected");
        assert_eq!(
            convolution_error.reason,
            "Qwen4 convolution input must have a feature dimension"
        );
        assert_eq!(&*history.as_ref().unwrap().to_float32().unwrap(), &[2., 3.]);
    }
}

mod compact_recurrence_tests {
    use super::super::*;

    fn input(shape: &[i64], phase: f32) -> MxArray {
        let n: i64 = shape.iter().product();
        let values: Vec<_> = (0..n)
            .map(|i| (i as f32 * 0.017 + phase).sin() * 0.03)
            .collect();
        MxArray::from_float32(&values, shape)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    #[test]
    fn prepared_gdn_matches_separate_ops_across_short_and_wide_history() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let conv = input(&[10240, 4], 0.2).astype(DType::Float32).unwrap();
        let scale = input(&[48], 0.8)
            .astype(DType::Float32)
            .unwrap()
            .sub_scalar(0.5)
            .unwrap();
        let dt = input(&[48], 1.3).astype(DType::Float32).unwrap();
        let mut expected_state = Some(input(&[3, 10240], 2.4));
        let mut actual_state = expected_state.clone();
        let mut compact_state = expected_state.clone();
        for tokens in [1, 2, 7, 1024, 3] {
            let x = input(&[1, tokens, 10240], 0.1);
            let a = input(&[1, tokens, 48], 0.7).astype(DType::Float32).unwrap();
            let b = input(&[1, tokens, 48], 1.1).astype(DType::Float32).unwrap();
            let y = conv_sequence(&x, &conv, &mut expected_state, 4).unwrap();
            let q = l2(&y
                .slice_axis(2, 0, 2048)
                .unwrap()
                .reshape(&[1, tokens, 16, 128])
                .unwrap())
            .unwrap()
            .mul_scalar(128f64.powf(-0.5))
            .unwrap();
            let k = l2(&y
                .slice_axis(2, 2048, 4096)
                .unwrap()
                .reshape(&[1, tokens, 16, 128])
                .unwrap())
            .unwrap();
            let v = y
                .slice_axis(2, 4096, 10240)
                .unwrap()
                .reshape(&[1, tokens, 48, 128])
                .unwrap();
            let (decay, beta) = gdn_gates(&a, &b, &scale, &dt).unwrap();
            let actual = gdn_prepare(&x, &a, &b, &conv, &mut actual_state, &scale, &dt)
                .unwrap()
                .unwrap();
            let compact = gdn_prepare(
                &x,
                &a.astype(DType::BFloat16).unwrap(),
                &b.astype(DType::BFloat16).unwrap(),
                &conv,
                &mut compact_state,
                &scale,
                &dt,
            )
            .unwrap()
            .unwrap();
            for (field, (want, got)) in actual.iter().zip(&compact).enumerate() {
                assert_eq!(want.dtype().unwrap(), got.dtype().unwrap());
                assert_eq!(
                    want.to_float32().unwrap().to_vec(),
                    got.to_float32().unwrap().to_vec(),
                    "compact gate preparation field={field}, tokens={tokens}"
                );
            }
            assert_eq!(
                actual_state
                    .as_ref()
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .to_vec(),
                compact_state
                    .as_ref()
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .to_vec()
            );
            for (field, (want, got)) in [q, k, v, decay, beta].iter().zip(&actual).enumerate() {
                assert_eq!(want.dtype().unwrap(), got.dtype().unwrap());
                assert_eq!(
                    &*want.astype(DType::Float32).unwrap().to_float32().unwrap(),
                    &*got.astype(DType::Float32).unwrap().to_float32().unwrap(),
                    "GDN preparation field={field}, tokens={tokens}"
                );
            }
            assert_eq!(
                &*expected_state
                    .as_ref()
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap(),
                &*actual_state
                    .as_ref()
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap()
            );
        }
    }

    #[test]
    fn bf16_inputs_preserve_fp32_recurrence_and_continuation() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        // Compact GGUF head modulo, expanded heads, and the smaller 2-row
        // dispatch all retain the same arithmetic and FP32 output/state.
        for (kh, nh, d) in [(16, 48, 128), (48, 48, 128), (2, 4, 32)] {
            let mut expected = input(&[1, nh, d, d], 0.3).astype(DType::Float32).unwrap();
            let mut actual = expected.clone();
            for tokens in [7, 1024, 1] {
                let q = input(&[1, tokens, kh, d], 0.1);
                let k = input(&[1, tokens, kh, d], 0.7);
                let v = input(&[1, tokens, nh, d], 1.3);
                let decay = input(&[1, tokens, nh], 2.1)
                    .astype(DType::Float32)
                    .unwrap()
                    .add_scalar(0.95)
                    .unwrap();
                let beta = input(&[1, tokens, nh], 3.4)
                    .astype(DType::Float32)
                    .unwrap()
                    .add_scalar(0.5)
                    .unwrap();
                let (want, next) = recurrent_sequence(
                    &q.astype(DType::Float32).unwrap(),
                    &k.astype(DType::Float32).unwrap(),
                    &v.astype(DType::Float32).unwrap(),
                    &decay,
                    &beta,
                    &expected,
                )
                .unwrap();
                let (got, state) = recurrent_sequence(&q, &k, &v, &decay, &beta, &actual).unwrap();
                assert_eq!(got.dtype().unwrap(), DType::Float32);
                assert_eq!(state.dtype().unwrap(), DType::Float32);
                assert_eq!(
                    &*got.to_float32().unwrap(),
                    &*want.to_float32().unwrap(),
                    "output kh={kh}, nh={nh}, d={d}, tokens={tokens}"
                );
                assert_eq!(
                    &*state.to_float32().unwrap(),
                    &*next.to_float32().unwrap(),
                    "state kh={kh}, nh={nh}, d={d}, tokens={tokens}"
                );
                expected = next;
                actual = state;
            }
        }
    }
}

mod rotary_window_tests {
    use super::super::*;

    /// The Flash Next tables take Splash's YaRN inverse frequencies and its
    /// attention factor on cos and sin; without YaRN they are today's
    /// angles bit for bit. Mutation: the attention factor left out of
    /// `cos_sin` → red (4.5% off).
    #[test]
    fn flash_next_rotary_takes_the_yarn_table() {
        let mut config = crate::models::qwen4_exp::memory::flash_next_config();
        let (dims, theta) = (config.rope_dims(), config.rope_theta());
        let positions = vec![[0i64; 3], [3; 3], [409_000; 3]];
        let (plain_cos, plain_sin) =
            rotary_tables(&positions, dims, theta, [0; 3], true, DType::Float32, None).unwrap();
        let old: Vec<f32> = positions
            .iter()
            .flat_map(|p| {
                (0..dims / 2)
                    .map(|i| (p[0] as f64 / theta.powf((2 * i) as f64 / dims as f64)) as f32)
            })
            .collect();
        let old = MxArray::from_float32(&old, &[1, 1, 3, dims as i64 / 2]).unwrap();
        let bits = |a: &MxArray| {
            a.to_float32()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&plain_cos), bits(&old.cos().unwrap()));
        assert_eq!(bits(&plain_sin), bits(&old.sin().unwrap()));

        crate::models::qwen4_exp::load_plan::serve_context("Qwen4Exp", &mut config, Some(409_600))
            .unwrap();
        let yarn = Yarn::of(&config)
            .unwrap()
            .expect("YaRN past the native window");
        let inverse = crate::nn::rope_scaling::rotary_inverse_frequencies(
            theta as f32,
            (dims / 2) as u32,
            &config.rope_scaling,
        )
        .unwrap();
        let factor = f64::from(config.rope_scaling.attention_factor());
        assert_eq!(factor as f32, f32::from_bits(0x3f85b665));
        let (cos, sin) = rotary_tables(
            &positions,
            dims,
            theta,
            [0; 3],
            true,
            DType::Float32,
            Some(&yarn),
        )
        .unwrap();
        let (cos, sin) = (cos.to_float32().unwrap(), sin.to_float32().unwrap());
        for (row, position) in positions.iter().enumerate() {
            for (pair, &inverse) in inverse.iter().enumerate() {
                let angle = f64::from((position[0] as f64 * f64::from(inverse)) as f32);
                let index = row * dims / 2 + pair;
                let tolerance = if position[0] > 1000 { 2e-3 } else { 1e-5 };
                assert!((f64::from(cos[index]) - angle.cos() * factor).abs() <= tolerance);
                assert!((f64::from(sin[index]) - angle.sin() * factor).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn shared_tables_preserve_scalar_rounding_and_media_axes() {
        let mut cache = RotaryWindowCache::default();
        for dtype in [DType::BFloat16, DType::Float32] {
            for interleaved in [true, false] {
                for (count, stride, base) in [(1, 1, 1023), (7, 4, 1), (256, 4, 2045), (1024, 1, 0)]
                {
                    let positions: Vec<_> = (0..count)
                        .map(|i| {
                            let p = base + i * stride;
                            [p, p / 7 + 11, p / 3 - 5]
                        })
                        .collect();
                    for width in [128, 256] {
                        let data: Vec<_> = (0..count * width)
                            .map(|i| (i as f32 * 0.17).sin() * 7.1)
                            .collect();
                        let x = MxArray::from_float32(&data, &[1, 1, count, width])
                            .unwrap()
                            .astype(dtype)
                            .unwrap();
                        let got = cache
                            .apply(
                                &x,
                                positions.clone(),
                                64,
                                10_000_000.,
                                [16, 8, 8],
                                interleaved,
                                None,
                            )
                            .unwrap();
                        let rows: Vec<_> = positions
                            .iter()
                            .enumerate()
                            .map(|(i, &p)| {
                                mrope(
                                    &x.slice_axis(2, i as i64, i as i64 + 1).unwrap(),
                                    p,
                                    64,
                                    10_000_000.,
                                    [16, 8, 8],
                                    interleaved,
                                    None,
                                )
                                .unwrap()
                            })
                            .collect();
                        let expected =
                            MxArray::concatenate_many(rows.iter().collect(), Some(2)).unwrap();
                        assert_eq!(
                            &*got.astype(DType::Float32).unwrap().to_float32().unwrap(),
                            &*expected
                                .astype(DType::Float32)
                                .unwrap()
                                .to_float32()
                                .unwrap(),
                            "{dtype:?}, {interleaved}, {count}, {width}"
                        );
                        assert!(cache.tables.len() <= 2);
                    }
                }
            }
        }
        cache.clear();
        assert!(cache.tables.is_empty());
    }
}

mod attention_gate_tests {
    use super::super::*;

    fn values(n: i64, shape: &[i64], phase: f32) -> MxArray {
        MxArray::from_float32(
            &(0..n)
                .map(|i| (i as f32 * 0.137 + phase).sin() * 3.1)
                .collect::<Vec<_>>(),
            shape,
        )
        .unwrap()
    }

    fn equal(a: &MxArray, b: &MxArray) {
        assert_eq!(a.dtype().unwrap(), b.dtype().unwrap());
        assert_eq!(a.shape().unwrap().to_vec(), b.shape().unwrap().to_vec());
        assert_eq!(
            a.astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec(),
            b.astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec()
        );
    }

    #[test]
    fn window_convolution_preserves_rounding_and_chunk_history() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        for width in [17, 256, 10240] {
            // Transposed weights and offset activation views also exercise
            // the custom primitive's contiguous-input preparation.
            let w = values(4 * width, &[4, width], 0.7).transpose(None).unwrap();
            let x = values(73 * width, &[1, 73, width], 1.3)
                .astype(DType::BFloat16)
                .unwrap();
            let old = values(6 * width, &[6, width], 2.1)
                .astype(DType::BFloat16)
                .unwrap()
                .slice_axis(0, 3, 6)
                .unwrap();
            let mut expected = Some(old.clone());
            let mut actual = Some(old);
            let mut offset = 0;
            for count in [1, 2, 3, 7, 60] {
                let input = x.slice_axis(1, offset, offset + count).unwrap();
                let want = conv_sequence_reference(&input, &w, &mut expected, 4).unwrap();
                // Call the primitive directly so an unsupported/fallback
                // dispatch cannot make a kernel regression pass silently.
                let (got, next) =
                    crate::kernels::qwen4::metal::window_conv(&input, actual.as_ref().unwrap(), &w)
                        .unwrap()
                        .expect("test window conv");
                equal(&got, &want);
                equal(&next, expected.as_ref().unwrap());
                actual = Some(next);
                offset += count;
            }
            let mut whole_state = Some(
                values(6 * width, &[6, width], 2.1)
                    .astype(DType::BFloat16)
                    .unwrap()
                    .slice_axis(0, 3, 6)
                    .unwrap(),
            );
            conv_sequence(&x, &w, &mut whole_state, 4).unwrap();
            equal(whole_state.as_ref().unwrap(), actual.as_ref().unwrap());
        }
    }
}

mod prefill_epilogue_tests {
    use super::super::*;

    #[test]
    fn replay_keeps_bf16_norm_and_output_gate_boundaries() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        for tokens in [1i64, 7, 128, 512, 692, 1024] {
            let shape = [1, tokens, 2, 128];
            let n = (tokens * 2 * 128) as usize;
            let values: Vec<_> = (0..n).map(|i| (i as f32 * 0.013).sin() * 5.3).collect();
            let gates: Vec<_> = (0..n).map(|i| (i as f32 * 0.177).cos() * 9.1).collect();
            let scales: Vec<_> = (0..128).map(|i| (i as f32 * 0.103).cos() * 1.3).collect();
            let out = MxArray::from_float32(&values, &shape).unwrap();
            let z = MxArray::from_float32(&gates, &shape)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let w = MxArray::from_float32(&scales, &[128]).unwrap();
            let normalized =
                norm(&out.astype(DType::BFloat16).unwrap(), &w, 128, 1e-6, false).unwrap();
            let expected = sigmoid_mul(
                &z.astype(DType::Float32).unwrap(),
                &normalized.astype(DType::Float32).unwrap(),
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let actual = gdn_epilogue(&out, &z, &w, 1e-6).unwrap().unwrap();
            assert_eq!(
                &*actual.to_float32().unwrap(),
                &*expected.to_float32().unwrap(),
                "tokens={tokens}"
            );
        }
    }
}

mod pass_length_tests {
    use crate::array::{DType, MxArray};

    const LENGTHS: [i64; 4] = [7, 30, 448, 1024];

    /// FNV-1a over every output's bits, captured from the kernels that took
    /// the row count as a template constant, on an M5 (GPU generation 17).
    const GOLDEN: &[(&str, i64, u64)] = &[
        ("gdn-qwen4", 7, 0xfe150edfdd978794),
        ("gdn-qwen35", 7, 0x27fa31557b3a9daf),
        ("conv-10240", 7, 0x672337c677cd603c),
        ("conv-17", 7, 0x8e499dea1a9a2bc7),
        ("gdn-qwen4", 30, 0xe7d6215836e8c100),
        ("gdn-qwen35", 30, 0x744f531c5db4bc1f),
        ("conv-10240", 30, 0xb720278ad82a338b),
        ("conv-17", 30, 0xf9a8246846de77da),
        ("gdn-qwen4", 448, 0xf614225f1219ca08),
        ("gdn-qwen35", 448, 0x28916257a045dd60),
        ("conv-10240", 448, 0x5275a4b967d5997f),
        ("conv-17", 448, 0xc09682b1cc1ba3bf),
        ("gdn-qwen4", 1024, 0x4cd731152850a4a3),
        ("gdn-qwen35", 1024, 0xbdae17d0365efd9d),
        ("conv-10240", 1024, 0x899094d57fc6f0c3),
        ("conv-17", 1024, 0x4abc164be203232d),
    ];
    const GOLDEN_GPU_GEN: i32 = 17;

    fn input(shape: &[i64], phase: f32, dtype: DType) -> MxArray {
        let n: i64 = shape.iter().product();
        let values: Vec<_> = (0..n)
            .map(|i| (i as f32 * 0.0173 + phase).sin() * 0.9 + (i as f32 * 0.31).cos() * 0.05)
            .collect();
        MxArray::from_float32(&values, shape)
            .unwrap()
            .astype(dtype)
            .unwrap()
    }

    /// A strided view of `width` columns out of a wider last axis.
    fn view(shape: &[i64], width: i64, phase: f32, dtype: DType) -> MxArray {
        let mut wide = shape.to_vec();
        let last = wide.len() - 1;
        wide[last] = width + 64;
        input(&wide, phase, dtype)
            .slice_axis(last, 32, 32 + width)
            .unwrap()
    }

    fn fnv(hash: &mut u64, bytes: impl IntoIterator<Item = u8>) {
        for byte in bytes {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }

    fn digest(outputs: &[MxArray]) -> u64 {
        let refs: Vec<&MxArray> = outputs.iter().collect();
        MxArray::eval_arrays(&refs).unwrap();
        let mut hash = 0xcbf2_9ce4_8422_2325;
        for out in outputs {
            fnv(
                &mut hash,
                out.shape().unwrap().iter().flat_map(|d| d.to_le_bytes()),
            );
            match out.dtype().unwrap() {
                DType::BFloat16 => fnv(
                    &mut hash,
                    out.to_uint16_native()
                        .unwrap()
                        .into_iter()
                        .flat_map(u16::to_le_bytes),
                ),
                DType::Float32 => fnv(
                    &mut hash,
                    out.to_float32()
                        .unwrap()
                        .iter()
                        .flat_map(|v| v.to_bits().to_le_bytes()),
                ),
                other => panic!("unexpected output dtype {other:?}"),
            }
        }
        hash
    }

    fn bits(x: &MxArray) -> Vec<u32> {
        let f = x.astype(DType::Float32).unwrap();
        f.to_float32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    struct Gdn {
        conv: MxArray,
        scale: MxArray,
        dt: MxArray,
        mean_eps: bool,
        beta_input_dtype: bool,
        gate_dtype: DType,
    }

    impl Gdn {
        fn new(mean_eps: bool, beta_input_dtype: bool, gate_dtype: DType) -> Self {
            Self {
                conv: input(&[10240, 4], 0.2, DType::Float32),
                scale: input(&[48], 0.8, DType::Float32).sub_scalar(0.5).unwrap(),
                dt: input(&[48], 1.3, DType::Float32),
                mean_eps,
                beta_input_dtype,
                gate_dtype,
            }
        }

        fn inputs(&self, t: i64, phase: f32) -> [MxArray; 3] {
            [
                view(&[1, t, 10240], 10240, phase, DType::BFloat16),
                view(&[1, t, 48], 48, phase + 0.6, self.gate_dtype),
                view(&[1, t, 48], 48, phase + 1.0, self.gate_dtype),
            ]
        }

        /// q, k, v, decay, beta, next history.
        fn run(&self, [qkv, a, b]: &[MxArray; 3], history: &MxArray) -> Vec<MxArray> {
            crate::kernels::qwen4::metal::gdn_prepare(
                qkv,
                a,
                b,
                &self.conv,
                history,
                &self.scale,
                &self.dt,
                self.mean_eps,
                self.beta_input_dtype,
            )
            .unwrap()
            .expect("test GDN preparation")
            .into_iter()
            .collect()
        }
    }

    fn conv_run(x: &MxArray, history: &MxArray, weight: &MxArray) -> Vec<MxArray> {
        let (out, next) = crate::kernels::qwen4::metal::window_conv(x, history, weight)
            .unwrap()
            .expect("test window conv");
        vec![out, next]
    }

    /// Splitting a pass in two, the second piece starting from the first
    /// piece's history, gives the whole pass's rows and history.
    fn assert_split_equals_whole(
        what: &str,
        x: &[MxArray],
        cut: i64,
        history: &MxArray,
        run: &dyn Fn(&[MxArray], &MxArray) -> Vec<MxArray>,
    ) {
        let t = x[0].shape_at(1).unwrap();
        let piece = |from: i64, to: i64| -> Vec<MxArray> {
            x.iter()
                .map(|v| v.slice_axis(1, from, to).unwrap())
                .collect()
        };
        let whole = run(x, history);
        let first = run(&piece(0, cut), history);
        let second = run(&piece(cut, t), first.last().unwrap());
        let rows = whole.len() - 1;
        for i in 0..rows {
            let joined = MxArray::concatenate_many(vec![&first[i], &second[i]], Some(1)).unwrap();
            assert_eq!(
                bits(&whole[i]),
                bits(&joined),
                "{what}: output {i}, t={t}, cut={cut}"
            );
        }
        assert_eq!(
            bits(&whole[rows]),
            bits(&second[rows]),
            "{what}: next history, t={t}, cut={cut}"
        );
    }

    /// Both pass-length kernels keep the bits they had when the row count was
    /// a template constant, at every pass length.
    #[test]
    fn gdn_kernels_keep_their_bits_at_any_length() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let gdns = [
            ("gdn-qwen4", Gdn::new(false, false, DType::Float32)),
            ("gdn-qwen35", Gdn::new(true, true, DType::BFloat16)),
        ];
        // Transposed weights and offset views also exercise the primitive's
        // contiguous-input preparation.
        let weights = [
            (
                "conv-10240",
                10240i64,
                input(&[4, 10240], 0.7, DType::Float32),
            ),
            ("conv-17", 17, input(&[4, 17], 0.9, DType::Float32)),
        ]
        .map(|(name, width, w)| (name, width, w.transpose(None).unwrap()));
        // Bits are pinned on the generation they were captured on; the split
        // check below holds on every GPU.
        if unsafe { mlx_sys::mlx_gpu_architecture_gen() } == GOLDEN_GPU_GEN {
            let mut seen = Vec::new();
            for t in LENGTHS {
                for (name, gdn) in &gdns {
                    let history = view(&[3, 10240], 10240, 2.4, DType::BFloat16);
                    seen.push((*name, t, digest(&gdn.run(&gdn.inputs(t, 0.1), &history))));
                }
                for (name, width, weight) in &weights {
                    let x = view(&[1, t, *width], *width, 0.4, DType::BFloat16);
                    let history = view(&[3, *width], *width, 2.1, DType::BFloat16);
                    seen.push((*name, t, digest(&conv_run(&x, &history, weight))));
                }
            }
            assert_eq!(seen, GOLDEN, "kernel bits changed");
        }
        for t in LENGTHS {
            for (name, gdn) in &gdns {
                let history = view(&[3, 10240], 10240, 2.4, DType::BFloat16);
                let run = |x: &[MxArray], h: &MxArray| {
                    gdn.run(&[x[0].clone(), x[1].clone(), x[2].clone()], h)
                };
                assert_split_equals_whole(name, &gdn.inputs(t, 0.1), t / 3, &history, &run);
            }
            for (name, width, weight) in &weights {
                let x = view(&[1, t, *width], *width, 0.4, DType::BFloat16);
                let history = view(&[3, *width], *width, 2.1, DType::BFloat16);
                let run = |x: &[MxArray], h: &MxArray| conv_run(&x[0], h, weight);
                assert_split_equals_whole(name, &[x], t / 3, &history, &run);
            }
        }
    }

    /// One pipeline serves every pass length: a new length builds no kernel.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_new_pass_length_builds_no_pipeline() {
        const NAME: &str = "models::qwen4_exp::math::tests::pass_length_tests::a_new_pass_length_builds_no_pipeline";
        if !crate::engine::persistence::compiled_forward_backend_available()
            || !crate::test_support::in_own_process(NAME)
        {
            return;
        }
        assert!(
            crate::engine::pipeline_counter::install(),
            "the counter installs in tests"
        );
        let gdn = Gdn::new(false, false, DType::Float32);
        let weight = input(&[10240, 4], 0.7, DType::Float32);
        let history = input(&[3, 10240], 2.4, DType::BFloat16);
        let passes: Vec<_> = [7i64, 30, 448]
            .iter()
            .map(|&t| gdn.inputs(t, 0.1))
            .collect();
        for pass in &passes {
            MxArray::eval_arrays(&pass.iter().collect::<Vec<_>>()).unwrap();
        }
        MxArray::eval_arrays(&[&gdn.conv, &gdn.scale, &gdn.dt, &weight, &history]).unwrap();
        let run = |pass: &[MxArray; 3]| {
            let mut outputs = gdn.run(pass, &history);
            outputs.extend(conv_run(&pass[0], &history, &weight));
            MxArray::eval_arrays(&outputs.iter().collect::<Vec<_>>()).unwrap();
        };
        run(&passes[0]);
        let first = crate::engine::pipeline_counter::read().expect("installed");
        let _ = mlx_metal::pipeline_counter::drain_serving();
        let mut created = Vec::new();
        for pass in &passes[1..] {
            run(pass);
            created.extend(
                mlx_metal::pipeline_counter::drain_serving()
                    .into_iter()
                    .map(|c| c.name),
            );
        }
        let after = crate::engine::pipeline_counter::read().expect("installed");
        let per_kernel = |kernel: &str| created.iter().filter(|n| n.contains(kernel)).count();
        assert_eq!(
            (
                per_kernel("qwen4_gdn_prepare"),
                per_kernel("qwen4_window_conv"),
                after.created_serving - first.created_serving,
            ),
            (0, 0, 0),
            "pipelines built for new pass lengths: {created:?}"
        );
    }
}
