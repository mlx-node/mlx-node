#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::engine::backend::ChatBackend;
    use crate::engine::types::ChatConfig;
    use crate::models::quantized_linear::LinearProj;
    use crate::nn::Linear;

    fn tiny_dflash_inner(seed: u64) -> Qwen35Inner {
        tiny_dflash_inner_with_config(seed, |_| {})
    }

    fn tiny_dflash_inner_with_config(
        seed: u64,
        adjust: impl FnOnce(&mut crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig),
    ) -> Qwen35Inner {
        let mut inner =
            super::super::super::model::scheduled_mtp::seeded_moe_inner_with_config(seed, adjust);
        let mut head = Linear::new(
            inner.config.hidden_size as u32,
            inner.config.vocab_size as u32,
            Some(false),
        )
        .unwrap();
        head.set_weight(
            &MxArray::random_normal(
                &[
                    inner.config.vocab_size as i64,
                    inner.config.hidden_size as i64,
                ],
                0.0,
                0.02,
                Some(DType::BFloat16),
            )
            .unwrap(),
        )
        .unwrap();
        inner.lm_head = Some(LinearProj::Standard(head));
        let dense = inner.config.clone();
        let draft =
            crate::models::qwen3_5::dflash2::tiny_dflash2_model_for_stepper_test(&dense).unwrap();
        draft.validate_target(&dense).unwrap();
        inner.dflash2 = Some(draft);
        inner
    }

    #[test]
    fn full_width_context_policy_crosses_over_at_keep_seven() {
        for keep in 0..=9 {
            assert_eq!(
                prefer_full_width_context(keep),
                matches!(keep, 7 | 8),
                "keep={keep}"
            );
        }
    }

    #[test]
    fn moe_dflash_plan_and_params_use_companion_width() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let inner = tiny_dflash_inner(0xDFA5_3503);
        let plan = ChatBackend::execution_plan(&inner).speculative.unwrap();
        assert_eq!(plan.kind, crate::engine::plan::SpeculativeKind::DraftModel);
        assert!(plan.supports_paged_attention);
        assert!(plan.supports_streaming);
        assert!(plan.supported_input_media.is_empty());
        let params = ChatBackend::resolve_params(
            &inner,
            &ChatConfig {
                mtp_depth: Some(99),
                mtp_adaptive_depth: Some(true),
                temperature: Some(0.0),
                max_new_tokens: Some(19),
                ..ChatConfig::default()
            },
        );
        assert_eq!(params.mtp_depth, 2);
        assert!(params.mtp_adaptive_depth);
        assert_eq!(params.max_new_tokens, 19);
        Ok(())
    }
}

#[cfg(test)]
mod paged_dflash2_tests {
    use super::super::*;
    use crate::engine::backend::{PagedBackend, PagedPrefix};
    use crate::engine::dspark_turn::PagedDsparkBackend;
    use crate::models::quantized_linear::LinearProj;
    use crate::models::qwen3_5::dflash2::DFLASH2_DRAFT_PROPOSALS;
    use crate::models::qwen3_5::gdn_blob::trace::{GdnTrace, cells};
    use crate::nn::Linear;
    use crate::stream::{Stream, StreamContext};
    use rand::SeedableRng;

    const VOCAB: u32 = 16;

    /// The seeded MoE paged fixture on an int8 pool, a sensitive random head
    /// and a draft that proposes the 7 tokens a cycle drafts.
    fn tiny_paged_int8_moe_dflash_inner(seed: u64) -> Qwen35Inner {
        let mut inner = super::super::super::model::scheduled_mtp::seeded_moe_inner_with_config(
            seed,
            |config| {
                config.head_dim = 256;
                config.hidden_size = 256;
                config.kv_format = Some("int8".to_string());
                // The GDN head geometry whose verify takes the complete kernel
                // that writes the spare cell.
                config.linear_key_head_dim = 128;
                config.linear_value_head_dim = 128;
                config.linear_num_key_heads = 2;
                config.linear_num_value_heads = 2;
            },
        );
        let mut head =
            Linear::new(inner.config.hidden_size as u32, VOCAB, Some(false)).expect("tiny head");
        head.set_weight(
            &MxArray::random_normal(
                &[i64::from(VOCAB), inner.config.hidden_size as i64],
                0.0,
                0.02,
                Some(DType::BFloat16),
            )
            .expect("tiny head weights"),
        )
        .expect("install tiny head");
        inner.lm_head = Some(LinearProj::Standard(head));
        let dense = inner.config.clone();
        let draft = crate::models::qwen3_5::dflash2::tiny_dflash2_model_with_block(
            &dense,
            DFLASH2_DRAFT_PROPOSALS,
        )
        .expect("tiny 7-proposal draft");
        draft.validate_target(&dense).expect("draft fits target");
        inner.dflash2 = Some(draft);
        assert_eq!(
            inner.paged_adapter.as_ref().expect("pool").cache_dtype(),
            mlx_paged_attn::metal::MetalDtype::Int8,
            "the fixture's pool must be int8"
        );
        inner
    }

    fn prompt(len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| (i * 7 + 3) % VOCAB).collect()
    }

    fn argmax_rows(logits: &MxArray) -> Result<Vec<u32>> {
        let ids = logits.argmax(-1, None)?;
        ids.eval();
        Ok(ids.to_int32()?.iter().map(|&id| id as u32).collect())
    }

    fn cursor(inner: &Qwen35Inner) -> u32 {
        inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .current_token_count()
    }

    /// Every prefill chunk launches the in-place attention with the prefill
    /// split rule (Splash `prefillSplits`, max 32) and the 8-row DFlash2
    /// verify with the verify rule (Splash `kv::verifyAttentionSplits`, max
    /// 128), one launch per full-attention layer. Mutation R1:
    /// `paged_verify_core` passes `PagedPass::Prefill` → red (no verify
    /// launch).
    #[test]
    fn verify_takes_the_verify_split_rule() -> Result<()> {
        // The launch probe counts every paged prefill in the process.
        if !crate::test_support::in_own_process(
            "models::qwen3_5::dflash2_decode::dflash2_decode_moe_tests::paged_dflash2_tests::verify_takes_the_verify_split_rule",
        ) {
            return Ok(());
        }
        if crate::test_support::paged_prefill_unsupported("verify_takes_the_verify_split_rule") {
            return Ok(());
        }
        let probe = |kind| unsafe { mlx_sys::mlx_paged_prefill_test_probe_count(kind) };
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8010);
        let layers = inner.config.full_attention_layer_count() as u64;
        let prompt = prompt(40);
        let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
        unsafe { mlx_sys::mlx_paged_prefill_test_probe_reset() };
        let logits = inner.paged_prefill_with_draft_state(&prompt, &prefix, stream)?;
        let anchor = argmax_rows(&logits)?[0];
        let chunks = probe(0);
        assert!(
            chunks > 0 && chunks % layers == 0,
            "{chunks} prefill launches over {layers} layers"
        );
        assert_eq!(probe(1), 0, "a prefill chunk is no verify");
        assert_eq!(
            unsafe { mlx_sys::mlx_paged_prefill_test_last_max_splits() },
            32
        );
        let open = cursor(&inner);
        let params = crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
            temperature: Some(0.0),
            ..Default::default()
        });
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
        let mut proposal = step.propose(anchor, DFLASH2_DRAFT_PROPOSALS, &params, &mut rng)?;
        proposal.materialize_draft_ids()?;
        let mut ids = vec![anchor];
        ids.extend(proposal.draft_ids.iter().map(|&id| id as u32));
        step.verify(&ids)?.logits.eval();
        assert_eq!(probe(1), layers, "one verify launch per layer");
        assert_eq!(probe(0), chunks);
        assert_eq!(
            unsafe { mlx_sys::mlx_paged_prefill_test_last_splits() },
            mlx_paged_attn::prefill_plan::verify_splits(open)
        );
        assert_eq!(
            unsafe { mlx_sys::mlx_paged_prefill_test_last_max_splits() },
            128
        );
        step.commit_with_provenance(1, ids.len(), &ids)?;
        step.finish()?;
        Ok(())
    }

    fn greedy() -> crate::engine::params::ChatParams {
        crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
            temperature: Some(0.0),
            ..Default::default()
        })
    }

    /// Greedy paged AR: `n` tokens and the cursor after them.
    fn paged_ar(inner: &mut Qwen35Inner, prompt: &[u32], n: usize) -> Result<(Vec<u32>, u32)> {
        let prefix = inner.prime_dense_prefix(prompt, 0, None)?;
        let suffix = &prompt[prefix.effective_cached_prefix_len()..];
        let logits = inner.dense_paged_prefill(suffix, &prefix, None)?.0;
        let mut tokens = argmax_rows(&logits)?;
        while tokens.len() < n {
            let last = *tokens.last().expect("one token");
            let embed = inner.embedding.clone();
            let caches = inner.caches.as_mut().expect("caches");
            let adapter = inner.paged_adapter.as_mut().expect("pool");
            let logits = crate::models::qwen3_5::paged_forward::run_paged_decode_step(
                last,
                &embed,
                &mut inner.layers,
                caches,
                &inner.final_norm,
                &inner.lm_head,
                &inner.layer_kinds,
                adapter,
                0,
            )?;
            tokens.extend(argmax_rows(&logits)?);
        }
        Ok((tokens, cursor(inner)))
    }

    /// Paged MoE DFlash2 cycles where the first `correct` proposals of each
    /// are the reference's tokens and the next one is wrong.
    fn paged_dflash2(
        inner: &mut Qwen35Inner,
        prompt: &[u32],
        reference: &[u32],
        correct: usize,
    ) -> Result<(Vec<u32>, u32, Vec<u64>, u64, GdnTrace)> {
        let n = reference.len();
        let linear = inner
            .layer_kinds
            .iter()
            .filter(|kind| {
                matches!(
                    kind,
                    crate::models::qwen3_5::decoder_layer::Qwen3_5LayerKind::Linear
                )
            })
            .count() as u64;
        let prefix = inner.prime_dspark_prefix_state(prompt, true, 0)?;
        let suffix = prompt[prefix.effective_cached_prefix_len()..].to_vec();
        let logits =
            inner.paged_prefill_with_draft_state(&suffix, &prefix, Stream::generation())?;
        let mut generated = argmax_rows(&logits)?;
        let params = greedy();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
        let mut gdn = GdnTrace::begin(
            step.gdn_blobs.as_ref().expect("cells"),
            step.inner.caches.as_ref().expect("caches"),
        );
        let mut deferred = Vec::new();
        while generated.len() < n {
            let anchor = *generated.last().expect("one token");
            let at = generated.len();
            let l_cap = DFLASH2_DRAFT_PROPOSALS.min(n - at - 1);
            if l_cap > 0 {
                step.propose(anchor, l_cap, &params, &mut rng)?;
            }
            let mut ids = vec![anchor];
            for i in 0..l_cap {
                let reference_id = reference[at + i];
                ids.push(if i < correct {
                    reference_id
                } else {
                    (reference_id + 1) % VOCAB
                });
            }
            unsafe { mlx_sys::mlx_test_kquant_counting(true) };
            let rows = step.verify(&ids).and_then(|out| argmax_rows(&out.logits));
            deferred.push(unsafe {
                mlx_sys::mlx_test_kquant_family_count(c"gdn_fused_deferred".as_ptr())
            });
            unsafe { mlx_sys::mlx_test_kquant_counting(false) };
            let rows = rows?;
            assert_eq!(rows.len(), ids.len(), "the driver sees 1 + L logits rows");
            let mut accepted = 0;
            while accepted < l_cap && rows[accepted] == ids[accepted + 1] {
                accepted += 1;
            }
            let tape = step.tape.clone().expect("verify tape");
            step.commit_with_provenance(accepted + 1, ids.len(), &ids)?;
            gdn.cycles.push((tape, accepted + 1));
            generated.extend_from_slice(&ids[1..=accepted]);
            generated.push(rows[accepted]);
        }
        gdn.before_finish(
            step.gdn_blobs.as_ref().expect("cells"),
            step.inner.caches.as_ref().expect("caches"),
        );
        step.finish()?;
        gdn.after = cells(inner.caches.as_ref().expect("caches"), &gdn.layers);
        Ok((generated, cursor(inner), deferred, linear, gdn))
    }

    /// MoE speculation on the int8 pool, every keep: paged DFlash2 lands on
    /// the AR cursor, every cycle launches one deferred GDN kernel per linear
    /// layer, and the cells after `finish` equal every cycle's kept rows
    /// replayed from the packed start and the deferred kernel's fold of the
    /// last pending rows. Its tokens are not compared with paged AR: the
    /// verify reads through the in-place prefill attention, the AR decode
    /// through the decode kernel. Mutation MV1 (the verify core passes no
    /// pending tape) turns it red.
    #[test]
    fn paged_moe_dflash2_lands_on_the_ar_cursor_for_every_keep() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "paged_moe_dflash2_lands_on_the_ar_cursor_for_every_keep",
        ) {
            return Ok(());
        }
        use crate::models::qwen3_5::gdn_blob::trace::{assert_cells, kernel_fold, replay_chain};
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prompt = prompt(40);
        let n = 64;
        let mut ar_inner = tiny_paged_int8_moe_dflash_inner(0x5_8101);
        let (reference, ar_cursor) = paged_ar(&mut ar_inner, &prompt, n)?;
        for correct in 0..=DFLASH2_DRAFT_PROPOSALS {
            let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8101);
            let (tokens, end, deferred, linear, gdn) =
                paged_dflash2(&mut inner, &prompt, &reference, correct)?;
            assert!(tokens.len() >= n, "{correct}: {} tokens", tokens.len());
            assert_eq!(
                end, ar_cursor,
                "paged MoE DFlash2 with {correct} correct proposals left the AR cursor"
            );
            assert!(
                deferred.iter().all(|&launches| launches == linear),
                "{correct}: deferred launches per cycle {deferred:?} for {linear} linear layers"
            );
            let tag = format!("MoE correct={correct}");
            assert_cells(&gdn.after, &replay_chain(&gdn), &tag);
            if correct == 3 {
                assert_cells(&gdn.after, &kernel_fold(&gdn), &tag);
            }
        }
        Ok(())
    }

    /// Splash `decode_accept_dflash`: the verify's row budget is
    /// `min(budget, 1 + L)` for a driver handed L proposals. Mutation B2 (the
    /// clamp removed) turns it red.
    #[test]
    fn paged_moe_verify_clamps_the_row_budget() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("paged_moe_verify_clamps_the_row_budget")
        {
            return Ok(());
        }
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8102);
        let prompt = prompt(40);
        let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
        let logits = inner.paged_prefill_with_draft_state(&prompt, &prefix, stream)?;
        let anchor = argmax_rows(&logits)?[0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
        let mut proposal = step.propose(anchor, 2, &greedy(), &mut rng)?;
        assert_eq!(proposal.draft_len()?, 2, "the driver gets its cap");
        proposal.materialize_draft_ids()?;
        let mut ids = vec![anchor as i32];
        ids.extend(proposal.draft_ids.iter().copied());
        step.verify_row_budget = Some(MxArray::from_int32(&[5], &[1])?);
        let device = MxArray::from_int32(&ids, &[ids.len() as i64])?;
        let logits = step.verify_device(&device)?.logits;
        assert_eq!(logits.shape()?.as_ref(), &[1, 3, i64::from(VOCAB)]);
        logits.eval();
        let budget = step
            .device_verification_budget()
            .expect("a budgeted verify keeps its budget")
            .to_int32()?;
        assert_eq!(budget.as_ref(), &[3], "min(budget 5, 1 + 2 proposals)");
        let host: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        step.commit_with_provenance(1, host.len(), &host)?;
        step.finish()?;
        Ok(())
    }

    /// A poolless load refuses the companion.
    #[test]
    fn moe_dflash2_companion_needs_the_pool() {
        let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8103);
        inner
            .require_dflash2_pool()
            .expect("a pool admits the companion");
        inner.paged_adapter = None;
        let error = inner
            .require_dflash2_pool()
            .expect_err("no pool refuses the companion");
        assert!(
            error.reason.contains("block-paged KV pool"),
            "{}",
            error.reason
        );
    }

    /// The MoE turn parks its prompt's replay point once its blocks
    /// register, and the next DFlash2 turn on that prefix resumes there
    /// though the live window ends elsewhere (Splash `Cache::lookup`).
    /// Mutation MS1 (the prime skips the parked-state lookup) → red (hit 0).
    #[test]
    fn moe_parked_replay_point_resumes_the_next_session() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "moe_parked_replay_point_resumes_the_next_session",
        ) {
            return Ok(());
        }
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let a1 = prompt(40);
        let mut ar_inner = tiny_paged_int8_moe_dflash_inner(0x5_8104);
        let (reference, _) = paged_ar(&mut ar_inner, &a1, 12)?;
        let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8104);
        paged_dflash2(&mut inner, &a1, &reference, 2)?;
        inner.finalize_paged_turn(true, 0);
        let parked: Vec<usize> = inner
            .dflash2_states
            .store
            .entries()
            .map(|entry| entry.depth())
            .collect();
        assert_eq!(parked, vec![32]);
        let mut a2 = a1[..32].to_vec();
        a2.extend((0..20u32).map(|i| (i * 5 + 2) % VOCAB));
        let prefix = inner.prime_dspark_prefix_state(&a2, true, 0)?;
        assert_eq!(prefix.effective_cached_prefix_len(), 32);
        let logits = inner.paged_prefill_with_draft_state(&a2[32..], &prefix, stream)?;
        logits.eval();
        Ok(())
    }
    /// An autoregressive MoE turn on a draft load primes and prefills as a
    /// DFlash2 turn (Splash `configureDraftStatePlan` runs for every
    /// request): it parks its replay point with the draft window, its first
    /// token is plain paged AR's, and the next autoregressive turn of the
    /// prefix resumes there.
    /// Mutation MDL1m (`draft_lane_turn` false) → red (nothing parked).
    #[test]
    fn moe_ar_turn_on_a_draft_load_publishes_draft_states() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "moe_ar_turn_on_a_draft_load_publishes_draft_states",
        ) {
            return Ok(());
        }
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let a1 = prompt(40);
        let mut plain = tiny_paged_int8_moe_dflash_inner(0x5_8105);
        let (reference, _) = paged_ar(&mut plain, &a1, 1)?;
        let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8105);
        let prefix = PagedBackend::prime_prefix_state(&mut inner, &a1, true, 0, &[], 0)?;
        let logits = PagedBackend::paged_prefill(
            &mut inner,
            &a1[prefix.effective_cached_prefix_len()..],
            &prefix,
            stream,
        )?;
        assert_eq!(argmax_rows(&logits)?[..1], reference[..1]);
        inner.finalize_paged_turn(true, 0);
        let parked: Vec<(usize, bool)> = inner
            .dflash2_states
            .store
            .entries()
            .map(|entry| {
                (
                    entry.depth(),
                    entry.ram().is_some_and(|state| state.window.is_some()),
                )
            })
            .collect();
        assert_eq!(parked, vec![(32, true)]);
        let mut a2 = a1[..32].to_vec();
        a2.extend((0..20u32).map(|i| (i * 5 + 2) % VOCAB));
        let prefix = PagedBackend::prime_prefix_state(&mut inner, &a2, true, 0, &[], 0)?;
        assert_eq!(prefix.effective_cached_prefix_len(), 32);
        Ok(())
    }

    /// Splash `Runtime::suspend` → `QwenStateStorage::releaseLane` on the
    /// MoE serial lane: a verify the host refuses (critical) suspends the
    /// lane; its two GDN cells and pending rows are gone before the wait,
    /// its draft ring waits in the idle lane; the replay's lane verifies.
    /// Mutation R1m (the stepper keeps its lane across the claim) → red.
    #[test]
    fn moe_suspended_dflash2_lane_drops_its_cells_and_idles_its_ring() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "moe_suspended_dflash2_lane_drops_its_cells_and_idles_its_ring",
        ) {
            return Ok(());
        }
        use crate::engine::governor::governor::Governor;
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        use crate::engine::governor::host::Pressure;
        use crate::models::qwen3_5::dflash2_decode::lane_probe;
        use std::sync::{Arc, Mutex};
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let io = FakeIo::new(stats(0), host(32 * G, Pressure::Normal));
        let governor: &'static Governor = Box::leak(Box::new(Governor::new(io.clone())));
        let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8105);
        // A pool of two blocks that grows by two through the governor.
        let adapter = inner.paged_adapter.take().expect("pool");
        let block_size = adapter.block_size();
        let pool = adapter.layer_kv_pool();
        let (config, dtype) = (pool.config().clone(), pool.cache_dtype());
        drop(adapter);
        let pool =
            mlx_paged_attn::LayerKVPool::new(config, 2, 8, dtype).map_err(Error::from_reason)?;
        let allocator = Arc::new(Mutex::new(mlx_paged_attn::BlockAllocator::with_recency(
            2,
            8,
            block_size,
            Arc::clone(&inner.cache_recency),
        )));
        let mut adapter = crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter::new(
            allocator,
            Arc::new(pool),
            block_size,
        )
        .map_err(Error::from_reason)?;
        adapter.set_grow_headroom_probe_override(Some(Arc::new(Ok)));
        adapter.set_admission(Arc::new(governor));
        adapter.set_extent_growth(2);
        inner.paged_adapter = Some(adapter);
        inner.memory_registration =
            Some(governor.register_model(model(16 * G, 0)).expect("register"));
        type Held = ((usize, usize), (usize, usize));
        let in_wait: Arc<Mutex<Option<Held>>> = Arc::default();
        {
            let seen = Arc::clone(&in_wait);
            io.with(|state| {
                state.on_sleep = Some(Box::new(move |state| {
                    let mut seen = seen.lock().unwrap();
                    if seen.is_none() {
                        *seen = Some(lane_probe::held());
                    }
                    if state.sleeps >= 3 {
                        state.host = host(32 * G, Pressure::Normal);
                    }
                }));
            });
        }
        // Eight rows left: the first verify fills the second block, the
        // second needs a third.
        let prompt = prompt(2 * block_size as usize - 8);
        let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
        let logits = inner.paged_prefill_with_draft_state(&prompt, &prefix, stream)?;
        let mut anchor = argmax_rows(&logits)?[0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
        for cycle in 0..2 {
            if cycle == 1 {
                io.with(|state| state.host = host(32 * G, Pressure::Critical));
            }
            let mut proposal =
                step.propose(anchor, DFLASH2_DRAFT_PROPOSALS, &greedy(), &mut rng)?;
            proposal.materialize_draft_ids()?;
            let mut ids = vec![anchor];
            ids.extend(proposal.draft_ids.iter().map(|&id| id as u32));
            let rows = argmax_rows(&step.verify(&ids)?.logits)?;
            step.commit_with_provenance(1, ids.len(), &ids)?;
            anchor = rows[0];
        }
        step.finish()?;
        assert_eq!(governor.suspensions(7), 1);
        assert_eq!(governor.resumptions(7), 1);
        let ((cells, held), (ring, idle)) = in_wait.lock().unwrap().expect("the lane waited");
        assert!(cells >= 4, "{cells} cell arrays recorded");
        assert_eq!(
            held, 0,
            "{held} of the lane's {cells} cell arrays outlived the vacate"
        );
        assert!(ring > 0, "{ring} ring arrays recorded");
        assert_eq!(idle, ring, "the idle lane keeps the ring");
        Ok(())
    }

    /// The MoE napi object's handle names the namespace directory the
    /// model's persistent state tier opened, read from the tier; after
    /// `closeCache`, and for a temporary tier, it names none.
    /// Mutation: the tier names its namespace after its close → red.
    #[test]
    fn moe_tier_handle_names_the_open_namespace() -> Result<()> {
        use crate::models::qwen3_5::dflash2_state::disk::close_cache;
        use crate::models::qwen3_5::dflash2_state::disk::tests::{
            attach_tier, persistent_test_tier, test_tier,
        };
        use crate::models::qwen3_5::dflash2_turn::DFlash2Target;
        let _ctx = StreamContext::new(Stream::generation());
        for persistent in [true, false] {
            let tier = if persistent {
                persistent_test_tier("state-tier-moe-namespace", 1 << 30)
            } else {
                test_tier("state-tier-moe-namespace-temp", 1 << 30)
            };
            let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8261);
            let dense = DFlash2Target::dense_config(&inner).into_owned();
            let window = inner
                .dflash2
                .as_ref()
                .expect("draft")
                .config
                .context_window()?;
            attach_tier(
                &mut inner.dflash2_states,
                &dense,
                inner.paged_adapter.as_ref().expect("pool"),
                &tier,
                window.nbytes(),
                window.slots(),
            );
            let handle = inner.dflash2_states.tier_handle();
            let dir = tier.tier.dir();
            let want = persistent.then(|| dir.file_name().and_then(|name| name.to_str()));
            assert_eq!(handle.namespace().as_deref(), want.flatten());
            assert!(close_cache(
                &mut inner,
                std::time::Instant::now() + std::time::Duration::from_secs(2)
            ));
            assert_eq!(handle.namespace(), None);
        }
        Ok(())
    }

    /// The MoE model's `closeCache` command at a clean stop, its deadline in
    /// milliseconds on `Date.now()`'s clock: a persistent tier writes the
    /// waiting point with its chain before that deadline and reports it.
    /// Mutation CC1 (the deadline is read as now) → red.
    #[test]
    fn moe_close_cache_command_flushes_by_its_deadline() -> Result<()> {
        use crate::engine::model_command::FamilyCommand;
        use crate::models::qwen3_5::dflash2_state::disk::tests::{
            attach_tier, persistent_test_tier, writable_anchor,
        };
        use crate::models::qwen3_5::dflash2_state::store::{Lineage, StateClass};
        use crate::models::qwen3_5::dflash2_state::write_behind::{MINIMUM_TOKENS, published};
        use crate::models::qwen3_5::dflash2_turn::DFlash2Target;
        use crate::models::qwen3_5::model::Qwen35FamilyCommand;
        let _ctx = StreamContext::new(Stream::generation());
        let tier = persistent_test_tier("state-tier-moe-close-command", 1 << 30);
        let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8260);
        let dense = DFlash2Target::dense_config(&inner).into_owned();
        let window = inner
            .dflash2
            .as_ref()
            .expect("draft")
            .config
            .context_window()?;
        attach_tier(
            &mut inner.dflash2_states,
            &dense,
            inner.paged_adapter.as_ref().expect("pool"),
            &tier,
            window.nbytes(),
            window.slots(),
        );
        let adapter = inner.paged_adapter.as_ref().expect("pool");
        let block_size = adapter.block_size();
        let blocks = MINIMUM_TOKENS.div_ceil(block_size);
        let tokens: Vec<u32> = (0..blocks * block_size)
            .map(|i| (i * 3 + 1) % VOCAB)
            .collect();
        {
            let mut allocator = adapter.allocator().lock().expect("allocator");
            let held: Vec<_> = (0..blocks)
                .map(|_| allocator.allocate().expect("a block for the chain"))
                .collect();
            allocator
                .cache_full_blocks_with_keys(
                    &tokens,
                    &held,
                    block_size,
                    mlx_paged_attn::PrefixKeys::Uniform(&[]),
                    0,
                )
                .expect("cache the chain");
            for block in held {
                allocator.free(block);
            }
        }
        let anchor = writable_anchor(&dense, window);
        let p = inner
            .dflash2_states
            .store
            .insert(Lineage::text(&tokens, 0), anchor, StateClass::Ordinary)
            .expect("a whole-block state")
            .id;
        inner.sync_gdn_state_marks();
        published(&mut inner.dflash2_states, p, blocks * block_size, 0.0);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            * 1000.0;
        let (reply, mut flushed) = tokio::sync::oneshot::channel();
        Qwen35FamilyCommand::CloseCache {
            deadline_ms: now_ms + 6000.0,
            reply,
        }
        .execute(&mut inner);
        assert!(flushed.try_recv().expect("a reply")?);
        let copy = inner
            .dflash2_states
            .store
            .get(p)
            .and_then(|entry| entry.disk().copied());
        assert!(copy.is_some_and(|copy| copy.landed && !copy.chain_owed));
        Ok(())
    }

    /// Splash `Engine::vacateLane` on a MoE DFlash2 turn: the host refuses
    /// the tapped prefill's chunk after 64 of a 100-token prompt, then (a
    /// second run) a verify once the lane fed 113. Each time the lane is
    /// vacated and replayed, streams an unsuspended run's tokens, and its
    /// draft ring goes to the idle lane and comes back: no ring is
    /// allocated beyond the unsuspended run's.
    /// Mutations MVC1 (the MoE vacate drops the lane's buffers) and MVP1
    /// (the MoE verify keeps its ring off the lane during its claim) → red.
    #[test]
    fn moe_suspension_replays_on_the_lane_buffers() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "moe_suspension_replays_on_the_lane_buffers",
        ) {
            return Ok(());
        }
        use crate::engine::governor::governor::tests::{G, model};
        use crate::engine::governor::host::Pressure;
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::models::qwen3_5::dflash2::RING_ALLOCATIONS;
        use crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device;
        use crate::models::qwen3_5::dflash2_turn::YIELD_FROM;
        use crate::models::qwen3_5::model::dflash2_park_tests::{
            LoggingSink, StreamLog, fake_governor, run_streamed_turn_for,
        };
        use crate::models::qwen3_5::paged_forward::PREFILL_CHUNK_ROWS_OVERRIDE;
        use std::sync::atomic::AtomicBool;
        if !metal_is_default_device() {
            return Ok(());
        }
        const MAX_NEW: u32 = 32;
        let decoder = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let prompt = prompt(100);
        let run = |yield_from: Option<usize>| -> Result<(StreamLog, u64, u64, u64)> {
            let (_io, governor) = fake_governor(Pressure::Normal);
            let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8160);
            inner.memory_registration =
                Some(governor.register_model(model(16 * G, 0)).expect("register"));
            inner
                .paged_adapter
                .as_mut()
                .expect("adapter")
                .set_admission(std::sync::Arc::new(governor));
            RING_ALLOCATIONS.with(|count| count.set(0));
            YIELD_FROM.with(|from| from.set(yield_from));
            let sink = LoggingSink::new(None);
            let ran = run_streamed_turn_for(
                &mut inner,
                decoder,
                &prompt,
                &sink,
                &AtomicBool::new(false),
                MAX_NEW,
            );
            YIELD_FROM.with(|from| from.set(None));
            ran?;
            let log = std::mem::take(&mut *sink.log.lock().unwrap());
            Ok((
                log,
                RING_ALLOCATIONS.with(std::cell::Cell::get),
                governor.suspensions(7),
                governor.resumptions(7),
            ))
        };
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(4)));
        let result = (|| -> Result<()> {
            let (solo, rings, suspensions, _) = run(None)?;
            assert_eq!(suspensions, 0);
            assert_eq!(solo.tokens, MAX_NEW);
            assert!(rings > 0);
            for (from, what) in [(64, "prefill"), (113, "decode")] {
                let (log, suspended_rings, suspensions, resumptions) = run(Some(from))?;
                assert_eq!((suspensions, resumptions), (1, 1), "{what}");
                assert_eq!(suspended_rings, rings, "{what}: the idle ring comes back");
                assert_eq!(log.done, solo.done, "{what}: the unsuspended tokens");
            }
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    fn bits(array: &MxArray) -> Result<Vec<u32>> {
        let f = array.astype(DType::Float32)?;
        f.eval();
        Ok(f.to_float32()?
            .iter()
            .map(|value| value.to_bits())
            .collect())
    }

    /// One stored state: its depth, its GDN slots' bits, and its window's
    /// bounds and rows' bits.
    type DraftState = (usize, Vec<Vec<u32>>, (u32, u32), Vec<u32>);

    fn window_state(
        window: &crate::models::qwen3_5::draft_context::DraftContextWindow,
    ) -> Result<((u32, u32), Vec<u32>)> {
        let rows = bits(&window.load_rows(window.begin(), window.end() - window.begin())?)?;
        Ok(((window.begin(), window.end()), rows))
    }

    /// The states on `tokens`' lineage, by depth.
    fn draft_states(inner: &Qwen35Inner, tokens: &[u32]) -> Result<Vec<DraftState>> {
        let mut states = Vec::new();
        for entry in inner.dflash2_states.store.entries() {
            if !tokens.starts_with(entry.tokens()) {
                continue;
            }
            let anchor = entry.ram().expect("a RAM copy");
            let mut arrays = Vec::new();
            for cache in &anchor.linear {
                if matches!(
                    cache,
                    crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache::Linear(_)
                ) {
                    cache.collect_arrays(&mut arrays);
                }
            }
            let cell = arrays.into_iter().map(bits).collect::<Result<Vec<_>>>()?;
            let (bounds, rows) = window_state(anchor.window.as_ref().expect("a draft window"))?;
            states.push((entry.depth(), cell, bounds, rows));
        }
        states.sort_by_key(|state| state.0);
        Ok(states)
    }

    /// The fixture with a `rows`-row draft window, so a point's window
    /// spans prefill slices.
    fn wide_draft_inner(seed: u64, rows: usize) -> Qwen35Inner {
        let mut inner = tiny_paged_int8_moe_dflash_inner(seed);
        inner.dflash2.as_mut().expect("draft").config.sliding_window = rows;
        inner
    }

    /// The states a whole autoregressive MoE turn of `tokens` on a draft
    /// load publishes (its tapped prime and prefill), and its logits.
    fn whole_turn_states(seed: u64, tokens: &[u32]) -> Result<(Vec<DraftState>, Vec<u32>)> {
        let mut inner = wide_draft_inner(seed, 40);
        let prefix = inner.prime_dspark_prefix_state(tokens, true, 0)?;
        assert_eq!(prefix.effective_cached_prefix_len(), 0);
        let logits = inner.paged_prefill_with_draft_state(tokens, &prefix, Stream::generation())?;
        Ok((draft_states(&inner, tokens)?, bits(&logits)?))
    }

    /// The scheduled prime of `tokens` on `seq`, the sequence parked after
    /// it.
    fn scheduled_prime(
        inner: &mut Qwen35Inner,
        seq: u32,
        tokens: &[u32],
    ) -> Result<<Qwen35Inner as PagedBackend>::PrefixState> {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        HybridSchedulerBackend::activate_scheduled_recurrent(inner, seq)?;
        let base = PagedBackend::prime_prefix_state(inner, tokens, true, 16, &[], 0)?;
        HybridSchedulerBackend::park_active_scheduled_recurrent(inner)?;
        Ok(base)
    }

    /// The prefill slice `[start, end)` of `seq`'s request, the sequence
    /// parked after it; returns the slice's logits.
    fn scheduled_slice(
        inner: &mut Qwen35Inner,
        seq: u32,
        tokens: &[u32],
        base: &<Qwen35Inner as PagedBackend>::PrefixState,
        start: usize,
        end: usize,
    ) -> Result<MxArray> {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        HybridSchedulerBackend::activate_paged_seq(inner, seq)?;
        let prefix = HybridSchedulerBackend::build_scheduled_prefix(
            inner,
            base,
            start,
            end - start,
            tokens.to_vec(),
            start == base.effective_cached_prefix_len(),
        );
        let logits = PagedBackend::paged_prefill(
            inner,
            &tokens[start..end],
            &prefix,
            Stream::default(crate::stream::DeviceType::Gpu),
        )?;
        logits.eval();
        HybridSchedulerBackend::park_active_scheduled_recurrent(inner)?;
        Ok(logits)
    }

    /// The dense `scheduled_ar_on_a_draft_load_publishes_draft_states` on a
    /// MoE load: two scheduled sequences whose prefill slices interleave
    /// each publish the states, GDN cells and draft windows a whole turn of
    /// their prompt does; the turns' ends give the windows back.
    /// Mutation SD1m (a scheduled MoE prefill on a draft load keeps no
    /// sink) → red.
    #[test]
    fn moe_scheduled_ar_on_a_draft_load_publishes_draft_states() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "moe_scheduled_ar_on_a_draft_load_publishes_draft_states",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        use crate::models::qwen3_5::paged_forward::PREFILL_CHUNK_ROWS_OVERRIDE;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8180;
        let prompts = [
            (1u32, prompt(150)),
            (
                2u32,
                (0..150u32).map(|i| (i * 5 + 1) % VOCAB).collect::<Vec<_>>(),
            ),
        ];
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            let mut references = Vec::new();
            for (_, tokens) in &prompts {
                let (states, _) = whole_turn_states(seed, tokens)?;
                assert_eq!(
                    states.iter().map(|state| state.0).collect::<Vec<_>>(),
                    vec![80, 144]
                );
                references.push(states);
            }
            let mut inner = wide_draft_inner(seed, 40);
            let mut bases = Vec::new();
            for (seq, tokens) in &prompts {
                let base = scheduled_prime(&mut inner, *seq, tokens)?;
                assert_eq!(base.effective_cached_prefix_len(), 0);
                bases.push(base);
            }
            for (start, end) in [(0, 48), (48, 112), (112, 150)] {
                for ((seq, tokens), base) in prompts.iter().zip(&bases) {
                    scheduled_slice(&mut inner, *seq, tokens, base, start, end)?;
                }
            }
            assert!(inner.dflash2.is_some(), "the draft is back");
            for ((seq, tokens), reference) in prompts.iter().zip(&references) {
                assert_eq!(
                    &draft_states(&inner, tokens)?,
                    reference,
                    "sequence {seq} publishes the whole turn's states"
                );
            }
            assert_eq!(inner.dflash2_states.seq_windows.len(), 2);
            for (seq, _) in &prompts {
                HybridSchedulerBackend::scheduled_turn_closed(&mut inner, *seq);
            }
            assert!(inner.dflash2_states.seq_windows.is_empty());
            assert!(inner.dflash2_states.idle_lane.window.is_some());
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// The states a whole autoregressive MoE turn of `tokens` on a draft
    /// load publishes when its generation prompt is the template's.
    fn whole_turn_states_with_prompt(seed: u64, tokens: &[u32]) -> Result<Vec<DraftState>> {
        let mut inner = wide_draft_inner(seed, 40);
        inner.dflash2_states.generation_suffix = Some(std::sync::Arc::from(
            &crate::models::qwen3_5::dflash2_decode::tests::GENERATION_PROMPT[..],
        ));
        let prefix = inner.prime_dspark_prefix_state(tokens, true, 0)?;
        assert_eq!(prefix.effective_cached_prefix_len(), 0);
        inner.paged_prefill_with_draft_state(tokens, &prefix, Stream::generation())?;
        draft_states(&inner, tokens)
    }

    /// The dense `scheduler_runs_draft_load_ar_sessions_on_the_state_tier`
    /// on a MoE load: through the model thread's scheduler, two sessions
    /// started together run as two live sequences and each publishes the
    /// states, GDN cells and draft windows a whole turn of its own prompt
    /// does; after a restart a session extending the first one's prompt
    /// resumes at its replay point (144) from disk, and the states past it
    /// are the whole turn's of the longer prompt.
    /// Mutation DS1m (the scheduled MoE prime on a draft load primes plain)
    /// → red.
    #[test]
    fn moe_scheduler_runs_draft_load_ar_sessions_on_the_state_tier() -> Result<()> {
        use crate::models::qwen3_5::dflash2_decode::tests::{
            content_tokenizer, drive_session_starts, metal_is_default_device, rendered_prompt,
        };
        if crate::test_support::paged_prefill_unsupported(
            "moe_scheduler_runs_draft_load_ar_sessions_on_the_state_tier",
        ) || !metal_is_default_device()
        {
            return Ok(());
        }
        use crate::engine::governor::governor::Governor;
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        use crate::engine::hybrid_scheduler::HybridSchedulerState;
        use crate::models::qwen3_5::dflash2_state::disk::tests::{
            TestTier, attach_tier, persistent_test_tier,
        };
        use crate::models::qwen3_5::dflash2_state::disk::{adopt_all, close_cache};
        use crate::models::qwen3_5::paged_forward::PREFILL_CHUNK_ROWS_OVERRIDE;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_81B0;
        let scheduler_on = |tier: &TestTier| -> Result<HybridSchedulerState<Qwen35Inner>> {
            let mut inner = wide_draft_inner(seed, 40);
            let window = inner
                .dflash2
                .as_ref()
                .expect("draft")
                .config
                .context_window()?;
            let dense = inner.config.clone();
            attach_tier(
                &mut inner.dflash2_states,
                &dense,
                inner.paged_adapter.as_ref().expect("pool"),
                tier,
                window.nbytes(),
                window.slots(),
            );
            inner.paged_adapter.as_mut().expect("pool").set_cold_tier(
                crate::models::qwen3_5::dflash2_state::disk::state_context(&tier.tier),
            );
            inner.tokenizer = Some(content_tokenizer());
            let io = FakeIo::new(
                stats(0),
                host(32 * G, crate::engine::governor::host::Pressure::Normal),
            );
            let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
            inner.memory_registration =
                Some(governor.register_model(model(16 * G, 0)).expect("register"));
            let mut state = HybridSchedulerState::new(inner)?;
            state.enable_continuous_batching();
            Ok(state)
        };
        let first = prompt(150);
        let second: Vec<u32> = (0..150u32).map(|i| (i * 5 + 1) % VOCAB).collect();
        let longer = prompt(180);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            let references = [
                whole_turn_states_with_prompt(seed, &rendered_prompt(&first))?,
                whole_turn_states_with_prompt(seed, &rendered_prompt(&second))?,
            ];
            for reference in &references {
                assert_eq!(
                    reference.iter().map(|state| state.0).collect::<Vec<_>>(),
                    vec![80, 144]
                );
            }
            let longer_reference = whole_turn_states_with_prompt(seed, &rendered_prompt(&longer))?;
            let tier = persistent_test_tier("state-tier-moe-sched-drive-draft", 1 << 30);
            let mut state = scheduler_on(&tier)?;
            let (results, most_live) =
                drive_session_starts(&mut state, &[("a", &first), ("b", &second)], 2)?;
            assert_eq!(most_live, 2, "the two sessions ran as two live sequences");
            for (result, (body, reference)) in results
                .iter()
                .zip([&first, &second].into_iter().zip(&references))
            {
                assert_eq!(result.cached_tokens, 0);
                assert_eq!(
                    &draft_states(&state.inner, &rendered_prompt(body))?,
                    reference,
                    "each session publishes its own whole turn's states"
                );
            }
            assert!(state.inner.dflash2.is_some(), "the draft is back");
            assert!(state.inner.dflash2_states.seq_windows.is_empty());
            assert!(state.inner.dflash2_states.turns.is_empty());
            let replay = draft_states(&state.inner, &rendered_prompt(&first))?
                .into_iter()
                .find(|stored| stored.0 == 144)
                .expect("the replay point at 144");
            assert!(close_cache(
                &mut state.inner,
                std::time::Instant::now() + std::time::Duration::from_secs(30)
            ));
            drop(state);

            let tier = tier.reopen("state-tier-moe-sched-drive-draft", 1 << 30);
            let mut state = scheduler_on(&tier)?;
            assert!(adopt_all(&mut state.inner) >= 1);
            let (results, _) = drive_session_starts(&mut state, &[("c", &longer)], 2)?;
            assert_eq!(
                results[0].cached_tokens, 144,
                "the session resumed from disk"
            );
            let disk = state
                .inner
                .dflash2_states
                .disk
                .as_ref()
                .expect("tier")
                .counters;
            assert_eq!((disk.promotions, disk.invalidations), (1, 0));
            let states = draft_states(&state.inner, &rendered_prompt(&longer))?;
            assert!(
                states.contains(&replay),
                "the promoted state is the stored one"
            );
            let past: Vec<_> = states.into_iter().filter(|stored| stored.0 > 144).collect();
            let expected: Vec<_> = longer_reference
                .into_iter()
                .filter(|stored| stored.0 > 144)
                .collect();
            assert!(expected.iter().any(|stored| stored.2.0 < 144));
            assert_eq!(
                past, expected,
                "the states past the resume continue the stored window"
            );
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// The dense `scheduled_ar_on_a_draft_load_resumes_after_a_restart` on
    /// a MoE load: after a restart a scheduled request extending a stored
    /// prompt resumes at its replay point (144) from disk with the stored
    /// window; the states it publishes past it and its logits equal a whole
    /// turn's of the longer prompt.
    /// Mutation SR1 (the scheduled prime resumes at no state) → red.
    #[test]
    fn moe_scheduled_ar_on_a_draft_load_resumes_after_a_restart() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "moe_scheduled_ar_on_a_draft_load_resumes_after_a_restart",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        use crate::models::qwen3_5::dflash2_state::disk::tests::{
            TestTier, attach_tier, persistent_test_tier,
        };
        use crate::models::qwen3_5::dflash2_state::disk::{adopt_all, close_cache};
        use crate::models::qwen3_5::paged_forward::PREFILL_CHUNK_ROWS_OVERRIDE;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8190;
        let on_tier = |tier: &TestTier| -> Result<Qwen35Inner> {
            let mut inner = wide_draft_inner(seed, 40);
            let window = inner
                .dflash2
                .as_ref()
                .expect("draft")
                .config
                .context_window()?;
            let dense = inner.config.clone();
            attach_tier(
                &mut inner.dflash2_states,
                &dense,
                inner.paged_adapter.as_ref().expect("pool"),
                tier,
                window.nbytes(),
                window.slots(),
            );
            inner.paged_adapter.as_mut().expect("pool").set_cold_tier(
                crate::models::qwen3_5::dflash2_state::disk::state_context(&tier.tier),
            );
            Ok(inner)
        };
        let first = prompt(150);
        let longer = prompt(180);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            let (reference, reference_logits) = whole_turn_states(seed, &longer)?;
            let tier = persistent_test_tier("state-tier-moe-sched-draft-restart", 1 << 30);
            let mut inner = on_tier(&tier)?;
            let base = scheduled_prime(&mut inner, 1, &first)?;
            for (start, end) in [(0, 64), (64, 150)] {
                scheduled_slice(&mut inner, 1, &first, &base, start, end)?;
            }
            HybridSchedulerBackend::release_scheduled_cache(&mut inner, 1)?;
            HybridSchedulerBackend::scheduled_turn_closed(&mut inner, 1);
            let replay = draft_states(&inner, &first)?
                .into_iter()
                .find(|state| state.0 == 144)
                .expect("the replay point at 144");
            assert!(close_cache(
                &mut inner,
                std::time::Instant::now() + std::time::Duration::from_secs(30)
            ));
            drop(inner);

            let tier = tier.reopen("state-tier-moe-sched-draft-restart", 1 << 30);
            let mut inner = on_tier(&tier)?;
            assert!(adopt_all(&mut inner) >= 1);
            let base = scheduled_prime(&mut inner, 4, &longer)?;
            assert_eq!(base.effective_cached_prefix_len(), 144);
            assert_eq!(
                window_state(
                    inner
                        .dflash2_states
                        .seq_windows
                        .get(&4)
                        .expect("the lane's window")
                )?,
                (replay.2, replay.3.clone()),
                "the lane holds the stored window"
            );
            let logits = scheduled_slice(&mut inner, 4, &longer, &base, 144, 180)?;
            let disk = inner.dflash2_states.disk.as_ref().expect("tier").counters;
            assert_eq!((disk.promotions, disk.invalidations), (1, 0));
            let states = draft_states(&inner, &longer)?;
            assert!(
                states.contains(&replay),
                "the promoted state is the stored one"
            );
            let past: Vec<_> = states.into_iter().filter(|state| state.0 > 144).collect();
            let expected: Vec<_> = reference
                .into_iter()
                .filter(|state| state.0 > 144)
                .collect();
            assert!(expected.iter().any(|state| state.0 == 176));
            assert_eq!(
                past, expected,
                "the state past the resume continues the window"
            );
            assert_eq!(bits(&logits)?, reference_logits);
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    fn sampled_config(max_new_tokens: i32) -> crate::engine::types::ChatConfig {
        crate::engine::types::ChatConfig {
            temperature: Some(0.6),
            top_p: Some(0.95),
            top_k: Some(20),
            min_p: Some(0.0),
            max_new_tokens: Some(max_new_tokens),
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        }
    }

    /// The sampled cycle of `run_dspark_turn` on the device rows: propose
    /// (the MoE draft is walked on the host), verify the host ids, queue the
    /// rows, accept, commit, queue the boundary. Returns the tokens, the
    /// cached prefix and the cycles the rows took.
    fn sampled_rows_cycles(
        inner: &mut Qwen35Inner,
        prompt: &[u32],
        n: usize,
        seed: u64,
    ) -> Result<(Vec<u32>, usize, usize)> {
        let prefix = inner.prime_dspark_prefix_state(prompt, true, 0)?;
        let cached = prefix.effective_cached_prefix_len();
        let suffix = prompt[cached..].to_vec();
        let logits =
            inner.paged_prefill_with_draft_state(&suffix, &prefix, Stream::generation())?;
        let mut generated = argmax_rows(&logits)?;
        generated.drain(..generated.len() - 1);
        let params = crate::engine::extract_chat_params(&sampled_config(n as i32));
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut cycles = 0;
        let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
        while generated.len() < n {
            let anchor = *generated.last().expect("one token");
            let l_cap = DFLASH2_DRAFT_PROPOSALS.min(n - generated.len() - 1);
            if l_cap == 0 {
                break;
            }
            let mut proposal = step.propose(anchor, l_cap, &params, &mut rng)?;
            if proposal.device_draft_ids.is_some() {
                return Err(Error::from_reason(
                    "a sampled MoE draft is walked on the host",
                ));
            }
            let mut ids = vec![anchor];
            ids.extend(proposal.draft_ids.iter().map(|&id| id as u32));
            let verified = step.verify(&ids)?;
            let rows = crate::engine::dspark_turn::submit_sampled_rows(
                &verified.logits,
                &proposal,
                &params,
                &mut rng,
            )?
            .ok_or_else(|| Error::from_reason("the device rows take a sampled MoE cycle"))?;
            let (k, boundary) = rows.accept(&mut proposal, None)?;
            cycles += 1;
            step.commit_with_provenance(k + 1, ids.len(), &ids)?;
            generated.extend_from_slice(&ids[1..=k]);
            generated.push(boundary);
            DsparkStepper::eval_boundary(&step, &MxArray::from_int32(&[boundary as i32], &[1])?);
        }
        step.finish()?;
        generated.truncate(n);
        Ok((generated, cached, cycles))
    }

    /// One product whole turn (`run_paged_turn`, the draft-model plan) on
    /// `config`: (device rows cycles, (device, host) accepts).
    fn moe_product_turn(
        inner: &mut Qwen35Inner,
        tokens: &[u32],
        config: &crate::engine::types::ChatConfig,
    ) -> Result<(usize, (usize, usize))> {
        use crate::engine::backend::{ChatBackend, TurnOutput, WholeTurnArgs};
        use crate::engine::dspark_turn::{TEST_SAMPLED_ACCEPTS, TEST_SAMPLED_ROWS};
        use crate::engine::plan::{
            DecoderPlan, MediaCapabilities, MediaInputs, SpeculativeKind, TurnPlan,
        };
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let params = ChatBackend::resolve_params(inner, config);
        let mut args = WholeTurnArgs {
            tokens,
            tokenizer: &tokenizer,
            eos_id: u32::MAX,
            config,
            params: &params,
            thinking: ChatBackend::thinking_setup(inner, config),
            plan: TurnPlan {
                is_delta: false,
                input_media: MediaCapabilities::NONE,
                context_media: MediaCapabilities::NONE,
                use_paged_attention: true,
                decoder: DecoderPlan::Speculative(SpeculativeKind::DraftModel),
            },
            sink: None,
            cancelled: None,
            media: MediaInputs {
                images: &[],
                audio: &[],
            },
        };
        TEST_SAMPLED_ROWS.with(|c| c.set(0));
        TEST_SAMPLED_ACCEPTS.with(|c| c.set((0, 0)));
        let TurnOutput::Complete(result) = ChatBackend::run_paged_turn(inner, &mut args)? else {
            return Err(Error::from_reason("a sync turn completes"));
        };
        assert_eq!(result.num_tokens, config.max_new_tokens.unwrap_or(0) as u32);
        Ok((
            TEST_SAMPLED_ROWS.with(|c| c.get()),
            TEST_SAMPLED_ACCEPTS.with(|c| c.get()),
        ))
    }

    /// A sampled MoE DFlash2 whole turn takes the route the stepper
    /// documents: an unpenalized turn with `top_k` 1-256 walks its draft on
    /// the host and accepts every cycle on the device rows; a penalized turn
    /// and a `top_k` 0 turn keep the filter chain. On the rows, a seed
    /// repeats its tokens, another seed draws others, and a restored prefix
    /// gives the tokens of a cold one.
    #[test]
    fn a_sampled_moe_dflash2_turn_takes_the_device_rows() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_sampled_moe_dflash2_turn_takes_the_device_rows",
        ) || !unsafe { mlx_sys::mlx_metal_is_available() }
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8101;
        let first = prompt(40);
        let (rows, accepts) = moe_product_turn(
            &mut tiny_paged_int8_moe_dflash_inner(seed),
            &first,
            &sampled_config(24),
        )?;
        assert!(
            rows >= 3 && accepts == (0, 0),
            "the device rows: {rows} cycles, (device, host) accepts {accepts:?}"
        );
        let mut penalized = sampled_config(24);
        penalized.repetition_penalty = Some(1.1);
        let mut unbounded = sampled_config(24);
        unbounded.top_k = Some(0);
        for (tag, config) in [("penalized", penalized), ("top_k 0", unbounded)] {
            let (rows, (device, host)) =
                moe_product_turn(&mut tiny_paged_int8_moe_dflash_inner(seed), &first, &config)?;
            assert!(
                rows == 0 && device == 0 && host >= 3,
                "{tag}: the filter chain: {rows} rows cycles, accepts ({device}, {host})"
            );
        }
        let run = |draws: u64| -> Result<(Vec<u32>, usize, usize)> {
            sampled_rows_cycles(
                &mut tiny_paged_int8_moe_dflash_inner(seed),
                &first,
                48,
                draws,
            )
        };
        let a = run(5)?;
        assert!(a.2 >= 6, "{} rows cycles", a.2);
        assert_eq!(a, run(5)?, "same seed, other tokens");
        assert_ne!(a.0, run(6)?.0, "the stream does not follow the seed");
        let mut second = first.clone();
        second.extend((0..8u32).map(|i| (i * 3 + 2) % VOCAB));
        let mut warm = tiny_paged_int8_moe_dflash_inner(seed);
        moe_product_turn(&mut warm, &first, &sampled_config(12))?;
        let restored = sampled_rows_cycles(&mut warm, &second, 32, 21)?;
        let fresh =
            sampled_rows_cycles(&mut tiny_paged_int8_moe_dflash_inner(seed), &second, 32, 21)?;
        assert!(restored.1 > 0, "the second turn restored nothing");
        assert_eq!(fresh.1, 0);
        assert_eq!(restored.0, fresh.0, "restored {} rows", restored.1);
        Ok(())
    }
}
