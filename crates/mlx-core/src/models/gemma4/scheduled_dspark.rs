//! Request-owned DSpark context with cycle-atomic, packed target verification.

use super::dspark::{DsparkContextCache, DsparkTap};
use super::dspark_decode::{
    Gemma4DsparkState, Gemma4DsparkStepper, dspark_confidence_threshold_from_env,
};
use super::model::{Gemma4Inner, Gemma4SpecPagedCache};
use crate::array::MxArray;
use crate::engine::backend::{DsparkProposal, DsparkStepper};
use crate::engine::hybrid_scheduler::{ScheduledVerifyCommit, ScheduledVerifyRow};
use crate::engine::params::ChatParams;
use crate::engine::scheduled_verify::{ScheduledDraftVerify, ScheduledVerifyBatch};
#[cfg(test)]
use crate::engine::spec_paged::SpecPagedCache;
use crate::transformer::paged_kv_cache_adapter::PagedRaggedRow;
use napi::bindgen_prelude::{Error, Result};

pub(crate) type Gemma4ScheduledVerify = ScheduledVerifyBatch;

impl Gemma4Inner {
    pub(crate) fn begin_scheduled_dspark(&mut self, seq_id: u32, position: u32) -> Result<()> {
        if self.scheduled_dspark_verify.is_some() {
            return Err(Error::from_reason(
                "cannot seed DSpark during an open verify batch",
            ));
        }
        let draft = self
            .dspark_draft()
            .ok_or_else(|| Error::from_reason("no DSpark draft loaded"))?;
        let state = Gemma4DsparkState {
            seq_id,
            ctx: DsparkContextCache::new(draft.num_layers()),
            next_pos: i32::try_from(position)
                .map_err(|_| Error::from_reason("DSpark position overflow"))?,
            ar_fallback: false,
            layer_ids: draft
                .config
                .target_layer_ids
                .iter()
                .map(|&id| id as usize)
                .collect(),
            layer_kinds: self.compute_layer_kinds()?,
            confidence_threshold: dspark_confidence_threshold_from_env(),
            adaptive_verification: false,
            open_cycle: None,
            tapped: None,
            ar_probe_pending: false,
        };
        self.scheduled_dspark_states.insert(seq_id, state);
        Ok(())
    }

    /// The owned state is removed during the borrow and returned only on
    /// success. A failed context mutation cannot be reused by a later cycle.
    fn with_scheduled_dspark<T>(
        &mut self,
        seq_id: u32,
        run: impl FnOnce(&mut Gemma4DsparkStepper<'_>) -> Result<T>,
    ) -> Result<T> {
        let state = self
            .scheduled_dspark_states
            .remove(&seq_id)
            .ok_or_else(|| {
                Error::from_reason(format!("DSpark sequence {seq_id} has no draft context"))
            })?;
        let mut stepper = Gemma4DsparkStepper { inner: self, state };
        let result = run(&mut stepper);
        let state = stepper.state;
        if result.is_ok() {
            self.scheduled_dspark_states.insert(seq_id, state);
        }
        result
    }

    pub(crate) fn propose_scheduled_dspark(
        &mut self,
        seq_id: u32,
        anchor: u32,
        max_drafts: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
        confidence: bool,
    ) -> Result<DsparkProposal> {
        self.with_scheduled_dspark(seq_id, |step| {
            step.set_adaptive_verification(confidence && step.supports_adaptive_verification());
            step.propose(anchor, max_drafts, params, rng)
        })
    }

    /// Collect only freshly computed target rows, including the last-token
    /// prefill split. Cached target prefixes remain authoritative without
    /// requiring a second full-prefix host or drafter cache restore.
    pub(crate) fn scheduled_dspark_tap(&self, seq_id: u32) -> Option<Vec<usize>> {
        self.scheduled_dspark_states
            .get(&seq_id)
            .map(|s| s.layer_ids.clone())
    }

    pub(crate) fn append_scheduled_dspark_prefill(
        &mut self,
        seq_id: u32,
        position: u32,
        tap: DsparkTap<'_>,
        count: usize,
    ) -> Result<()> {
        self.with_scheduled_dspark(seq_id, |step| {
            if step.state.next_pos != position as i32 {
                return Err(Error::from_reason(
                    "scheduled DSpark prefill frontier mismatch",
                ));
            }
            step.append_tapped_prefix(&tap.captured, count, "scheduled prefill")
        })
    }

    pub(crate) fn verify_scheduled_dspark(
        &mut self,
        rows: &[ScheduledVerifyRow],
    ) -> Result<MxArray> {
        self.verify_scheduled_rows(rows)
    }
    pub(crate) fn commit_scheduled_dspark(
        &mut self,
        rows: &[ScheduledVerifyCommit],
    ) -> Result<Vec<Result<()>>> {
        self.commit_scheduled_rows(rows)
    }
}

impl ScheduledDraftVerify for Gemma4Inner {
    type Cache<'a> = Gemma4SpecPagedCache<'a>;
    fn verify_cache(&mut self) -> Self::Cache<'_> {
        Gemma4SpecPagedCache::new(self)
    }
    fn take_verify_batch(&mut self) -> Option<ScheduledVerifyBatch> {
        self.scheduled_dspark_verify.take()
    }
    fn store_verify_batch(&mut self, batch: ScheduledVerifyBatch) {
        self.scheduled_dspark_verify = Some(batch);
    }
    fn validate_draft_owner(&self, row: &ScheduledVerifyRow) -> Result<()> {
        let state = self.scheduled_dspark_states.get(&row.seq_id);
        if row.speculative != state.is_some() {
            return Err(Error::from_reason(
                "scheduled verifier draft ownership mismatch",
            ));
        }
        if let Some(state) = state
            && (state.next_pos != row.first_position as i32 || state.open_cycle.is_some())
        {
            return Err(Error::from_reason(
                "scheduled verifier draft frontier mismatch",
            ));
        }
        Ok(())
    }
    fn run_packed_target(
        &mut self,
        rows: &[(PagedRaggedRow, Vec<u32>)],
    ) -> Result<(MxArray, Vec<MxArray>)> {
        let ids = self
            .dspark_draft()
            .ok_or_else(|| Error::from_reason("no DSpark draft loaded"))?
            .config
            .target_layer_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        self.run_paged_ragged_verify(rows, &ids)
    }
    fn append_committed_taps(&mut self, seq_id: u32, taps: &[MxArray], keep: usize) -> Result<()> {
        self.with_scheduled_dspark(seq_id, |step| {
            step.append_tapped_prefix(taps, keep, "scheduled commit")
        })
    }
    fn discard_draft_owner(&mut self, seq_id: u32) {
        self.scheduled_dspark_states.remove(&seq_id);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::engine::hybrid_scheduler::{HybridSchedulerBackend, ScheduledPrefixAdmission};
    use crate::models::gemma4::dspark_decode::tests::{
        committed_paged_tokens, seeded_tiny_paged_inner_with_draft,
    };
    use crate::stream::{DeviceType, Stream};

    fn seed(inner: &mut Gemma4Inner, seq: u32, tokens: &[u32], draft: bool) {
        let prefix = match inner
            .prepare_scheduled_prefix(seq, tokens, &[], false, 0, 8)
            .unwrap()
        {
            ScheduledPrefixAdmission::Ready(prefix) => prefix,
            _ => unreachable!(),
        };
        inner.activate_paged_seq(seq).unwrap();
        if draft {
            inner.begin_scheduled_dspark(seq, 0).unwrap();
        }
        let logits = inner
            .run_scheduled_prefill_slice(
                seq,
                tokens,
                &prefix,
                0,
                tokens.len(),
                Stream::default(DeviceType::Gpu),
                true,
            )
            .unwrap()
            .unwrap();
        logits.eval();
    }

    #[test]
    fn ragged_verify_matches_independent_owners_and_commits_only_accepted_prefix() {
        let Some(mut packed) = seeded_tiny_paged_inner_with_draft(912) else {
            return;
        };
        let Some(mut scalar) = seeded_tiny_paged_inner_with_draft(912) else {
            return;
        };
        let prompts = [
            vec![0, 1, 2, 3, 4, 5, 6],
            vec![4, 5, 6, 7, 8, 9, 10, 11, 12],
        ];
        for inner in [&mut packed, &mut scalar] {
            for (i, tokens) in prompts.iter().enumerate() {
                seed(inner, i as u32 + 11, tokens, true);
            }
        }
        let rows = vec![
            ScheduledVerifyRow {
                seq_id: 11,
                first_position: 7,
                tokens: vec![7, 8, 9],
                speculative: true,
            },
            ScheduledVerifyRow {
                seq_id: 12,
                first_position: 9,
                tokens: vec![13],
                speculative: true,
            },
        ];
        let actual = packed.verify_scheduled_dspark(&rows).unwrap();
        actual.eval();
        let actual = actual.to_float32().unwrap();
        let mut expected = Vec::new();
        for (row, keep) in rows.iter().zip([2, 1]) {
            let logits = scalar
                .verify_scheduled_dspark(std::slice::from_ref(row))
                .unwrap();
            logits.eval();
            expected.extend_from_slice(&logits.to_float32().unwrap());
            scalar
                .commit_scheduled_dspark(&[ScheduledVerifyCommit {
                    seq_id: row.seq_id,
                    keep,
                }])
                .unwrap()
                .pop()
                .unwrap()
                .unwrap();
        }
        assert_eq!(actual.len(), expected.len());
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_error < 0.015,
            "ragged versus scalar logits differ by {max_error}"
        );
        let outcomes = packed
            .commit_scheduled_dspark(&[
                ScheduledVerifyCommit {
                    seq_id: 11,
                    keep: 2,
                },
                ScheduledVerifyCommit {
                    seq_id: 12,
                    keep: 1,
                },
            ])
            .unwrap();
        for result in outcomes {
            result.unwrap();
        }
        for (seq, frontier) in [(11, 9), (12, 10)] {
            assert_eq!(
                committed_paged_tokens(&packed, seq),
                committed_paged_tokens(&scalar, seq)
            );
            assert_eq!(packed.scheduled_dspark_states[&seq].next_pos, frontier);
        }
        // A rejected token is absent from both the target and the next cycle's
        // context, even when the accepted prefix crossed a sliding block edge.
        assert_eq!(
            committed_paged_tokens(&packed, 11).unwrap(),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8]
        );
    }

    #[test]
    fn mixed_ar_owner_and_zero_draft_owner_keep_context_ownership() {
        let Some(mut inner) = seeded_tiny_paged_inner_with_draft(914) else {
            return;
        };
        seed(&mut inner, 21, &[0, 1, 2], true);
        seed(&mut inner, 22, &[3, 4, 5, 6], false);
        let wrong = [ScheduledVerifyRow {
            seq_id: 21,
            first_position: 3,
            tokens: vec![7],
            speculative: false,
        }];
        assert!(inner.verify_scheduled_dspark(&wrong).is_err());
        let rows = [
            ScheduledVerifyRow {
                seq_id: 21,
                first_position: 3,
                tokens: vec![7],
                speculative: true,
            },
            ScheduledVerifyRow {
                seq_id: 22,
                first_position: 4,
                tokens: vec![8],
                speculative: false,
            },
        ];
        inner.verify_scheduled_dspark(&rows).unwrap().eval();
        for result in inner
            .commit_scheduled_dspark(&[
                ScheduledVerifyCommit {
                    seq_id: 21,
                    keep: 1,
                },
                ScheduledVerifyCommit {
                    seq_id: 22,
                    keep: 1,
                },
            ])
            .unwrap()
        {
            result.unwrap();
        }
        assert_eq!(inner.scheduled_dspark_states[&21].next_pos, 4);
        assert!(!inner.scheduled_dspark_states.contains_key(&22));
    }

    #[test]
    fn invalid_commit_retracts_entire_batch_without_publishing_any_draft() {
        let Some(mut inner) = seeded_tiny_paged_inner_with_draft(916) else {
            return;
        };
        seed(&mut inner, 31, &[0, 1, 2], true);
        seed(&mut inner, 32, &[3, 4, 5, 6], true);
        let before = [
            committed_paged_tokens(&inner, 31),
            committed_paged_tokens(&inner, 32),
        ];
        let rows = [
            ScheduledVerifyRow {
                seq_id: 31,
                first_position: 3,
                tokens: vec![7, 8],
                speculative: true,
            },
            ScheduledVerifyRow {
                seq_id: 32,
                first_position: 4,
                tokens: vec![9],
                speculative: true,
            },
        ];
        inner.verify_scheduled_dspark(&rows).unwrap().eval();
        assert!(
            inner
                .commit_scheduled_dspark(&[ScheduledVerifyCommit {
                    seq_id: 31,
                    keep: 3
                }])
                .is_err()
        );
        assert_eq!(
            before,
            [
                committed_paged_tokens(&inner, 31),
                committed_paged_tokens(&inner, 32)
            ]
        );
        assert!(inner.scheduled_dspark_verify.is_none());
    }

    #[test]
    fn peer_content_and_row_order_cannot_change_an_owners_logits_or_continuation() {
        fn run(change_peer: bool, reverse: bool) -> Option<(Vec<f32>, Vec<f32>, Vec<u32>)> {
            let mut inner = seeded_tiny_paged_inner_with_draft(920)?;
            seed(&mut inner, 41, &[0, 1, 2, 3, 4, 5, 6], true);
            let peer = if change_peer {
                [14, 13, 12, 11, 10, 9, 8, 7, 6]
            } else {
                [4, 5, 6, 7, 8, 9, 10, 11, 12]
            };
            seed(&mut inner, 42, &peer, true);
            let mut rows = vec![
                ScheduledVerifyRow {
                    seq_id: 41,
                    first_position: 7,
                    tokens: vec![7, 8, 9],
                    speculative: true,
                },
                ScheduledVerifyRow {
                    seq_id: 42,
                    first_position: 9,
                    tokens: vec![if change_peer { 4 } else { 13 }],
                    speculative: true,
                },
            ];
            if reverse {
                rows.reverse();
            }
            let logits = inner.verify_scheduled_dspark(&rows).unwrap();
            logits.eval();
            let flat = logits.to_float32().unwrap();
            let vocab = inner.config.vocab_size as usize;
            let start = usize::from(reverse) * vocab;
            let first = flat[start..start + 3 * vocab].to_vec();
            for result in inner
                .commit_scheduled_dspark(
                    &rows
                        .iter()
                        .map(|row| ScheduledVerifyCommit {
                            seq_id: row.seq_id,
                            keep: if row.seq_id == 41 { 2 } else { 1 },
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            {
                result.unwrap();
            }
            // Owner 41 rejects token 9 and continues with a different boundary.
            // Recycling its peer must not alter retained attention or draft state.
            inner.release_scheduled_cache(42).unwrap();
            seed(&mut inner, 42, &[2, 2, 2, 2], false);
            let next = [ScheduledVerifyRow {
                seq_id: 41,
                first_position: 9,
                tokens: vec![10, 11],
                speculative: true,
            }];
            let logits = inner.verify_scheduled_dspark(&next).unwrap();
            logits.eval();
            let second = logits.to_float32().unwrap();
            inner
                .commit_scheduled_dspark(&[ScheduledVerifyCommit {
                    seq_id: 41,
                    keep: 1,
                }])
                .unwrap()
                .pop()
                .unwrap()
                .unwrap();
            Some((
                first,
                second.to_vec(),
                committed_paged_tokens(&inner, 41).unwrap(),
            ))
        }
        let Some(expected) = run(false, false) else {
            return;
        };
        for (change_peer, reverse) in [(true, false), (false, true), (true, true)] {
            let actual = run(change_peer, reverse).unwrap();
            assert_eq!(actual.0, expected.0, "peer/order changed verifier logits");
            assert_eq!(
                actual.1, expected.1,
                "peer/order changed rejection continuation"
            );
            assert_eq!(actual.2, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 10]);
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod real_model_tests {
    use super::*;
    use crate::engine::backend::ChatBackend;
    use crate::engine::hybrid_scheduler::{HybridSchedulerBackend, ScheduledPrefixAdmission};
    use crate::stream::{DeviceType, Stream};

    #[test]
    #[ignore = "requires local Gemma4 and DSpark checkpoints; numerical batch-shape audit"]
    fn compare_real_packed_and_independent_verifier_logits() {
        let model = std::env::var("MLX_TEST_GEMMA4_MODEL_PATH").unwrap();
        let draft = std::env::var("MLX_TEST_GEMMA4_DSPARK_PATH").unwrap();
        let output = std::env::var("MLX_TEST_SCHEDULED_OUTPUT_PATH").unwrap();
        let data: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
        let run = data["runs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["rows"] == 2)
            .unwrap();
        let (mut inner, _) = Gemma4Inner::load_from_dir(&model, Some(&draft)).unwrap();
        let tok = inner.tokenizer.as_ref().unwrap().clone();
        let mut config =
            crate::models::gemma4::dspark_decode::tests::tiny_turn_config(Some(7), 128);
        config.reasoning_effort = Some("none".into());
        let prompts = [
            "Give a simple recipe for pancakes with numbered steps.",
            "Explain how a computer runs a program, using numbered steps and short examples.",
        ];
        let mut full = Vec::new();
        for (i, text) in prompts.iter().enumerate() {
            let msg =
                serde_json::from_value(serde_json::json!({"role":"user","content":text})).unwrap();
            let prefix = inner.render_prompt(&tok, &[msg], &config, false).unwrap();
            let suffix = tok
                .encode_sync(run["results"][i]["rawText"].as_str().unwrap(), Some(false))
                .unwrap();
            full.push((prefix, suffix));
        }
        for generated_prefix in [0usize, 32, 64, 96] {
            let mut rows = Vec::new();
            for (i, (prompt, generated)) in full.iter().enumerate() {
                let mut prefix = prompt.clone();
                prefix.extend_from_slice(&generated[..generated_prefix]);
                let width = if i == 0 { 8 } else { 5 };
                for seq in [101 + i as u32, 201 + i as u32] {
                    let admitted = match inner
                        .prepare_scheduled_prefix(seq, &prefix, &[], false, 0, 16)
                        .unwrap()
                    {
                        ScheduledPrefixAdmission::Ready(prefix) => prefix,
                        _ => unreachable!(),
                    };
                    inner.activate_paged_seq(seq).unwrap();
                    inner
                        .run_scheduled_prefill_slice(
                            seq,
                            &prefix,
                            &admitted,
                            0,
                            prefix.len(),
                            Stream::default(DeviceType::Gpu),
                            true,
                        )
                        .unwrap()
                        .unwrap()
                        .eval();
                }
                rows.push(ScheduledVerifyRow {
                    seq_id: 101 + i as u32,
                    first_position: prefix.len() as u32,
                    tokens: generated[generated_prefix..generated_prefix + width].to_vec(),
                    speculative: false,
                });
            }
            let actual = inner.verify_scheduled_dspark(&rows).unwrap();
            actual.eval();
            let actual_top = actual.argmax(-1, None).unwrap();
            actual_top.eval();
            let actual_argmax = (0..rows.iter().map(|row| row.tokens.len()).sum())
                .map(|i| actual_top.item_at_int32(i).unwrap())
                .collect::<Vec<_>>();
            let actual = actual.to_float32().unwrap();
            for outcome in inner
                .commit_scheduled_dspark(
                    &rows
                        .iter()
                        .map(|r| ScheduledVerifyCommit {
                            seq_id: r.seq_id,
                            keep: r.tokens.len(),
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            {
                outcome.unwrap();
            }
            // Hold total query width and each owner's context length fixed.
            // Changing peer content isolates cross-owner addressing from the
            // width-dependent GEMV/GEMM arithmetic in the scalar comparison.
            for (seq_base, change_peer, reverse) in [(301u32, true, false), (401, false, true)] {
                let mut probe_rows = Vec::new();
                for (i, (prompt, generated)) in full.iter().enumerate() {
                    let seq = seq_base + i as u32;
                    let mut prefix = prompt.clone();
                    prefix.extend_from_slice(&generated[..generated_prefix]);
                    if i == 1 && change_peer {
                        prefix.reverse();
                    }
                    let admitted = match inner
                        .prepare_scheduled_prefix(seq, &prefix, &[], false, 0, 16)
                        .unwrap()
                    {
                        ScheduledPrefixAdmission::Ready(prefix) => prefix,
                        _ => unreachable!(),
                    };
                    inner.activate_paged_seq(seq).unwrap();
                    inner
                        .run_scheduled_prefill_slice(
                            seq,
                            &prefix,
                            &admitted,
                            0,
                            prefix.len(),
                            Stream::default(DeviceType::Gpu),
                            true,
                        )
                        .unwrap()
                        .unwrap()
                        .eval();
                    let mut tokens = rows[i].tokens.clone();
                    if i == 1 && change_peer {
                        tokens.reverse();
                    }
                    probe_rows.push(ScheduledVerifyRow {
                        seq_id: seq,
                        first_position: prefix.len() as u32,
                        tokens,
                        speculative: false,
                    });
                }
                if reverse {
                    probe_rows.reverse();
                }
                let probe = inner.verify_scheduled_dspark(&probe_rows).unwrap();
                probe.eval();
                let probe = probe.to_float32().unwrap();
                let vocab = inner.config.vocab_size as usize;
                let start = if reverse {
                    rows[1].tokens.len() * vocab
                } else {
                    0
                };
                let owner_error = probe[start..start + rows[0].tokens.len() * vocab]
                    .iter()
                    .zip(&actual[..rows[0].tokens.len() * vocab])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                for result in inner
                    .commit_scheduled_dspark(
                        &probe_rows
                            .iter()
                            .map(|row| ScheduledVerifyCommit {
                                seq_id: row.seq_id,
                                keep: 0,
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap()
                {
                    result.unwrap();
                }
                for seq in [seq_base, seq_base + 1] {
                    inner.release_scheduled_cache(seq).unwrap();
                }
                assert_eq!(
                    owner_error, 0.0,
                    "owner logits changed with peer/order at prefix={generated_prefix} reverse={reverse}"
                );
            }
            let mut expected = Vec::new();
            let mut expected_argmax = Vec::new();
            for row in &rows {
                // Match the existing whole-turn DSpark verifier exactly:
                // record the transaction, run the [1,Q,H] paged layer loop,
                // and project every row. Only packing changes in the other arm.
                let seq = row.seq_id + 100;
                inner.activate_paged_seq(seq).unwrap();
                let ticket = Gemma4SpecPagedCache::new(&mut inner)
                    .record_verify(seq, &row.tokens)
                    .unwrap();
                let kinds = inner.compute_layer_kinds().unwrap();
                let hidden = inner
                    .run_paged_prefill_layer_loop(
                        &row.tokens,
                        row.first_position,
                        row.first_position,
                        &kinds,
                        None,
                    )
                    .unwrap();
                let logits = inner.project_paged_hidden(&hidden, false).unwrap();
                logits.eval();
                let top = logits.argmax(-1, None).unwrap();
                top.eval();
                expected_argmax
                    .extend((0..row.tokens.len()).map(|i| top.item_at_int32(i).unwrap()));
                expected.extend_from_slice(&logits.to_float32().unwrap());
                Gemma4SpecPagedCache::new(&mut inner)
                    .commit_cycle(seq, ticket, row.tokens.len())
                    .unwrap();
                let frontier = Gemma4SpecPagedCache::new(&mut inner).frontier(seq).unwrap();
                Gemma4SpecPagedCache::new(&mut inner)
                    .settle_committed(seq, frontier.attn_tokens)
                    .unwrap();
            }
            let vocab = inner.config.vocab_size as usize;
            let mut max_error = 0f32;
            let mut dot = 0f64;
            let mut aa = 0f64;
            let mut bb = 0f64;
            for (&a, &b) in actual.iter().zip(&expected) {
                assert!(a.is_finite() && b.is_finite());
                max_error = max_error.max((a - b).abs());
                dot += f64::from(a) * f64::from(b);
                aa += f64::from(a) * f64::from(a);
                bb += f64::from(b) * f64::from(b);
            }
            let cosine = dot / (aa * bb).sqrt();
            eprintln!(
                "packed_logits prefix={generated_prefix} max_abs_error={max_error} cosine={cosine}"
            );
            let mut owner_start = 0;
            for row in &rows {
                let owner_end = owner_start + row.tokens.len() * vocab;
                let error = actual[owner_start..owner_end]
                    .iter()
                    .zip(&expected[owner_start..owner_end])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!("owner_logits seq={} max_abs_error={error}", row.seq_id);
                owner_start = owner_end;
            }
            for (pos, (a, b)) in actual
                .chunks_exact(vocab)
                .zip(expected.chunks_exact(vocab))
                .enumerate()
            {
                let at = actual_argmax[pos] as usize;
                let bt = expected_argmax[pos] as usize;
                if at != bt {
                    eprintln!(
                        "argmax_change row={pos} packed={at} scalar={bt} packed_gap={} scalar_gap={}",
                        a[at] - a[bt],
                        b[bt] - b[at]
                    );
                }
            }
            assert!(
                cosine > 0.999,
                "packed verifier materially changes logits: {cosine}"
            );
            for seq in [101, 102, 201, 202] {
                inner.release_scheduled_cache(seq).unwrap();
            }
        }
    }
}
