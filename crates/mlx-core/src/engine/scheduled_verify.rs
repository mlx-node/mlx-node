//! Shared transaction driver for paged draft-model verification.
//! Model backends provide target math and draft context; record/rollback,
//! rejection frontiers, and completion-gated settlement live here once.
use crate::array::MxArray;
use crate::engine::hybrid_scheduler::{ScheduledVerifyCommit, ScheduledVerifyRow};
use crate::engine::spec_paged::{SpecPagedCache, VerifyTicket};
use crate::transformer::paged_kv_cache_adapter::PagedRaggedRow;
use napi::bindgen_prelude::{Error, Result};
use std::collections::HashSet;

pub(crate) struct ScheduledVerifyBatch {
    entries: Vec<VerifyEntry>,
}

struct VerifyEntry {
    seq_id: u32,
    ticket: VerifyTicket,
    tapped: Option<Vec<MxArray>>,
}

pub(crate) trait ScheduledDraftVerify: Sized {
    type Cache<'a>: SpecPagedCache
    where
        Self: 'a;
    fn verify_cache(&mut self) -> Self::Cache<'_>;
    fn take_verify_batch(&mut self) -> Option<ScheduledVerifyBatch>;
    fn store_verify_batch(&mut self, batch: ScheduledVerifyBatch);
    fn validate_draft_owner(&self, row: &ScheduledVerifyRow) -> Result<()>;
    fn run_packed_target(
        &mut self,
        rows: &[(PagedRaggedRow, Vec<u32>)],
    ) -> Result<(MxArray, Vec<MxArray>)>;
    fn append_committed_taps(&mut self, seq_id: u32, taps: &[MxArray], keep: usize) -> Result<()>;
    fn discard_draft_owner(&mut self, seq_id: u32);

    fn retract_verify_batch(&mut self, batch: ScheduledVerifyBatch) -> Result<()> {
        let mut failure = None;
        for entry in batch.entries.into_iter().rev() {
            if let Err(error) = self
                .verify_cache()
                .commit_cycle(entry.seq_id, entry.ticket, 0)
            {
                failure.get_or_insert_with(|| Error::from_reason(error));
            }
        }
        failure.map_or(Ok(()), Err)
    }

    fn verify_scheduled_rows(&mut self, rows: &[ScheduledVerifyRow]) -> Result<MxArray> {
        if let Some(abandoned) = self.take_verify_batch() {
            self.retract_verify_batch(abandoned)?;
            return Err(Error::from_reason(
                "previous scheduled verifier batch was not committed",
            ));
        }
        if rows.is_empty() {
            return Err(Error::from_reason("empty scheduled verifier batch"));
        }
        let mut seen = HashSet::with_capacity(rows.len());
        let mut packed = Vec::with_capacity(rows.len());
        for row in rows {
            if !seen.insert(row.seq_id) || row.tokens.is_empty() {
                return Err(Error::from_reason(
                    "duplicate or empty scheduled verifier row",
                ));
            }
            let frontier = self
                .verify_cache()
                .frontier(row.seq_id)
                .ok_or_else(|| Error::from_reason("scheduled verifier has no target frontier"))?;
            if frontier.attn_tokens != u64::from(row.first_position) {
                return Err(Error::from_reason(
                    "scheduled verifier target frontier mismatch",
                ));
            }
            self.validate_draft_owner(row)?;
            packed.push((
                PagedRaggedRow {
                    seq_id: row.seq_id,
                    first_logical_position: row.first_position,
                    query_len: u32::try_from(row.tokens.len())
                        .map_err(|_| Error::from_reason("verifier width overflow"))?,
                },
                row.tokens.clone(),
            ));
        }
        let mut batch = ScheduledVerifyBatch {
            entries: Vec::with_capacity(rows.len()),
        };
        for row in rows {
            let recorded = self.verify_cache().record_verify(row.seq_id, &row.tokens);
            match recorded {
                Ok(ticket) => batch.entries.push(VerifyEntry {
                    seq_id: row.seq_id,
                    ticket,
                    tapped: None,
                }),
                Err(error) => {
                    self.retract_verify_batch(batch)?;
                    return Err(Error::from_reason(error));
                }
            }
        }
        let forward = self
            .run_packed_target(&packed)
            .and_then(|(logits, tapped)| {
                let mut start = 0i64;
                for (entry, row) in batch.entries.iter_mut().zip(rows) {
                    let end = start + row.tokens.len() as i64;
                    if row.speculative {
                        entry.tapped = Some(
                            tapped
                                .iter()
                                .map(|h| {
                                    h.slice_axis(0, start, end)
                                        .and_then(|h| h.transpose(Some(&[1, 0, 2])))
                                })
                                .collect::<Result<Vec<_>>>()?,
                        );
                    }
                    start = end;
                }
                Ok(logits)
            });
        match forward {
            Ok(logits) => {
                self.store_verify_batch(batch);
                Ok(logits)
            }
            Err(error) => {
                self.retract_verify_batch(batch)?;
                Err(error)
            }
        }
    }

    fn commit_scheduled_rows(&mut self, rows: &[ScheduledVerifyCommit]) -> Result<Vec<Result<()>>> {
        let batch = self
            .take_verify_batch()
            .ok_or_else(|| Error::from_reason("no scheduled verifier batch to commit"))?;
        if batch.entries.len() != rows.len()
            || batch
                .entries
                .iter()
                .zip(rows)
                .any(|(entry, row)| entry.seq_id != row.seq_id || row.keep > entry.ticket.rows())
        {
            self.retract_verify_batch(batch)?;
            return Err(Error::from_reason(
                "scheduled verifier commit does not match its tickets",
            ));
        }
        let mut results = Vec::with_capacity(rows.len());
        let mut contexts = Vec::with_capacity(rows.len());
        // All target frontiers settle before any durable checkpoint can see
        // this batch. Failed owners keep zero rows; valid peers keep their own prefix.
        for (entry, row) in batch.entries.into_iter().zip(rows) {
            let result = self
                .verify_cache()
                .commit_cycle(row.seq_id, entry.ticket, row.keep)
                .map_err(Error::from_reason);
            contexts.push(entry.tapped);
            results.push(result);
        }
        for ((row, tapped), result) in rows.iter().zip(contexts).zip(&mut results) {
            if result.is_err() {
                self.discard_draft_owner(row.seq_id);
                continue;
            }
            if row.keep == 0 {
                self.discard_draft_owner(row.seq_id);
                continue;
            }
            if let Some(tapped) = tapped {
                *result = self.append_committed_taps(row.seq_id, &tapped, row.keep);
            }
            if result.is_ok() {
                let mut cache = self.verify_cache();
                *result = cache
                    .frontier(row.seq_id)
                    .ok_or_else(|| Error::from_reason("scheduled commit lost target frontier"))
                    .and_then(|frontier| {
                        cache
                            .settle_committed(row.seq_id, frontier.attn_tokens)
                            .map_err(Error::from_reason)
                    });
            }
        }
        for (row, result) in rows.iter().zip(&results) {
            if result.is_err() {
                self.discard_draft_owner(row.seq_id);
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::SpecFrontier;
    use std::collections::HashMap;

    #[derive(Default)]
    struct FaultModel {
        cursors: HashMap<u32, u64>,
        batch: Option<ScheduledVerifyBatch>,
        events: Vec<String>,
        fault: &'static str,
    }

    struct Cache<'a>(&'a mut FaultModel);

    impl SpecPagedCache for Cache<'_> {
        fn reserve_lookahead(&mut self, _: u32, _: usize) -> std::result::Result<bool, String> {
            Ok(true)
        }
        fn record_rows(&mut self, seq: u32, tokens: &[u32]) -> std::result::Result<(), String> {
            if seq == 2 && self.0.fault == "record" {
                return Err("injected record failure".into());
            }
            *self.0.cursors.get_mut(&seq).unwrap() += tokens.len() as u64;
            self.0.events.push(format!("record:{seq}"));
            Ok(())
        }
        fn rollback_rows(&mut self, seq: u32, rows: usize) -> std::result::Result<(), String> {
            self.0.events.push(format!("rollback:{seq}"));
            if seq == 1 && self.0.fault == "rollback" {
                return Err("injected rollback failure".into());
            }
            *self.0.cursors.get_mut(&seq).unwrap() -= rows as u64;
            Ok(())
        }
        fn settle_committed(&mut self, seq: u32, tokens: u64) -> std::result::Result<(), String> {
            assert_eq!(self.0.cursors[&seq], tokens);
            self.0.events.push(format!("settle:{seq}"));
            if seq == 1 && self.0.fault == "settle" {
                return Err("injected settlement failure".into());
            }
            Ok(())
        }
        fn settle_captures_durable_state(&self) -> bool {
            true
        }
        fn frontier(&self, seq: u32) -> Option<SpecFrontier> {
            self.0.cursors.get(&seq).map(|&attn_tokens| SpecFrontier {
                attn_tokens,
                recurrent_tokens: None,
            })
        }
    }

    impl ScheduledDraftVerify for FaultModel {
        type Cache<'a> = Cache<'a>;
        fn verify_cache(&mut self) -> Self::Cache<'_> {
            Cache(self)
        }
        fn take_verify_batch(&mut self) -> Option<ScheduledVerifyBatch> {
            self.batch.take()
        }
        fn store_verify_batch(&mut self, batch: ScheduledVerifyBatch) {
            self.batch = Some(batch);
        }
        fn validate_draft_owner(&self, _: &ScheduledVerifyRow) -> Result<()> {
            Ok(())
        }
        fn run_packed_target(
            &mut self,
            rows: &[(PagedRaggedRow, Vec<u32>)],
        ) -> Result<(MxArray, Vec<MxArray>)> {
            self.events.push("forward".into());
            if self.fault == "forward" {
                return Err(Error::from_reason("injected forward failure"));
            }
            let values = rows
                .iter()
                .flat_map(|(_, tokens)| tokens.iter().map(|&v| v as f32))
                .collect::<Vec<_>>();
            let array = MxArray::from_float32(&values, &[values.len() as i64, 1, 1])?;
            Ok((array.clone(), vec![array]))
        }
        fn append_committed_taps(&mut self, seq: u32, taps: &[MxArray], keep: usize) -> Result<()> {
            self.events.push(format!("append:{seq}"));
            if seq == 1 && self.fault == "append" {
                return Err(Error::from_reason("injected append failure"));
            }
            // The transaction slices the packed target by owner before the
            // family applies its kept prefix. Peers must never share taps.
            let kept = taps[0].slice_axis(1, 0, keep as i64)?.to_float32()?;
            assert_eq!(
                kept.as_ref(),
                if seq == 1 {
                    &[10.0, 11.0][..keep]
                } else {
                    &[20.0][..keep]
                }
            );
            Ok(())
        }
        fn discard_draft_owner(&mut self, seq: u32) {
            self.events.push(format!("discard:{seq}"));
        }
    }

    fn fixture(fault: &'static str) -> (FaultModel, Vec<ScheduledVerifyRow>) {
        let model = FaultModel {
            cursors: HashMap::from([(1, 7), (2, 9)]),
            fault,
            ..Default::default()
        };
        let rows = vec![
            ScheduledVerifyRow {
                seq_id: 1,
                first_position: 7,
                tokens: vec![10, 11, 12],
                speculative: true,
            },
            ScheduledVerifyRow {
                seq_id: 2,
                first_position: 9,
                tokens: vec![20, 21],
                speculative: true,
            },
        ];
        (model, rows)
    }

    #[test]
    fn failed_record_or_forward_retracts_every_open_ticket_without_publication() {
        for fault in ["record", "forward"] {
            let (mut model, rows) = fixture(fault);
            assert!(model.verify_scheduled_rows(&rows).is_err());
            assert_eq!(model.cursors, HashMap::from([(1, 7), (2, 9)]));
            assert!(model.batch.is_none());
            assert!(!model.events.iter().any(|event| event.starts_with("settle")));
        }
    }

    #[test]
    fn unequal_prefixes_commit_before_context_or_durable_publication() {
        let (mut model, rows) = fixture("");
        model.verify_scheduled_rows(&rows).unwrap().eval();
        let results = model
            .commit_scheduled_rows(&[
                ScheduledVerifyCommit { seq_id: 1, keep: 2 },
                ScheduledVerifyCommit { seq_id: 2, keep: 1 },
            ])
            .unwrap();
        assert!(results.iter().all(Result::is_ok));
        assert_eq!(model.cursors, HashMap::from([(1, 9), (2, 10)]));
        assert_eq!(
            &model.events[3..],
            [
                "rollback:1",
                "rollback:2",
                "append:1",
                "settle:1",
                "append:2",
                "settle:2"
            ]
        );
    }

    #[test]
    fn rejected_commit_order_retracts_the_entire_batch() {
        let (mut model, rows) = fixture("");
        model.verify_scheduled_rows(&rows).unwrap().eval();
        assert!(
            model
                .commit_scheduled_rows(&[
                    ScheduledVerifyCommit { seq_id: 2, keep: 1 },
                    ScheduledVerifyCommit { seq_id: 1, keep: 2 },
                ])
                .is_err()
        );
        assert_eq!(model.cursors, HashMap::from([(1, 7), (2, 9)]));
        assert!(model.batch.is_none());
    }

    #[test]
    fn failed_owner_does_not_block_healthy_peer_commit_or_settlement() {
        for fault in ["rollback", "append", "settle"] {
            let (mut model, rows) = fixture(fault);
            model.verify_scheduled_rows(&rows).unwrap().eval();
            let results = model
                .commit_scheduled_rows(&[
                    ScheduledVerifyCommit { seq_id: 1, keep: 2 },
                    ScheduledVerifyCommit { seq_id: 2, keep: 1 },
                ])
                .unwrap();
            assert!(results[0].is_err(), "{fault}");
            assert!(results[1].is_ok(), "{fault}");
            assert_eq!(model.cursors[&2], 10);
            assert!(model.events.contains(&"discard:1".into()));
            assert!(model.events.contains(&"settle:2".into()));
            if fault != "settle" {
                assert!(!model.events.contains(&"settle:1".into()));
            }
        }
    }

    #[test]
    fn cancelled_owner_keeps_zero_and_never_publishes() {
        let (mut model, rows) = fixture("");
        model.verify_scheduled_rows(&rows).unwrap().eval();
        let results = model
            .commit_scheduled_rows(&[
                ScheduledVerifyCommit { seq_id: 1, keep: 0 },
                ScheduledVerifyCommit { seq_id: 2, keep: 1 },
            ])
            .unwrap();
        assert!(results.iter().all(Result::is_ok));
        assert_eq!(model.cursors, HashMap::from([(1, 7), (2, 10)]));
        assert!(model.events.contains(&"discard:1".into()));
        assert!(!model.events.contains(&"append:1".into()));
        assert!(!model.events.contains(&"settle:1".into()));
    }
}
