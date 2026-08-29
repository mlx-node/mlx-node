//! The family-neutral half of an `mlx eval` run.
//!
//! A model family supplies only what is genuinely its own — raw tokenization,
//! the vocabulary width, one teacher-forced prefill and the head projection —
//! through [`EvalBackend`]. The capture and score loops, the position chunking
//! and the on-disk round trip live here, so adding a family is an adapter
//! rather than a second copy of the metric.

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};

use super::cache::{
    EvalCacheMeta, invalidate_meta, new_generation, read_meta, read_row, write_meta, write_row,
};
use super::scoring::{ScoreTotals, TeacherLogits, TeacherRow, capture_chunk, score_chunk};
use super::{EvalOutcome, EvalRequest};

/// What one model family supplies so [`run`] can drive `mlx eval` over it.
pub trait EvalBackend {
    fn vocab_size(&self) -> i32;

    /// Tokenize RAW — no chat template, no BOS. Perplexity is conventionally
    /// defined over raw text, and it matches the calibration path's contract.
    fn tokenize_raw(&self, text: &str) -> Result<Vec<u32>>;

    /// Prefill `tokens[..T-1]` on fresh turn-0 caches and return the
    /// post-final-norm hidden `[1, T-1, hidden]`.
    ///
    /// EVERY position is kept — the AR path computes this same state and then
    /// discards all but the last position before the head.
    fn teacher_forced_hidden(&mut self, tokens: &[u32]) -> Result<MxArray>;

    /// Project `hidden` through the head, preserving the leading dims
    /// (`[*, hidden] -> [*, vocab]`).
    fn project_head(&self, hidden: &MxArray) -> Result<MxArray>;
}

/// Run one eval request against `backend`.
///
/// Both modes share one shape: obtain a sequence of `T` token ids — RAW
/// tokenize + truncate (capture) or read them back from the cache (score) —
/// prefill `ids[..T-1]`, and project the head over the resulting hidden. Score
/// mode NEVER re-tokenizes: reading the ids from the cache is what makes a
/// dataset or tokenizer edit unable to silently produce a comparison against
/// different text.
pub fn run<B: EvalBackend>(backend: &mut B, request: EvalRequest) -> Result<EvalOutcome> {
    match request {
        EvalRequest::Capture {
            teacher_path,
            teacher_quantized,
            identity,
            texts,
            seq_len,
            top_k,
            logit_chunk,
            cache_dir,
        } => {
            std::fs::create_dir_all(&cache_dir)
                .map_err(|e| Error::from_reason(format!("create {}: {e}", cache_dir.display())))?;
            // Before the first row, not after the last: see `invalidate_meta`.
            invalidate_meta(&cache_dir)?;
            // The first token primes the forward and has no target of its own,
            // so a row under 2 tokens scores nothing.
            let cap = seq_len.max(2) as usize;
            let mut rows: u32 = 0;
            let mut positions: u64 = 0;

            // `capture_chunk` clamps the requested width to `[1, vocab]`, so the
            // rows can be narrower than `top_k`. Record what they hold.
            let mut effective_top_k = top_k;
            for text in &texts {
                let mut tokens = backend.tokenize_raw(text)?;
                tokens.truncate(cap);
                if tokens.len() < 2 {
                    continue;
                }
                let row = capture_row(backend, &tokens, top_k, logit_chunk)?;
                effective_top_k = row.logits.support()? as u32;
                write_row(&cache_dir, rows, &row)?;
                positions += tokens.len() as u64 - 1;
                rows += 1;
                crate::array::synchronize_and_clear_cache();
            }

            if rows == 0 {
                return Err(Error::from_reason(
                    "eval capture wrote 0 rows — dataset empty, or every row tokenized to under 2 \
                     tokens",
                ));
            }
            write_meta(
                &cache_dir,
                &EvalCacheMeta {
                    teacher_path,
                    identity,
                    generation: new_generation(),
                    teacher_quantized,
                    vocab_size: backend.vocab_size(),
                    // `cap` and `effective_top_k`, not the requested values:
                    // what the rows actually hold.
                    seq_len: cap as u32,
                    top_k: effective_top_k,
                    rows,
                    positions,
                },
            )?;
            Ok(EvalOutcome::Captured { rows, positions })
        }

        EvalRequest::Score {
            cache_dir,
            logit_chunk,
            identity,
        } => {
            let meta = read_meta(&cache_dir)?;
            identity.require_match(&meta.identity)?;
            if meta.vocab_size != backend.vocab_size() {
                return Err(Error::from_reason(format!(
                    "teacher cache was captured on a {}-entry vocabulary but this checkpoint has \
                     {} — the two are not comparable",
                    meta.vocab_size,
                    backend.vocab_size()
                )));
            }

            let mut totals = ScoreTotals::default();
            for index in 0..meta.rows {
                let row = read_row(&cache_dir, index)?;
                let scored = row.tokens.len() as i64 - 1;
                if row.logits.positions()? != scored {
                    return Err(Error::from_reason(format!(
                        "teacher row {index} has {} cached positions for {} tokens",
                        row.logits.positions()?,
                        row.tokens.len()
                    )));
                }
                let hidden = backend.teacher_forced_hidden(&row.tokens)?;
                let targets = MxArray::from_uint32(&row.tokens[1..], &[scored])?;
                let chunks =
                    fold_position_chunks(backend, &hidden, logit_chunk, |start, end, logits| {
                        score_chunk(
                            logits,
                            &targets.slice_axis(0, start as i64, end as i64)?,
                            &row.logits.slice(start as i64, end as i64)?,
                        )
                    })?;
                for chunk in &chunks {
                    totals.add(chunk);
                }
                crate::array::synchronize_and_clear_cache();
            }
            // `eval_guard` is a process-local mutex, so it does not serialize a
            // second `mlx eval` process. A capture running against this same
            // directory rewrites rows in place under fixed names, so the loop
            // above can mix two teachers' rows and still be labelled with the
            // meta read before the loop. `invalidate_meta` only catches the
            // reader that arrives AFTER the capture started.
            //
            // Re-reading is the whole fix: the capture removes the meta before
            // its first row and writes a new one after its last, so any capture
            // overlapping this loop leaves the meta either missing or changed.
            // That turns a plausible-looking mixed report into a refusal.
            let after = read_meta(&cache_dir).map_err(|e| {
                Error::from_reason(format!(
                    "teacher cache changed while it was being scored ({e}) — a capture ran \
                     against {} at the same time. Re-run the score.",
                    cache_dir.display()
                ))
            })?;
            // Compare the GENERATION, not the fields: a concurrent capture of the
            // same teacher over a different dataset produces identical
            // teacher_path/identity/rows/positions/top_k, so a field comparison
            // would see no difference while the rows underneath were replaced.
            if after.generation != meta.generation {
                return Err(Error::from_reason(format!(
                    "teacher cache was replaced while it was being scored — the rows read \
                     may come from two different captures of {}. Re-run the score.",
                    cache_dir.display()
                )));
            }

            Ok(EvalOutcome::Scored(totals.into_report(
                meta.rows,
                meta.top_k,
                meta.teacher_path,
                meta.teacher_quantized,
            )?))
        }
    }
}

/// Capture one ALREADY-tokenized sequence: prefill it, fold the head over
/// positions and reduce each chunk to the cacheable reference.
///
/// Split out of [`run`]'s capture arm so a caller holding token ids rather than
/// text reaches the same reduction without going through a tokenizer.
pub fn capture_row<B: EvalBackend>(
    backend: &mut B,
    tokens: &[u32],
    top_k: u32,
    logit_chunk: u32,
) -> Result<TeacherRow> {
    let hidden = backend.teacher_forced_hidden(tokens)?;
    let chunks = fold_position_chunks(backend, &hidden, logit_chunk, |start, end, logits| {
        let targets = MxArray::from_uint32(&tokens[start + 1..end + 1], &[(end - start) as i64])?;
        let chunk = capture_chunk(logits, &targets, top_k as i64)?;
        // Materialize before the loop clears the MLX cache — these are lazy
        // handles onto graph nodes.
        chunk.eval()?;
        Ok(chunk)
    })?;
    Ok(TeacherRow {
        logits: TeacherLogits::concat(&chunks)?,
        tokens: tokens.to_vec(),
    })
}

/// Project `hidden` through the head in POSITION chunks, folding each chunk's
/// `[chunk, vocab]` f32 logits.
///
/// The chunking is the whole reason this metric is affordable: teacher forcing
/// needs the head's output at EVERY position, and the AR path never pays for
/// that — it slices to the last position before the head. A `[positions, vocab]`
/// f32 array is hundreds of megabytes in ONE allocation on a large vocabulary;
/// chunking caps the transient at `chunk x vocab x 4` bytes and breaks the head
/// into per-chunk command buffers.
fn fold_position_chunks<B: EvalBackend, T>(
    backend: &B,
    hidden: &MxArray,
    logit_chunk: u32,
    mut fold: impl FnMut(usize, usize, &MxArray) -> Result<T>,
) -> Result<Vec<T>> {
    let positions = hidden.shape_at(1)?;
    let vocab = backend.vocab_size() as i64;
    let step = (logit_chunk.max(1) as i64).min(positions.max(1));
    let mut folded = Vec::with_capacity(((positions + step - 1) / step) as usize);
    let mut start: i64 = 0;
    while start < positions {
        let end = (start + step).min(positions);
        let logits = backend
            .project_head(&hidden.slice_axis(1, start, end)?)?
            .astype(DType::Float32)?
            .reshape(&[end - start, vocab])?;
        folded.push(fold(start as usize, end as usize, &logits)?);
        crate::array::synchronize_and_clear_cache();
        start = end;
    }
    Ok(folded)
}

/// Prefill `ids` in `chunk_size` position slices, keeping every position's
/// post-final-norm hidden, and join them into `[1, ids.len(), hidden]`.
///
/// `forward` runs one `[1, c]` id slice through the family's layer stack and
/// final norm, advancing the caches the caller has already re-initialized. Each
/// chunk's hidden is materialized before the MLX cache is cleared — it is a lazy
/// handle onto graph nodes `clear_cache` would free.
pub fn chunked_hidden(
    ids: &[u32],
    chunk_size: i64,
    mut forward: impl FnMut(&MxArray) -> Result<MxArray>,
) -> Result<MxArray> {
    if ids.is_empty() {
        return Err(Error::from_reason("eval prefill: empty sequence"));
    }
    let step = chunk_size.max(1) as usize;
    let mut chunks: Vec<MxArray> = Vec::with_capacity(ids.len().div_ceil(step));
    for slice in ids.chunks(step) {
        let input = MxArray::from_uint32(slice, &[1, slice.len() as i64])?;
        let hidden = forward(&input)?;
        hidden.eval();
        chunks.push(hidden);
        crate::array::clear_cache();
    }
    if chunks.len() == 1 {
        return chunks.pop().ok_or_else(|| {
            Error::from_reason("eval prefill: chunk list emptied between check and pop")
        });
    }
    MxArray::concatenate_many(chunks.iter().collect(), Some(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality::cache::EvalIdentity;
    use crate::quality::scoring::EvalReport;

    /// A backend with no model behind it: the hidden IS the logit block, so the
    /// head is the identity and every number the driver reports is exactly the
    /// fixture's own. What this exercises is everything the model families do
    /// NOT own — the capture/score loops, the safetensors round trip, the
    /// position chunking and the vocabulary guard.
    struct FixtureBackend {
        vocab: i64,
        rows: Vec<Vec<u32>>,
        /// Rewrite this cache's `meta.json` on the FIRST forward of a score
        /// run, standing in for a second `mlx eval` process that starts a
        /// capture while this one is walking the rows. `eval_guard` is
        /// process-local, so nothing else would serialize them.
        replace_meta_mid_scan: Option<(std::path::PathBuf, String)>,
    }

    const FIXTURE_VOCAB: i64 = 24;

    impl FixtureBackend {
        fn new() -> Self {
            Self {
                vocab: FIXTURE_VOCAB,
                rows: vec![vec![3, 9, 14, 2, 20, 7, 11], vec![5, 1, 18, 6]],
                replace_meta_mid_scan: None,
            }
        }
    }

    impl EvalBackend for FixtureBackend {
        fn vocab_size(&self) -> i32 {
            self.vocab as i32
        }

        fn tokenize_raw(&self, text: &str) -> Result<Vec<u32>> {
            let index: usize = text.parse().unwrap();
            Ok(self.rows[index].clone())
        }

        fn teacher_forced_hidden(&mut self, tokens: &[u32]) -> Result<MxArray> {
            if let Some((dir, teacher)) = self.replace_meta_mid_scan.take() {
                let mut meta = read_meta(&dir)?;
                meta.teacher_path = teacher;
                // A real capture stamps a fresh generation; that is the only
                // field this stand-in is guaranteed to move.
                meta.generation = new_generation();
                write_meta(&dir, &meta)?;
            }
            let positions = tokens.len() - 1;
            let mut values = Vec::with_capacity(positions * self.vocab as usize);
            for (pos, &id) in tokens[..positions].iter().enumerate() {
                for v in 0..self.vocab {
                    let x = ((pos as i64) * 5 + (id as i64) * 3 + v * 7) % 19;
                    values.push((x as f32) * 0.27 - 2.5);
                }
            }
            MxArray::from_float32(&values, &[1, positions as i64, self.vocab])
        }

        fn project_head(&self, hidden: &MxArray) -> Result<MxArray> {
            Ok(hidden.clone())
        }
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "mlx_eval_runner_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    /// The CALL SITE, not just the helper. A capture that fails must leave no
    /// readable cache: rows keep fixed names and are overwritten in place, so a
    /// `meta.json` surviving from an earlier teacher would let `score` read that
    /// teacher's late rows beside this one's early rows, under that teacher's
    /// identity and row count, and report a number that looks valid.
    #[test]
    fn a_failed_capture_leaves_no_readable_cache() {
        let dir = scratch("failed_capture");
        std::fs::create_dir_all(&dir).unwrap();
        write_meta(
            &dir,
            &EvalCacheMeta {
                teacher_path: "/models/teacher-a".to_string(),
                identity: EvalIdentity::fixture("a"),
                generation: new_generation(),
                teacher_quantized: false,
                vocab_size: FIXTURE_VOCAB as i32,
                seq_len: 8,
                top_k: 4,
                rows: 99,
                positions: 600,
            },
        )
        .unwrap();
        read_meta(&dir).expect("the stale cache must be readable to begin with");

        // One token per text, so no row is scorable and the capture fails after
        // `create_dir_all` but before `write_meta`.
        let mut backend = FixtureBackend {
            vocab: FIXTURE_VOCAB,
            replace_meta_mid_scan: None,
            rows: vec![vec![4]],
        };
        let outcome = run(
            &mut backend,
            EvalRequest::Capture {
                teacher_path: "/models/teacher-b".to_string(),
                identity: EvalIdentity::fixture("a"),
                teacher_quantized: false,
                texts: vec!["0".to_string()],
                seq_len: 8,
                top_k: 4,
                logit_chunk: 8,
                cache_dir: dir.clone(),
            },
        );
        let Err(err) = outcome else {
            panic!("a capture with no scorable row must fail");
        };
        assert!(
            err.reason.contains("0 rows"),
            "unexpected failure: {}",
            err.reason
        );

        assert!(
            read_meta(&dir).is_err(),
            "teacher A's meta must not survive teacher B's failed capture"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A `seq_len` under the teacher-forcing floor is raised, and the cache
    /// records what the rows actually hold rather than what was asked for.
    ///
    /// Without this the meta would claim `seq_len: 1` beside rows of 2 tokens,
    /// and the provenance a later A/B reads back would be wrong.
    #[test]
    fn a_seq_len_below_the_floor_is_recorded_as_the_floor() {
        let dir = std::env::temp_dir().join(format!("mlx-eval-floor-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut backend = FixtureBackend {
            vocab: FIXTURE_VOCAB,
            replace_meta_mid_scan: None,
            rows: vec![vec![4, 5, 6, 7]],
        };

        run(
            &mut backend,
            EvalRequest::Capture {
                teacher_path: "fixture".to_string(),
                identity: EvalIdentity::fixture("a"),
                teacher_quantized: false,
                texts: vec!["0".to_string()],
                seq_len: 1,
                top_k: 6,
                logit_chunk: 8,
                cache_dir: dir.clone(),
            },
        )
        .unwrap();

        let meta = read_meta(&dir).unwrap();
        assert_eq!(
            meta.seq_len, 2,
            "a 1-token request must be recorded as the 2-token floor it ran at"
        );
        assert_eq!(meta.positions, 1, "2 tokens score exactly one position");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A candidate whose tokenizer differs from the teacher's is refused, even
    /// at the same vocabulary width.
    ///
    /// Width is not identity. Score reads its ids FROM the cache and indexes
    /// the candidate's logits with the teacher's cached vocabulary indices, so
    /// a different id-to-token map yields a finite, plausible, entirely wrong
    /// report rather than an error.
    #[test]
    fn a_candidate_with_another_tokenizer_is_refused_at_equal_vocab_width() {
        let dir = std::env::temp_dir().join(format!("mlx-eval-tok-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut backend = FixtureBackend {
            vocab: FIXTURE_VOCAB,
            replace_meta_mid_scan: None,
            rows: vec![vec![4, 5, 6, 7]],
        };

        run(
            &mut backend,
            EvalRequest::Capture {
                teacher_path: "fixture".to_string(),
                identity: EvalIdentity::fixture("a"),
                teacher_quantized: false,
                texts: vec!["0".to_string()],
                seq_len: 512,
                top_k: 6,
                logit_chunk: 8,
                cache_dir: dir.clone(),
            },
        )
        .unwrap();

        // Same vocabulary width, different tokenizer.
        let outcome = run(
            &mut backend,
            EvalRequest::Score {
                cache_dir: dir.clone(),
                identity: EvalIdentity::fixture("b"),
                logit_chunk: 8,
            },
        );
        let Err(err) = outcome else {
            panic!("a different tokenizer must be refused");
        };
        assert!(
            err.reason.contains("different tokenizer"),
            "message must name the cause: {}",
            err.reason
        );

        // The matching candidate still scores.
        run(
            &mut backend,
            EvalRequest::Score {
                cache_dir: dir.clone(),
                identity: EvalIdentity::fixture("a"),
                logit_chunk: 8,
            },
        )
        .expect("the teacher's own identity must still score");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A capture that lands while a score is walking the rows is refused, not
    /// reported.
    ///
    /// `eval_guard` is process-local, so a second `mlx eval` process can
    /// overwrite rows under fixed names mid-scan. The score re-reads the meta
    /// after the loop; here the replacement is simulated by capturing a
    /// different teacher into the same directory.
    #[test]
    fn a_cache_replaced_mid_score_is_refused() {
        let dir = std::env::temp_dir().join(format!("mlx-eval-race-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut backend = FixtureBackend {
            vocab: FIXTURE_VOCAB,
            replace_meta_mid_scan: None,
            rows: vec![vec![4, 5, 6, 7], vec![1, 2, 3, 4]],
        };

        run(
            &mut backend,
            EvalRequest::Capture {
                teacher_path: "/models/teacher-a".to_string(),
                identity: EvalIdentity::fixture("a"),
                teacher_quantized: false,
                texts: vec!["0".to_string(), "1".to_string()],
                seq_len: 512,
                top_k: 6,
                logit_chunk: 8,
                cache_dir: dir.clone(),
            },
        )
        .unwrap();

        // Arm the stand-in: the meta is rewritten during the FIRST forward of
        // the score below, i.e. after `run` has already read it and before the
        // row loop ends. Only the post-loop re-read can catch that.
        //
        // The replacement keeps the SAME teacher path, so every field the old
        // shape comparison looked at is identical and only the generation
        // moves — the case the field comparison missed.
        backend.replace_meta_mid_scan = Some((dir.clone(), "/models/teacher-a".to_string()));

        let outcome = run(
            &mut backend,
            EvalRequest::Score {
                cache_dir: dir.clone(),
                identity: EvalIdentity::fixture("a"),
                logit_chunk: 8,
            },
        );
        let Err(err) = outcome else {
            panic!("a cache replaced mid-score must be refused");
        };
        assert!(
            err.reason.contains("replaced while it was being scored"),
            "message must name the cause: {}",
            err.reason
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A `top_k` wider than the vocabulary is clamped by `capture_chunk`, and
    /// the cache must record the width the rows actually carry.
    ///
    /// Otherwise a report labels its KL support `K=4096` while each position
    /// holds 24 entries, and the support width — the one number that says how
    /// far a top-K KL can be trusted — is wrong.
    #[test]
    fn an_oversized_top_k_is_recorded_at_the_width_the_rows_hold() {
        let dir = std::env::temp_dir().join(format!("mlx-eval-topk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut backend = FixtureBackend {
            vocab: FIXTURE_VOCAB,
            replace_meta_mid_scan: None,
            rows: vec![vec![4, 5, 6, 7]],
        };

        run(
            &mut backend,
            EvalRequest::Capture {
                teacher_path: "fixture".to_string(),
                identity: EvalIdentity::fixture("a"),
                teacher_quantized: false,
                texts: vec!["0".to_string()],
                seq_len: 512,
                top_k: 4096,
                logit_chunk: 8,
                cache_dir: dir.clone(),
            },
        )
        .unwrap();

        let meta = read_meta(&dir).unwrap();
        assert_eq!(
            meta.top_k, FIXTURE_VOCAB as u32,
            "a request wider than the vocabulary must be recorded at the clamped width"
        );

        let outcome = run(
            &mut backend,
            EvalRequest::Score {
                cache_dir: dir.clone(),
                identity: EvalIdentity::fixture("a"),
                logit_chunk: 8,
            },
        )
        .unwrap();
        let EvalOutcome::Scored(report) = outcome else {
            panic!("score returned a capture outcome");
        };
        assert_eq!(
            report.top_k, FIXTURE_VOCAB as u32,
            "the report must label the support at its real width"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A quantized teacher is captured, not refused — and the fact rides the
    /// cache into every report scored against it.
    #[test]
    fn a_quantized_teacher_is_recorded_not_refused() {
        let dir = std::env::temp_dir().join(format!("mlx-eval-quant-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut backend = FixtureBackend {
            vocab: FIXTURE_VOCAB,
            replace_meta_mid_scan: None,
            rows: vec![vec![4, 5, 6, 7], vec![1, 2, 3, 4]],
        };

        run(
            &mut backend,
            EvalRequest::Capture {
                teacher_path: "fixture".to_string(),
                identity: EvalIdentity::fixture("a"),
                teacher_quantized: true,
                texts: vec!["0".to_string(), "1".to_string()],
                seq_len: 512,
                top_k: 6,
                logit_chunk: 8,
                cache_dir: dir.clone(),
            },
        )
        .unwrap();

        assert!(
            read_meta(&dir).unwrap().teacher_quantized,
            "the capture must not silently drop the teacher's quantization state"
        );

        let outcome = run(
            &mut backend,
            EvalRequest::Score {
                cache_dir: dir.clone(),
                identity: EvalIdentity::fixture("a"),
                logit_chunk: 8,
            },
        )
        .unwrap();
        let EvalOutcome::Scored(report) = outcome else {
            panic!("score returned a capture outcome");
        };
        assert!(
            report.teacher_quantized,
            "a report anchored on a quantized teacher must say so"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn capture(backend: &mut FixtureBackend, dir: &std::path::Path, logit_chunk: u32) -> u32 {
        match run(
            backend,
            EvalRequest::Capture {
                teacher_path: "fixture".to_string(),
                identity: EvalIdentity::fixture("a"),
                teacher_quantized: false,
                texts: vec!["0".to_string(), "1".to_string()],
                seq_len: 512,
                top_k: 6,
                logit_chunk,
                cache_dir: dir.to_path_buf(),
            },
        )
        .unwrap()
        {
            EvalOutcome::Captured { rows, .. } => rows,
            EvalOutcome::Scored(_) => panic!("capture returned a report"),
        }
    }

    fn score(backend: &mut FixtureBackend, dir: &std::path::Path, logit_chunk: u32) -> EvalReport {
        match run(
            backend,
            EvalRequest::Score {
                cache_dir: dir.to_path_buf(),
                identity: EvalIdentity::fixture("a"),
                logit_chunk,
            },
        )
        .unwrap()
        {
            EvalOutcome::Scored(report) => report,
            EvalOutcome::Captured { .. } => panic!("score returned a capture count"),
        }
    }

    /// The driver-level self-identity property: capture, go through disk, score
    /// the SAME backend back, and every divergence must be exactly zero. The two
    /// calls use different `logit_chunk` values so a chunk boundary that
    /// reweighted or misaligned a position could not survive.
    #[test]
    fn a_backend_scored_against_its_own_capture_is_exactly_zero() {
        let dir = scratch("identity");
        let mut backend = FixtureBackend::new();
        assert_eq!(capture(&mut backend, &dir, 3), 2);

        let report = score(&mut backend, &dir, 4);
        assert_eq!(report.rows, 2);
        assert_eq!(report.positions, 9, "6 + 3 scored positions");
        assert_eq!(report.mean_kl_topk, 0.0, "self-KL must be exactly zero");
        assert_eq!(report.top1_agreement, 1.0);
        assert_eq!(report.mean_nll, report.teacher_mean_nll);
        assert_eq!(report.teacher_path, "fixture");
        // Top-6 of 24 leaves real mass out, so the zero KL above is not the
        // vacuous "everything is in K" case.
        assert!(
            report.teacher_tail_mass > 0.01,
            "top-6 of 24 must leave real mass out, got {}",
            report.teacher_tail_mass
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A candidate whose head is a different size answers a different question,
    /// and the cached token ids cannot catch it.
    #[test]
    fn a_cache_from_another_vocabulary_is_refused() {
        let dir = scratch("vocab");
        let mut backend = FixtureBackend::new();
        capture(&mut backend, &dir, 3);

        backend.vocab += 1;
        let refused = run(
            &mut backend,
            EvalRequest::Score {
                cache_dir: dir.clone(),
                identity: EvalIdentity::fixture("a"),
                logit_chunk: 3,
            },
        );
        let reason = match refused {
            Err(err) => err.reason,
            Ok(_) => panic!("a cache from another vocabulary must be refused"),
        };
        assert!(
            reason.contains("vocabulary"),
            "unexpected refusal: {reason}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
