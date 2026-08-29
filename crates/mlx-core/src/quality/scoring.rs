//! Teacher-forced quality metrics — the pure-numeric half of `mlx eval`.
//!
//! Nothing here loads a model or a tokenizer: every function takes logits that
//! are already in hand and returns either a reference row to cache or a bundle
//! of running sums. That is what makes the self-identity property
//! ([`score_chunk`] of a chunk against its own [`capture_chunk`] output) an
//! EXACT assertion in the unit tests rather than a tolerance.
//!
//! The metrics, per scored position `i` with target id `t`:
//!
//! ```text
//! nll_i   = logsumexp(student_i) - student_i[t]        exact, full vocab
//! top1_i  = argmax(student_i) == teacher argmax_i      exact, full vocab
//! kl_i    = SUM_{v in S} p_v * (log p_v - log q_v)     restricted to S
//! tail_i  = 1 - SUM_{v in S} p_v
//! ```
//!
//! where `S` is the teacher's cached top-`K` support, `p_v` uses the teacher's
//! TRUE full-vocab normaliser (the cached `lse`) and `q_v` the student's own
//! live one. Restricting the SUPPORT but keeping both normalisers exact is what
//! separates this from mlx-lm's DWQ loss, which renormalises both sides over
//! `K` and so cannot report how much probability mass it ignored.
//! `teacher_tail_mass` is that ignored mass, reported alongside the KL so a
//! reader can tell a trustworthy top-`K` KL from a meaningless one.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::{DType, MxArray};

/// The teacher's cached next-token distribution over a span of scored
/// positions. `P` is the span length, `K` the retained support width.
pub struct TeacherLogits {
    /// `[P, K]` f32 — raw (unnormalised) teacher logits at [`Self::indices`].
    pub logits: MxArray,
    /// `[P, K]` u32 — vocabulary ids of the retained support.
    pub indices: MxArray,
    /// `[P]` f32 — the teacher's FULL-vocab logsumexp. Cached because it cannot
    /// be recovered from the top-`K` slice, and without it the KL would have to
    /// renormalise over `K`.
    pub lse: MxArray,
    /// `[P]` f32 — teacher `log p(target)`, exact over the full vocab.
    pub target_logprob: MxArray,
    /// `[P]` u32 — the teacher's top-1 id, exact over the full vocab.
    pub argmax: MxArray,
}

/// One cached sequence: the exact token ids that were scored, plus the
/// teacher's distribution at every position that has a target.
///
/// `tokens` has `T` ids; the forward consumes `tokens[..T-1]` and the targets
/// are `tokens[1..]`, so [`TeacherLogits`] spans `P = T - 1` positions.
pub struct TeacherRow {
    pub tokens: Vec<u32>,
    pub logits: TeacherLogits,
}

impl TeacherLogits {
    /// Number of scored positions (`P`).
    pub fn positions(&self) -> Result<i64> {
        self.lse.shape_at(0)
    }

    /// Support width (`K`).
    pub fn support(&self) -> Result<i64> {
        self.logits.shape_at(1)
    }

    /// Materialize every array. Callers that clear the MLX cache between
    /// position chunks MUST call this first — the arrays are lazy handles onto
    /// graph nodes a `clear_cache` would free.
    pub fn eval(&self) -> Result<()> {
        MxArray::eval_arrays(&[
            &self.logits,
            &self.indices,
            &self.lse,
            &self.target_logprob,
            &self.argmax,
        ])
    }

    /// Positions `[start, end)` of this span.
    pub fn slice(&self, start: i64, end: i64) -> Result<Self> {
        Ok(Self {
            logits: self.logits.slice_axis(0, start, end)?,
            indices: self.indices.slice_axis(0, start, end)?,
            lse: self.lse.slice_axis(0, start, end)?,
            target_logprob: self.target_logprob.slice_axis(0, start, end)?,
            argmax: self.argmax.slice_axis(0, start, end)?,
        })
    }

    /// Join consecutive position chunks back into one span.
    pub fn concat(chunks: &[Self]) -> Result<Self> {
        if chunks.is_empty() {
            return Err(Error::from_reason(
                "TeacherLogits::concat: no position chunks",
            ));
        }
        let join = |pick: fn(&Self) -> &MxArray| -> Result<MxArray> {
            MxArray::concatenate_many(chunks.iter().map(pick).collect(), Some(0))
        };
        Ok(Self {
            logits: join(|c| &c.logits)?,
            indices: join(|c| &c.indices)?,
            lse: join(|c| &c.lse)?,
            target_logprob: join(|c| &c.target_logprob)?,
            argmax: join(|c| &c.argmax)?,
        })
    }
}

/// Reduce one chunk of teacher logits to the cacheable reference.
///
/// `logits` is `[C, V]` f32, `targets` `[C]` u32. `top_k` is clamped to `V`, so
/// a vocabulary smaller than the requested support degrades to full-vocab
/// (exact) KL rather than erroring.
pub fn capture_chunk(logits: &MxArray, targets: &MxArray, top_k: i64) -> Result<TeacherLogits> {
    let vocab = logits.shape_at(1)?;
    let k = top_k.clamp(1, vocab);

    let lse = logits.logsumexp(Some(&[1]), Some(false))?;
    // `argpartition(V - k)` puts the k largest entries last, in no particular
    // order — order does not matter, the KL is a sum over the support.
    let order = logits.argpartition((vocab - k) as i32, Some(1))?;
    let indices = order.slice_axis(1, vocab - k, vocab)?;

    let target_column = targets.expand_dims(1)?;
    let target_logit = logits
        .take_along_axis(&target_column, 1)?
        .squeeze(Some(&[1]))?;

    Ok(TeacherLogits {
        logits: logits.take_along_axis(&indices, 1)?,
        indices,
        target_logprob: target_logit.sub(&lse)?,
        lse,
        argmax: logits.argmax(1, Some(false))?,
    })
}

/// Running metric sums over an arbitrary set of scored positions.
///
/// Sums, never means: rows have unequal length after truncation, so a
/// mean-of-means would silently reweight the short ones. Every mean is taken
/// exactly once, in [`Self::into_report`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ScoreTotals {
    pub positions: u64,
    pub nll_sum: f64,
    pub teacher_nll_sum: f64,
    pub kl_sum: f64,
    pub tail_mass_sum: f64,
    pub top1_matches: f64,
}

impl ScoreTotals {
    pub fn add(&mut self, other: &Self) {
        self.positions += other.positions;
        self.nll_sum += other.nll_sum;
        self.teacher_nll_sum += other.teacher_nll_sum;
        self.kl_sum += other.kl_sum;
        self.tail_mass_sum += other.tail_mass_sum;
        self.top1_matches += other.top1_matches;
    }

    pub fn into_report(
        self,
        rows: u32,
        top_k: u32,
        teacher_path: String,
        teacher_quantized: bool,
    ) -> Result<EvalReport> {
        if self.positions == 0 {
            return Err(Error::from_reason(
                "eval scored 0 positions — the teacher cache is empty",
            ));
        }
        let n = self.positions as f64;
        let mean_nll = self.nll_sum / n;
        let teacher_mean_nll = self.teacher_nll_sum / n;
        Ok(EvalReport {
            rows,
            positions: self.positions as u32,
            top_k,
            teacher_path,
            teacher_quantized,
            mean_nll,
            perplexity: mean_nll.exp(),
            teacher_mean_nll,
            teacher_perplexity: teacher_mean_nll.exp(),
            mean_kl_topk: self.kl_sum / n,
            teacher_tail_mass: self.tail_mass_sum / n,
            top1_agreement: self.top1_matches / n,
        })
    }
}

/// Score one chunk of student logits against the teacher's reference for the
/// same positions.
///
/// `student` is `[C, V]` f32, `targets` `[C]` u32, `reference` spans exactly
/// those `C` positions. Exactly ONE host readback per chunk: the five sums are
/// packed into a single `[5]` array first.
pub fn score_chunk(
    student: &MxArray,
    targets: &MxArray,
    reference: &TeacherLogits,
) -> Result<ScoreTotals> {
    let positions = student.shape_at(0)?;
    if positions != reference.positions()? {
        return Err(Error::from_reason(format!(
            "score_chunk: {positions} student positions vs {} teacher positions",
            reference.positions()?
        )));
    }

    let student_lse = student.logsumexp(Some(&[1]), Some(false))?;
    let target_column = targets.expand_dims(1)?;
    let student_target = student
        .take_along_axis(&target_column, 1)?
        .squeeze(Some(&[1]))?;
    let nll = student_lse.sub(&student_target)?;
    let teacher_nll = reference.target_logprob.negative()?;

    // Both sides keep their OWN true full-vocab normaliser; only the support is
    // shared. `p` therefore sums to <= 1 and the shortfall is the tail mass.
    let teacher_logprob = reference.logits.sub(&reference.lse.expand_dims(1)?)?;
    let student_logprob = student
        .take_along_axis(&reference.indices, 1)?
        .sub(&student_lse.expand_dims(1)?)?;
    let p = teacher_logprob.exp()?;
    let q = student_logprob.exp()?;
    let head = p
        .mul(&teacher_logprob.sub(&student_logprob)?)?
        .sum(Some(&[1]), Some(false))?;
    let tail_mass = p
        .sum(Some(&[1]), Some(false))?
        .sub_scalar(1.0)?
        .negative()?;
    let student_tail = q
        .sum(Some(&[1]), Some(false))?
        .sub_scalar(1.0)?
        .negative()?;
    // The head terms alone are NOT a divergence: both sides carry their own
    // full-vocab normaliser, so a student holding more mass on the teacher's
    // top-K than the teacher does makes the sum negative — by the log-sum
    // inequality the head is bounded below by `P·log(P/Q)`, which is < 0 exactly
    // when Q > P. Folding everything outside the support into one bucket makes
    // this a real KL over a (K+1)-way partition, and therefore non-negative.
    // The floor keeps `log` finite; a bucket under it contributes nothing a
    // finite-precision score could resolve anyway.
    const TAIL_FLOOR: f64 = 1e-12;
    let tail_term = tail_mass.mul(
        &tail_mass
            .clip(Some(TAIL_FLOOR), None)?
            .log()?
            .sub(&student_tail.clip(Some(TAIL_FLOOR), None)?.log()?)?,
    )?;
    let kl = head.add(&tail_term)?;

    let top1 = student
        .argmax(1, Some(false))?
        .equal(&reference.argmax)?
        .astype(DType::Float32)?;

    let total = |x: &MxArray| x.sum(Some(&[0]), Some(true));
    let packed = MxArray::concatenate_many(
        vec![
            &total(&nll)?,
            &total(&teacher_nll)?,
            &total(&kl)?,
            &total(&tail_mass)?,
            &total(&top1)?,
        ],
        Some(0),
    )?;
    packed.eval();
    let sums = packed.to_float32()?;

    Ok(ScoreTotals {
        positions: positions as u64,
        nll_sum: sums[0] as f64,
        teacher_nll_sum: sums[1] as f64,
        kl_sum: sums[2] as f64,
        tail_mass_sum: sums[3] as f64,
        top1_matches: sums[4] as f64,
    })
}

/// Teacher-forced quality of one checkpoint against a cached reference.
///
/// `mean_nll`, `perplexity` and `top1_agreement` are EXACT full-vocab numbers.
/// `mean_kl_topk` is exact only over the teacher's cached support — read it
/// together with `teacher_tail_mass`, which is the probability mass that support
/// leaves out.
#[napi(object)]
pub struct EvalReport {
    pub rows: u32,
    pub positions: u32,
    pub top_k: u32,
    /// Teacher the cache was captured from, carried through from its metadata.
    pub teacher_path: String,
    /// That teacher was itself quantized, so every number below measures
    /// divergence from it rather than from the bf16 model.
    pub teacher_quantized: bool,
    pub mean_nll: f64,
    pub perplexity: f64,
    pub teacher_mean_nll: f64,
    pub teacher_perplexity: f64,
    pub mean_kl_topk: f64,
    pub teacher_tail_mass: f64,
    pub top1_agreement: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::losses::Losses;

    const POSITIONS: i64 = 8;
    const VOCAB: i64 = 32;

    /// A deterministic, non-degenerate `[8, 32]` logit block. The spread is wide
    /// enough that the top-8 support carries most but not all of the mass, so
    /// `teacher_tail_mass` is a meaningful non-zero number rather than an
    /// underflowed zero.
    fn fixture_logits() -> MxArray {
        let mut values = Vec::with_capacity((POSITIONS * VOCAB) as usize);
        for pos in 0..POSITIONS {
            for v in 0..VOCAB {
                let x = (pos * 7 + v * 13) % 29;
                values.push((x as f32) * 0.31 - 4.0);
            }
        }
        MxArray::from_float32(&values, &[POSITIONS, VOCAB]).unwrap()
    }

    fn fixture_targets() -> MxArray {
        let ids: Vec<u32> = (0..POSITIONS)
            .map(|p| ((p * 5 + 3) % VOCAB) as u32)
            .collect();
        MxArray::from_uint32(&ids, &[POSITIONS]).unwrap()
    }

    fn read_row(array: &MxArray) -> Vec<f32> {
        array.eval();
        array.to_float32().unwrap().to_vec()
    }

    /// THE gating property: a checkpoint scored against a reference captured
    /// from ITSELF has zero divergence. Asserted EXACTLY — every KL term is
    /// `p_v * (x - x)` on bit-identical inputs, so any tolerance here would be
    /// hiding a defect rather than absorbing noise.
    #[test]
    fn scoring_a_capture_against_its_own_source_is_exactly_zero() {
        let logits = fixture_logits();
        let targets = fixture_targets();
        let reference = capture_chunk(&logits, &targets, 8).unwrap();
        let totals = score_chunk(&logits, &targets, &reference).unwrap();

        assert_eq!(totals.positions, POSITIONS as u64);
        assert_eq!(totals.kl_sum, 0.0, "self-KL must be exactly zero");
        assert_eq!(
            totals.top1_matches, POSITIONS as f64,
            "a checkpoint must agree with itself at every position"
        );
        assert_eq!(
            totals.nll_sum, totals.teacher_nll_sum,
            "student NLL must equal the cached teacher NLL bit for bit"
        );

        let report = totals.into_report(1, 8, "self".to_string(), false).unwrap();
        assert_eq!(report.mean_kl_topk, 0.0);
        assert_eq!(report.top1_agreement, 1.0);
        assert_eq!(report.mean_nll, report.teacher_mean_nll);
        assert_eq!(report.perplexity, report.teacher_perplexity);
        // The support is 8 of 32 entries, so a real, sizeable tail is left out —
        // proving the zero KL above is not a vacuous "everything is in K" case.
        assert!(
            report.teacher_tail_mass > 0.01,
            "top-8 of 32 must leave real mass out, got {}",
            report.teacher_tail_mass
        );
    }

    /// The mutation guard for the test above: a scorer hardwired to return zero,
    /// or one that ignores the student entirely and re-reads the cache as its own
    /// answer, passes self-identity. Perturb the student and every metric must
    /// move.
    ///
    /// Mutation: at position 3 the LOWEST logit is lifted to just under the
    /// winner (a large distribution change that leaves the top-1 alone), and at
    /// position 5 a different id is pushed past the winner (a top-1 flip that
    /// barely moves the distribution). Together they prove the KL and the
    /// agreement count are independently live.
    #[test]
    fn scoring_a_perturbed_student_moves_every_metric() {
        let logits = fixture_logits();
        let targets = fixture_targets();
        let reference = capture_chunk(&logits, &targets, 8).unwrap();

        let teacher_argmax = reference.argmax.to_uint32().unwrap().to_vec();
        let mut values = logits.to_float32().unwrap().to_vec();
        let row = |pos: usize| pos * VOCAB as usize;

        let winner_at_3 = values[row(3) + teacher_argmax[3] as usize];
        let lowest_at_3 = (0..VOCAB as usize)
            .min_by(|a, b| values[row(3) + a].total_cmp(&values[row(3) + b]))
            .unwrap();
        values[row(3) + lowest_at_3] = winner_at_3 - 0.25;

        let winner_at_5 = teacher_argmax[5] as usize;
        let challenger_at_5 = (winner_at_5 + 1) % VOCAB as usize;
        values[row(5) + challenger_at_5] = values[row(5) + winner_at_5] + 1.0;
        let student = MxArray::from_float32(&values, &[POSITIONS, VOCAB]).unwrap();

        let totals = score_chunk(&student, &targets, &reference).unwrap();
        assert!(
            totals.kl_sum > 1e-3,
            "a perturbed student must diverge, got {}",
            totals.kl_sum
        );
        assert_eq!(
            totals.top1_matches,
            (POSITIONS - 1) as f64,
            "exactly the swapped position must disagree"
        );
        assert_ne!(totals.nll_sum, totals.teacher_nll_sum);
    }

    /// Sum only the top-K terms, exactly as this file did before the tail bucket
    /// existed. The reference for the anti-vacuity half of the test below.
    fn head_only_kl(teacher: &MxArray, student: &MxArray, indices: &MxArray) -> f64 {
        let t = teacher.to_float32().unwrap().to_vec();
        let st = student.to_float32().unwrap().to_vec();
        let idx = indices.to_uint32().unwrap().to_vec();
        let vocab = VOCAB as usize;
        let k = idx.len() / POSITIONS as usize;
        let lse = |v: &[f32], row: usize| -> f64 {
            let m = v[row..row + vocab]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max) as f64;
            m + v[row..row + vocab]
                .iter()
                .map(|x| ((*x as f64) - m).exp())
                .sum::<f64>()
                .ln()
        };
        let mut sum = 0.0f64;
        for pos in 0..POSITIONS as usize {
            let row = pos * vocab;
            let (tl, sl) = (lse(&t, row), lse(&st, row));
            for j in 0..k {
                let v = idx[pos * k + j] as usize;
                let lp = t[row + v] as f64 - tl;
                let lq = st[row + v] as f64 - sl;
                sum += lp.exp() * (lp - lq);
            }
        }
        sum
    }

    /// A student SHARPER than its teacher holds more mass on the teacher's own
    /// top-K than the teacher does. The head terms then sum BELOW zero — they
    /// are bounded below by `P·log(P/Q)`, which is negative exactly when
    /// `Q > P` — and a candidate would score better than the teacher's own zero.
    /// The aggregated tail bucket is what makes the reported number a real KL
    /// over a (K+1)-way partition, and so non-negative.
    #[test]
    fn a_sharper_student_cannot_score_below_zero() {
        // Small on purpose: 4 of 32 in the support leaves the tail real mass.
        const TOP_K: i64 = 4;
        let logits = fixture_logits();
        let targets = fixture_targets();
        let reference = capture_chunk(&logits, &targets, TOP_K).unwrap();

        // Same ordering, lower temperature.
        let sharper: Vec<f32> = logits
            .to_float32()
            .unwrap()
            .to_vec()
            .iter()
            .map(|v| v / 0.9)
            .collect();
        let student = MxArray::from_float32(&sharper, &[POSITIONS, VOCAB]).unwrap();

        // Anti-vacuity: this fixture must actually exercise the defect, or the
        // assertion below would hold for a scorer that never had the bug.
        let head = head_only_kl(&logits, &student, &reference.indices);
        assert!(
            head < 0.0,
            "fixture proves nothing unless the head-only sum is negative, got {head}"
        );

        let totals = score_chunk(&student, &targets, &reference).unwrap();
        assert!(
            totals.kl_sum >= 0.0,
            "a KL cannot be negative, got {} (head-only would be {head})",
            totals.kl_sum
        );
    }

    /// Cross-check the fifteen lines of reductions above against the in-tree
    /// loss implementations. At full support the partial KL IS the KL, so
    /// `Losses::kl_divergence` over both log-softmaxes must agree; and the NLL is
    /// cross-entropy with class-index targets. Catches an argument-order flip in
    /// the KL or an NLL that dropped the logsumexp normaliser.
    #[test]
    fn reductions_agree_with_the_in_tree_losses() {
        let teacher = fixture_logits();
        let targets = fixture_targets();
        let mut values = teacher.to_float32().unwrap().to_vec();
        for (i, v) in values.iter_mut().enumerate() {
            *v += ((i % 5) as f32) * 0.17;
        }
        let student = MxArray::from_float32(&values, &[POSITIONS, VOCAB]).unwrap();

        // Full support: the top-K restriction is the identity here.
        let reference = capture_chunk(&teacher, &targets, VOCAB).unwrap();
        let totals = score_chunk(&student, &targets, &reference).unwrap();
        let report = totals
            .into_report(1, VOCAB as u32, "oracle".to_string(), false)
            .unwrap();

        let oracle_kl = read_row(
            &Losses::kl_divergence(
                &teacher.log_softmax(-1).unwrap(),
                &student.log_softmax(-1).unwrap(),
            )
            .unwrap(),
        )[0] as f64;
        let oracle_nll =
            read_row(&Losses::cross_entropy(&student, &targets, None, None, None).unwrap())[0]
                as f64;

        assert!(
            (report.mean_kl_topk - oracle_kl).abs() < 1e-6,
            "KL {} vs Losses::kl_divergence {oracle_kl}",
            report.mean_kl_topk
        );
        assert!(
            (report.mean_nll - oracle_nll).abs() < 1e-6,
            "NLL {} vs Losses::cross_entropy {oracle_nll}",
            report.mean_nll
        );
        assert!(
            report.teacher_tail_mass.abs() < 1e-6,
            "full support must leave no tail, got {}",
            report.teacher_tail_mass
        );
    }

    /// Position chunking must not change the answer: rows have unequal length
    /// after truncation, so anything that divides per chunk instead of summing
    /// would reweight them.
    #[test]
    fn chunked_positions_sum_to_the_whole_span() {
        let logits = fixture_logits();
        let targets = fixture_targets();
        let reference = capture_chunk(&logits, &targets, 8).unwrap();
        let whole = score_chunk(&logits, &targets, &reference).unwrap();

        let mut chunked = ScoreTotals::default();
        for (start, end) in [(0i64, 3i64), (3, 5), (5, POSITIONS)] {
            chunked.add(
                &score_chunk(
                    &logits.slice_axis(0, start, end).unwrap(),
                    &targets.slice_axis(0, start, end).unwrap(),
                    &reference.slice(start, end).unwrap(),
                )
                .unwrap(),
            );
        }
        assert_eq!(chunked.positions, whole.positions);
        assert!((chunked.nll_sum - whole.nll_sum).abs() < 1e-4);
        assert_eq!(chunked.top1_matches, whole.top1_matches);

        // The same split, reassembled, must reproduce the reference span.
        let rejoined = TeacherLogits::concat(&[
            reference.slice(0, 3).unwrap(),
            reference.slice(3, 5).unwrap(),
            reference.slice(5, POSITIONS).unwrap(),
        ])
        .unwrap();
        assert_eq!(rejoined.positions().unwrap(), POSITIONS);
        assert_eq!(
            read_row(&rejoined.lse),
            read_row(&reference.lse),
            "concatenated chunks must reproduce the whole span"
        );
    }
}
