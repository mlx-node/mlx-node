//! Confidence-based verification lengths using costs measured on this host.
//!
//! Conditional keep probabilities become prefix survival probabilities. For a
//! fixed total query budget, choosing the largest remaining survival probability
//! maximizes expected accepted tokens while preserving each row's prefix. Only
//! measured query shapes compete; CUDA graph assumptions do not enter the policy.

#[derive(Debug)]
pub(crate) struct VerificationBudget {
    costs_ns: Vec<Option<f64>>,
    calibration: Vec<usize>,
    fallback: bool,
    context_band: Option<usize>,
}

impl VerificationBudget {
    pub fn new(max_drafts: usize) -> Self {
        let mut calibration = vec![0, 1, max_drafts.div_ceil(2), max_drafts];
        calibration.sort_unstable();
        calibration.dedup();
        Self {
            costs_ns: vec![None; max_drafts.saturating_add(2)],
            calibration,
            fallback: false,
            context_band: None,
        }
    }

    pub fn observe_context(&mut self, tokens: usize) {
        let band = tokens / 2048;
        if self.context_band.is_some_and(|previous| previous != band) && !self.fallback {
            self.costs_ns.fill(None);
        }
        self.context_band = Some(band);
    }

    /// A short tail never starts calibration for a shape it cannot execute.
    pub fn calibration_cap(&self, cap: usize) -> Option<usize> {
        if self.fallback {
            return None;
        }
        self.calibration
            .iter()
            .copied()
            .find(|&n| n <= cap && self.costs_ns[n + 1].is_none())
    }

    pub fn is_fallback(&self) -> bool {
        self.fallback
    }

    pub fn record(&mut self, drafts: usize, elapsed_ns: u64) {
        let Some(cost) = self.costs_ns.get_mut(drafts.saturating_add(1)) else {
            return;
        };
        let sample = elapsed_ns.max(1) as f64;
        *cost = Some(cost.map_or(sample, |old| old * 0.8 + sample * 0.2));
    }

    /// Returns None for missing/malformed confidence or incomplete calibration.
    /// Zero means measured target-only AR wins, including saved future draft work.
    pub fn choose(&mut self, conditional: &[f32], proposal_ns: u64) -> Option<usize> {
        if self.calibration_cap(conditional.len()).is_some() {
            return None;
        }
        let ar_ns = self.costs_ns.get(1).copied().flatten()?;
        let lengths = choose_verification_lengths(
            &[conditional],
            &self.costs_ns,
            proposal_ns as f64,
            ar_ns,
            1.05,
        )?;
        let keep = lengths[0];
        self.fallback = keep == 0;
        Some(keep)
    }
}

/// Shared total-query budget optimizer. Costs are indexed by total verifier
/// queries, including one anchor per active row. The caller owns separate cost
/// profiles for different models, context bands and active batch sizes.
pub(crate) fn choose_verification_lengths(
    conditional: &[&[f32]],
    costs_ns: &[Option<f64>],
    proposal_ns: f64,
    ar_batch_ns: f64,
    minimum_speedup: f64,
) -> Option<Vec<usize>> {
    if conditional.is_empty()
        || !proposal_ns.is_finite()
        || proposal_ns < 0.0
        || !ar_batch_ns.is_finite()
        || ar_batch_ns <= 0.0
        || !minimum_speedup.is_finite()
        || minimum_speedup < 1.0
    {
        return None;
    }
    let survival = conditional
        .iter()
        .map(|row| {
            let mut product = 1.0f64;
            row.iter()
                .map(|&p| {
                    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                        return None;
                    }
                    product *= f64::from(p);
                    Some(product)
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect::<Option<Vec<_>>>()?;
    let mut lengths = vec![0; conditional.len()];
    let mut best = lengths.clone();
    let mut expected = conditional.len() as f64; // one boundary per row
    let mut best_rate = expected / ar_batch_ns * minimum_speedup;
    let mut queries = conditional.len();
    loop {
        let next = survival
            .iter()
            .enumerate()
            .filter_map(|(row, p)| p.get(lengths[row]).map(|&p| (row, p)))
            .max_by(|(a, pa), (b, pb)| pa.total_cmp(pb).then_with(|| b.cmp(a)));
        let Some((row, probability)) = next else {
            break;
        };
        lengths[row] += 1;
        queries += 1;
        expected += probability;
        let Some(verify_ns) = costs_ns.get(queries).copied().flatten() else {
            continue;
        };
        if !verify_ns.is_finite() || verify_ns <= 0.0 {
            continue;
        }
        let rate = expected / (proposal_ns + verify_ns);
        if rate > best_rate {
            best_rate = rate;
            best.clone_from(&lengths);
        }
    }
    Some(best)
}

/// A scheduled profile is valid only for this ordered set of owners, context
/// bands and available draft widths. A new owner or a short tail recalibrates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScheduledBudgetRow {
    pub seq_id: u32,
    pub context_band: u32,
    pub cap: usize,
}

#[derive(Default)]
pub(crate) struct ScheduledVerificationBudget {
    rows: Vec<ScheduledBudgetRow>,
    // Exact per-owner draft counts, never just their sum. Proposal work uses
    // each owner's full cap for every nonzero shape, including calibration.
    costs: Vec<(Vec<usize>, f64)>,
}

impl ScheduledVerificationBudget {
    #[cfg(test)]
    pub fn has_measurements(&self) -> bool {
        !self.costs.is_empty()
    }

    pub fn configure(&mut self, rows: &[ScheduledBudgetRow]) {
        if self.rows != rows {
            self.rows = rows.to_vec();
            self.costs.clear();
        }
    }

    pub fn clear(&mut self) {
        self.rows.clear();
        self.costs.clear();
    }

    fn cost(&self, shape: &[usize]) -> Option<f64> {
        self.costs
            .iter()
            .find_map(|(known, cost)| (known == shape).then_some(*cost))
    }

    pub fn next_probe(&self) -> Option<Vec<usize>> {
        [0, 1, 2, 3].into_iter().find_map(|level| {
            let shape = self
                .rows
                .iter()
                .map(|row| match level {
                    0 => 0,
                    1 => row.cap.min(1),
                    2 => row.cap.div_ceil(2),
                    _ => row.cap,
                })
                .collect::<Vec<_>>();
            self.cost(&shape).is_none().then_some(shape)
        })
    }

    pub fn record(&mut self, shape: &[usize], elapsed_ns: u64) {
        if shape.len() != self.rows.len()
            || shape.iter().zip(&self.rows).any(|(&n, row)| n > row.cap)
        {
            return;
        }
        let sample = elapsed_ns.max(1) as f64;
        if let Some((_, cost)) = self.costs.iter_mut().find(|(known, _)| known == shape) {
            *cost = *cost * 0.8 + sample * 0.2;
        } else if self.costs.len() < 8 {
            self.costs.push((shape.to_vec(), sample));
        }
    }

    /// The bool identifies an unmeasured calibration shape. Only a measured
    /// zero allocation can trigger permanent fallback; calibration zero keeps
    /// every owner's draft context alive for the next wave.
    pub fn choose(&self, conditional: &[&[f32]]) -> Option<(Vec<usize>, bool)> {
        if conditional.len() != self.rows.len() || self.next_probe().is_some() {
            return None;
        }
        let survival = conditional
            .iter()
            .zip(&self.rows)
            .map(|(values, row)| {
                if values.len() != row.cap {
                    return None;
                }
                let mut product = 1.0;
                values
                    .iter()
                    .map(|&value| {
                        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                            return None;
                        }
                        product *= f64::from(value);
                        Some(product)
                    })
                    .collect::<Option<Vec<_>>>()
            })
            .collect::<Option<Vec<_>>>()?;
        let total = self.rows.iter().map(|row| row.cap).sum::<usize>();
        // Probe a bounded number of confidence-allocated prefixes. Their cost
        // must be observed before they can compete with any measured shape.
        if self.costs.len() < 8 {
            for budget in [total / 3, total * 2 / 3] {
                let mut shape = vec![0; self.rows.len()];
                for _ in 0..budget {
                    let next = survival
                        .iter()
                        .enumerate()
                        .filter_map(|(i, row)| row.get(shape[i]).map(|&p| (i, p)))
                        .max_by(|(a, pa), (b, pb)| pa.total_cmp(pb).then_with(|| b.cmp(a)));
                    if let Some((i, _)) = next {
                        shape[i] += 1;
                    }
                }
                if self.cost(&shape).is_none() {
                    return Some((shape, true));
                }
            }
        }
        let mut best = vec![0; self.rows.len()];
        let boundaries = self.rows.len() as f64;
        let mut rate = boundaries / self.cost(&best)? * 1.05;
        for (shape, cost) in &self.costs {
            if shape.iter().all(|&n| n == 0) {
                continue;
            }
            let expected = boundaries
                + survival
                    .iter()
                    .zip(shape)
                    .map(|(row, &n)| row[..n].iter().sum::<f64>())
                    .sum::<f64>();
            if expected / cost > rate {
                best.clone_from(shape);
                rate = expected / cost;
            }
        }
        Some((best, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_cost_and_survival_choose_a_shorter_profitable_prefix() {
        let costs = [None, Some(100.0), Some(110.0), Some(115.0), Some(400.0)];
        assert_eq!(
            choose_verification_lengths(&[&[0.9, 0.9, 0.9]], &costs, 20.0, 100.0, 1.05),
            Some(vec![2])
        );
        assert_eq!(
            choose_verification_lengths(&[&[0.01, 0.9, 0.9]], &costs, 20.0, 100.0, 1.05),
            Some(vec![0])
        );
        assert_eq!(
            choose_verification_lengths(&[&[0.9, f32::NAN]], &costs, 20.0, 100.0, 1.05),
            None
        );
    }

    #[test]
    fn global_budget_preserves_prefixes_and_credits_each_boundary_once() {
        let costs = [None, None, Some(100.0), Some(110.0), Some(115.0)];
        assert_eq!(
            choose_verification_lengths(&[&[0.9, 0.9], &[0.6, 1.0]], &costs, 10.0, 100.0, 1.05),
            Some(vec![2, 0])
        );
        assert_eq!(
            choose_verification_lengths(&[&[0.8, 0.1], &[0.9, 0.5]], &costs, 10.0, 100.0, 1.05),
            Some(vec![1, 1])
        );
    }

    #[test]
    fn unknown_shapes_do_not_compete_and_calibration_respects_tail_capacity() {
        let mut policy = VerificationBudget::new(7);
        for n in [0, 1, 4, 7] {
            assert_eq!(policy.calibration_cap(7), Some(n));
            policy.record(n, 100 + n as u64 * 5);
        }
        assert_eq!(policy.calibration_cap(7), None);
        assert_eq!(policy.choose(&[0.99; 7], 10), Some(7));
        assert!(!policy.is_fallback());
        let tail = VerificationBudget::new(7);
        assert_eq!(tail.calibration_cap(0), Some(0));
        assert_eq!(
            choose_verification_lengths(&[&[0.9; 7]], &[None; 9], 10.0, 100.0, 1.05),
            Some(vec![0])
        );
    }

    #[test]
    fn scheduled_costs_distinguish_equal_total_queries_and_reset_on_owner_changes() {
        let rows = [
            ScheduledBudgetRow {
                seq_id: 1,
                context_band: 0,
                cap: 7,
            },
            ScheduledBudgetRow {
                seq_id: 2,
                context_band: 0,
                cap: 7,
            },
        ];
        let mut policy = ScheduledVerificationBudget::default();
        policy.configure(&rows);
        for shape in [vec![0, 0], vec![1, 1], vec![4, 4], vec![7, 7]] {
            assert_eq!(policy.next_probe(), Some(shape.clone()));
            policy.record(&shape, if shape == [0, 0] { 100 } else { 1000 });
        }
        policy.record(&[7, 0], 100);
        policy.record(&[4, 3], 10000);
        assert_eq!(policy.cost(&[7, 0]), Some(100.0));
        assert_eq!(policy.cost(&[4, 3]), Some(10000.0));
        // Fill the bounded calibration set before selecting measured costs.
        policy.record(&[3, 1], 1000);
        policy.record(&[6, 3], 1000);
        assert_eq!(
            policy.choose(&[&[0.99; 7], &[0.01; 7]]),
            Some((vec![7, 0], false))
        );
        let mut changed = rows.clone();
        changed[1].seq_id = 3;
        policy.configure(&changed);
        assert_eq!(policy.next_probe(), Some(vec![0, 0]));
        assert_eq!(policy.cost(&[7, 0]), None);
        policy.record(&[0, 0], 10);
        changed[0].context_band = 1;
        policy.configure(&changed);
        assert_eq!(policy.next_probe(), Some(vec![0, 0]));
    }

    #[test]
    fn scheduled_zero_probe_is_temporary_and_unmeasured_shapes_are_calibration() {
        let mut policy = ScheduledVerificationBudget::default();
        policy.configure(&[ScheduledBudgetRow {
            seq_id: 1,
            context_band: 0,
            cap: 3,
        }]);
        assert_eq!(policy.next_probe(), Some(vec![0]));
        assert_eq!(policy.choose(&[&[0.1; 3]]), None);
        for n in 0..=3 {
            policy.record(&[n], if n == 0 { 100 } else { 1000 });
        }
        assert_eq!(policy.choose(&[&[0.1; 3]]), Some((vec![0], false)));
        assert_eq!(policy.choose(&[&[f32::NAN; 3]]), None);
    }
}
