use super::DRAFT_SLOTS;

// ---------------------------------------------------------------------------
// Logistic Acceptance Calibration (Splash `VerifyBudget.{hpp,cpp}`)
// ---------------------------------------------------------------------------

#[inline]
pub(super) fn logit_f64(p: f64) -> f64 {
    let p = p.clamp(1.0e-6, 1.0 - 1.0e-6);
    (p / (1.0 - p)).ln()
}

/// Logistic calibration parameters for draft proposal acceptance:
/// `logit(p) = slope * logit(q) + intercepts[position]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BudgetAcceptance {
    pub slope: f32,
    pub intercepts: [f32; DRAFT_SLOTS],
}

impl BudgetAcceptance {
    /// Splash `VerifyBudget::kPrior` fitted on Qwen3.8 / Qwen3.6-35B-A3B decode cycles.
    pub(crate) const PRIOR: Self = Self {
        slope: 0.7,
        intercepts: [0.0, -0.35, -0.8, -0.75, -0.75, -0.85, -0.45],
    };

    /// Calibrated conditional acceptance probability for a draft candidate of
    /// probability `q` at 0-based proposal `position`.
    #[inline]
    pub(crate) fn estimate(&self, q: f32, position: usize) -> f32 {
        let q = q.clamp(1.0e-6, 1.0 - 1.0e-6);
        let intercept = self.intercepts.get(position).copied().unwrap_or(0.0);
        let z = self.slope * (q / (1.0 - q)).ln() + intercept;
        1.0 / (1.0 + (-z).exp())
    }
}

impl Default for BudgetAcceptance {
    fn default() -> Self {
        Self::PRIOR
    }
}

/// One labeled draft node from a completed verification walk, used to refit
/// [`VerifyBudget`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BudgetLabel {
    pub position: usize,
    pub feature: f32,
    pub accepted: bool,
    pub cap: f32,
}

impl BudgetLabel {
    #[inline]
    fn is_valid(&self) -> bool {
        self.position < DRAFT_SLOTS
            && self.feature.is_finite()
            && self.cap.is_finite()
            && (0.0..=1.0).contains(&self.cap)
    }

    #[inline]
    fn is_capped(&self, z: f64) -> bool {
        self.cap < 1.0 && z > logit_f64(f64::from(self.cap))
    }
}

/// Online logistic calibrator for draft tree probabilities (Splash `VerifyBudget`).
#[derive(Clone, Debug)]
pub(crate) struct VerifyBudget {
    acceptance: [BudgetAcceptance; 2],
    labels: [Vec<BudgetLabel>; 2],
    next: [usize; 2],
    cycles: u64,
}

impl Default for VerifyBudget {
    fn default() -> Self {
        Self::new()
    }
}

impl VerifyBudget {
    pub(crate) const REFIT_CYCLES: u64 = 64;
    pub(crate) const MAX_LABELS: usize = 4096;
    pub(crate) const SLOPE_PRECISION: f64 = 50.0;
    pub(crate) const INTERCEPT_PRECISION: f64 = 4.0;
    const PARAMS: usize = 1 + DRAFT_SLOTS;
    const NEWTON_ITERS: usize = 16;

    pub(crate) fn new() -> Self {
        Self {
            acceptance: [BudgetAcceptance::PRIOR, BudgetAcceptance::PRIOR],
            labels: [Vec::new(), Vec::new()],
            next: [0, 0],
            cycles: 0,
        }
    }

    pub(crate) fn acceptance(&self, samples: bool) -> &BudgetAcceptance {
        &self.acceptance[usize::from(samples)]
    }

    /// Record a cycle's draft labels and trigger a MAP refit every `REFIT_CYCLES`.
    pub(crate) fn observe(&mut self, samples: bool, labels: &[BudgetLabel]) {
        let slot = usize::from(samples);
        let kept = &mut self.labels[slot];
        let next = &mut self.next[slot];
        for &label in labels {
            if !label.is_valid() {
                continue;
            }
            if kept.len() < Self::MAX_LABELS {
                kept.push(label);
            } else {
                kept[*next] = label;
                *next = (*next + 1) % Self::MAX_LABELS;
            }
        }
        self.cycles = self.cycles.wrapping_add(1);
        if self.cycles.is_multiple_of(Self::REFIT_CYCLES) {
            for s in 0..2 {
                if !self.labels[s].is_empty() {
                    let ordered = self.ordered_labels(s == 1);
                    self.acceptance[s] = Self::fit(&ordered, &self.acceptance[s]);
                }
            }
        }
    }

    /// Kept labels for greedy (`false`) or sampled (`true`) lanes, oldest first.
    pub(crate) fn ordered_labels(&self, samples: bool) -> Vec<BudgetLabel> {
        let slot = usize::from(samples);
        let kept = &self.labels[slot];
        let split = self.next[slot].min(kept.len());
        let mut out = Vec::with_capacity(kept.len());
        out.extend_from_slice(&kept[split..]);
        out.extend_from_slice(&kept[..split]);
        out
    }

    #[inline]
    fn prior_precision(param: usize) -> f64 {
        if param == 0 {
            Self::SLOPE_PRECISION
        } else {
            Self::INTERCEPT_PRECISION
        }
    }

    fn params_of(a: &BudgetAcceptance) -> [f64; Self::PARAMS] {
        let mut p = [0.0; Self::PARAMS];
        p[0] = f64::from(a.slope);
        for i in 0..DRAFT_SLOTS {
            p[1 + i] = f64::from(a.intercepts[i]);
        }
        p
    }

    fn objective(
        labels: &[BudgetLabel],
        theta: &[f64; Self::PARAMS],
        prior: &[f64; Self::PARAMS],
    ) -> f64 {
        let mut value = 0.0;
        for label in labels {
            if !label.is_valid() {
                continue;
            }
            let z = theta[0] * f64::from(label.feature) + theta[1 + label.position];
            if label.is_capped(z) {
                let cap = f64::from(label.cap).clamp(1.0e-6, 1.0 - 1.0e-6);
                value -= if label.accepted {
                    cap.ln()
                } else {
                    (-cap).ln_1p()
                };
                continue;
            }
            let margin = if label.accepted { z } else { -z };
            value += if margin > 0.0 {
                (-margin).exp().ln_1p()
            } else {
                -margin + margin.exp().ln_1p()
            };
        }
        for i in 0..Self::PARAMS {
            let diff = theta[i] - prior[i];
            value += 0.5 * Self::prior_precision(i) * diff * diff;
        }
        value
    }

    fn solve(
        mut h: [[f64; Self::PARAMS]; Self::PARAMS],
        mut g: [f64; Self::PARAMS],
    ) -> [f64; Self::PARAMS] {
        for col in 0..Self::PARAMS {
            let mut pivot = col;
            for row in (col + 1)..Self::PARAMS {
                if h[row][col].abs() > h[pivot][col].abs() {
                    pivot = row;
                }
            }
            h.swap(col, pivot);
            g.swap(col, pivot);
            let diag = h[col][col];
            if diag.abs() < 1.0e-30 {
                continue;
            }
            for row in (col + 1)..Self::PARAMS {
                let factor = h[row][col] / diag;
                for k in col..Self::PARAMS {
                    h[row][k] -= factor * h[col][k];
                }
                g[row] -= factor * g[col];
            }
        }
        let mut x = [0.0; Self::PARAMS];
        for row in (0..Self::PARAMS).rev() {
            let mut sum = g[row];
            for k in (row + 1)..Self::PARAMS {
                sum -= h[row][k] * x[k];
            }
            let diag = h[row][row];
            x[row] = if diag.abs() < 1.0e-30 {
                0.0
            } else {
                sum / diag
            };
        }
        x
    }

    /// Maximum a posteriori logistic calibration via damped Newton iterations.
    pub(crate) fn fit(labels: &[BudgetLabel], start: &BudgetAcceptance) -> BudgetAcceptance {
        let prior = Self::params_of(&BudgetAcceptance::PRIOR);
        let mut theta = Self::params_of(start);
        let mut current = Self::objective(labels, &theta, &prior);

        for _ in 0..Self::NEWTON_ITERS {
            let mut grad = [0.0; Self::PARAMS];
            let mut hess = [[0.0; Self::PARAMS]; Self::PARAMS];
            for label in labels {
                if !label.is_valid() {
                    continue;
                }
                let b = 1 + label.position;
                let f = f64::from(label.feature);
                let z = theta[0] * f + theta[b];
                if label.is_capped(z) {
                    continue;
                }
                let p = 1.0 / (1.0 + (-z).exp());
                let residual = p - if label.accepted { 1.0 } else { 0.0 };
                let weight = p * (1.0 - p);
                grad[0] += residual * f;
                grad[b] += residual;
                hess[0][0] += weight * f * f;
                hess[0][b] += weight * f;
                hess[b][0] += weight * f;
                hess[b][b] += weight;
            }
            for i in 0..Self::PARAMS {
                let prec = Self::prior_precision(i);
                grad[i] += prec * (theta[i] - prior[i]);
                hess[i][i] += prec;
            }
            let step = Self::solve(hess, grad);
            let mut next = [0.0; Self::PARAMS];
            let mut next_val = current;
            let mut scale = 1.0;
            let mut moved = false;
            for _ in 0..30 {
                for i in 0..Self::PARAMS {
                    next[i] = theta[i] - scale * step[i];
                }
                next_val = Self::objective(labels, &next, &prior);
                if next_val <= current {
                    moved = true;
                    break;
                }
                scale *= 0.5;
            }
            if !moved {
                break;
            }
            let mut largest = 0.0f64;
            for i in 0..Self::PARAMS {
                largest = largest.max((next[i] - theta[i]).abs());
            }
            theta = next;
            current = next_val;
            if largest < 1.0e-9 {
                break;
            }
        }

        let mut intercepts = [0.0f32; DRAFT_SLOTS];
        for (i, slot) in intercepts.iter_mut().enumerate() {
            *slot = theta[1 + i] as f32;
        }
        BudgetAcceptance {
            slope: theta[0] as f32,
            intercepts,
        }
    }
}
