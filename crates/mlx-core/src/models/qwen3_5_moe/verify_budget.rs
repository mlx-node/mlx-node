// Single-lane port of Splash runtime/model/VerifyBudget.{hpp,cpp} at
// 1482f6bbc2abf12375332725302019594150b6cb (Apache-2.0).
// Keep the source attribution in THIRD_PARTY_NOTICES when integrating.
// CPU policy only: the GPU consumes parameters() and selects the live prefix.

pub(crate) const POSITIONS: usize = 7;
pub(crate) const BLOCK_ROWS: usize = 8;
const REFIT_CYCLES: u64 = 64;
const LABELS: usize = 4096;
const EXPLORE_CYCLES: u64 = 32;
const DECAY: f64 = 0.97;
const PRIOR_CYCLES: f64 = 5.0;
const PARAMETERS: usize = POSITIONS + 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Acceptance {
    pub slope: f32,
    pub intercepts: [f32; POSITIONS],
}

pub(crate) const PRIOR: Acceptance = Acceptance {
    slope: 0.7,
    intercepts: [0.0, -0.35, -0.8, -0.75, -0.75, -0.85, -0.45],
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct BudgetParameters {
    pub acceptance: Acceptance,
    /// Index zero is unused; index r gives the cost of r live target rows.
    pub costs: [f32; BLOCK_ROWS + 1],
    pub every_row: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CycleObservation {
    pub confidence: [f32; POSITIONS],
    pub verify_rows: u32,
    /// Raw prefix matches, before clamping retained tokens. A valid final
    /// row can label one more proposal than the retained prefix includes.
    pub accepted: u32,
    /// False for replaced/forced thinking proposals or other non-draft ids.
    pub labels: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Label {
    position: usize,
    feature: f32,
    accepted: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct Cost {
    weight: f64,
    rows: f64,
    rows_squared: f64,
    time: f64,
    rows_time: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct VerifyBudget {
    acceptance: Acceptance,
    labels: Vec<Label>,
    next: usize,
    cycles: u64,
    // Constrained and unconstrained cycles have distinct wall-time models.
    tables: [Cost; 2],
    table_cycles: [u64; 2],
}

impl Default for VerifyBudget {
    fn default() -> Self {
        Self {
            acceptance: PRIOR,
            labels: Vec::new(),
            next: 0,
            cycles: 0,
            tables: [Cost::default(); 2],
            table_cycles: [0; 2],
        }
    }
}

impl VerifyBudget {
    pub(crate) fn parameters(&self, constrained: bool, adaptive: bool) -> BudgetParameters {
        BudgetParameters {
            acceptance: self.acceptance,
            costs: self.costs(constrained),
            every_row: !adaptive || self.explores(constrained),
        }
    }

    fn line(&self, constrained: bool) -> Option<(f64, f64)> {
        let cost = self.tables[usize::from(constrained)];
        if cost.weight <= 0.0 {
            return None;
        }
        let mean_rows = cost.rows / cost.weight;
        let mean_time = cost.time / cost.weight;
        let spread = (cost.rows_squared / cost.weight - mean_rows * mean_rows).max(0.0);
        let covariance = cost.rows_time / cost.weight - mean_rows * mean_time;
        let seed_slope = (seed(BLOCK_ROWS) - seed(1)) / (BLOCK_ROWS - 1) as f64;
        let prior_slope = seed_slope * mean_time / seed(mean_rows.round() as usize);
        let slope = ((cost.weight * covariance + PRIOR_CYCLES * prior_slope)
            / (cost.weight * spread + PRIOR_CYCLES))
            .max(0.0);
        Some((mean_time - slope * mean_rows, slope))
    }

    pub(crate) fn costs(&self, constrained: bool) -> [f32; BLOCK_ROWS + 1] {
        let mut result = [0.0; BLOCK_ROWS + 1];
        let fit = self.line(constrained);
        let cost = self.tables[usize::from(constrained)];
        for (rows, value) in result.iter_mut().enumerate().skip(1) {
            *value = match fit {
                None => seed(rows) as f32,
                Some((intercept, slope)) => {
                    (intercept + slope * rows as f64).max(0.05 * cost.time / cost.weight) as f32
                }
            };
        }
        result
    }

    pub(crate) fn explores(&self, constrained: bool) -> bool {
        self.table_cycles[usize::from(constrained)] % EXPLORE_CYCLES == EXPLORE_CYCLES - 1
    }

    /// Call only after the whole cycle successfully commits. Wall time must
    /// describe the same draft + verification work whose row budget is chosen.
    pub(crate) fn observe(
        &mut self,
        lane: CycleObservation,
        constrained: bool,
        mut milliseconds: f64,
    ) -> Result<(), &'static str> {
        if lane.verify_rows == 0
            || lane.verify_rows > BLOCK_ROWS as u32
            || lane.accepted > POSITIONS as u32
        {
            return Err("invalid verify budget lane");
        }
        let index = usize::from(constrained);
        let rows = f64::from(lane.verify_rows);
        self.table_cycles[index] += 1;
        if milliseconds.is_finite() && milliseconds > 0.0 {
            if let Some((intercept, slope)) = self.line(constrained) {
                let expected = (intercept + slope * rows).max(1.0e-3);
                milliseconds = milliseconds.clamp(0.5 * expected, 2.0 * expected);
            }
            let cost = &mut self.tables[index];
            cost.weight = DECAY * cost.weight + 1.0;
            cost.rows = DECAY * cost.rows + rows;
            cost.rows_squared = DECAY * cost.rows_squared + rows * rows;
            cost.time = DECAY * cost.time + milliseconds;
            cost.rows_time = DECAY * cost.rows_time + rows * milliseconds;
        }
        if lane.labels {
            let last = lane
                .accepted
                .min(lane.verify_rows - 1)
                .min(POSITIONS as u32 - 1) as usize;
            for position in 0..=last {
                let confidence = lane.confidence[position];
                if !(0.0..=1.0).contains(&confidence) {
                    continue;
                }
                let label = Label {
                    position,
                    feature: logit(f64::from(confidence)) as f32,
                    accepted: position < lane.accepted as usize,
                };
                if self.labels.len() < LABELS {
                    self.labels.push(label);
                } else {
                    self.labels[self.next] = label;
                    self.next = (self.next + 1) % LABELS;
                }
            }
        }
        self.cycles += 1;
        if self.cycles.is_multiple_of(REFIT_CYCLES) && !self.labels.is_empty() {
            self.acceptance = fit(&self.ordered_labels(), self.acceptance);
        }
        Ok(())
    }

    fn ordered_labels(&self) -> Vec<Label> {
        self.labels[self.next..]
            .iter()
            .chain(&self.labels[..self.next])
            .copied()
            .collect()
    }
}

fn seed(rows: usize) -> f64 {
    assert!((1..=BLOCK_ROWS).contains(&rows));
    let knots = [1.0, 2.0, 4.0, 8.0];
    let shape = [0.60, 0.69, 0.85, 1.0];
    let mut upper = 1;
    while upper < 3 && rows as f64 > knots[upper] {
        upper += 1;
    }
    let t = (rows as f64 - knots[upper - 1]) / (knots[upper] - knots[upper - 1]);
    shape[upper - 1] + t * (shape[upper] - shape[upper - 1])
}

fn logit(p: f64) -> f64 {
    let p = p.clamp(1.0e-6, 1.0 - 1.0e-6);
    (p / (1.0 - p)).ln()
}

type Parameters = [f64; PARAMETERS];
fn prior_precision(parameter: usize) -> f64 {
    if parameter == 0 { 50.0 } else { 4.0 }
}

fn parameters_of(acceptance: Acceptance) -> Parameters {
    let mut result = [0.0; PARAMETERS];
    result[0] = f64::from(acceptance.slope);
    for position in 0..POSITIONS {
        result[1 + position] = f64::from(acceptance.intercepts[position]);
    }
    result
}

fn objective(labels: &[Label], theta: &Parameters, prior: &Parameters) -> f64 {
    let mut value = 0.0;
    for label in labels {
        let z = theta[0] * f64::from(label.feature) + theta[1 + label.position];
        let margin = if label.accepted { z } else { -z };
        value += if margin > 0.0 {
            (-margin).exp().ln_1p()
        } else {
            -margin + margin.exp().ln_1p()
        };
    }
    for i in 0..PARAMETERS {
        value += 0.5 * prior_precision(i) * (theta[i] - prior[i]) * (theta[i] - prior[i]);
    }
    value
}

fn solve(mut h: [Parameters; PARAMETERS], mut g: Parameters) -> Parameters {
    for column in 0..PARAMETERS {
        let mut pivot = column;
        for row in column + 1..PARAMETERS {
            if h[row][column].abs() > h[pivot][column].abs() {
                pivot = row;
            }
        }
        h.swap(column, pivot);
        g.swap(column, pivot);
        for row in column + 1..PARAMETERS {
            let factor = h[row][column] / h[column][column];
            let (pivot_rows, remaining_rows) = h.split_at_mut(row);
            let pivot = &pivot_rows[column];
            let current = &mut remaining_rows[0];
            for (value, pivot) in current[column..].iter_mut().zip(&pivot[column..]) {
                *value -= factor * pivot;
            }
            g[row] -= factor * g[column];
        }
    }
    let mut result = [0.0; PARAMETERS];
    for row in (0..PARAMETERS).rev() {
        let mut sum = g[row];
        for k in row + 1..PARAMETERS {
            sum -= h[row][k] * result[k];
        }
        result[row] = sum / h[row][row];
    }
    result
}

fn fit(labels: &[Label], start: Acceptance) -> Acceptance {
    let prior = parameters_of(PRIOR);
    let mut theta = parameters_of(start);
    let mut current = objective(labels, &theta, &prior);
    for _ in 0..16 {
        let mut gradient = [0.0; PARAMETERS];
        let mut hessian = [[0.0; PARAMETERS]; PARAMETERS];
        for label in labels {
            let b = 1 + label.position;
            let feature = f64::from(label.feature);
            let p = 1.0 / (1.0 + (-(theta[0] * feature + theta[b])).exp());
            let residual = p - f64::from(label.accepted);
            let weight = p * (1.0 - p);
            gradient[0] += residual * feature;
            gradient[b] += residual;
            hessian[0][0] += weight * feature * feature;
            hessian[0][b] += weight * feature;
            hessian[b][0] += weight * feature;
            hessian[b][b] += weight;
        }
        for i in 0..PARAMETERS {
            gradient[i] += prior_precision(i) * (theta[i] - prior[i]);
            hessian[i][i] += prior_precision(i);
        }
        let step = solve(hessian, gradient);
        let mut next = [0.0; PARAMETERS];
        let mut value = current;
        let mut scale = 1.0;
        let mut moved = false;
        for _ in 0..30 {
            for i in 0..PARAMETERS {
                next[i] = theta[i] - scale * step[i];
            }
            value = objective(labels, &next, &prior);
            if value <= current {
                moved = true;
                break;
            }
            scale *= 0.5;
        }
        if !moved {
            break;
        }
        let mut largest: f64 = 0.0;
        for i in 0..PARAMETERS {
            largest = largest.max((next[i] - theta[i]).abs());
        }
        theta = next;
        current = value;
        if largest < 1.0e-9 {
            break;
        }
    }
    let mut result = Acceptance {
        slope: theta[0] as f32,
        intercepts: [0.0; POSITIONS],
    };
    for position in 0..POSITIONS {
        result.intercepts[position] = theta[1 + position] as f32;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cycle(rows: u32, accepted: u32) -> CycleObservation {
        CycleObservation {
            confidence: [0.7; POSITIONS],
            verify_rows: rows,
            accepted,
            labels: true,
        }
    }

    #[test]
    fn source_prior_seed_and_exploration_are_preserved() {
        let mut policy = VerifyBudget::default();
        let parameters = policy.parameters(false, true);
        assert_eq!(parameters.acceptance, PRIOR);
        assert_eq!(
            parameters.costs,
            [0.0, 0.60, 0.69, 0.77, 0.85, 0.8875, 0.925, 0.9625, 1.0]
        );
        assert!(!parameters.every_row);
        assert!(policy.parameters(false, false).every_row);
        for index in 0..64 {
            assert_eq!(policy.explores(false), index % 32 == 31);
            assert!(!policy.explores(true));
            policy.observe(cycle(8, 2), false, 20.0).unwrap();
        }
    }

    #[test]
    fn last_valid_target_row_is_an_extra_label_not_a_retained_draft() {
        let mut policy = VerifyBudget::default();
        policy.observe(cycle(1, 1), false, 10.0).unwrap();
        assert_eq!(policy.labels.len(), 1);
        assert_eq!(policy.labels[0].position, 0);
        assert!(policy.labels[0].accepted);
        policy.observe(cycle(3, 7), false, 10.0).unwrap();
        assert_eq!(policy.labels.len(), 4);
        assert_eq!(policy.labels[3].position, 2);
        policy.observe(cycle(8, 0), false, 10.0).unwrap();
        assert_eq!(policy.labels.len(), 5);
        assert!(!policy.labels[4].accepted);
    }

    #[test]
    fn forced_and_nonfinite_confidence_do_not_train_acceptance() {
        let mut policy = VerifyBudget::default();
        let mut observation = cycle(8, 7);
        observation.labels = false;
        policy.observe(observation, false, 10.0).unwrap();
        assert!(policy.labels.is_empty());
        observation.labels = true;
        observation.confidence = [f32::NAN, f32::INFINITY, -0.1, 1.1, 0.0, 1.0, 0.5];
        policy.observe(observation, false, f64::NAN).unwrap();
        assert_eq!(
            policy.labels.iter().map(|x| x.position).collect::<Vec<_>>(),
            [4, 5, 6]
        );
        assert!(policy.labels.iter().all(|x| x.feature.is_finite()));
        assert_eq!(policy.tables[0].weight, 1.0);
    }

    #[test]
    fn cost_tables_are_separate_positive_and_stall_bounded() {
        let mut policy = VerifyBudget::default();
        policy.observe(cycle(8, 7), false, 20.0).unwrap();
        let before = policy.costs(false);
        assert_eq!(before[8], 20.0);
        assert_eq!(policy.costs(true)[8], 1.0);
        policy.observe(cycle(8, 7), false, 1.0e9).unwrap();
        assert!((policy.tables[0].time - (0.97 * 20.0 + 40.0)).abs() < 1.0e-12);
        let costs = policy.costs(false);
        assert!(costs[1..].iter().all(|x| x.is_finite() && *x > 0.0));
        assert!(costs[1..].windows(2).all(|x| x[0] <= x[1]));
    }

    #[test]
    fn fit_retains_prior_without_labels_and_learns_rejections() {
        assert_eq!(fit(&[], PRIOR), PRIOR);
        let mut policy = VerifyBudget::default();
        for _ in 0..63 {
            policy.observe(cycle(8, 0), false, 10.0).unwrap();
        }
        assert_eq!(policy.acceptance, PRIOR);
        policy.observe(cycle(8, 0), false, 10.0).unwrap();
        assert!(policy.acceptance.intercepts[0] < PRIOR.intercepts[0]);
        assert_eq!(policy.acceptance.intercepts[1..], PRIOR.intercepts[1..]);
    }

    #[test]
    fn labels_retain_only_the_latest_4096_in_order() {
        let mut policy = VerifyBudget::default();
        // Every cycle has seven labels and a distinct first confidence.
        for index in 0..600 {
            let mut observation = cycle(8, 7);
            observation.confidence[0] = (index as f32 + 1.0) / 1000.0;
            policy.observe(observation, false, 10.0).unwrap();
        }
        let labels = policy.ordered_labels();
        assert_eq!(labels.len(), LABELS);
        assert_eq!(labels[0].position, (4200 - LABELS) % 7);
        assert_eq!(labels[LABELS - 7].feature, logit(f64::from(0.6_f32)) as f32);
        assert_eq!(labels[LABELS - 1].position, 6);
    }

    #[test]
    fn malformed_observation_does_not_mutate_policy() {
        let mut policy = VerifyBudget::default();
        for observation in [cycle(0, 0), cycle(9, 0), cycle(8, 8)] {
            assert!(policy.observe(observation, false, 1.0).is_err());
        }
        assert_eq!(policy.cycles, 0);
        assert!(policy.labels.is_empty());
    }

    #[test]
    fn source_cpp_golden_after_3000_mixed_cycles_and_label_ring_wrap() {
        // These bits were emitted by the unmodified Splash VerifyBudget.cpp
        // compiled with clang++ -std=c++20 -O2 -ffp-contract=off. This fixture
        // includes separate constrained timings, a clipped long stall,
        // forced-label exclusions, refits and a wrapped 4096-label ring.
        let mut policy = VerifyBudget::default();
        for i in 0..3000 {
            let mut observation = CycleObservation {
                verify_rows: (1 + i % 8) as u32,
                accepted: ((i * 3) % 8) as u32,
                labels: i % 7 != 0,
                confidence: [0.0; POSITIONS],
            };
            for j in 0..POSITIONS {
                observation.confidence[j] = ((i * 17 + j * 13) % 99 + 1) as f32 / 100.0;
            }
            let ms = if i == 55 {
                1.0e9
            } else {
                10.0 + 0.2 * f64::from(observation.verify_rows) + 0.3 * (i % 5) as f64
            };
            policy.observe(observation, i % 3 == 0, ms).unwrap();
        }
        assert_eq!(policy.acceptance.slope.to_bits(), 1014826058);
        assert_eq!(
            policy.acceptance.intercepts.map(f32::to_bits),
            [
                1072821919, 1071411176, 1068014245, 1079067444, 1059317619, 3176085071, 3202770534
            ]
        );
        assert_eq!(
            policy.costs(false).map(f32::to_bits),
            [
                0, 1093442207, 1093658283, 1093874360, 1094090436, 1094306512, 1094522589,
                1094738665, 1094954741
            ]
        );
        assert_eq!(
            policy.costs(true).map(f32::to_bits),
            [
                0, 1093355170, 1093595531, 1093835891, 1094076252, 1094316612, 1094556972,
                1094797333, 1095037693
            ]
        );
        assert_eq!(policy.labels.len(), 4096);
    }
}
