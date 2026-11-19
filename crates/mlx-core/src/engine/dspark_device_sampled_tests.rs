//! Device-drawn DFlash2 proposals and their device acceptance: the selector
//! draw follows the rows it reports, the emitted tokens follow the target
//! distribution (the speculative law), and a seed repeats its tokens.

use napi::Result;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::array::{DType, MxArray};
use crate::engine::backend::DsparkProposal;
use crate::engine::dspark_turn::{accept_device_sampled, accept_dspark_proposal};
use crate::engine::params::ChatParams;
use crate::models::qwen3_5::dflash2::{SelectorPath, device_sampled_path};

pub(super) const V: usize = 64;
const K: usize = 16;
pub(super) const L: usize = 7;
const DRAFT_TEMPERATURE: f64 = 0.6;

fn metal() -> bool {
    unsafe { mlx_sys::mlx_metal_is_available() }
}

struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A fixed draft (candidates `[1,L,K]`, score tables `[L,K,K]`) and target
/// (`[1,L+1,V]` logits) pair. Each row's target leans toward half of that
/// position's candidates, so drafts are accepted, refused and corrected.
pub(super) struct Fixture {
    candidates: MxArray,
    scores: MxArray,
    pub(super) logits: MxArray,
    candidate_ids: Vec<i32>,
    score_values: Vec<f32>,
    pub(super) params: ChatParams,
    /// The product's target distribution of each verify row.
    pub(super) targets: Vec<Vec<f64>>,
}

fn app_params() -> ChatParams {
    crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
        temperature: Some(DRAFT_TEMPERATURE),
        top_p: Some(0.95),
        top_k: Some(20),
        min_p: Some(0.0),
        ..Default::default()
    })
}

pub(super) fn fixture(seed: u64) -> Result<Fixture> {
    let mut rng = Xorshift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut candidate_ids = Vec::with_capacity(L * K);
    for _ in 0..L {
        let mut order: Vec<(f64, i32)> = (0..V as i32).map(|id| (rng.next(), id)).collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        candidate_ids.extend(order[..K].iter().map(|&(_, id)| id));
    }
    let score_values: Vec<f32> = (0..L * K * K)
        .map(|_| (3.0 * rng.next() - 1.5) as f32)
        .collect();
    let mut logit_values = Vec::with_capacity((L + 1) * V);
    for row in 0..=L {
        for id in 0..V as i32 {
            let favored = row < L && candidate_ids[row * K..row * K + K / 2].contains(&id);
            let lean = if favored { 3.5 } else { 0.0 };
            logit_values.push((1.5 * rng.next() + lean) as f32);
        }
    }
    let params = app_params();
    let logits = MxArray::from_float32(&logit_values, &[1, (L + 1) as i64, V as i64])?;
    let mut targets = Vec::with_capacity(L + 1);
    for row in 0..=L {
        let one = MxArray::from_float32(&logit_values[row * V..(row + 1) * V], &[V as i64])?;
        let p = crate::sampling::sampling_distribution(&one, params.sampling_config)?
            .astype(DType::Float32)?;
        p.eval();
        targets.push(p.to_float32()?.iter().map(|&v| f64::from(v)).collect());
    }
    Ok(Fixture {
        candidates: MxArray::from_int32(&candidate_ids, &[1, L as i64, K as i64])?,
        scores: MxArray::from_float32(&score_values, &[L as i64, K as i64, K as i64])?,
        logits,
        candidate_ids,
        score_values,
        params,
        targets,
    })
}

pub(super) fn device_proposal(f: &Fixture, rng: &mut StdRng) -> Result<DsparkProposal> {
    let Some(SelectorPath::Sampled { ids, rows }) =
        device_sampled_path(&f.candidates, &f.scores, V, DRAFT_TEMPERATURE, rng)
    else {
        return Err(napi::Error::from_reason("the device walk declined"));
    };
    Ok(DsparkProposal {
        draft_ids: Vec::new(),
        device_draft_ids: Some(ids),
        device_draft_sparse: Some(rows),
        draft_dists: Vec::new(),
        draft_sparse_dists: Vec::new(),
        keep_probabilities: None,
    })
}

/// One cycle: the emitted tokens (accepted drafts, then the boundary) and
/// the drafts.
fn cycle(f: &Fixture, rng: &mut StdRng, host_accept: bool) -> Result<(Vec<usize>, Vec<i32>)> {
    let mut proposal = device_proposal(f, rng)?;
    let (k, boundary) = if host_accept {
        accept_dspark_proposal(&f.logits, &mut proposal, &[], &f.params, rng, None)?
    } else {
        accept_device_sampled(&f.logits, &mut proposal, &f.params, rng)?
    };
    let mut emitted: Vec<usize> = proposal.draft_ids[..k]
        .iter()
        .map(|&id| id as usize)
        .collect();
    emitted.push(boundary as usize);
    Ok((emitted, proposal.draft_ids))
}

fn ln_gamma(x: f64) -> f64 {
    let g = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = 0.999_999_999_999_809_9;
    for (i, gi) in g.iter().enumerate() {
        a += gi / (x + i as f64 + 1.0);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Upper regularized gamma Q(s, x).
pub(super) fn gamma_q(s: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < s + 1.0 {
        let (mut sum, mut term, mut n) = (1.0 / s, 1.0 / s, s);
        for _ in 0..10_000 {
            n += 1.0;
            term *= x / n;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        1.0 - (sum.ln() - x + s * x.ln() - ln_gamma(s)).exp()
    } else {
        let (mut b, mut c, mut d) = (x + 1.0 - s, 1e300, 1.0 / (x + 1.0 - s));
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - s);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = b + an / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        (-x + s * x.ln() - ln_gamma(s)).exp() * h
    }
}

/// Chi-square of counts against probabilities: bins expecting >= 5 apart,
/// the rest pooled; a count where the law has no mass fails outright.
pub(super) fn chi_square(counts: &[u64], probs: &[f64]) -> (f64, usize) {
    let n = counts.iter().sum::<u64>() as f64;
    let (mut stat, mut bins, mut pool_o, mut pool_e) = (0.0, 0usize, 0.0, 0.0);
    for (&c, &p) in counts.iter().zip(probs) {
        let e = p * n;
        if p <= 0.0 && c > 0 {
            stat += 1e9;
        } else if e >= 5.0 {
            stat += (c as f64 - e).powi(2) / e;
            bins += 1;
        } else {
            pool_o += c as f64;
            pool_e += e;
        }
    }
    if pool_e >= 5.0 {
        stat += (pool_o - pool_e).powi(2) / pool_e;
        bins += 1;
    }
    (stat, bins.saturating_sub(1).max(1))
}

/// Pooled p-value of the emitted tokens per verify row against the target,
/// and of the first draft against the selector row it is drawn from.
fn law_p(f: &Fixture, draws: usize, seed: u64, host_accept: bool) -> Result<(f64, Vec<String>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut counts = vec![vec![0u64; V]; L + 1];
    let mut first_draft = vec![0u64; V];
    let mut emitted_total = 0usize;
    while emitted_total < draws {
        let (emitted, drafts) = cycle(f, &mut rng, host_accept)?;
        first_draft[drafts[0] as usize] += 1;
        for (row, &token) in emitted.iter().enumerate() {
            counts[row][token] += 1;
        }
        emitted_total += emitted.len();
    }
    let mut lines = Vec::new();
    let (mut stat, mut dof) = (0.0, 0usize);
    for (row, (c, p)) in counts.iter().zip(&f.targets).enumerate() {
        let n: u64 = c.iter().sum();
        if n < 200 {
            continue;
        }
        let (s, d) = chi_square(c, p);
        lines.push(format!(
            "row {row}: n {n} chi2 {s:.1} dof {d} p {:.3e}",
            gamma_q(d as f64 / 2.0, s / 2.0)
        ));
        stat += s;
        dof += d;
    }
    let q0 = selector_probs(&f.score_values[..K], DRAFT_TEMPERATURE);
    let mut q_vocab = vec![0.0; V];
    for (j, &id) in f.candidate_ids[..K].iter().enumerate() {
        q_vocab[id as usize] = q0[j];
    }
    let (s, d) = chi_square(&first_draft, &q_vocab);
    lines.push(format!(
        "draft 0: chi2 {s:.1} dof {d} p {:.3e}",
        gamma_q(d as f64 / 2.0, s / 2.0)
    ));
    stat += s;
    dof += d;
    Ok((gamma_q(dof as f64 / 2.0, stat / 2.0), lines))
}

/// The host selector's softmax of one score row (`normalized_selector_probs`).
fn selector_probs(row: &[f32], temperature: f64) -> Vec<f64> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f64> = row
        .iter()
        .map(|&s| ((f64::from(s) - f64::from(max)) / temperature).exp())
        .collect();
    let total: f64 = w.iter().sum();
    w.iter().map(|x| x / total).collect()
}

/// 200,000 emitted tokens through the product's device draw and device
/// acceptance: per verify row they follow the target distribution, and the
/// first draft follows its selector row. `MLX_TEST_SAMPLED_DRAWS` sets the count.
#[test]
fn device_sampled_acceptance_keeps_the_target_law() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    let f = fixture(0x51)?;
    let draws = std::env::var("MLX_TEST_SAMPLED_DRAWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    let started = std::time::Instant::now();
    let (p, lines) = law_p(&f, draws, 0xA11CE, false)?;
    eprintln!(
        "sampled law device draws {draws} pooled p {p:.3e} in {:?}",
        started.elapsed()
    );
    for line in &lines {
        eprintln!("sampled law   {line}");
    }
    assert!(
        p >= 1e-6,
        "device sampled acceptance breaks the law: p {p:.3e}"
    );
    Ok(())
}

/// The host acceptance of the same device-drawn proposals (the route a
/// thinking cut takes) keeps the law too.
#[test]
fn host_acceptance_of_device_drafts_keeps_the_target_law() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    let f = fixture(0x52)?;
    let (p, lines) = law_p(&f, 40_000, 0xB0B, true)?;
    eprintln!("sampled law host draws 40000 pooled p {p:.3e}");
    for line in &lines {
        eprintln!("sampled law   {line}");
    }
    assert!(
        p >= 1e-6,
        "host acceptance of device drafts breaks the law: p {p:.3e}"
    );
    Ok(())
}

/// The rows the device walk reports are the host selector's softmax of the
/// predecessor's score row, and each draft has positive probability there.
#[test]
fn device_walk_reports_the_rows_it_drew_from() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    let f = fixture(0x53)?;
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..64 {
        let mut proposal = device_proposal(&f, &mut rng)?;
        let rows = proposal.device_draft_sparse.take().expect("rows");
        let probs = rows.probs.to_float32()?.to_vec();
        proposal.materialize_draft_ids()?;
        let mut predecessor = 0usize;
        for (position, &draft) in proposal.draft_ids.iter().enumerate() {
            let start = (position * K + predecessor) * K;
            let want = selector_probs(&f.score_values[start..start + K], DRAFT_TEMPERATURE);
            let got = &probs[position * K..(position + 1) * K];
            for (g, w) in got.iter().zip(&want) {
                assert!(
                    (f64::from(*g) - w).abs() < 1e-6,
                    "position {position}: q {g} vs {w}"
                );
            }
            let slot = f.candidate_ids[position * K..(position + 1) * K]
                .iter()
                .position(|&id| id == draft)
                .expect("the draft is a candidate");
            assert!(got[slot] > 0.0, "a zero-probability candidate was drawn");
            predecessor = slot;
        }
    }
    Ok(())
}

/// One seed gives one token stream, run after run; another seed another.
#[test]
fn a_seed_repeats_its_device_sampled_tokens() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    let f = fixture(0x54)?;
    let run = |seed: u64| -> Result<Vec<usize>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::new();
        for _ in 0..400 {
            out.extend(cycle(&f, &mut rng, false)?.0);
        }
        Ok(out)
    };
    let a = run(9)?;
    assert_eq!(a, run(9)?, "same seed, other tokens");
    assert_ne!(a, run(10)?, "the stream does not follow the seed");
    Ok(())
}

/// A NaN or +inf selector score fails the device walk, as it fails the host
/// walk; a -inf score is a candidate with no mass.
#[test]
fn a_nan_or_inf_selector_score_fails_the_device_walk() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    for (name, bad) in [
        ("NaN", f32::NAN),
        ("+inf", f32::INFINITY),
        ("-inf", f32::NEG_INFINITY),
    ] {
        let mut f = fixture(0x55)?;
        f.score_values[2] = bad;
        f.scores = MxArray::from_float32(&f.score_values, &[L as i64, K as i64, K as i64])?;
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..64 {
            let mut proposal = device_proposal(&f, &mut rng)?;
            let result = proposal.materialize_draft_ids();
            if bad == f32::NEG_INFINITY {
                result?;
                assert_ne!(proposal.draft_ids[0], f.candidate_ids[2], "{name}: drawn");
            } else {
                assert!(
                    result.is_err(),
                    "{name}: the device walk drew {:?}",
                    proposal.draft_ids
                );
            }
        }
    }
    Ok(())
}
