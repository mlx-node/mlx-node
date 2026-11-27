//! The device rows of a sampled DFlash2 cycle against the filter chain
//! (`sampling::sampling_distribution`): the tokens a row keeps, their
//! probabilities, the draws, and the turns the kernels do not take.

use std::time::Instant;

use napi::Result;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::array::{DType, MxArray};
use crate::engine::backend::DsparkProposal;
use crate::engine::dspark_device_sampled_tests as device_sampled;
use crate::engine::dspark_turn::{
    TEST_SAMPLED_ROWS, accept_dspark_proposal, sampled_rows_take, submit_sampled_rows,
};
use crate::engine::params::ChatParams;
use crate::sampling::{SamplingConfig, SparseDistribution, sampling_distribution};

const VOCAB: usize = 248_320;
/// f32 softmax of up to 256 kept terms: the chain itself is up to 13 ulp
/// from the f64 law on these rows.
const ULP_BOUND: u64 = 32;
const ROWS: usize = 8;

fn metal() -> bool {
    unsafe { mlx_sys::mlx_metal_is_available() }
}

struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, bound: usize) -> usize {
        ((self.next() * bound as f64) as usize).min(bound - 1)
    }
}

fn config(temperature: f64, top_k: i32, top_p: f64, min_p: f64) -> SamplingConfig {
    SamplingConfig {
        temperature: Some(temperature),
        top_k: Some(top_k),
        top_p: Some(top_p),
        min_p: Some(min_p),
    }
}

fn params(config: SamplingConfig) -> ChatParams {
    crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        min_p: config.min_p,
        ..Default::default()
    })
}

/// A bell of logits with `peaks` tokens far above it.
fn peaked(vocab: usize, peaks: usize, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift::new(seed);
    let mut row: Vec<f32> = (0..vocab)
        .map(|_| (2.0 * (rng.next() + rng.next() + rng.next() + rng.next() - 2.0)) as f32)
        .collect();
    for _ in 0..peaks {
        let token = rng.below(vocab);
        row[token] = (8.0 + 6.0 * rng.next()) as f32;
    }
    row
}

/// A bell of logits with twelve tokens above it, a quarter apart.
fn spread(vocab: usize, seed: u64) -> Vec<f32> {
    let mut row = peaked(vocab, 0, seed);
    let mut rng = Xorshift::new(seed ^ 0x55);
    let mut placed = Vec::new();
    while placed.len() < 12 {
        let token = rng.below(vocab);
        if !placed.contains(&token) {
            row[token] = 14.0 - 0.25 * placed.len() as f32;
            placed.push(token);
        }
    }
    row
}

fn chain(row: &[f32], config: SamplingConfig) -> Result<Vec<f32>> {
    let logits = MxArray::from_float32(row, &[row.len() as i64])?;
    let p = sampling_distribution(&logits, Some(config))?.astype(DType::Float32)?;
    p.eval();
    Ok(p.to_float32()?.to_vec())
}

/// The kernels' probability of each of `tokens` on `row`, and the row's kept
/// count: `ROWS` copies of the row per dispatch, each following one token.
fn device(row: &[f32], config: SamplingConfig, tokens: &[i32]) -> Result<(Vec<f32>, usize)> {
    let vocab = row.len();
    let mut stacked = Vec::with_capacity(ROWS * vocab);
    for _ in 0..ROWS {
        stacked.extend_from_slice(row);
    }
    let logits = MxArray::from_float32(&stacked, &[ROWS as i64, vocab as i64])?;
    let uniforms = MxArray::from_float32(&[0.5; ROWS], &[ROWS as i64])?;
    let candidates = MxArray::from_int32(&[-1; ROWS], &[ROWS as i64, 1])?;
    let probabilities = MxArray::from_float32(&[0.0; ROWS], &[ROWS as i64, 1])?;
    let mut out = Vec::with_capacity(tokens.len());
    let mut kept = None;
    for chunk in tokens.chunks(ROWS) {
        let mut ids = chunk.to_vec();
        ids.resize(ROWS, chunk[0]);
        let ids = MxArray::from_int32(&ids, &[ROWS as i64])?;
        let mut output = std::ptr::null_mut();
        let taken = unsafe {
            mlx_sys::mlx_dflash2_sampled_rows(
                logits.handle.0,
                uniforms.handle.0,
                ids.handle.0,
                candidates.handle.0,
                probabilities.handle.0,
                config.temperature.unwrap_or(1.0) as f32,
                config.top_k.unwrap_or(0),
                config.top_p.unwrap_or(1.0) as f32,
                config.min_p.unwrap_or(0.0) as f32,
                &mut output,
            )
        };
        assert_eq!(taken, 1, "the kernels take the fixed row");
        let records = MxArray::from_handle(output, "rows")?.to_float32()?;
        for slot in 0..ROWS {
            let count = records[slot * 4 + 3] as usize;
            assert_eq!(
                *kept.get_or_insert(count),
                count,
                "every copy of a row keeps the same count"
            );
        }
        out.extend((0..chunk.len()).map(|slot| records[slot * 4]));
    }
    Ok((out, kept.unwrap_or(0)))
}

fn ulps(a: f32, b: f32) -> u64 {
    (i64::from(a.to_bits()) - i64::from(b.to_bits())).unsigned_abs()
}

/// The f64 law on the tokens the chain keeps: softmax of `logit / T`.
fn law(row: &[f32], kept: &[usize], temperature: f64) -> Vec<f64> {
    let peak = kept
        .iter()
        .map(|&t| f64::from(row[t]))
        .fold(f64::MIN, f64::max);
    let weights: Vec<f64> = kept
        .iter()
        .map(|&t| ((f64::from(row[t]) - peak) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

struct Case {
    name: &'static str,
    row: Vec<f32>,
    config: SamplingConfig,
    /// Tokens to ask about beside the ones the chain keeps.
    probes: Vec<usize>,
}

fn tie_at_top_k(tied: usize) -> (Vec<f32>, Vec<usize>, Vec<usize>) {
    let mut rng = Xorshift::new(41);
    let mut row: Vec<f32> = (0..VOCAB).map(|_| (-4.0 * rng.next()) as f32).collect();
    let mut heads = Vec::new();
    while heads.len() < 10 {
        let token = rng.below(VOCAB);
        if !heads.contains(&token) {
            heads.push(token);
        }
    }
    let mut ties = Vec::new();
    while ties.len() < tied {
        let token = rng.below(VOCAB);
        if !heads.contains(&token) && !ties.contains(&token) {
            ties.push(token);
        }
    }
    for &token in &heads {
        row[token] = 6.0;
    }
    for &token in &ties {
        row[token] = 4.0;
    }
    ties.sort_unstable();
    (row, heads, ties)
}

fn tie_at_top_p() -> (Vec<f32>, [usize; 3]) {
    let mut row = vec![f32::NEG_INFINITY; VOCAB];
    let tied = [1_000, 50_000, 200_000];
    for (token, weight) in [
        (7usize, 0.4f64),
        (120_000, 0.2),
        (tied[0], 0.1),
        (tied[1], 0.1),
        (tied[2], 0.1),
        (33, 0.05),
        (248_319, 0.05),
    ] {
        row[token] = weight.ln() as f32;
    }
    (row, tied)
}

fn cases() -> Vec<Case> {
    let app = config(0.6, 20, 0.95, 0.0);
    let base = peaked(VOCAB, 30, 7);
    let mut holes = base.clone();
    let mut best = 0;
    for (token, value) in holes.iter_mut().enumerate() {
        if token % 3 == 0 {
            *value = f32::NEG_INFINITY;
        }
    }
    for (token, &value) in holes.iter().enumerate() {
        if value > holes[best] {
            best = token;
        }
    }
    holes[best] = f32::NEG_INFINITY;
    let (edge_k, _, edge_k_ties) = tie_at_top_k(30);
    let (wide_k, _, wide_k_ties) = tie_at_top_k(3_000);
    let (edge_p, edge_p_ties) = tie_at_top_p();
    vec![
        Case {
            name: "peaked",
            row: base.clone(),
            config: app,
            probes: vec![0, 1, VOCAB - 1],
        },
        Case {
            name: "spread",
            row: spread(VOCAB, 19),
            config: app,
            probes: vec![0, 2],
        },
        Case {
            name: "spread top_k 256",
            row: spread(VOCAB, 21),
            config: config(1.0, 256, 1.0, 0.0),
            probes: vec![0],
        },
        Case {
            name: "peaked top_p 1",
            row: base.clone(),
            config: config(0.6, 20, 1.0, 0.0),
            probes: vec![0],
        },
        Case {
            name: "peaked temperature 1.3 top_k 64",
            row: base.clone(),
            config: config(1.3, 64, 0.9, 0.0),
            probes: vec![0],
        },
        Case {
            name: "peaked min_p 0.05",
            row: base.clone(),
            config: config(0.6, 20, 0.95, 0.05),
            probes: vec![0],
        },
        Case {
            name: "peaked min_p 0.2 top_p 1",
            row: base.clone(),
            config: config(0.8, 40, 1.0, 0.2),
            probes: vec![0],
        },
        Case {
            name: "-inf holes",
            row: holes,
            config: app,
            probes: vec![0, 3, best],
        },
        Case {
            name: "flat top_p 1",
            row: vec![1.5; VOCAB],
            config: config(0.6, 20, 1.0, 0.0),
            probes: vec![19, 20, 21, VOCAB - 1],
        },
        Case {
            name: "flat top_p 0.95",
            row: vec![1.5; VOCAB],
            config: app,
            probes: vec![0, 19, 20],
        },
        Case {
            name: "tie at the top_k edge",
            row: edge_k,
            config: config(0.6, 20, 1.0, 0.0),
            probes: edge_k_ties,
        },
        Case {
            name: "wide tie at the top_k edge",
            row: wide_k,
            config: config(0.6, 20, 1.0, 0.0),
            probes: wide_k_ties[..40].to_vec(),
        },
        Case {
            name: "tie at the top_p edge",
            row: edge_p,
            config: config(0.6, 20, 0.75, 0.0),
            probes: edge_p_ties.to_vec(),
        },
        Case {
            name: "vocabulary 64",
            row: peaked(64, 4, 11),
            config: app,
            probes: (0..64).collect(),
        },
        Case {
            name: "vocabulary 16 under top_k",
            row: peaked(16, 2, 13),
            config: app,
            probes: (0..16).collect(),
        },
    ]
}

#[test]
fn device_rows_keep_the_tokens_the_filter_chain_keeps() -> Result<()> {
    if !metal()
        || crate::test_support::apple_gpu_family_absent(
            "device_rows_keep_the_tokens_the_filter_chain_keeps",
        )
    {
        return Ok(());
    }
    for case in cases() {
        let chain_p = chain(&case.row, case.config)?;
        let kept: Vec<usize> = (0..case.row.len()).filter(|&t| chain_p[t] > 0.0).collect();
        let empty = kept.is_empty();
        let mut tokens = kept.clone();
        tokens.extend(case.probes.iter().copied().filter(|t| !kept.contains(t)));
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let (device_p, device_kept) = device(&case.row, case.config, &ids)?;
        if empty {
            assert_eq!(
                device_kept, 0,
                "{}: the chain keeps no token, the kernels keep none",
                case.name
            );
            assert!(
                device_p.iter().all(|p| p.is_nan()),
                "{}: a row with no kept token records NaN",
                case.name
            );
            eprintln!(
                "ROWS {}: chain keeps no token (NaN row); kernels keep 0",
                case.name
            );
            continue;
        }
        let reference = law(&case.row, &kept, case.config.temperature.unwrap_or(1.0));
        let mut worst_chain = 0;
        let mut worst_law = 0;
        let mut chain_law = 0;
        for (slot, &token) in tokens.iter().enumerate() {
            assert_eq!(
                device_p[slot] > 0.0,
                chain_p[token] > 0.0,
                "{}: token {token} kept by the kernels {} by the chain {} (a tie keeps its \
                 smaller ids at the top_k edge and its larger ids at the top_p edge)",
                case.name,
                device_p[slot],
                chain_p[token]
            );
            worst_chain = worst_chain.max(ulps(device_p[slot], chain_p[token]));
            if let Some(index) = kept.iter().position(|&t| t == token) {
                worst_law = worst_law.max(ulps(device_p[slot], reference[index] as f32));
                chain_law = chain_law.max(ulps(chain_p[token], reference[index] as f32));
            }
        }
        assert_eq!(device_kept, kept.len(), "{}: kept count", case.name);
        eprintln!(
            "ROWS {}: kept {} ulp device-chain {worst_chain} device-law {worst_law} chain-law \
             {chain_law}",
            case.name,
            kept.len()
        );
        assert!(
            worst_law <= ULP_BOUND && worst_chain <= ULP_BOUND,
            "{}: a kept token is {worst_law} ulp from the f64 law, {worst_chain} from the chain",
            case.name
        );
    }
    Ok(())
}

#[test]
fn a_tie_keeps_smaller_ids_at_the_top_k_edge_and_larger_ids_at_the_top_p_edge() -> Result<()> {
    if !metal()
        || crate::test_support::apple_gpu_family_absent(
            "a_tie_keeps_smaller_ids_at_the_top_k_edge_and_larger_ids_at_the_top_p_edge",
        )
    {
        return Ok(());
    }
    let (row, _, ties) = tie_at_top_k(30);
    let ids: Vec<i32> = ties.iter().map(|&t| t as i32).collect();
    let (p, kept) = device(&row, config(0.6, 20, 1.0, 0.0), &ids)?;
    assert_eq!(kept, 20);
    for (slot, &token) in ties.iter().enumerate() {
        assert_eq!(
            p[slot] > 0.0,
            slot < 10,
            "top_k edge: tied token {token} (place {slot} by id)"
        );
    }
    let (row, tied) = tie_at_top_p();
    let ids: Vec<i32> = tied.iter().map(|&t| t as i32).collect();
    let (p, kept) = device(&row, config(0.6, 20, 0.75, 0.0), &ids)?;
    assert_eq!(kept, 4);
    assert_eq!(
        p.iter().map(|&p| p > 0.0).collect::<Vec<_>>(),
        [false, true, true],
        "top_p edge: the tie drops its smallest id"
    );
    Ok(())
}

fn host_proposal(
    vocab: usize,
    drafts: &[i32],
    candidates: &[(Vec<i32>, Vec<f64>)],
) -> Result<DsparkProposal> {
    Ok(DsparkProposal {
        draft_ids: drafts.to_vec(),
        device_draft_ids: None,
        device_draft_sparse: None,
        draft_dists: Vec::new(),
        draft_sparse_dists: candidates
            .iter()
            .map(|(ids, probs)| SparseDistribution::from_parts(ids.clone(), probs.clone(), vocab))
            .collect::<Result<Vec<_>>>()?,
        keep_probabilities: None,
    })
}

/// The first `count` tokens of `p` by (probability descending, id ascending).
fn top(p: &[f32], count: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..p.len()).filter(|&t| p[t] > 0.0).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    order.truncate(count);
    order
}

#[test]
fn device_rows_draw_by_inverse_cdf_and_correct_from_the_residual() -> Result<()> {
    if !metal()
        || crate::test_support::apple_gpu_family_absent(
            "device_rows_draw_by_inverse_cdf_and_correct_from_the_residual",
        )
    {
        return Ok(());
    }
    let vocab = 4_096;
    let cfg = config(0.6, 20, 0.95, 0.0);
    let p = params(cfg);
    let mut values = Vec::with_capacity(2 * vocab);
    values.extend(spread(vocab, 3));
    values.extend(spread(vocab, 5));
    let logits = MxArray::from_float32(&values, &[1, 2, vocab as i64])?;
    let first = chain(&values[..vocab], cfg)?;
    let last = chain(&values[vocab..], cfg)?;
    let kept_first = top(&first, 20);
    let kept_last = top(&last, 20);
    assert!(kept_first.len() >= 3 && kept_last.len() >= 2);
    let heaviest = kept_first[0];
    let outside = (0..vocab).find(|&t| first[t] == 0.0).unwrap_or(0);

    // A draft the target refuses: its token has no target mass. The draft
    // also holds more than the target's mass on the heaviest token, so the
    // residual has none there.
    let refused = host_proposal(
        vocab,
        &[outside as i32],
        &[(vec![outside as i32, heaviest as i32], vec![0.01, 0.99])],
    )?;
    let mut counts = vec![0u64; vocab];
    let mut rng = StdRng::seed_from_u64(17);
    let draws = 4_000;
    for _ in 0..draws {
        let mut proposal = host_proposal(
            vocab,
            &refused.draft_ids,
            &[(vec![outside as i32, heaviest as i32], vec![0.01, 0.99])],
        )?;
        let rows = submit_sampled_rows(&logits, &proposal, &p, &mut rng)?
            .ok_or_else(|| napi::Error::from_reason("the kernels declined"))?;
        let (accepted, token) = rows.accept(&mut proposal, None)?;
        assert_eq!(accepted, 0, "a draft with no target mass is refused");
        counts[token as usize] += 1;
    }
    let residual: Vec<f64> = (0..vocab)
        .map(|t| {
            let q = if t == heaviest { 0.99 } else { 0.0 };
            (f64::from(first[t]) - q).max(0.0)
        })
        .collect();
    let mass: f64 = residual.iter().sum();
    assert!(mass > 0.0);
    for token in 0..vocab {
        let expected = residual[token] / mass * draws as f64;
        if residual[token] == 0.0 {
            assert_eq!(counts[token], 0, "token {token} has no residual mass");
        } else if expected >= 200.0 {
            let sigma = (expected * (1.0 - residual[token] / mass)).sqrt();
            assert!(
                (counts[token] as f64 - expected).abs() <= 6.0 * sigma,
                "token {token}: {} draws, expected {expected:.1}",
                counts[token]
            );
        }
    }

    // A draft the target always keeps (the draft holds almost no mass on
    // it): the cycle ends on the last row's draw from the target itself.
    let mut counts = vec![0u64; vocab];
    for _ in 0..draws {
        let mut proposal = host_proposal(
            vocab,
            &[heaviest as i32],
            &[(vec![heaviest as i32, outside as i32], vec![1e-9, 1.0])],
        )?;
        let rows = submit_sampled_rows(&logits, &proposal, &p, &mut rng)?
            .ok_or_else(|| napi::Error::from_reason("the kernels declined"))?;
        let (accepted, token) = rows.accept(&mut proposal, None)?;
        assert_eq!(accepted, 1, "a draft the target outweighs is kept");
        counts[token as usize] += 1;
    }
    for token in 0..vocab {
        let expected = f64::from(last[token]) * draws as f64;
        if last[token] == 0.0 {
            assert_eq!(
                counts[token], 0,
                "token {token} is outside the last row's kept set"
            );
        } else if expected >= 200.0 {
            let sigma = (expected * (1.0 - f64::from(last[token]))).sqrt();
            assert!(
                (counts[token] as f64 - expected).abs() <= 6.0 * sigma,
                "last row token {token}: {} draws, expected {expected:.1}",
                counts[token]
            );
        }
    }

    // A seed repeats its cycle.
    let cycle = |seed: u64| -> Result<Vec<(usize, u32)>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..32)
            .map(|_| {
                let mut proposal = host_proposal(
                    vocab,
                    &[kept_first[1] as i32],
                    &[(
                        vec![kept_first[1] as i32, kept_first[2] as i32],
                        vec![0.7, 0.3],
                    )],
                )?;
                let rows = submit_sampled_rows(&logits, &proposal, &p, &mut rng)?
                    .ok_or_else(|| napi::Error::from_reason("the kernels declined"))?;
                rows.accept(&mut proposal, None)
            })
            .collect()
    };
    assert_eq!(cycle(5)?, cycle(5)?, "a seed repeats its tokens");
    Ok(())
}

#[test]
fn a_turn_outside_the_kernels_list_keeps_the_filter_chain() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    let vocab = 256;
    let row = peaked(vocab, 3, 9);
    let logits = MxArray::from_float32(&row, &[1, 1, vocab as i64])?;
    use rand::RngExt;
    let empty = || host_proposal(vocab, &[], &[]);
    let mut penalized = params(config(0.6, 20, 0.95, 0.0));
    penalized.repetition_penalty = 1.1;
    for (name, p) in [
        ("top_k 0", params(config(0.6, 0, 0.95, 0.0))),
        ("top_k 257", params(config(0.6, 257, 0.95, 0.0))),
        ("greedy", params(config(0.0, 20, 0.95, 0.0))),
        ("penalized", penalized),
    ] {
        assert!(!sampled_rows_take(&p), "{name}");
        let mut rng = StdRng::seed_from_u64(1);
        let before = TEST_SAMPLED_ROWS.with(|c| c.get());
        assert!(
            submit_sampled_rows(&logits, &empty()?, &p, &mut rng)?.is_none(),
            "{name}: not taken"
        );
        assert_eq!(
            TEST_SAMPLED_ROWS.with(|c| c.get()),
            before,
            "{name}: no kernel cycle"
        );
        assert_eq!(
            rng.random::<u64>(),
            StdRng::seed_from_u64(1).random::<u64>(),
            "{name}: the rng is untouched"
        );
    }
    if crate::test_support::apple_gpu_family_absent(
        "a_turn_outside_the_kernels_list_keeps_the_filter_chain: the taken turn",
    ) {
        return Ok(());
    }
    let taken = params(config(0.6, 20, 0.95, 0.0));
    assert!(sampled_rows_take(&taken));
    let mut rng = StdRng::seed_from_u64(1);
    let before = TEST_SAMPLED_ROWS.with(|c| c.get());
    let rows = submit_sampled_rows(&logits, &empty()?, &taken, &mut rng)?
        .ok_or_else(|| napi::Error::from_reason("the kernels declined"))?;
    let (accepted, token) = rows.accept(&mut empty()?, None)?;
    assert_eq!(accepted, 0);
    assert!(chain(&row, config(0.6, 20, 0.95, 0.0))?[token as usize] > 0.0);
    assert_eq!(TEST_SAMPLED_ROWS.with(|c| c.get()), before + 1);
    Ok(())
}

/// A GPU of no Apple family declines the device rows and its turn keeps the
/// filter chain; every Apple GPU family takes them.
#[test]
fn the_device_rows_take_a_turn_only_on_a_gpu_that_launches_them() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    use rand::RngExt;
    let p = params(config(0.6, 20, 0.95, 0.0));
    // SAFETY: nullary predicate that catches internally.
    let lacks = unsafe { mlx_sys::mlx_test_gpu_lacks_apple_family() };
    assert_eq!(
        sampled_rows_take(&p),
        !lacks,
        "no Apple GPU family: {lacks}"
    );
    if !lacks {
        return Ok(());
    }
    let vocab = 256;
    let logits = MxArray::from_float32(&peaked(vocab, 3, 9), &[1, 1, vocab as i64])?;
    let mut rng = StdRng::seed_from_u64(1);
    let before = TEST_SAMPLED_ROWS.with(|c| c.get());
    let proposal = host_proposal(vocab, &[], &[])?;
    assert!(submit_sampled_rows(&logits, &proposal, &p, &mut rng)?.is_none());
    assert_eq!(TEST_SAMPLED_ROWS.with(|c| c.get()), before);
    assert_eq!(
        rng.random::<u64>(),
        StdRng::seed_from_u64(1).random::<u64>(),
        "the rng is untouched"
    );
    Ok(())
}

/// The call a cycle's acceptance comes through.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Call {
    /// The whole turn: a device-drawn proposal, `[1, 1 + L, V]` logits.
    WholeTurn,
    /// A scheduler lane: host ids and sparse rows, the lane's packed rows.
    Lane,
}

/// A sampler defect the law test must see.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Planted {
    None,
    /// The rows run 25 % hotter than the turn's temperature.
    Hot,
    /// The rows keep the whole top_k (no nucleus).
    NoNucleus,
    /// Acceptance reads a uniform draft distribution, not the one the drafts
    /// were drawn from.
    WrongDraft,
}

/// One cycle through the device rows: the emitted tokens (accepted drafts,
/// then the boundary).
fn rows_cycle(
    f: &device_sampled::Fixture,
    rng: &mut StdRng,
    call: Call,
    planted: Planted,
) -> Result<Vec<usize>> {
    let mut proposal = device_sampled::device_proposal(f, rng)?;
    let logits = match call {
        Call::WholeTurn => {
            f.logits
                .reshape(&[1, (device_sampled::L + 1) as i64, device_sampled::V as i64])?
        }
        Call::Lane => {
            proposal.materialize_draft_ids()?;
            f.logits
                .reshape(&[(device_sampled::L + 1) as i64, 1, device_sampled::V as i64])?
        }
    };
    let app = f.params.sampling_config.unwrap_or_default();
    let p = match planted {
        Planted::Hot => params(config(0.75, 20, 0.95, 0.0)),
        Planted::NoNucleus => params(config(app.temperature.unwrap_or(1.0), 20, 1.0, 0.0)),
        _ => params(app),
    };
    if planted == Planted::WrongDraft {
        proposal.materialize_draft_ids()?;
        proposal.draft_sparse_dists = proposal
            .draft_sparse_dists
            .iter()
            .map(|row| {
                let (ids, probs) = row.candidates();
                SparseDistribution::from_parts(
                    ids.to_vec(),
                    probs.iter().map(|_| 1.0).collect(),
                    device_sampled::V,
                )
            })
            .collect::<Result<Vec<_>>>()?;
    }
    let rows = submit_sampled_rows(&logits, &proposal, &p, rng)?
        .ok_or_else(|| napi::Error::from_reason("the kernels declined"))?;
    let (accepted, boundary) = rows.accept(&mut proposal, None)?;
    let mut emitted: Vec<usize> = proposal.draft_ids[..accepted]
        .iter()
        .map(|&id| id as usize)
        .collect();
    emitted.push(boundary as usize);
    Ok(emitted)
}

/// Pooled p-value and per-row hit counts over either a draw budget or a cycle
/// budget against the filter chain's distribution of each verify row.
fn rows_law_with(
    draws: usize,
    cycles_limit: Option<usize>,
    seed: u64,
    call: Call,
    planted: Planted,
    align_scores: bool,
) -> Result<(f64, Vec<u64>, Vec<String>)> {
    let f = device_sampled::fixture_with(seed, device_sampled::L, align_scores)?;
    let mut rng = StdRng::seed_from_u64(seed ^ 0xA11CE);
    let mut counts = vec![vec![0u64; device_sampled::V]; device_sampled::L + 1];
    let (mut total, mut cycles) = (0usize, 0usize);
    while match cycles_limit {
        Some(limit) => cycles < limit,
        None => total < draws,
    } {
        let emitted = rows_cycle(&f, &mut rng, call, planted)?;
        for (row, &token) in emitted.iter().enumerate() {
            counts[row][token] += 1;
        }
        total += emitted.len();
        cycles += 1;
    }
    let mut lines = Vec::new();
    let mut row_hits = Vec::with_capacity(device_sampled::L + 1);
    let (mut stat, mut dof) = (0.0, 0usize);
    for (row, (c, p)) in counts.iter().zip(&f.targets).enumerate() {
        let n: u64 = c.iter().sum();
        row_hits.push(n);
        if n < 200 {
            continue;
        }
        let (s, d) = device_sampled::chi_square(c, p);
        lines.push(format!(
            "row {row}: n {n} chi2 {s:.1} dof {d} p {:.3e}",
            device_sampled::gamma_q(d as f64 / 2.0, s / 2.0)
        ));
        stat += s;
        dof += d;
    }
    Ok((
        device_sampled::gamma_q(dof as f64 / 2.0, stat / 2.0),
        row_hits,
        lines,
    ))
}

fn rows_law(draws: usize, seed: u64, call: Call, planted: Planted) -> Result<(f64, Vec<String>)> {
    let (p, _, lines) = rows_law_with(draws, None, seed, call, planted, false)?;
    Ok((p, lines))
}

/// 8,000 emitted tokens through each call keep the target law, row by
/// row; a full-width high-acceptance arm verifies every verify row 0..=L
/// (including late rows 4..=6 and bonus row 7) with >= 200 hits each; and
/// three planted sampler defects break it. `MLX_TEST_SAMPLED_DRAWS` sets the count.
#[test]
fn device_rows_keep_the_target_law_through_the_whole_turn_and_the_lane() -> Result<()> {
    if !metal()
        || crate::test_support::apple_gpu_family_absent(
            "device_rows_keep_the_target_law_through_the_whole_turn_and_the_lane",
        )
    {
        return Ok(());
    }
    let draws = std::env::var("MLX_TEST_SAMPLED_DRAWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8_000);
    for (call, seed) in [(Call::WholeTurn, 0x61), (Call::Lane, 0x62)] {
        let started = Instant::now();
        let (p, lines) = rows_law(draws, seed, call, Planted::None)?;
        eprintln!(
            "LAW {call:?} draws {draws} pooled p {p:.3e} in {:?}",
            started.elapsed()
        );
        for line in &lines {
            eprintln!("LAW   {line}");
        }
        assert!(
            p >= 1e-6,
            "{call:?}: the device rows break the law: p {p:.3e}"
        );

        // High-acceptance full-width (L = 7) arm guarantees every verify row
        // 0..=L (including late rows and the bonus row) reaches >= 200 hits.
        const FULL_CYCLES: usize = 1_500;
        let started = Instant::now();
        let (full_p, row_hits, full_lines) =
            rows_law_with(0, Some(FULL_CYCLES), seed ^ 0x40, call, Planted::None, true)?;
        eprintln!(
            "LAW {call:?} full-width cycles {FULL_CYCLES} hits {:?} pooled p {full_p:.3e} in {:?}",
            row_hits,
            started.elapsed()
        );
        for line in &full_lines {
            eprintln!("LAW   {line}");
        }
        for (row, &hits) in row_hits.iter().enumerate() {
            assert!(
                hits >= 200,
                "{call:?}: full-width row {row} got only {hits} hits (< 200)"
            );
        }
        assert!(
            full_p >= 1e-6,
            "{call:?}: full-width device rows break the law: p {full_p:.3e}"
        );
    }
    for planted in [Planted::Hot, Planted::NoNucleus, Planted::WrongDraft] {
        let (p, _) = rows_law(draws.min(6_000), 0x63, Call::Lane, planted)?;
        eprintln!("LAW planted {planted:?} pooled p {p:.3e}");
        assert!(
            p < 1e-6,
            "the law test misses the planted {planted:?}: p {p:.3e}"
        );
    }
    Ok(())
}

/// One lane's cycle as the scheduler holds it: `[8, 1, V]` logits and a host
/// proposal whose first `accepted` drafts the target keeps and whose next
/// one it refuses.
fn lane_cycle(rows: &[Vec<f32>], targets: &[Vec<f32>], accepted: usize) -> Result<DsparkProposal> {
    let mut drafts = Vec::with_capacity(ROWS - 1);
    let mut candidates = Vec::with_capacity(ROWS - 1);
    for row in 0..ROWS - 1 {
        let kept = top(&targets[row], 16);
        let outside: Vec<i32> = (0..rows[row].len())
            .filter(|&t| targets[row][t] == 0.0)
            .take(15)
            .map(|t| t as i32)
            .collect();
        let mut ids = vec![kept[0] as i32];
        ids.extend(&outside);
        let mut probs = vec![1.0f64; ids.len()];
        if row < accepted {
            probs[0] = 1e-9;
            drafts.push(kept[0] as i32);
        } else {
            for prob in probs.iter_mut().skip(2) {
                *prob = 1e-9;
            }
            probs[0] = 1e-9;
            drafts.push(outside[0]);
        }
        candidates.push((ids, probs));
    }
    host_proposal(rows[0].len(), &drafts, &candidates)
}

#[test]
#[ignore = "timing row: run alone"]
fn sampled_rows_against_the_filter_chain_timing() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    let cfg = config(0.6, 20, 0.95, 0.0);
    let p = params(cfg);
    let rows: Vec<Vec<f32>> = (0..ROWS)
        .map(|row| peaked(VOCAB, 30, 100 + row as u64))
        .collect();
    let targets: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| chain(row, cfg))
        .collect::<Result<_>>()?;
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let base = MxArray::from_float32(&flat, &[ROWS as i64, 1, VOCAB as i64])?;
    let zero = MxArray::scalar_float(0.0)?.astype(DType::Float32)?;
    // The verify head's product form: a fresh f32 `[rows, 1, V]` array per cycle.
    let fresh = || -> Result<MxArray> {
        let logits = base.add(&zero)?;
        logits.eval();
        assert_eq!(logits.dtype()?, DType::Float32);
        Ok(logits)
    };
    let pattern = [0usize, 1, 1, 2, 2, 2, 3, 3, 4, 7];
    let mut rng = StdRng::seed_from_u64(23);
    let today = |accepted: usize, rng: &mut StdRng| -> Result<(f64, usize, u32)> {
        let logits = fresh()?;
        let mut proposal = lane_cycle(&rows, &targets, accepted)?;
        let start = Instant::now();
        let lane = logits
            .slice_axis(0, 0, ROWS as i64)?
            .transpose(Some(&[1, 0, 2]))?;
        let (k, token) = accept_dspark_proposal(&lane, &mut proposal, &[], &p, rng, None)?;
        Ok((start.elapsed().as_secs_f64() * 1e3, k, token))
    };
    let device = |accepted: usize, rng: &mut StdRng| -> Result<(f64, usize, u32)> {
        let logits = fresh()?;
        let mut proposal = lane_cycle(&rows, &targets, accepted)?;
        let start = Instant::now();
        let lane = logits.slice_axis(0, 0, ROWS as i64)?;
        let submitted = submit_sampled_rows(&lane, &proposal, &p, rng)?
            .ok_or_else(|| napi::Error::from_reason("the kernels declined"))?;
        let (k, token) = submitted.accept(&mut proposal, None)?;
        Ok((start.elapsed().as_secs_f64() * 1e3, k, token))
    };
    for round in 0..6 {
        today(pattern[round % pattern.len()], &mut rng)?;
        device(pattern[round % pattern.len()], &mut rng)?;
    }
    let mut savings = Vec::with_capacity(20);
    let mut chain_ms = Vec::with_capacity(20);
    let mut device_ms = Vec::with_capacity(20);
    for round in 0..20 {
        let accepted = pattern[round % pattern.len()];
        let (a, b) = if round % 2 == 0 {
            let a = today(accepted, &mut rng)?;
            (a, device(accepted, &mut rng)?)
        } else {
            let b = device(accepted, &mut rng)?;
            (today(accepted, &mut rng)?, b)
        };
        assert_eq!(a.1, accepted, "the chain keeps the planned drafts");
        assert_eq!(b.1, accepted, "the kernels keep the planned drafts");
        eprintln!(
            "TIMING round {round} accepted {accepted} chain_ms {:.3} device_ms {:.3} saving_ms {:.3}",
            a.0,
            b.0,
            a.0 - b.0
        );
        savings.push(a.0 - b.0);
        chain_ms.push(a.0);
        device_ms.push(b.0);
    }
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        (values[9] + values[10]) / 2.0
    };
    eprintln!(
        "TIMING median chain_ms {:.3} device_ms {:.3} saving_ms {:.3} rounds 20",
        median(&mut chain_ms),
        median(&mut device_ms),
        median(&mut savings)
    );
    Ok(())
}
