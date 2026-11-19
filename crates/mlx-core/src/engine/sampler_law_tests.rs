//! The sampler's law, with no model: the filter chain's distribution against
//! an f64 reference of the documented chain, the sampler's draws against
//! that distribution, and the speculative law (each emitted token follows
//! its verify row's distribution) through the host acceptance and the
//! device rows.

use napi::{Error, Result};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::array::{DType, MxArray};
use crate::engine::backend::DsparkProposal;
use crate::engine::dspark_device_sampled_tests::{chi_square, gamma_q};
use crate::engine::dspark_turn::{accept_dspark_proposal, submit_sampled_rows};
use crate::engine::params::ChatParams;
use crate::sampling::{SamplingConfig, SparseDistribution};

const V: usize = 2048;
const DRAFTS: usize = 7;
const CYCLES: usize = 2048;
const MIN_P_VALUE: f64 = 1e-6;

fn metal() -> bool {
    unsafe { mlx_sys::mlx_metal_is_available() }
}

/// The f64 reference of the default-order filter chain: logprobs of the raw
/// logits; top-k; top-p on the ascending cumulative of their probabilities
/// (keep where the cumulative passes `1 - top_p`); min-p against the top
/// logprob; softmax of the kept logprobs over the temperature.
fn reference(logits: &[f32], config: &SamplingConfig) -> Vec<f64> {
    let t = config.temperature.unwrap_or(1.0);
    let top_k = config.top_k.unwrap_or(0);
    let top_p = config.top_p.unwrap_or(1.0);
    let min_p = config.min_p.unwrap_or(0.0);
    let x: Vec<f64> = logits.iter().map(|v| f64::from(*v)).collect();
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lse = m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
    let mut lp: Vec<f64> = x.iter().map(|v| v - lse).collect();
    let mut order: Vec<usize> = (0..lp.len()).collect();
    order.sort_by(|a, b| lp[*b].total_cmp(&lp[*a]));
    if top_k > 0 && (top_k as usize) < lp.len() {
        for &i in &order[top_k as usize..] {
            lp[i] = f64::NEG_INFINITY;
        }
    }
    if top_p > 0.0 && top_p < 1.0 {
        let mut cumulative = 0.0;
        let threshold = 1.0 - (top_p - 1e-7);
        let mut keep = vec![false; lp.len()];
        for &i in order.iter().rev() {
            cumulative += lp[i].exp();
            keep[i] = cumulative > threshold;
        }
        for (value, kept) in lp.iter_mut().zip(&keep) {
            if !kept {
                *value = f64::NEG_INFINITY;
            }
        }
    }
    if min_p > 0.0 {
        let top = lp[order[0]];
        for (i, value) in lp.iter_mut().enumerate() {
            if i != order[0] && *value < top + min_p.ln() {
                *value = f64::NEG_INFINITY;
            }
        }
    }
    let scaled: Vec<f64> = lp.iter().map(|v| v / t).collect();
    let m = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = scaled.iter().map(|v| (v - m).exp()).sum();
    scaled.iter().map(|v| (v - m).exp() / z).collect()
}

/// A logits row: a Zipf-like head over a flat tail; `sharp` sets the
/// entropy.
fn row(seed: u64, sharp: f64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..V)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            let rank = ((i * 7919 + seed as usize) % V) as f64 + 1.0;
            (-sharp * rank.ln() + 0.7 * u) as f32
        })
        .collect()
}

fn product_distribution(logits: &[f32], config: &SamplingConfig) -> Result<Vec<f64>> {
    let row = MxArray::from_float32(logits, &[logits.len() as i64])?;
    let p = crate::sampling::sampling_distribution(&row, Some(*config))?.astype(DType::Float32)?;
    p.eval();
    Ok(p.to_float32()?.iter().map(|v| f64::from(*v)).collect())
}

fn presets() -> Vec<(&'static str, SamplingConfig)> {
    let preset = |t: f64, k: i32, p: f64, m: f64| SamplingConfig {
        temperature: Some(t),
        top_k: Some(k),
        top_p: Some(p),
        min_p: Some(m),
    };
    vec![
        ("t0.8-k40-p0.95-m0.02", preset(0.8, 40, 0.95, 0.02)),
        ("t1.0-p0.9", preset(1.0, 0, 0.9, 0.0)),
        ("t0.6-k20", preset(0.6, 20, 1.0, 0.0)),
        ("t1.0-plain", preset(1.0, 0, 1.0, 0.0)),
    ]
}

fn params(config: &SamplingConfig) -> ChatParams {
    crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
        temperature: config.temperature,
        top_k: config.top_k,
        top_p: config.top_p,
        min_p: config.min_p,
        ..Default::default()
    })
}

fn p_value(counts: &[u64], probs: &[f64]) -> (f64, usize, f64) {
    let (stat, dof) = chi_square(counts, probs);
    (stat, dof, gamma_q(dof as f64 / 2.0, stat / 2.0))
}

/// The acceptance a cycle of the speculative law runs.
#[derive(Clone, Copy, Debug)]
enum Accept {
    Host,
    DeviceRows,
}

/// One verify block of 8 rows: its logits, each row's product
/// distribution, and draft rows over each target's top 16, flattened
/// toward uniform.
struct Block {
    logits: MxArray,
    targets: Vec<Vec<f64>>,
    drafts: Vec<SparseDistribution>,
}

fn block(state: u64, config: &SamplingConfig) -> Result<Block> {
    let rows: Vec<Vec<f32>> = (0..=DRAFTS as u64)
        .map(|j| row(100 + state * 10 + j, 0.9 + 0.05 * j as f64))
        .collect();
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let logits = MxArray::from_float32(&flat, &[1, DRAFTS as i64 + 1, V as i64])?;
    let targets = rows
        .iter()
        .map(|r| product_distribution(r, config))
        .collect::<Result<Vec<_>>>()?;
    let drafts = targets[..DRAFTS]
        .iter()
        .map(|p| {
            let mut ids: Vec<usize> = (0..V).collect();
            ids.sort_by(|a, b| p[*b].total_cmp(&p[*a]));
            SparseDistribution::from_parts(
                ids[..16].iter().map(|&i| i as i32).collect(),
                ids[..16].iter().map(|&i| p[i].sqrt() + 1e-3).collect(),
                V,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Block {
        logits,
        targets,
        drafts,
    })
}

/// `CYCLES` cycles of one block through `accept`: the fails of the per-row
/// law (emitted token j, given j accepted drafts, follows row j's
/// distribution) and of lag-1 independence of the first emitted token.
fn speculative_law(
    tag: &str,
    state: u64,
    config: &SamplingConfig,
    accept: Accept,
    pool: &mut (f64, usize),
) -> Result<Vec<String>> {
    let b = block(state, config)?;
    let p = params(config);
    let mut rng = StdRng::seed_from_u64(0xd3 + state);
    let mut counts = vec![vec![0u64; V]; DRAFTS + 1];
    let mut firsts = Vec::with_capacity(CYCLES);
    for _ in 0..CYCLES {
        let draft = b
            .drafts
            .iter()
            .map(|q| q.as_row().sample(&mut rng))
            .collect::<Result<Vec<_>>>()?;
        let mut proposal = DsparkProposal {
            draft_ids: draft.clone(),
            device_draft_ids: None,
            device_draft_sparse: None,
            draft_dists: Vec::new(),
            draft_sparse_dists: b.drafts.clone(),
            keep_probabilities: None,
        };
        let (k, boundary) = match accept {
            Accept::Host => {
                accept_dspark_proposal(&b.logits, &mut proposal, &[], &p, &mut rng, None)?
            }
            Accept::DeviceRows => submit_sampled_rows(&b.logits, &proposal, &p, &mut rng)?
                .ok_or_else(|| Error::from_reason(format!("{tag}: the device rows declined")))?
                .accept(&mut proposal, None)?,
        };
        for (j, bins) in counts.iter_mut().enumerate().take(k + 1) {
            let token = if j < k { draft[j] } else { boundary as i32 };
            bins[token as usize] += 1;
        }
        firsts.push(if k > 0 {
            draft[0] as usize
        } else {
            boundary as usize
        });
    }
    let mut fails = Vec::new();
    for (j, (c, target)) in counts.iter().zip(&b.targets).enumerate() {
        let n: u64 = c.iter().sum();
        if n < 200 {
            continue;
        }
        let (stat, dof, pv) = p_value(c, target);
        eprintln!("{tag} state {state} row {j}: n {n} chi2 {stat:.1} dof {dof} p {pv:.3e}");
        pool.0 += stat;
        pool.1 += dof;
    }
    let top = (0..V)
        .max_by(|a, c| b.targets[0][*a].total_cmp(&b.targets[0][*c]))
        .unwrap_or(0);
    let mut table = [[0u64; 2]; 2];
    for w in firsts.windows(2) {
        table[usize::from(w[0] == top)][usize::from(w[1] == top)] += 1;
    }
    let total = (CYCLES - 1) as f64;
    let row_sums = [table[0][0] + table[0][1], table[1][0] + table[1][1]];
    let column_sums = [table[0][0] + table[1][0], table[0][1] + table[1][1]];
    let mut stat = 0.0;
    for (row, row_sum) in table.iter().zip(row_sums) {
        for (&observed, column_sum) in row.iter().zip(column_sums) {
            let expected = row_sum as f64 * column_sum as f64 / total;
            if expected > 0.0 {
                stat += (observed as f64 - expected).powi(2) / expected;
            }
        }
    }
    let pv = gamma_q(0.5, stat / 2.0);
    if pv.is_nan() || pv < MIN_P_VALUE {
        fails.push(format!("{tag} state {state}: lag-1 dependence, p {pv:.3e}"));
    }
    Ok(fails)
}

/// The filter chain's distribution is the f64 reference's (total variation
/// at most 1e-4, the same support) on four presets and three entropies; the
/// sampler's draws follow it; and the speculative law holds through the host
/// acceptance of sparse draft rows and, for the presets whose `top_k` the
/// kernels take, through the device rows.
#[test]
fn the_sampler_keeps_its_law() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    let mut fails: Vec<String> = Vec::new();
    let mut draws_pool = (0.0f64, 0usize);
    for (name, config) in presets() {
        for (r, sharp) in [0.6f64, 1.0, 1.6].into_iter().enumerate() {
            let logits = row(17 + r as u64, sharp);
            let p = product_distribution(&logits, &config)?;
            let q = reference(&logits, &config);
            let tv = 0.5 * p.iter().zip(&q).map(|(a, b)| (a - b).abs()).sum::<f64>();
            let support = p
                .iter()
                .zip(&q)
                .filter(|(a, b)| (**a > 0.0) != (**b > 0.0))
                .count();
            if tv.is_nan() || tv > 1e-4 || support > 0 {
                fails.push(format!(
                    "distribution {name} row {r}: tv {tv:.3e}, support diff {support}"
                ));
            }
            let tiled = MxArray::concatenate_many(
                vec![&MxArray::from_float32(&logits, &[1, V as i64])?; 64],
                Some(0),
            )?;
            let mut counts = vec![0u64; V];
            for _ in 0..32 {
                let ids = crate::sampling::sample(&tiled, Some(config))?;
                ids.eval();
                for id in ids.to_int32()?.iter() {
                    counts[*id as usize] += 1;
                }
            }
            let (stat, dof, pv) = p_value(&counts, &p);
            eprintln!("draws {name} row {r}: chi2 {stat:.1} dof {dof} p {pv:.3e}");
            draws_pool.0 += stat;
            draws_pool.1 += dof;
        }
    }
    let draws_p = gamma_q(draws_pool.1 as f64 / 2.0, draws_pool.0 / 2.0);
    if draws_p.is_nan() || draws_p < MIN_P_VALUE {
        fails.push(format!("draws: pooled p {draws_p:.3e}"));
    }
    let all = presets();
    let mut arms = vec![
        ("host", all[0].1, Accept::Host, 0u64),
        ("host", all[0].1, Accept::Host, 1),
    ];
    for (name, config) in &all {
        if crate::engine::dspark_turn::sampled_rows_take(&params(config)) {
            arms.push((*name, *config, Accept::DeviceRows, 0));
        }
    }
    assert!(
        arms.len() >= 4,
        "the device rows take the top-k presets: {} arms",
        arms.len()
    );
    for (tag, config, accept, state) in arms {
        let tag = format!("{tag} {accept:?}");
        let mut pool = (0.0f64, 0usize);
        fails.extend(speculative_law(&tag, state, &config, accept, &mut pool)?);
        let pooled = gamma_q(pool.1 as f64 / 2.0, pool.0 / 2.0);
        eprintln!("{tag} state {state}: pooled p {pooled:.3e}");
        if pooled.is_nan() || pooled < MIN_P_VALUE {
            fails.push(format!("{tag} state {state}: pooled p {pooled:.3e}"));
        }
    }
    assert!(fails.is_empty(), "{fails:#?}");
    Ok(())
}
