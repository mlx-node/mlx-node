//! A check that a state written to the SSD tier before the
//! host-built block table, the row-split GDN prefill scan and the GEMM store
//! epilogues is read after them, and the reverse: the same stored bytes, the
//! same resumed prefix, the same tokens.

use super::*;
use crate::models::qwen3_5::dflash2_state::disk::{adopt_all, state_context};
use crate::models::qwen3_5::gated_delta::HEAD_SCAN_PREFILL;
use crate::models::qwen3_5::paged_forward::PREFILL_SEPARATE_OPS;
use crate::stream::{Stream, StreamContext};
use crate::transformer::paged_metadata_cache::RESHAPED_BLOCK_TABLE;

type PoolBytes = Vec<Vec<(Vec<u8>, Vec<u8>)>>;

/// The 27B's GDN head width, without which the fused prefill declines.
fn wide_gdn(config: &mut crate::models::qwen3_5::config::Qwen3_5Config) {
    config.linear_key_head_dim = 128;
    config.linear_value_head_dim = 128;
    config.linear_num_key_heads = 2;
    config.linear_num_value_heads = 4;
}

/// The GDN gate scalars in float32, as every checkpoint stores them: the
/// tiny fixture casts them to bfloat16, which the fused prefill declines.
fn f32_gates(inner: &mut Qwen35Inner) -> Result<()> {
    for layer in inner.layers.iter_mut() {
        if let crate::models::qwen3_5::decoder_layer::AttentionType::Linear(gdn) = &mut layer.attn {
            let bias = gdn.get_dt_bias().astype(crate::array::DType::Float32)?;
            gdn.set_dt_bias(&bias);
            let a_log = gdn.get_a_log().astype(crate::array::DType::Float32)?;
            gdn.set_a_log(&a_log)?;
        }
    }
    Ok(())
}

/// `true`: the reference order on this thread (reshaped block table,
/// head-kernel scan, separate ops).
fn set_order(before: bool) {
    RESHAPED_BLOCK_TABLE.with(|flag| flag.set(before));
    HEAD_SCAN_PREFILL.with(|flag| flag.set(before));
    PREFILL_SEPARATE_OPS.with(|flag| flag.set(before));
}

/// A state window and prefill chunks of `rows` rows on this thread.
struct Plan;

impl Plan {
    fn new(rows: i32) -> Self {
        STATE_WINDOW_OVERRIDE.with(|window| window.set(Some(rows as u32)));
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|slot| slot.set(Some(rows)));
        Self
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        STATE_WINDOW_OVERRIDE.with(|window| window.set(None));
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|slot| slot.set(None));
        STATE_CLAIM_SCRIPT.with(|script| script.borrow_mut().clear());
    }
}

fn all_pool_bytes(inner: &mut Qwen35Inner) -> Result<PoolBytes> {
    let adapter = inner
        .paged_adapter
        .as_mut()
        .ok_or_else(|| Error::from_reason("no pool"))?;
    adapter
        .eval_pending_pool_writes()
        .map_err(Error::from_reason)?;
    crate::array::synchronize_and_clear_cache();
    let pool = adapter.layer_kv_pool();
    (0..pool.num_blocks())
        .map(|id| pool.read_block_all_layers(id).map_err(Error::from_reason))
        .collect()
}

fn encoded_states(inner: &Qwen35Inner) -> Vec<(usize, Vec<u8>)> {
    let mut states: Vec<_> = inner
        .dflash2_states
        .store
        .entries()
        .filter_map(|entry| {
            let anchor = entry.ram()?;
            Some((
                entry.depth(),
                encode_anchor(&inner.dflash2_states, &inner.config, anchor)?,
            ))
        })
        .collect();
    states.sort();
    states
}

fn scans() -> (u64, u64) {
    // SAFETY: NUL-terminated names; test-only counters.
    unsafe {
        (
            mlx_sys::mlx_test_kquant_family_count(c"gdn_prefill_row_scan".as_ptr()),
            mlx_sys::mlx_test_kquant_family_count(c"gdn_prefill_head_scan".as_ptr()),
        )
    }
}

/// A turn in one process, a restart, a turn in the next process.
#[derive(Debug, PartialEq)]
struct Restart {
    cold: Vec<u32>,
    cold_text: String,
    written: Vec<(usize, Vec<u8>)>,
    adopted: u64,
    cached: u32,
    warm: Vec<u32>,
    warm_text: String,
    states: Vec<(usize, Vec<u8>)>,
    pool: PoolBytes,
}

/// The restart, and the (row scan, head scan) GDN prefill dispatches of the
/// writer and of the reader.
fn restart(
    tag: &str,
    writer_before: bool,
    reader_before: bool,
    load: impl Fn(&TestTier) -> Result<Qwen35Inner>,
    turn: impl Fn(&mut Qwen35Inner, Option<&[u32]>) -> Result<ChatResult>,
) -> Result<(Restart, (u64, u64), (u64, u64))> {
    let result = (|| {
        let tier = persistent_test_tier(tag, 1 << 30);
        set_order(writer_before);
        let mut first = load(&tier)?;
        // SAFETY: test-only thread-local counter switch.
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        let cold_result = turn(&mut first, None);
        let writer_scans = scans();
        // SAFETY: as above.
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        let cold_result = cold_result?;
        let cold = generated();
        let history = first.cached_token_history.clone();
        let written = encoded_states(&first);
        assert!(close_cache(
            &mut first,
            std::time::Instant::now() + std::time::Duration::from_secs(30)
        ));
        drop(first);

        let tier = tier.reopen(tag, 1 << 30);
        set_order(reader_before);
        let mut second = load(&tier)?;
        let adopted = adopt_all(&mut second);
        // SAFETY: as above.
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        let warm_result = turn(&mut second, Some(&history));
        let reader_scans = scans();
        // SAFETY: as above.
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        let warm_result = warm_result?;
        let warm = generated();
        Ok((
            Restart {
                cold,
                cold_text: cold_result.raw_text,
                written,
                adopted,
                cached: warm_result.cached_tokens,
                warm,
                warm_text: warm_result.raw_text,
                states: encoded_states(&second),
                pool: all_pool_bytes(&mut second)?,
            },
            writer_scans,
            reader_scans,
        ))
    })();
    set_order(false);
    result
}

/// The four writer / reader pairs of the two orders leave one outcome.
fn restarts_agree(
    tag: &str,
    reader_prefills_a_window: bool,
    load: impl Fn(&TestTier) -> Result<Qwen35Inner>,
    turn: impl Fn(&mut Qwen35Inner, Option<&[u32]>) -> Result<ChatResult>,
) -> Result<Restart> {
    let (reference, writer_scans, _) = restart(tag, true, true, &load, &turn)?;
    assert!(
        reference.adopted >= 1,
        "{tag}: the restart adopted no state"
    );
    assert!(
        reference.cached > 0,
        "{tag}: the next process resumed at no state"
    );
    assert!(!reference.written.is_empty(), "{tag}: no state was stored");
    assert!(
        !reference.warm.is_empty(),
        "{tag}: the turn decoded nothing"
    );
    assert!(
        writer_scans.1 > 0 && writer_scans.0 == 0,
        "{tag}: the reference order ran (row, head) scans {writer_scans:?}"
    );
    for (writer_before, reader_before) in [(true, false), (false, true), (false, false)] {
        let (pair, writer_scans, reader_scans) =
            restart(tag, writer_before, reader_before, &load, &turn)?;
        let name = format!(
            "{tag}: written in the {} order, read in the {} order",
            if writer_before { "reference" } else { "new" },
            if reader_before { "reference" } else { "new" },
        );
        if !writer_before {
            assert!(
                writer_scans.0 > 0 && writer_scans.1 == 0,
                "{name}: the writer ran (row, head) scans {writer_scans:?}"
            );
        }
        if reader_prefills_a_window {
            let (own, other) = if reader_before {
                (reader_scans.1, reader_scans.0)
            } else {
                (reader_scans.0, reader_scans.1)
            };
            assert!(
                own > 0 && other == 0,
                "{name}: the reader ran (row, head) scans {reader_scans:?} on the restored state"
            );
        }
        eprintln!(
            "V2_RESTART {name}: writer (row, head) scans {writer_scans:?}, reader \
             {reader_scans:?}, adopted {}, resumed at {}",
            pair.adopted, pair.cached
        );
        assert_eq!(
            pair.written, reference.written,
            "{name}: stored state bytes"
        );
        assert_eq!(pair.cold, reference.cold, "{name}: the writer's tokens");
        assert_eq!(
            pair.cold_text, reference.cold_text,
            "{name}: the writer's text"
        );
        assert_eq!(
            (pair.adopted, pair.cached),
            (reference.adopted, reference.cached),
            "{name}: (adopted states, resumed prefix)"
        );
        assert_eq!(
            pair.warm, reference.warm,
            "{name}: the continuation's tokens"
        );
        assert_eq!(
            pair.warm_text, reference.warm_text,
            "{name}: the continuation's text"
        );
        assert_eq!(pair.states, reference.states, "{name}: the reader's states");
        assert!(
            pair.pool == reference.pool,
            "{name}: the reader's pool bytes"
        );
    }
    Ok(reference)
}

/// A plain autoregressive turn of 290 prompt tokens in 96-row chunks (the
/// fused GDN prefill runs on every chunk) whose state and KV chain are
/// written to the SSD tier in one order and read after a restart in the
/// other, for every pair of the reference order (reshaped block table,
/// head-kernel scan, separate ops) and the new order:
/// the same stored state bytes, the same adopted states and resumed prefix,
/// the same tokens over the same pool bytes; and the same again for the
/// conversation's next message on the saved history.
#[test]
fn a_plain_state_crosses_a_restart_between_the_orders() -> Result<()> {
    const TEST: &str = "a_plain_state_crosses_a_restart_between_the_orders";
    if plain(0x5A13_E201, TEST).is_none() {
        return Ok(());
    }
    let _plan = Plan::new(96);
    let tokens = prompt(11, 288);
    let config = ChatConfig {
        max_new_tokens: Some(6),
        ..config()
    };
    let load = |tier: &TestTier| {
        let mut inner = tiny_dflash_inner_with_config(0x5A13_E201, wide_gdn);
        inner.dflash2 = None;
        f32_gates(&mut inner)?;
        install_pool(&mut inner, 32, 32, passing_probe())?;
        attach(&mut inner, tier);
        Ok(inner)
    };
    let reference = restarts_agree("state-tier-restart-plain", false, load, |inner, _| {
        whole_turn(inner, &tokens, &config, DecoderPlan::Autoregressive)
    })?;
    assert_eq!(
        reference.warm, reference.cold,
        "the restart moved the tokens"
    );
    let continued = restarts_agree(
        "state-tier-restart-plain-next",
        true,
        load,
        |inner, history| match history {
            None => whole_turn(inner, &tokens, &config, DecoderPlan::Autoregressive),
            Some(history) => {
                let mut next = history.to_vec();
                next.extend(prompt(5, 150));
                whole_turn(inner, &next, &config, DecoderPlan::Autoregressive)
            }
        },
    )?;
    eprintln!(
        "V2_RESTART plain: same prompt resumes at {}, the next message at {}",
        reference.cached, continued.cached
    );
    Ok(())
}

/// A greedy DFlash2 turn (tapped prefill in 96-row chunks, device-id verify
/// cycles) crosses a restart between the two orders, both ways: the same
/// state bytes, resumed prefix, tokens, text and pool bytes; and the same
/// for the conversation's next message.
#[test]
fn a_draft_turn_crosses_a_restart_between_the_orders() -> Result<()> {
    const TEST: &str = "a_draft_turn_crosses_a_restart_between_the_orders";
    if !metal_is_default_device() || crate::test_support::paged_prefill_unsupported(TEST) {
        return Ok(());
    }
    use crate::models::qwen3_5::dflash2_decode::paged_dflash2_tests::tiny_paged_int8_dflash_inner;
    let _ctx = StreamContext::new(Stream::generation());
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|slot| slot.set(Some(96)));
    let tokens = prompt(11, 288);
    let config = ChatConfig {
        max_new_tokens: Some(20),
        ..config()
    };
    let draft_turn = |inner: &mut Qwen35Inner, tokens: &[u32]| {
        whole_turn(
            inner,
            tokens,
            &config,
            DecoderPlan::Speculative(crate::engine::plan::SpeculativeKind::DraftModel),
        )
    };
    let load = |tier: &TestTier| -> Result<Qwen35Inner> {
        let mut inner = tiny_paged_int8_dflash_inner(0x5A13_E202);
        f32_gates(&mut inner)?;
        let window = inner
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("no draft"))?
            .config
            .context_window()?;
        attach_tier(
            &mut inner.dflash2_states,
            &inner.config,
            inner
                .paged_adapter
                .as_ref()
                .ok_or_else(|| Error::from_reason("no pool"))?,
            tier,
            window.nbytes(),
            window.slots(),
        );
        inner
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("no pool"))?
            .set_cold_tier(state_context(&tier.tier));
        Ok(inner)
    };
    let ran = (|| {
        let reference = restarts_agree("state-tier-restart-draft", false, load, |inner, _| {
            draft_turn(inner, &tokens)
        })?;
        assert_eq!(reference.warm.len(), 20, "the continuation's length");
        let continued = restarts_agree(
            "state-tier-restart-draft-next",
            true,
            load,
            |inner, history| match history {
                None => draft_turn(inner, &tokens),
                Some(history) => {
                    let mut next = history.to_vec();
                    next.extend(prompt(5, 150));
                    draft_turn(inner, &next)
                }
            },
        )?;
        eprintln!(
            "V2_RESTART draft: same prompt resumes at {}, the next message at {}",
            reference.cached, continued.cached
        );
        Ok(())
    })();
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|slot| slot.set(None));
    ran
}
