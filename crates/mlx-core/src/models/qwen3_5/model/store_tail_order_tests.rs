//! The store-only prefill tail, the merged sink eval and the device-id
//! verify against the reference order (every last layer in full, the sink
//! evaluated apart, ids read back first): shapes the other tail tests leave
//! out, and states that cross a restart from one order to the other.

use super::*;
use crate::models::qwen3_5::decoder_layer::compute_layer_kinds;
use crate::models::qwen3_5::dflash2_decode::verify_order::READ_BACK_FIRST;
use crate::models::qwen3_5::dflash2_state::disk::{adopt_all, state_context};
use crate::models::qwen3_5::paged_forward::{
    MaterializedGdnPrefixCheckpoint, PREFILL_CHUNK_BLOCKS, PREFILL_SERIAL, PREFILL_TAIL_FULL,
    PREFILL_TAILS_STORED, PrefillParts, run_paged_prefill_chunk_with_hidden_with_size,
    run_paged_prefill_chunk_with_size, run_paged_vlm_prefill,
};
use crate::models::qwen3_5::prefill_tail_tests as tail;
use crate::stream::{Stream, StreamContext};

type PoolBytes = Vec<Vec<(Vec<u8>, Vec<u8>)>>;

/// `true`: the reference order on this thread.
fn set_order(before: bool) {
    PREFILL_TAIL_FULL.with(|flag| flag.set(before));
    PREFILL_SERIAL.with(|flag| flag.set(before));
    READ_BACK_FIRST.with(|flag| flag.set(before));
}

/// One prefill, or a prefill and a continuation of the live request.
#[derive(Clone, Debug)]
struct Shape {
    layers: i32,
    rows: i32,
    first: usize,
    /// Rows a second prefill adds on the live request.
    more: usize,
    /// The with-hidden driver keeping this many rows.
    keep: Option<Option<usize>>,
    /// Blocks come at each chunk's booking, not before the prefill.
    lazy_blocks: bool,
}

#[derive(Debug, PartialEq)]
struct Left {
    logits: Vec<Vec<u32>>,
    hidden: Vec<u32>,
    pool: PoolBytes,
    gdn: Vec<Vec<u32>>,
    checkpoints: Vec<(u32, Vec<Vec<u32>>)>,
}

fn note(left: &mut Left, points: &[MaterializedGdnPrefixCheckpoint]) -> Result<()> {
    for point in points {
        left.checkpoints
            .push((point.prefix_len, tail::gdn_bits(&point.caches)?));
    }
    Ok(())
}

/// What the prefills leave, their chunk count and their store-only tails.
fn drive(shape: &Shape, before: bool) -> Result<(Left, usize, usize)> {
    let mut inner = tail::fixture(0xE1_7001, shape.layers);
    let tokens = tail::prompt(shape.first + shape.more);
    let booked = if shape.lazy_blocks { 1 } else { tokens.len() };
    tail::begin_request(&mut inner, &tokens[..booked])?;
    let kinds = compute_layer_kinds(inner.config.num_layers as usize, |i| {
        inner.config.is_linear_layer(i)
    });
    set_order(before);
    PREFILL_TAILS_STORED.with(|count| count.set(0));
    PREFILL_CHUNK_BLOCKS.with(|log| *log.borrow_mut() = Some(Vec::new()));
    let embed = inner.embedding.clone();
    let mut left = Left {
        logits: Vec::new(),
        hidden: Vec::new(),
        pool: Vec::new(),
        gdn: Vec::new(),
        checkpoints: Vec::new(),
    };
    let ran = (|| -> Result<()> {
        let mut cached = 0usize;
        for end in [shape.first, shape.first + shape.more] {
            if end == cached {
                continue;
            }
            let caches = inner
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("no caches"))?;
            let adapter = inner
                .paged_adapter
                .as_mut()
                .ok_or_else(|| Error::from_reason("no pool"))?;
            let mut target = PrefillParts {
                layers: &mut inner.layers,
                caches,
                final_norm: &inner.final_norm,
                lm_head: &inner.lm_head,
                adapter,
            };
            let logits = match shape.keep {
                None => {
                    let (logits, points) = run_paged_prefill_chunk_with_size(
                        &tokens[..end],
                        &tokens[cached..end],
                        cached as u32,
                        cached > 0,
                        &embed,
                        &mut target,
                        &kinds,
                        shape.rows,
                        0,
                        None,
                        None,
                    )?;
                    note(&mut left, &points)?;
                    logits
                }
                Some(keep) => {
                    let (logits, hidden, points) = run_paged_prefill_chunk_with_hidden_with_size(
                        &tokens[..end],
                        &tokens[cached..end],
                        cached as u32,
                        cached > 0,
                        &embed,
                        &mut target,
                        &kinds,
                        shape.rows,
                        keep,
                        0,
                        None,
                        None,
                    )?;
                    note(&mut left, &points)?;
                    left.hidden.extend(tail::bits(&hidden)?);
                    logits
                }
            };
            left.logits.push(tail::bits(&logits)?);
            cached = end;
        }
        Ok(())
    })();
    set_order(false);
    let chunks = PREFILL_CHUNK_BLOCKS
        .with(|log| log.borrow_mut().take())
        .map_or(0, |log| log.len());
    let stored = PREFILL_TAILS_STORED.with(std::cell::Cell::get);
    ran?;
    left.gdn = tail::gdn_bits(inner.caches.as_deref().unwrap_or_default())?;
    left.pool = tail::request_pool_bytes(&mut inner)?;
    Ok((left, chunks, stored))
}

/// The store-only tail and the merged sink eval keep the bits of the
/// reference order on shapes the other tail tests leave out: a model whose only
/// full-attention layer is the last one (no in-loop barrier before it), 12
/// layers, a prompt that ends at a chunk edge, a one-row final chunk, 8-row
/// and one-row chunks, blocks booked chunk by chunk, a continuation of the live request (several chunks, and one short
/// chunk), and a kept tail that starts at a chunk edge.
/// Mutation: the store-only tail rotates its keys one row off -> red.
#[test]
fn tail_and_pipeline_shapes_keep_the_bits_before() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "tail_and_pipeline_shapes_keep_the_bits_before",
    ) {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let shape = |layers, rows, first, more, keep, lazy_blocks| Shape {
        layers,
        rows,
        first,
        more,
        keep,
        lazy_blocks,
    };
    // (shape, store-only tails of the with-hidden driver)
    let cases = [
        (shape(4, 32, 100, 0, None, false), None),
        (shape(12, 32, 100, 0, None, false), None),
        (shape(8, 32, 64, 0, None, false), None),
        (shape(8, 32, 96, 0, None, false), None),
        (shape(8, 32, 33, 0, None, false), None),
        (shape(8, 8, 40, 0, None, false), None),
        (shape(8, 1, 5, 0, None, false), None),
        (shape(8, 16, 100, 0, None, true), None),
        (shape(8, 32, 70, 50, None, false), None),
        (shape(8, 32, 70, 5, None, true), None),
        (shape(4, 16, 40, 40, None, true), None),
        (shape(8, 32, 100, 0, Some(Some(36)), false), Some(2)),
        (shape(8, 32, 100, 0, Some(Some(4)), false), Some(3)),
        (shape(8, 32, 100, 0, Some(Some(0)), false), Some(3)),
        (shape(8, 32, 70, 50, Some(Some(8)), false), None),
    ];
    for (shape, kept_tails) in cases {
        let (before, before_chunks, before_stored) = drive(&shape, true)?;
        let (after, chunks, stored) = drive(&shape, false)?;
        assert_eq!(
            before_stored, 0,
            "{shape:?}: the reference order stored a tail"
        );
        assert_eq!(before_chunks, chunks, "{shape:?}: the chunk schedule moved");
        let calls = 1 + usize::from(shape.more > 0);
        match (shape.keep, kept_tails) {
            (None, _) => assert_eq!(
                stored,
                chunks - calls,
                "{shape:?}: store-only tails over {chunks} chunks"
            ),
            (Some(_), Some(tails)) => {
                assert_eq!(stored, tails, "{shape:?}: store-only tails")
            }
            (Some(_), None) => assert!(stored > 0, "{shape:?}: no store-only tail"),
        }
        assert!(
            after.pool.iter().any(|block| block
                .last()
                .is_some_and(|(k, v)| k.iter().chain(v).any(|&b| b != 0))),
            "{shape:?}: the last layer stored nothing"
        );
        assert!(before == after, "{shape:?}: the prefill's bits changed");
    }
    Ok(())
}

/// A media prefill's chunks on an M-RoPE grid: what it leaves and its
/// store-only tails.
fn media(before: bool) -> Result<(Left, usize)> {
    let mut inner = tail::fixture(0xE1_7002, 8);
    inner.init_mrope_layers(
        vec![11, 11, 10],
        inner.config.rope_theta,
        inner.config.max_position_embeddings,
    )?;
    let tokens = tail::prompt(100);
    tail::begin_request(&mut inner, &tokens)?;
    let kinds = compute_layer_kinds(inner.config.num_layers as usize, |i| {
        inner.config.is_linear_layer(i)
    });
    let rows = tokens.len() as i64;
    // Three axes that differ, so the rotation is not the scalar one.
    let grid: Vec<i32> = (0..rows as i32)
        .chain((0..rows as i32).map(|row| row / 4))
        .chain((0..rows as i32).map(|row| row % 4))
        .collect();
    let merge = crate::engine::vision::VisionMerge {
        inputs_embeds: inner
            .embedding
            .forward(&MxArray::from_uint32(&tokens, &[1, rows])?)?,
        position_ids: MxArray::from_int32(&grid, &[3, 1, rows])?,
        rope_deltas: 0,
    };
    set_order(before);
    PREFILL_TAILS_STORED.with(|count| count.set(0));
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(32)));
    let embed = inner.embedding.clone();
    let ran = (|| {
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("no caches"))?;
        let adapter = inner
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("no pool"))?;
        let mut target = PrefillParts {
            layers: &mut inner.layers,
            caches,
            final_norm: &inner.final_norm,
            lm_head: &inner.lm_head,
            adapter,
        };
        run_paged_vlm_prefill(
            &tokens,
            &merge,
            0,
            false,
            &embed,
            &mut target,
            &kinds,
            None,
            None,
        )
    })();
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
    set_order(false);
    let stored = PREFILL_TAILS_STORED.with(std::cell::Cell::get);
    let (logits, points) = ran?;
    let mut checkpoints = Vec::new();
    for point in &points {
        checkpoints.push((point.prefix_len, tail::gdn_bits(&point.caches)?));
    }
    let left = Left {
        logits: vec![tail::bits(&logits)?],
        hidden: Vec::new(),
        gdn: tail::gdn_bits(inner.caches.as_deref().unwrap_or_default())?,
        pool: tail::request_pool_bytes(&mut inner)?,
        checkpoints,
    };
    Ok((left, stored))
}

/// A media (M-RoPE) prefill of several chunks runs every last layer in full
/// and keeps its bits: the store-only rule declines M-RoPE positions.
/// Mutation: the rule ignores the M-RoPE positions -> red.
#[test]
fn media_chunks_run_the_last_layer_in_full() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("media_chunks_run_the_last_layer_in_full") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let (before, before_stored) = media(true)?;
    let (after, stored) = media(false)?;
    assert_eq!((before_stored, stored), (0, 0), "store-only tails");
    assert!(
        after.checkpoints.iter().any(|(row, _)| *row == 96),
        "the media prefill split at no rung: {:?}",
        after
            .checkpoints
            .iter()
            .map(|(row, _)| *row)
            .collect::<Vec<_>>()
    );
    assert!(before == after, "the media prefill's bits changed");
    Ok(())
}

/// `rows`-row prefill chunks on this thread.
struct ChunkRows;

impl ChunkRows {
    fn new(rows: i32) -> Self {
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|slot| slot.set(Some(rows)));
        Self
    }
}

impl Drop for ChunkRows {
    fn drop(&mut self) {
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|slot| slot.set(None));
    }
}

/// The bytes of every block of the pool.
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

/// Each state with a RAM copy as the tier encodes it, by depth.
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

/// A turn in one process, a restart, a turn in the next process.
#[derive(Debug, PartialEq)]
struct Restart {
    /// The first process: its tokens, its text and its states as encoded.
    cold: Vec<u32>,
    cold_text: String,
    written: Vec<(usize, Vec<u8>)>,
    /// The next process: the states it adopted, the prefix it resumed at,
    /// its tokens, its text, its states and its pool.
    adopted: u64,
    cached: u32,
    warm: Vec<u32>,
    warm_text: String,
    states: Vec<(usize, Vec<u8>)>,
    pool: PoolBytes,
}

fn restart(
    tag: &str,
    writer_before: bool,
    reader_before: bool,
    load: impl Fn(&TestTier) -> Result<Qwen35Inner>,
    turn: impl Fn(&mut Qwen35Inner, Option<&[u32]>) -> Result<ChatResult>,
) -> Result<Restart> {
    let result = (|| -> Result<Restart> {
        let tier = persistent_test_tier(tag, 1 << 30);
        set_order(writer_before);
        let mut first = load(&tier)?;
        let cold_result = turn(&mut first, None)?;
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
        let warm_result = turn(&mut second, Some(&history))?;
        let warm = generated();
        Ok(Restart {
            cold,
            cold_text: cold_result.raw_text,
            written,
            adopted,
            cached: warm_result.cached_tokens,
            warm,
            warm_text: warm_result.raw_text,
            states: encoded_states(&second),
            pool: all_pool_bytes(&mut second)?,
        })
    })();
    set_order(false);
    result
}

/// The four writer / reader pairs of the two orders leave one outcome.
/// `turn` runs the writer's turn with `None` and the reader's with the
/// history the writer's turn saved.
fn restarts_agree(
    tag: &str,
    load: impl Fn(&TestTier) -> Result<Qwen35Inner>,
    turn: impl Fn(&mut Qwen35Inner, Option<&[u32]>) -> Result<ChatResult>,
) -> Result<Restart> {
    let reference = restart(tag, true, true, &load, &turn)?;
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
    for (writer_before, reader_before) in [(true, false), (false, true), (false, false)] {
        let pair = restart(tag, writer_before, reader_before, &load, &turn)?;
        let name = format!(
            "{tag}: written in the {} order, read in the {} order",
            if writer_before { "reference" } else { "new" },
            if reader_before { "reference" } else { "new" },
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

/// A plain autoregressive turn's state (the GDN cell at its replay point)
/// and its KV chain, written to the SSD tier in one order and read after a
/// restart in the other: every pair of the reference order and the
/// new order stores the same state bytes, adopts the same states,
/// resumes at the same prefix (112) and decodes the same tokens over the
/// same pool bytes; so does the conversation's next message on the saved
/// history.
/// Mutation: the store-only tail rotates its keys one row off -> red.
#[test]
fn a_plain_state_crosses_a_restart_between_the_orders() -> Result<()> {
    const TEST: &str = "a_plain_state_crosses_a_restart_between_the_orders";
    if plain(0x5A13_E101, TEST).is_none() {
        return Ok(());
    }
    let _plan = SmallPlan::new();
    let tokens = prompt(11, 127);
    let config = ChatConfig {
        max_new_tokens: Some(6),
        ..config()
    };
    let reference = restarts_agree(
        "state-tier-tail-order-plain",
        |tier| {
            let mut inner =
                plain(0x5A13_E101, TEST).ok_or_else(|| Error::from_reason("no fixture"))?;
            attach(&mut inner, tier);
            Ok(inner)
        },
        |inner, _| whole_turn(inner, &tokens, &config, DecoderPlan::Autoregressive),
    )?;
    assert_eq!(reference.cached, 112, "the resumed prefix");
    assert_eq!(
        reference.warm, reference.cold,
        "the restart moved the tokens"
    );
    // The next process continues the conversation: the saved history, one
    // more message and the generation prompt.
    let continued = restarts_agree(
        "state-tier-tail-order-plain-next",
        |tier| {
            let mut inner =
                plain(0x5A13_E101, TEST).ok_or_else(|| Error::from_reason("no fixture"))?;
            attach(&mut inner, tier);
            Ok(inner)
        },
        |inner, history| match history {
            None => whole_turn(inner, &tokens, &config, DecoderPlan::Autoregressive),
            Some(history) => {
                let mut next = history.to_vec();
                next.extend([4, 5, 6, 2, 3]);
                whole_turn(inner, &next, &config, DecoderPlan::Autoregressive)
            }
        },
    )?;
    eprintln!(
        "tail order plain: same prompt resumes at {}, the next message at {}",
        reference.cached, continued.cached
    );
    Ok(())
}

/// A greedy DFlash2 turn (tapped prefill, device-id verify cycles) whose
/// states and KV chain cross a restart between the reference order
/// and the new order, both ways: the same state bytes, the same
/// resumed prefix, the same tokens and text, the same pool bytes.
/// The next process runs the same prompt, then the conversation's next
/// message on the saved history.
/// Mutation: the store-only tail rotates its keys one row off -> red.
#[test]
fn a_draft_turn_crosses_a_restart_between_the_orders() -> Result<()> {
    const TEST: &str = "a_draft_turn_crosses_a_restart_between_the_orders";
    if !metal_is_default_device() || crate::test_support::paged_prefill_unsupported(TEST) {
        return Ok(());
    }
    use crate::models::qwen3_5::dflash2_decode::paged_dflash2_tests::tiny_paged_int8_dflash_inner;
    let _ctx = StreamContext::new(Stream::generation());
    let _rows = ChunkRows::new(16);
    let tokens = prompt(11, 127);
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
        let mut inner = tiny_paged_int8_dflash_inner(0x5A13_E102);
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
    let reference = restarts_agree("state-tier-tail-order-draft", load, |inner, _| {
        draft_turn(inner, &tokens)
    })?;
    assert_eq!(reference.warm.len(), 20, "the continuation's length");
    // The next process continues the conversation: the saved history (its
    // last blocks hold decoded tokens, recorded by the device-id verify), one
    // more message and the generation prompt.
    let continued = restarts_agree(
        "state-tier-tail-order-draft-next",
        load,
        |inner, history| match history {
            None => draft_turn(inner, &tokens),
            Some(history) => {
                let mut next = history.to_vec();
                next.extend([4, 5, 6, 2, 3]);
                draft_turn(inner, &next)
            }
        },
    )?;
    eprintln!(
        "tail order draft: same prompt resumes at {}, the next message at {}",
        reference.cached, continued.cached
    );
    Ok(())
}
