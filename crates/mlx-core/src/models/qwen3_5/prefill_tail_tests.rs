//! A non-final paged prefill chunk runs its last layer as a K/V store only
//! (Splash `QwenTarget::planOutputRows`): everything the prefill leaves
//! behind keeps its bits, and the rule declines where a reader exists.

use napi::{Error, Result};

use crate::array::{DType, MxArray};
use crate::models::quantized_linear::LinearProj;
use crate::nn::Linear;
use crate::stream::{Stream, StreamContext};

use super::decoder_layer::compute_layer_kinds;
use super::layer_cache::Qwen3_5LayerCache;
use super::model::Qwen35Inner;
use super::paged_forward::{
    PREFILL_CHUNK_BLOCKS, PREFILL_CHUNK_ROWS_OVERRIDE, PREFILL_TAIL_FULL,
    PREFILL_TAIL_WRITES_READY, PREFILL_TAILS_STORED, PrefillParts, PrefillTapSink,
    run_paged_prefill_chunk_with_hidden_with_size, run_paged_prefill_chunk_with_size,
};

const VOCAB: u32 = 16;

/// The int8-pool fixture of the DFlash2 tests (head 256, GDN heads 128)
/// with `layers` decoder layers and a random head; full attention at every
/// 4th layer.
pub(super) fn fixture(seed: u64, layers: i32) -> Qwen35Inner {
    let mut inner = super::model::scheduled_mtp::seeded_inner_with_config(seed, |config| {
        config.head_dim = 256;
        config.hidden_size = 256;
        config.kv_format = Some("int8".to_string());
        config.linear_key_head_dim = 128;
        config.linear_value_head_dim = 128;
        config.linear_num_key_heads = 2;
        config.linear_num_value_heads = 2;
        config.num_layers = layers;
    });
    let mut head = Linear::new(inner.config.hidden_size as u32, VOCAB, Some(false)).expect("head");
    head.set_weight(
        &MxArray::random_normal(
            &[i64::from(VOCAB), inner.config.hidden_size as i64],
            0.0,
            0.02,
            Some(DType::BFloat16),
        )
        .expect("head weights"),
    )
    .expect("install head");
    inner.lm_head = Some(LinearProj::Standard(head));
    inner
}

pub(super) fn prompt(len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| (i * 7 + 3) % VOCAB).collect()
}

pub(super) fn bits(array: &MxArray) -> Result<Vec<u32>> {
    let f = array.astype(DType::Float32)?;
    f.eval();
    Ok(f.to_float32()?.iter().map(|v| v.to_bits()).collect())
}

pub(super) fn gdn_bits(caches: &[Qwen3_5LayerCache]) -> Result<Vec<Vec<u32>>> {
    let mut out = Vec::new();
    for cache in caches {
        if let Qwen3_5LayerCache::Linear(arrays) = cache {
            for i in 0..2 {
                if let Some(array) = arrays.get(i) {
                    out.push(bits(array)?);
                }
            }
        }
    }
    Ok(out)
}

/// The bytes of every layer of every block the active request holds.
pub(super) fn request_pool_bytes(inner: &mut Qwen35Inner) -> Result<Vec<Vec<(Vec<u8>, Vec<u8>)>>> {
    let adapter = inner
        .paged_adapter
        .as_mut()
        .ok_or_else(|| Error::from_reason("no pool"))?;
    adapter
        .eval_pending_pool_writes()
        .map_err(Error::from_reason)?;
    crate::array::synchronize_and_clear_cache();
    let ids = adapter
        .block_table()
        .map(|table| table.block_ids())
        .unwrap_or_default();
    let pool = adapter.layer_kv_pool();
    ids.into_iter()
        .map(|id| pool.read_block_all_layers(id).map_err(Error::from_reason))
        .collect()
}

/// Fresh caches and a cold request with the prompt's blocks.
pub(super) fn begin_request(inner: &mut Qwen35Inner, prompt: &[u32]) -> Result<()> {
    inner.caches = Some(
        (0..inner.config.num_layers as usize)
            .map(|i| {
                if inner.config.is_linear_layer(i) {
                    Qwen3_5LayerCache::new_linear()
                } else {
                    Qwen3_5LayerCache::new_full_attention()
                }
            })
            .collect(),
    );
    let adapter = inner
        .paged_adapter
        .as_mut()
        .ok_or_else(|| Error::from_reason("no pool"))?;
    adapter
        .reset_for_new_request(0)
        .map_err(Error::from_reason)?;
    adapter
        .allocate_suffix_blocks(prompt.len() as u32)
        .map_err(Error::from_reason)?;
    Ok(())
}

/// Taps and the GDN cells at its capture rows, as a DFlash2 prefill hands
/// them over.
pub(super) struct RecordingSink {
    pub(super) taps: Vec<usize>,
    pub(super) captures: Vec<usize>,
    pub(super) seen: Vec<MxArray>,
    pub(super) linear: Vec<(usize, Vec<Qwen3_5LayerCache>)>,
}

impl PrefillTapSink for RecordingSink {
    fn tap_layers(&self) -> &[usize] {
        &self.taps
    }

    fn chunk(&mut self, _first_row: usize, _chunk: &[u32], taps: &[MxArray]) -> Result<()> {
        self.seen.extend(taps.iter().cloned());
        Ok(())
    }

    fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        out.extend(self.seen.iter());
    }

    fn capture_rows(&self) -> &[usize] {
        &self.captures
    }

    fn capture_linear(&mut self, row: usize, linear: Vec<Qwen3_5LayerCache>) {
        self.linear.push((row, linear));
    }
}

#[derive(Clone, Debug)]
struct Case {
    layers: i32,
    len: usize,
    rows: i32,
    /// Tap layers and capture rows of a tapped prefill.
    sink: Option<(Vec<usize>, Vec<usize>)>,
    /// The with-hidden driver keeping this many rows.
    keep: Option<Option<usize>>,
}

/// What a prefill leaves behind, and how it ran.
#[derive(Debug, PartialEq)]
struct Run {
    logits: Vec<u32>,
    hidden: Vec<u32>,
    pool: Vec<Vec<(Vec<u8>, Vec<u8>)>>,
    gdn: Vec<Vec<u32>>,
    taps: Vec<Vec<u32>>,
    captured: Vec<(usize, Vec<Vec<u32>>)>,
    checkpoints: Vec<(u32, Vec<Vec<u32>>)>,
}

struct Ran {
    run: Run,
    chunks: usize,
    stored: usize,
    ready: Vec<bool>,
}

fn run_case(case: &Case, full: bool) -> Result<Ran> {
    let mut inner = fixture(0xE1_0001, case.layers);
    let tokens = prompt(case.len);
    begin_request(&mut inner, &tokens)?;
    let kinds = compute_layer_kinds(inner.config.num_layers as usize, |i| {
        inner.config.is_linear_layer(i)
    });
    let mut sink = case.sink.as_ref().map(|(taps, captures)| RecordingSink {
        taps: taps.clone(),
        captures: captures.clone(),
        seen: Vec::new(),
        linear: Vec::new(),
    });
    PREFILL_TAIL_FULL.with(|flag| flag.set(full));
    PREFILL_TAILS_STORED.with(|count| count.set(0));
    PREFILL_TAIL_WRITES_READY.with(|log| log.borrow_mut().clear());
    PREFILL_CHUNK_BLOCKS.with(|log| *log.borrow_mut() = Some(Vec::new()));
    let embed = inner.embedding.clone();
    let prefilled = {
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
        let taps = sink.as_mut().map(|sink| sink as &mut dyn PrefillTapSink);
        match case.keep {
            None => run_paged_prefill_chunk_with_size(
                &tokens,
                &tokens,
                0,
                false,
                &embed,
                &mut target,
                &kinds,
                case.rows,
                0,
                None,
                taps,
            )
            .map(|(logits, checkpoints)| (logits, None, checkpoints)),
            Some(keep) => run_paged_prefill_chunk_with_hidden_with_size(
                &tokens,
                &tokens,
                0,
                false,
                &embed,
                &mut target,
                &kinds,
                case.rows,
                keep,
                0,
                None,
                taps,
            )
            .map(|(logits, hidden, checkpoints)| (logits, Some(hidden), checkpoints)),
        }
    };
    PREFILL_TAIL_FULL.with(|flag| flag.set(false));
    let chunks = PREFILL_CHUNK_BLOCKS
        .with(|log| log.borrow_mut().take())
        .map_or(0, |log| log.len());
    let stored = PREFILL_TAILS_STORED.with(std::cell::Cell::get);
    let ready = PREFILL_TAIL_WRITES_READY.with(|log| std::mem::take(&mut *log.borrow_mut()));
    let (logits, hidden, checkpoints) = prefilled?;
    let checkpoints = checkpoints
        .iter()
        .map(|point| Ok((point.prefix_len, gdn_bits(&point.caches)?)))
        .collect::<Result<Vec<_>>>()?;
    let (taps, captured) = match sink {
        Some(sink) => (
            sink.seen.iter().map(bits).collect::<Result<Vec<_>>>()?,
            sink.linear
                .iter()
                .map(|(row, linear)| Ok((*row, gdn_bits(linear)?)))
                .collect::<Result<Vec<_>>>()?,
        ),
        None => (Vec::new(), Vec::new()),
    };
    let run = Run {
        logits: bits(&logits)?,
        hidden: hidden.as_ref().map(bits).transpose()?.unwrap_or_default(),
        gdn: gdn_bits(inner.caches.as_deref().unwrap_or_default())?,
        pool: request_pool_bytes(&mut inner)?,
        taps,
        captured,
        checkpoints,
    };
    Ok(Ran {
        run,
        chunks,
        stored,
        ready,
    })
}

/// Runs `case` with every last layer in full and with the rule on; the two
/// must leave the same bits. Returns the chunk count and the store-only
/// tails the rule ran.
fn same_bits(case: &Case) -> Result<(usize, usize)> {
    let full = run_case(case, true)?;
    let tail = run_case(case, false)?;
    assert_eq!(full.stored, 0, "{case:?}: the full arm skipped a tail");
    assert_eq!(
        full.chunks, tail.chunks,
        "{case:?}: the chunk schedule moved"
    );
    assert!(
        tail.ready.iter().all(|&ready| ready),
        "{case:?}: a non-final chunk left the last layer's K/V write unevaluated: {:?}",
        tail.ready
    );
    let last_layer = |run: &Run| -> Vec<(Vec<u8>, Vec<u8>)> {
        run.pool
            .iter()
            .filter_map(|block| block.last().cloned())
            .collect()
    };
    assert_eq!(
        last_layer(&full.run),
        last_layer(&tail.run),
        "{case:?}: the last layer's stored K/V changed"
    );
    assert!(
        last_layer(&tail.run)
            .iter()
            .any(|(k, v)| k.iter().chain(v).any(|&b| b != 0)),
        "{case:?}: the last layer stored nothing"
    );
    assert!(full.run == tail.run, "{case:?}: the prefill's bits changed");
    Ok((tail.chunks, tail.stored))
}

/// 1, 2 and 4 chunks, without and with a draft's taps and a capture row,
/// and on the with-hidden driver: every non-final chunk stores its last
/// layer's K/V only, and the final logits, the stored K/V of every layer,
/// the GDN state, the GDN checkpoint, the taps and the captured GDN cells
/// keep their bits.
/// Mutations, each → red: the store-only tail writes no K/V; its pool
/// write is not an eval root. Mutation: the final chunk's tail is skipped
/// too → the prefill fails.
#[test]
fn non_final_chunks_store_the_last_layer_kv_only() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "non_final_chunks_store_the_last_layer_kv_only",
    ) {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let draft = Some((vec![1, 3], vec![24, 48]));
    let cases = [
        (12, 64, None, None, (1, 0)),
        (40, 64, None, None, (2, 1)),
        (100, 32, None, None, (4, 3)),
        (12, 64, draft.clone(), None, (1, 0)),
        (40, 64, draft.clone(), None, (2, 1)),
        (100, 32, draft.clone(), None, (4, 3)),
        (100, 32, None, Some(Some(1)), (4, 3)),
        (100, 32, None, Some(None), (4, 0)),
        (100, 32, draft, Some(Some(20)), (4, 3)),
    ];
    for (len, rows, sink, keep, want) in cases {
        let case = Case {
            layers: 8,
            len,
            rows,
            sink,
            keep,
        };
        let got = same_bits(&case)?;
        assert_eq!(got, want, "{case:?}: (chunks, store-only tails)");
    }
    Ok(())
}

/// The rule declines a last layer that is a tap or a GDN layer: no
/// store-only tail runs and the bits are those of the full run.
/// Mutation: a tapped last layer runs store-only → the prefill fails.
#[test]
fn a_tapped_or_gdn_last_layer_runs_in_full() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("a_tapped_or_gdn_last_layer_runs_in_full") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let tapped_last = Case {
        layers: 8,
        len: 100,
        rows: 32,
        sink: Some((vec![1, 7], vec![48])),
        keep: None,
    };
    assert_eq!(same_bits(&tapped_last)?, (4, 0), "{tapped_last:?}");
    let gdn_last = Case {
        layers: 7,
        len: 100,
        rows: 32,
        sink: Some((vec![1, 3], vec![48])),
        keep: None,
    };
    let full = run_case(&gdn_last, true)?;
    let tail = run_case(&gdn_last, false)?;
    assert_eq!((tail.chunks, tail.stored), (4, 0), "{gdn_last:?}");
    assert!(
        full.run == tail.run,
        "{gdn_last:?}: the prefill's bits changed"
    );
    Ok(())
}

/// A DFlash2 turn's tapped prefill through the model: the logits, the
/// draft context, the GDN cell, every pool byte and every state it stores
/// (as the SSD tier writes it) keep their bits with the store-only tails.
/// Mutation: the store-only tail writes no K/V → red.
#[test]
fn draft_prefill_states_keep_their_bytes() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("draft_prefill_states_keep_their_bytes") {
        return Ok(());
    }
    use crate::engine::dspark_turn::PagedDsparkBackend;
    use crate::models::qwen3_5::dflash2_state::disk::encode_anchor;
    use crate::models::qwen3_5::dflash2_state::disk::tests::{attach_tier, test_tier};
    let stream = Stream::generation();
    let _ctx = StreamContext::new(stream);
    let tokens = prompt(100);
    let run = |full: bool| -> Result<(Run, Vec<(usize, Vec<u8>)>, usize)> {
        let tier = test_tier("e1-draft-states", 1 << 30);
        let mut inner =
            super::dflash2_decode::paged_dflash2_tests::tiny_paged_int8_dflash_inner(0xE1_0002);
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
            &tier,
            window.nbytes(),
            window.slots(),
        );
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        PREFILL_TAIL_FULL.with(|flag| flag.set(full));
        PREFILL_TAILS_STORED.with(|count| count.set(0));
        let logits = (|| -> Result<MxArray> {
            let prefix = inner.prime_dspark_prefix_state(&tokens, true, 0)?;
            inner.paged_prefill_with_draft_state(&tokens, &prefix, stream)
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        PREFILL_TAIL_FULL.with(|flag| flag.set(false));
        let stored = PREFILL_TAILS_STORED.with(std::cell::Cell::get);
        let logits = bits(&logits?)?;
        let states = inner
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
            .collect::<Vec<_>>();
        let mut context = Vec::new();
        if let Some(state) = inner.dflash2_turn_state.as_ref() {
            let mut arrays = Vec::new();
            state.context.collect_arrays(&mut arrays);
            for array in arrays {
                context.push(bits(array)?);
            }
        }
        let run = Run {
            logits,
            hidden: Vec::new(),
            gdn: gdn_bits(inner.caches.as_deref().unwrap_or_default())?,
            pool: request_pool_bytes(&mut inner)?,
            taps: context,
            captured: Vec::new(),
            checkpoints: Vec::new(),
        };
        Ok((run, states, stored))
    };
    let (full, full_states, full_stored) = run(true)?;
    let (tail, tail_states, tail_stored) = run(false)?;
    assert_eq!(full_stored, 0);
    assert!(tail_stored > 0, "the draft prefill ran no store-only tail");
    assert!(!tail_states.is_empty(), "the prefill stored no state");
    assert!(!tail.taps.is_empty(), "the turn kept no draft context");
    assert_eq!(full_states, tail_states, "a stored state's bytes changed");
    assert!(full == tail, "the draft prefill's bits changed");
    Ok(())
}
