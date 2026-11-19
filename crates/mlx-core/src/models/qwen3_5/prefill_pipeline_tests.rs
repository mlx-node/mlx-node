//! A tapped chunk of the chunked paged prefill evaluates its sink's arrays in
//! the chunk's own eval. Everything the prefill leaves behind keeps the bits
//! of the order that evaluated the sink apart.

use napi::{Error, Result};

use crate::array::MxArray;
use crate::stream::{Stream, StreamContext};

use super::paged_forward::{PREFILL_SERIAL, PREFILL_SINK_EVALS, PREFILL_SINK_READY};
use super::prefill_tail_tests::{bits, gdn_bits, prompt, request_pool_bytes};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    Serial,
    Merged,
}

/// What a prefill leaves behind.
#[derive(Debug, PartialEq)]
struct Left {
    gdn: Vec<Vec<u32>>,
    pool: Vec<Vec<(Vec<u8>, Vec<u8>)>>,
}

/// A real DFlash2 tapped prefill (16-row chunks, SSD tier attached): its
/// sink's draft context and lane window are evaluated in each chunk's eval;
/// the logits, the draft context, the GDN cell, every pool byte and every
/// stored state as the tier encodes it keep the serial order's bytes.
/// Mutation: the chunk's eval leaves the sink's arrays out → red.
#[test]
fn draft_prefill_states_keep_their_bytes_with_one_eval_per_chunk() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "draft_prefill_states_keep_their_bytes_with_one_eval_per_chunk",
    ) {
        return Ok(());
    }
    use super::paged_forward::PREFILL_CHUNK_ROWS_OVERRIDE;
    use crate::engine::dspark_turn::PagedDsparkBackend;
    use crate::models::qwen3_5::dflash2_state::disk::encode_anchor;
    use crate::models::qwen3_5::dflash2_state::disk::tests::{attach_tier, test_tier};
    let stream = Stream::generation();
    let _ctx = StreamContext::new(stream);
    let tokens = prompt(100);
    type Stored = Vec<(usize, Vec<u8>)>;
    let run = |order: Order| -> Result<(Vec<u32>, Vec<Vec<u32>>, Left, Stored, usize)> {
        let tier = test_tier("e0-draft-states", 1 << 30);
        let mut inner =
            super::dflash2_decode::paged_dflash2_tests::tiny_paged_int8_dflash_inner(0xE0_0002);
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
        PREFILL_SERIAL.with(|flag| flag.set(order == Order::Serial));
        PREFILL_SINK_EVALS.with(|count| count.set(0));
        PREFILL_SINK_READY.with(|log| log.borrow_mut().clear());
        let logits = (|| -> Result<MxArray> {
            let prefix = inner.prime_dspark_prefix_state(&tokens, true, 0)?;
            inner.paged_prefill_with_draft_state(&tokens, &prefix, stream)
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        PREFILL_SERIAL.with(|flag| flag.set(false));
        let sink_evals = PREFILL_SINK_EVALS.with(std::cell::Cell::get);
        let sink_ready = PREFILL_SINK_READY.with(|log| std::mem::take(&mut *log.borrow_mut()));
        assert!(
            !sink_ready.is_empty() && sink_ready.iter().all(|&ready| ready),
            "{order:?}: a chunk left its sink's arrays unevaluated: {sink_ready:?}"
        );
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
        let left = Left {
            gdn: gdn_bits(inner.caches.as_deref().unwrap_or_default())?,
            pool: request_pool_bytes(&mut inner)?,
        };
        Ok((logits, context, left, states, sink_evals))
    };
    let serial = run(Order::Serial)?;
    let merged = run(Order::Merged)?;
    assert!(serial.4 > 0, "the serial order ran no separate sink eval");
    assert_eq!(merged.4, 0, "a chunk evaluated its sink apart");
    assert!(!serial.3.is_empty(), "the prefill stored no state");
    assert!(!serial.1.is_empty(), "the turn kept no draft context");
    assert_eq!(serial.0, merged.0, "the logits changed");
    assert_eq!(serial.1, merged.1, "the draft context changed");
    assert!(serial.2 == merged.2, "the pool or the GDN cell changed");
    assert_eq!(serial.3, merged.3, "a stored state's bytes changed");
    Ok(())
}
