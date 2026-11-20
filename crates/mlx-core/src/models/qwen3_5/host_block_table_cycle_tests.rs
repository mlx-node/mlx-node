//! A check of the host-built varlen block table on verify
//! cycles that keep every count a cycle can keep (1 to 8 of its 8 rows),
//! against the order that reshaped and evaluated the table.

use super::*;
use crate::transformer::paged_metadata_cache::RESHAPED_BLOCK_TABLE;

/// Greedy device cycles whose driver accepts 0 to 7 of the 7 drafts (so each
/// cycle rolls back 7 to 0 of its 8 booked rows, across block edges) keep
/// the reshaped table's bits: pool records, logits, taps, keeps, tokens,
/// every pool byte and the GDN state. A cycle that accepts every draft
/// decodes the autoregressive tokens.
/// Mutation v-h1-seqlens (seq_lens holds the committed rows only) turns it
/// red.
#[test]
fn cycles_of_every_keep_have_the_reshaped_table_bits() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "cycles_of_every_keep_have_the_reshaped_table_bits",
    ) || !super::super::tests::metal_is_default_device()
    {
        return Ok(());
    }
    let stream = Stream::generation();
    let _ctx = StreamContext::new(stream);
    let n = 40;
    let mut ar_inner = tiny_paged_int8_dflash_inner(SEED);
    let (reference, _) = paged_ar(&mut ar_inner, &prompt(40), n)?;
    let mut kept = std::collections::BTreeSet::new();
    for correct in 0..=7 {
        let run = |reshaped: bool| {
            RESHAPED_BLOCK_TABLE.with(|flag| flag.set(reshaped));
            let cycles = device_cycles(false, correct, None, &reference);
            RESHAPED_BLOCK_TABLE.with(|flag| flag.set(false));
            cycles
        };
        let before = run(true)?;
        let after = run(false)?;
        let tag = format!("{correct} accepted drafts");
        assert!(!before.pool.is_empty(), "{tag}: the request holds blocks");
        assert_eq!(before.recorded, after.recorded, "{tag}: pool records");
        assert_eq!(before.logits, after.logits, "{tag}: logits");
        assert_eq!(before.taps, after.taps, "{tag}: taps");
        assert_eq!(before.keeps, after.keeps, "{tag}: keeps");
        assert_eq!(before.tokens, after.tokens, "{tag}: tokens");
        assert!(before.pool == after.pool, "{tag}: pool bytes");
        assert_eq!(before.gdn, after.gdn, "{tag}: GDN state");
        assert_eq!(
            after.tokens, reference,
            "{tag}: not the autoregressive tokens"
        );
        kept.extend(after.keeps.iter().copied());
    }
    assert_eq!(
        kept.into_iter().collect::<Vec<_>>(),
        (1..=8).collect::<Vec<_>>(),
        "the cycles did not keep every count from 1 to 8"
    );
    Ok(())
}
