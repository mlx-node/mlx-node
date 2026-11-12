use super::*;
use crate::models::qwen3_5::dflash2_state::disk::poll_offload;
use crate::models::qwen3_5::dflash2_state::disk::tests::{
    TestTier, disk_copy, drained, fixture_on, kv_key, park_blocks, persistent_test_tier, ram,
    test_tier,
};
use crate::models::qwen3_5::dflash2_state::store::StateClass;
use crate::models::qwen3_5::model::Qwen35Inner;
use mlx_paged_attn::ColdGroup;

fn snapshot(inner: &Qwen35Inner) -> WriteBehindSnapshot {
    inner
        .dflash2_states
        .disk
        .as_ref()
        .expect("tier")
        .write_behind
        .snapshot()
}

fn stopped(inner: &Qwen35Inner) -> Stop {
    inner
        .dflash2_states
        .disk
        .as_ref()
        .expect("tier")
        .write_behind
        .stopped()
}

fn wakeup(inner: &Qwen35Inner) -> Option<f64> {
    let states = &inner.dflash2_states;
    let disk = states.disk.as_ref().expect("tier");
    disk.write_behind
        .next_wakeup(&states.store, disk.tier.persistent)
}

fn enqueued(tier: &TestTier) -> u64 {
    tier.tier.manager.stats().enqueued
}

/// The blocks of a point deep enough to be copied.
fn point_blocks(inner: &Qwen35Inner) -> u32 {
    let block_size = inner.paged_adapter.as_ref().expect("pool").block_size();
    assert!(block_size < MINIMUM_TOKENS);
    MINIMUM_TOKENS.div_ceil(block_size) + 1
}

fn depth(inner: &Qwen35Inner, id: EntryId) -> u32 {
    inner
        .dflash2_states
        .store
        .get(id)
        .expect("the point")
        .depth() as u32
}

fn publish(inner: &mut Qwen35Inner, id: EntryId, now: f64) {
    let tokens = depth(inner, id);
    published(&mut inner.dflash2_states, id, tokens, now);
}

/// Splash `WriteBehind`: a persistent tier copies an ordinary point deep
/// enough once it waited the delay, its state first and then its chain,
/// the RAM copy kept, and counts it durable once all of it landed; a
/// shallow point and a checkpoint are never queued; a superseded point is
/// unneeded; nothing is copied while a claim waits. A temporary tier
/// copies only while the demand for copies exceeds what the units that
/// free at once hold (a free extent covers it alone), and skips a point
/// whose turn has not ended. The next wakeup is the due time while copies
/// are made, none once a step stalled on a landing, was covered or met a
/// waiting claim.
/// Mutations WB1 (the temporary tier copies without demand) and NW2 (the
/// next wakeup is the due time while stalled) → red.
#[test]
fn write_behind_copies_points_like_splash() {
    let Some((mut inner, tier)) = fixture_on(
        0x5713_1801,
        32,
        32,
        persistent_test_tier("s13-wb-persistent", 1 << 30),
    ) else {
        return;
    };
    let blocks = point_blocks(&inner);
    let p = park_blocks(&mut inner, 81, blocks, StateClass::Ordinary);
    publish(&mut inner, p, 0.0);
    let shallow = park_blocks(&mut inner, 82, 1, StateClass::Ordinary);
    publish(&mut inner, shallow, 0.0);
    let checkpoint = park_blocks(&mut inner, 83, blocks, StateClass::Checkpoint);
    publish(&mut inner, checkpoint, 0.0);
    assert_eq!(snapshot(&inner).waiting, 1, "only the deep ordinary point");
    let written = enqueued(&tier);

    assert!(!run(&mut inner, DELAY_SECONDS - 0.1, 0, false));
    assert!(disk_copy(&inner, p).is_none() && enqueued(&tier) == written);
    assert_eq!(wakeup(&inner), Some(DELAY_SECONDS));

    assert!(!run(&mut inner, DELAY_SECONDS, 0, true));
    assert_eq!(stopped(&inner), Stop::ClaimWaiting);
    assert_eq!(wakeup(&inner), None);
    assert!(disk_copy(&inner, p).is_none());

    assert!(!run(&mut inner, DELAY_SECONDS, 0, false));
    assert!(disk_copy(&inner, p).is_some_and(|copy| !copy.landed));
    assert!(ram(&inner, p), "a copy ahead keeps the RAM copy");
    assert_eq!(
        enqueued(&tier),
        written + 1 + u64::from(blocks),
        "the state, then its chain"
    );
    assert_eq!(stopped(&inner), Stop::Stalled);
    assert_eq!(wakeup(&inner), None, "its landings wake the model");
    drained(&tier);
    assert!(poll_offload(&mut inner));
    assert!(run(&mut inner, DELAY_SECONDS, 0, false));
    assert_eq!(snapshot(&inner).durable, 1);
    assert_eq!(snapshot(&inner).waiting, 0);
    let copy = disk_copy(&inner, p).expect("a disk copy");
    assert!(copy.landed && !copy.chain_owed && ram(&inner, p));
    assert!(
        tier.tier
            .manager
            .contains_in(&copy.key, ColdGroup::DFlash2State)
    );
    for index in 0..blocks as usize {
        assert!(
            tier.tier
                .manager
                .contains_in(&kv_key(&inner, &tier, p, index), ColdGroup::Kv)
        );
    }

    // Its conversation went on through one next block.
    let base = park_blocks(&mut inner, 84, blocks, StateClass::Ordinary);
    publish(&mut inner, base, 20.0);
    park_blocks(&mut inner, 84, blocks + 1, StateClass::Ordinary);
    assert!(run(&mut inner, 20.0 + DELAY_SECONDS, 0, false));
    assert_eq!(snapshot(&inner).unneeded, 1);
    assert!(disk_copy(&inner, base).is_none());
    drop(inner);
    drop(tier);

    // A temporary tier, its pool able to grow by an extent.
    let Some((mut inner, tier)) = fixture_on(0x5713_1802, 8, 32, test_tier("s13-wb-temp", 1 << 30))
    else {
        return;
    };
    let blocks = point_blocks(&inner);
    let p = park_blocks(&mut inner, 91, blocks, StateClass::Ordinary);
    publish(&mut inner, p, 0.0);
    let written = enqueued(&tier);
    assert!(!run(&mut inner, DELAY_SECONDS, 0, false));
    assert!(disk_copy(&inner, p).is_none(), "no copy without demand");
    assert_eq!(wakeup(&inner), None, "copies wait for demand");

    let extent = {
        let adapter = inner.paged_adapter.as_mut().expect("pool");
        adapter.set_extent_growth(4);
        assert!(adapter.try_grow_pool(8));
        adapter.free_extent_bytes(true)
    };
    assert!(extent > 0);
    assert!(!run(&mut inner, DELAY_SECONDS, extent, false));
    assert_eq!(stopped(&inner), Stop::Covered);
    assert_eq!(wakeup(&inner), None);
    assert!(disk_copy(&inner, p).is_none());

    assert!(!run(&mut inner, DELAY_SECONDS / 2.0, u64::MAX, false));
    assert_eq!(
        wakeup(&inner),
        Some(DELAY_SECONDS),
        "due later while copying"
    );

    let mark = inner.dflash2_states.store.get(p).expect("p").mark();
    inner.dflash2_states.store.use_state(mark);
    assert!(!run(&mut inner, DELAY_SECONDS, u64::MAX, false));
    assert!(disk_copy(&inner, p).is_none(), "a point in use waits");
    assert_eq!(wakeup(&inner), None);
    inner.dflash2_states.store.end_use(mark);

    assert!(!run(&mut inner, DELAY_SECONDS, u64::MAX, false));
    assert!(disk_copy(&inner, p).is_some_and(|copy| !copy.landed));
    assert_eq!(enqueued(&tier), written + 1 + u64::from(blocks));
    assert_eq!(stopped(&inner), Stop::Stalled);
}

/// Splash `closePersistentCache` with `WriteBehind::flush`: at a clean
/// stop a persistent tier writes every waiting point, none of them due,
/// newest first, each with its chain, before the deadline; a temporary
/// tier writes nothing, its directory goes at once, and it takes no write
/// afterwards.
/// Mutation FL3 (the flush reports done without writing) → red.
#[test]
fn close_at_stop_writes_every_waiting_point() {
    use crate::models::qwen3_5::dflash2_state::disk::close_cache;
    let deadline = || std::time::Instant::now() + std::time::Duration::from_secs(6);
    let Some((mut inner, tier)) = fixture_on(
        0x5713_2501,
        32,
        32,
        persistent_test_tier("s13-close-persistent", 1 << 30),
    ) else {
        return;
    };
    let blocks = point_blocks(&inner);
    let points: Vec<EntryId> = (0..8)
        .map(|seed| park_blocks(&mut inner, 120 + seed, blocks, StateClass::Ordinary))
        .collect();
    for &p in &points {
        publish(&mut inner, p, 0.0);
    }
    assert_eq!(snapshot(&inner).waiting, 8);
    assert!(close_cache(&mut inner, deadline()));
    assert_eq!(snapshot(&inner).durable, 8);
    assert_eq!(snapshot(&inner).waiting, 0);
    let mut keys = Vec::new();
    for &p in &points {
        let copy = disk_copy(&inner, p).expect("a disk copy");
        assert!(copy.landed && !copy.chain_owed);
        assert!(
            tier.tier
                .manager
                .contains_in(&copy.key, ColdGroup::DFlash2State)
        );
        for index in 0..blocks as usize {
            assert!(
                tier.tier
                    .manager
                    .contains_in(&kv_key(&inner, &tier, p, index), ColdGroup::Kv)
            );
        }
        keys.push(copy.key);
    }
    keys.reverse();
    let disk = inner.dflash2_states.disk.as_ref().expect("tier");
    assert_eq!(disk.landed_states, keys, "newest first");
    assert!(!disk.writable());
    drop(inner);
    drop(tier);

    let Some((mut inner, tier)) =
        fixture_on(0x5713_2502, 32, 32, test_tier("s13-close-temp", 1 << 30))
    else {
        return;
    };
    let p = park_blocks(&mut inner, 130, blocks, StateClass::Ordinary);
    publish(&mut inner, p, 0.0);
    let written = enqueued(&tier);
    let dir = tier.tier.dir();
    assert!(dir.exists());
    assert!(close_cache(&mut inner, std::time::Instant::now()));
    assert_eq!(enqueued(&tier), written, "a temporary tier writes nothing");
    assert!(!dir.exists(), "its directory went");
    assert!(!inner.dflash2_states.disk.as_ref().expect("tier").writable());
    assert!(disk_copy(&inner, p).is_none() && ram(&inner, p));
}
