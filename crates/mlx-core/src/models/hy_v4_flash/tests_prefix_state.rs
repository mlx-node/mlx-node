//! States at block rows: the rows a turn stores (fixture block-rows.json),
//! the T candidate, the RAM bound, the cut rule and shared bytes. CPU only.

use std::path::PathBuf;

use serde_json::Value;

use super::ladder::{Kind, RAM_STATES};
use super::prefill::{BLOCK_ROWS, Grid, StateRow, turn_row};
use super::prefix_state::{Key, PrefixState, States, Stored, Turn};

const OPEN: [u32; 3] = [120_000, 611, 10_372];
const GEN: [u32; 5] = [120_000, 611, 10_372, 120_001, 120_029];
const SALT: u64 = 7;

fn fixture() -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash/block-rows.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn u(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

/// `len` tokens with the generation prompt at the end.
fn prompt_of(len: usize) -> Vec<u32> {
    let mut p: Vec<u32> = (0..len).map(|i| 7 + (i % 97) as u32).collect();
    p[len - GEN.len()..].copy_from_slice(&GEN);
    p
}

fn bare(row: usize) -> PrefixState {
    PrefixState {
        row,
        layers: Vec::new(),
        draft: None,
    }
}

/// A turn that prefills `prompt_len` rows from `start` and decodes to
/// `kv_end`, holding a state at every block row it reaches past the
/// prompt's state rows; its stored rows and its T.
fn run_turn(g: &Grid, start: usize, prompt_len: usize, kv_end: usize) -> Vec<StateRow> {
    let p = prompt_of(prompt_len);
    let mut turn = Turn::new(g, start, &p);
    let mut rows = turn.rows().to_vec();
    let first = prompt_len.next_multiple_of(BLOCK_ROWS);
    for row in (first..=kv_end).step_by(BLOCK_ROWS) {
        turn.hold(bare(row));
    }
    if let Some((t, s)) = turn.close(kv_end) {
        assert_eq!(t.row, s.row);
        rows.push(t);
    }
    rows
}

/// Every block-rows.json sample: checkpoints every 8,192 below 32,768 and
/// every 32,768 above, P, then T = the block row below the KV end, not
/// stored again when it is P.
/// Mutation: `Turn::close` keeps `kv_end` instead of `turn_row` → red.
#[test]
fn state_rows_follow_the_block_rows_fixture() {
    let f = fixture();
    for s in f["samples"].as_array().unwrap() {
        let name = s["name"].as_str().unwrap();
        let g = Grid::new(u(&s["chunk"]), OPEN.to_vec()).unwrap();
        let got = run_turn(&g, u(&s["start"]), u(&s["prompt_len"]), u(&s["kv_end"]));
        let want: Vec<StateRow> = s["states"]
            .as_array()
            .unwrap()
            .iter()
            .map(|st| StateRow {
                row: u(&st["row"]),
                kind: match st["kind"].as_str().unwrap() {
                    "checkpoint" => Kind::Checkpoint,
                    "P" => Kind::Prompt,
                    "T" => Kind::Turn,
                    k => panic!("{name}: kind {k}"),
                },
            })
            .collect();
        assert_eq!(got, want, "{name}");
        assert!(got.iter().all(|r| r.row.is_multiple_of(BLOCK_ROWS)));
    }
}

/// T: a state at or below the prompt's last state row is not held, an
/// older one never replaces a newer one, a cut drops a held state past
/// it, and a held state past `turn_row(kv_end)` is not T; a commit that
/// stepped over that row leaves the deepest one below.
#[test]
fn turn_candidate_rules() {
    let g = Grid::new(8192, OPEN.to_vec()).unwrap();
    let p = prompt_of(900);
    let mut t = Turn::new(&g, 0, &p);
    assert_eq!(t.prompt_kind(864), Some(Kind::Prompt));
    t.hold(bare(864));
    assert_eq!(t.held_row(), None);
    t.hold(bare(960));
    t.hold(bare(928));
    assert_eq!(t.held_row(), Some(960));
    t.cut(950);
    assert_eq!(t.held_row(), None);
    t.hold(bare(992));
    assert!(t.close(980).is_none());
    t.hold(bare(1024));
    let (row, _) = t.close(1100).unwrap();
    assert_eq!(row.row, 1024);
    assert_ne!(row.row, turn_row(1100));
    assert_eq!(row.kind, Kind::Turn);
    assert_eq!(t.held_row(), None);
    assert!(t.wants(896));
    assert!(!t.wants(864));
}

struct Fake {
    own: u64,
    shared: Vec<(usize, u64)>,
}

impl Stored for Fake {
    fn bytes(&self) -> u64 {
        self.own + self.shared.iter().map(|s| s.1).sum::<u64>()
    }

    fn shared(&self) -> Vec<(usize, u64)> {
        self.shared.clone()
    }
}

fn fake(own: u64) -> Fake {
    Fake {
        own,
        shared: Vec::new(),
    }
}

/// `blocks` whole blocks of a lineage numbered `n`.
fn lineage(n: u32, blocks: usize) -> Vec<u32> {
    (0..blocks * BLOCK_ROWS)
        .map(|i| n * 1000 + i as u32)
        .collect()
}

fn publish(st: &mut States<Fake>, tokens: &[u32], kind: Kind, own: u64) -> Option<u64> {
    st.publish(Key::text(tokens, SALT), fake(own), kind)
        .unwrap()
}

fn hit(st: &States<Fake>, prompt: &[u32], cached: bool) -> Option<usize> {
    st.lookup(Key::text(prompt, SALT), &|_| cached).map(|h| h.1)
}

fn longer(tokens: &[u32]) -> Vec<u32> {
    let mut p = tokens.to_vec();
    p.push(9);
    p
}

/// Nine states: the oldest RAM copy goes, eight stay.
/// Mutation: the bound check uses `>` → red (nine kept).
#[test]
fn ram_bound_keeps_the_newest_eight() {
    let mut st = States::<Fake>::new();
    let all: Vec<Vec<u32>> = (1..=RAM_STATES as u32 + 1).map(|n| lineage(n, 1)).collect();
    for l in &all {
        assert!(publish(&mut st, l, Kind::Prompt, 10).is_some());
    }
    assert_eq!(st.ram_len(), RAM_STATES);
    assert_eq!(hit(&st, &longer(&all[0]), true), None);
    for l in &all[1..] {
        assert_eq!(hit(&st, &longer(l), true), Some(BLOCK_ROWS));
    }
}

/// A checkpoint goes before an older ordinary state.
#[test]
fn a_checkpoint_goes_first() {
    let mut st = States::<Fake>::with_cap(2);
    let (a, b, c) = (lineage(1, 1), lineage(2, 1), lineage(3, 1));
    publish(&mut st, &a, Kind::Prompt, 1);
    publish(&mut st, &b, Kind::Checkpoint, 1);
    publish(&mut st, &c, Kind::Turn, 1);
    assert_eq!(hit(&st, &longer(&a), true), Some(BLOCK_ROWS));
    assert_eq!(hit(&st, &longer(&b), true), None);
    assert_eq!(hit(&st, &longer(&c), true), Some(BLOCK_ROWS));
}

/// A cut below a state drops it unless its blocks stay cached; shallower
/// states stay.
/// Mutation: `cut` keeps every state → red.
#[test]
fn a_cut_drops_the_states_past_it() {
    let tokens = lineage(4, 3);
    let fill = |st: &mut States<Fake>| {
        for b in 1..=3 {
            publish(st, &tokens[..b * BLOCK_ROWS], Kind::Checkpoint, 1);
        }
    };
    let mut st = States::<Fake>::new();
    fill(&mut st);
    st.cut(&tokens, SALT, 40, &|_| false);
    assert_eq!(st.ram_rows(), vec![BLOCK_ROWS]);
    let mut kept = States::<Fake>::new();
    fill(&mut kept);
    kept.cut(&tokens, SALT, 40, &|_| true);
    assert_eq!(kept.ram_rows(), vec![32, 64, 96]);
    let mut other = States::<Fake>::new();
    fill(&mut other);
    other.cut(&tokens, SALT + 1, 0, &|_| false);
    assert_eq!(other.ram_rows(), vec![32, 64, 96]);
}

/// The deepest strict prefix with its blocks cached; none when the pool
/// lost them.
#[test]
fn lookup_takes_the_deepest_cached_prefix() {
    let tokens = lineage(5, 2);
    let mut st = States::<Fake>::new();
    publish(&mut st, &tokens[..BLOCK_ROWS], Kind::Checkpoint, 1);
    publish(&mut st, &tokens, Kind::Prompt, 1);
    assert_eq!(hit(&st, &longer(&tokens), true), Some(64));
    assert_eq!(hit(&st, &tokens, true), Some(32));
    let mut fork = tokens[..40].to_vec();
    fork.extend([1, 2, 3]);
    assert_eq!(hit(&st, &fork, true), Some(32));
    assert_eq!(hit(&st, &longer(&tokens), false), None);
    st.prune(&|_| false);
    assert_eq!(st.ram_len(), 0);
}

/// A reclaim gives up RAM copies in eviction order (checkpoints first, then
/// the least recently used) only until the shortfall is free.
/// Mutation: `reclaim` clears every state → red.
#[test]
fn reclaim_frees_the_shortfall_only() {
    let mut st = States::<Fake>::new();
    for n in 1..=4 {
        publish(&mut st, &lineage(n, 1), Kind::Prompt, 100 * u64::from(n)).unwrap();
    }
    assert_eq!(st.held_bytes(), 1000);
    assert_eq!(st.reclaim(0), 0);
    assert_eq!(st.reclaim(150), 300);
    assert_eq!(st.ram_len(), 2);
    assert_eq!(st.held_bytes(), 700);
    assert_eq!(st.reclaim(10_000), 700);
    assert_eq!(st.ram_len(), 0);
    assert_eq!(st.reclaim(1), 0);
}

/// Shared parts count once in the held bytes; off-block rows are refused.
#[test]
fn held_bytes_count_shared_parts_once() {
    let mut st = States::<Fake>::new();
    for (n, own) in [(6, 10), (7, 20)] {
        let s = Fake {
            own,
            shared: vec![(1, 100), (n as usize, 5)],
        };
        st.publish(Key::text(&lineage(n, 1), SALT), s, Kind::Prompt)
            .unwrap();
    }
    assert_eq!(st.held_bytes(), 10 + 20 + 100 + 5 + 5);
    let off = lineage(8, 1);
    assert!(
        st.publish(Key::text(&off[..31], SALT), fake(1), Kind::Prompt)
            .is_err()
    );
}

/// Ordinary eviction follows pure LRU (checkpoints still go first): with 7
/// idle short non-durable states and 1 hot durable state `A` in an 8-slot
/// store, promoting a second hot durable state `B` from disk evicts the oldest
/// idle short state rather than `A`, so alternating between `A` and `B` stays
/// in RAM instead of thrashing SSD.
#[test]
fn lru_evicts_idle_short_state_instead_of_thrashing_hot_durable_prefixes() {
    use crate::disk_tier::store::{DiskCopy, Lineage, StateClass};
    use mlx_paged_attn::{ColdCacheFingerprint, ColdCacheKey, ColdGroup};

    let disk_copy = |seed: u64| DiskCopy {
        key: ColdCacheKey::chain(
            ColdGroup::DFlash2State,
            ColdCacheFingerprint::from_components([&seed.to_le_bytes()[..]]),
            None,
            &[seed as u32],
            &[],
            0,
            0,
        ),
        bytes: 64,
        landed: true,
        window_begin: 0,
        chain_owed: false,
    };

    let mut st = States::<Fake>::with_cap(RAM_STATES);
    let idle: Vec<Vec<u32>> = (1..=7u32).map(|n| lineage(n, 1)).collect();
    let idle_ids: Vec<u64> = idle
        .iter()
        .map(|l| publish(&mut st, l, Kind::Prompt, 10).unwrap())
        .collect();
    let a = lineage(100, 1);
    let b = lineage(101, 1);
    let id_a = publish(&mut st, &a, Kind::Prompt, 10).unwrap();
    st.store_mut().set_disk(id_a, Some(disk_copy(1)));
    assert_eq!(st.ram_len(), RAM_STATES);

    // Park a second durable state `B` as disk-only (as after adoption or drop_ram).
    let id_b = st
        .store_mut()
        .insert_disk(
            Lineage::text(&b, SALT),
            10,
            StateClass::Ordinary,
            disk_copy(2),
        )
        .unwrap();
    st.touch(id_a);
    assert!(st.get(id_a).is_some() && st.get(id_b).is_none());

    // Promoting `B` evicts the oldest idle short state (`idle_ids[0]`), NOT hot `A`.
    assert!(st.promote(id_b, fake(10)));
    assert_eq!(st.ram_len(), RAM_STATES);
    assert!(
        st.get(idle_ids[0]).is_none(),
        "oldest idle short state goes"
    );
    assert!(st.get(id_a).is_some(), "hot durable `A` stays in RAM");
    assert!(st.get(id_b).is_some(), "promoted durable `B` is in RAM");

    // Alternating lookups between `A` and `B` hit RAM without SSD thrashing.
    for _ in 0..4 {
        assert_eq!(
            st.lookup(Key::text(&longer(&a), SALT), &|_| true),
            Some((id_a, BLOCK_ROWS))
        );
        st.touch(id_a);
        assert_eq!(
            st.lookup(Key::text(&longer(&b), SALT), &|_| true),
            Some((id_b, BLOCK_ROWS))
        );
        st.touch(id_b);
        assert!(
            st.disk_only_above(Key::text(&longer(&a), SALT), 0)
                .is_empty()
        );
        assert!(
            st.disk_only_above(Key::text(&longer(&b), SALT), 0)
                .is_empty()
        );
    }
}

/// Multi-lane RAM state budget enforcement: `reserve_bytes`,
/// `reserve_for_promote`, `promote_within`, and `publish_within` bound
/// resident bytes (counting shared segments once), evict LRU entries *before*
/// decoding from SSD, and refuse an entry that cannot fit without wiping out
/// healthy RAM states.
#[test]
fn byte_cap_enforces_pre_decode_reservation_and_unshared_mtp_segment_budget() {
    use crate::disk_tier::store::{DiskCopy, Lineage, StateClass};
    use mlx_paged_attn::{ColdCacheFingerprint, ColdCacheKey, ColdGroup};

    let disk_copy = |seed: u64| DiskCopy {
        key: ColdCacheKey::chain(
            ColdGroup::DFlash2State,
            ColdCacheFingerprint::from_components([&seed.to_le_bytes()[..]]),
            None,
            &[seed as u32],
            &[],
            0,
            0,
        ),
        bytes: 64,
        landed: true,
        window_begin: 0,
        chain_owed: false,
    };
    let mtp_fake = |own: u64, seg_id: usize, seg_bytes: u64| Fake {
        own,
        shared: vec![(seg_id, seg_bytes)],
    };

    // 4-lane slot cap (32) with a tight byte cap (300 B) where each unshared
    // MTP state costs 100 B (10 B own + 90 B unshared segments).
    let mut st = States::<Fake>::with_cap(RAM_STATES * 4);
    let mut ids = Vec::new();
    for i in 1..=3u32 {
        let id = st
            .publish_within(
                Key::text(&lineage(i, 1), SALT),
                mtp_fake(10, i as usize, 90),
                Kind::Prompt,
                Some(300),
            )
            .unwrap()
            .unwrap();
        ids.push(id);
    }
    assert_eq!(st.ram_len(), 3);
    assert_eq!(st.held_bytes(), 300);

    // Sharing an existing segment (`seg_id = 3`) only adds `own` (10 B), so
    // `added_bytes` deduplicates it and fits under a 310 B cap without evicting.
    let shared_state = mtp_fake(10, 3, 90);
    assert_eq!(st.added_bytes(&shared_state), 10);
    let shared_id = st
        .publish_within(
            Key::text(&lineage(4, 1), SALT),
            shared_state,
            Kind::Prompt,
            Some(310),
        )
        .unwrap()
        .unwrap();
    assert_eq!(st.ram_len(), 4);
    assert_eq!(st.held_bytes(), 310);

    // Park two disk-only MTP states: one that fits after evicting LRU entries
    // (`disk_fit`, 100 B unshared) and one larger than the entire byte cap
    // (`disk_huge`, 400 B unshared).
    let disk_fit = st
        .store_mut()
        .insert_disk(
            Lineage::text(&lineage(10, 1), SALT),
            100,
            StateClass::Ordinary,
            disk_copy(10),
        )
        .unwrap();
    let disk_huge = st
        .store_mut()
        .insert_disk(
            Lineage::text(&lineage(11, 1), SALT),
            400,
            StateClass::Ordinary,
            disk_copy(11),
        )
        .unwrap();

    // Oversized SSD candidate is rejected pre-decode WITHOUT evicting any RAM entry.
    assert!(!st.reserve_bytes(400, 300));
    assert!(!st.reserve_for_promote(disk_huge, 400, 300, None));
    assert_eq!(st.ram_len(), 4);
    assert_eq!(st.held_bytes(), 310);

    // Fitting SSD candidate evicts oldest LRU entries BEFORE decoding so
    // `held_bytes() + 100 <= 300`, then `promote_within` lands within cap.
    assert!(st.reserve_for_promote(disk_fit, 100, 300, None));
    assert!(st.held_bytes() + 100 <= 300);
    assert!(st.get(ids[0]).is_none(), "oldest LRU entry evicted first");
    assert!(
        st.get(ids[1]).is_none(),
        "second oldest evicted to fit 100 B"
    );
    assert!(st.get(ids[2]).is_some() && st.get(shared_id).is_some());
    assert!(st.promote_within(disk_fit, mtp_fake(10, 10, 90), Some(300), None));
    assert!(st.held_bytes() <= 300);
    assert!(st.get(disk_fit).is_some());

    // `reserve_bytes` evicts until `held_bytes() + incoming <= byte_cap`.
    assert!(st.reserve_bytes(100, 200));
    assert!(st.held_bytes() + 100 <= 200);

    // Publishing another unshared 100 B state evicts LRU entries to stay within
    // `byte_cap = 200`, while an oversized publish returns `Ok(None)` without
    // wiping existing RAM states.
    let before = st.ram_len();
    assert!(
        st.publish_within(
            Key::text(&lineage(12, 1), SALT),
            mtp_fake(10, 12, 250),
            Kind::Turn,
            Some(200),
        )
        .unwrap()
        .is_none()
    );
    assert_eq!(st.ram_len(), before);
    let fit_pub = st
        .publish_within(
            Key::text(&lineage(13, 1), SALT),
            mtp_fake(10, 13, 90),
            Kind::Turn,
            Some(200),
        )
        .unwrap();
    assert!(fit_pub.is_some());
    assert!(st.held_bytes() <= 200);
}

/// Protecting the current request's shallow RAM fallback during a deeper SSD
/// restore: `reserve_for_promote` and `promote_within` never evict `protect`
/// (even when it is a checkpoint), so an SSD `Busy`/`Failed` read or a tight
/// cap always preserves the RAM fallback.
#[test]
fn protected_ram_fallback_survives_tight_or_failed_ssd_promote() {
    use crate::disk_tier::store::{DiskCopy, Lineage, StateClass};
    use mlx_paged_attn::{ColdCacheFingerprint, ColdCacheKey, ColdGroup};

    let disk_copy = |seed: u64| DiskCopy {
        key: ColdCacheKey::chain(
            ColdGroup::DFlash2State,
            ColdCacheFingerprint::from_components([&seed.to_le_bytes()[..]]),
            None,
            &[seed as u32],
            &[],
            0,
            0,
        ),
        bytes: 64,
        landed: true,
        window_begin: 0,
        chain_owed: false,
    };

    // 1. Even though `fallback` is a Checkpoint (normally evicted first),
    //    `reserve_for_promote(..., Some(fallback))` evicts `other` instead and
    //    keeps `fallback` resident if the subsequent SSD read fails/times out.
    let mut st = States::<Fake>::with_cap(2);
    let fallback = publish(&mut st, &lineage(1, 1), Kind::Checkpoint, 100).unwrap();
    let other = publish(&mut st, &lineage(2, 1), Kind::Prompt, 100).unwrap();
    let disk_deep = st
        .store_mut()
        .insert_disk(
            Lineage::text(&lineage(3, 2), SALT),
            100,
            StateClass::Ordinary,
            disk_copy(3),
        )
        .unwrap();

    // Tight byte cap (150 B) where `fallback` (100 B) + `disk_deep` (100 B)
    // cannot both fit: refuses without evicting either `fallback` or `other`.
    assert!(!st.reserve_for_promote(disk_deep, 100, 150, Some(fallback)));
    assert!(st.get(fallback).is_some());
    assert!(st.get(other).is_some());

    // Under 200 B cap at slot limit 2: evicts `other`, NEVER `fallback`.
    assert!(st.reserve_for_promote(disk_deep, 100, 200, Some(fallback)));
    assert!(st.get(fallback).is_some(), "protected fallback must stay");
    assert!(st.get(other).is_none(), "unprotected entry is evicted");

    // 2. When `ram_cap == 1` and `fallback` is the sole resident entry,
    //    neither `reserve_for_promote` nor `promote_within` may evict it.
    let mut tiny = States::<Fake>::with_cap(1);
    let only_fallback = publish(&mut tiny, &lineage(4, 1), Kind::Checkpoint, 50).unwrap();
    let disk_only = tiny
        .store_mut()
        .insert_disk(
            Lineage::text(&lineage(5, 2), SALT),
            50,
            StateClass::Ordinary,
            disk_copy(5),
        )
        .unwrap();
    assert!(!tiny.reserve_for_promote(disk_only, 50, 500, Some(only_fallback)));
    assert!(!tiny.promote_within(disk_only, fake(50), Some(500), Some(only_fallback)));
    assert!(tiny.get(only_fallback).is_some());
    assert!(tiny.get(disk_only).is_none());
}

/// Upgrading an already-resident `Kind::Checkpoint` to `Kind::Prompt` mutates
/// `kind` in place without swapping the payload or duplicating segments, so
/// `held_bytes()` stays unchanged even at an exact `byte_cap`.
#[test]
fn checkpoint_to_prompt_upgrade_mutates_kind_in_place_without_duplicating_segments() {
    let mut st = States::<Fake>::with_cap(4);
    let ck_tok = lineage(1, 1);
    let turn_tok = lineage(2, 2);

    // Checkpoint and deeper Turn share segment id 1 (90 B); own bytes = 10 B each.
    let ck_id = st
        .publish_within(
            Key::text(&ck_tok, SALT),
            Fake {
                own: 10,
                shared: vec![(1, 90)],
            },
            Kind::Checkpoint,
            Some(110),
        )
        .unwrap()
        .unwrap();
    let turn_id = st
        .publish_within(
            Key::text(&turn_tok, SALT),
            Fake {
                own: 10,
                shared: vec![(1, 90)],
            },
            Kind::Turn,
            Some(110),
        )
        .unwrap()
        .unwrap();
    assert_eq!(st.held_bytes(), 110);

    // Upgrading the checkpoint to Prompt with a fresh payload carrying a new
    // unshared segment id 2 (90 B) must keep the original segment id 1 in place.
    let upgraded = st
        .publish_within(
            Key::text(&ck_tok, SALT),
            Fake {
                own: 10,
                shared: vec![(2, 90)],
            },
            Kind::Prompt,
            Some(110),
        )
        .unwrap()
        .unwrap();
    assert_eq!(upgraded, ck_id);
    let entry = st.store().get(ck_id).unwrap();
    assert_eq!(entry.ram().unwrap().kind, Kind::Prompt);
    assert!(!entry.checkpoint());
    assert_eq!(entry.ram().unwrap().state.shared, vec![(1, 90)]);
    assert_eq!(st.held_bytes(), 110, "in-place upgrade adds zero bytes");
    assert!(st.get(turn_id).is_some(), "deeper Turn state stays in RAM");
}
