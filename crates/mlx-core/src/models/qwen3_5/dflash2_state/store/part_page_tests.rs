use super::*;

struct NoRam;

impl StateBytes for NoRam {
    fn state_bytes(&self) -> u64 {
        0
    }
}

type Store = HybridStateStore<NoRam>;

fn copy(seed: u8) -> DiskCopy {
    DiskCopy {
        key: mlx_paged_attn::ColdCacheKey::chain(
            mlx_paged_attn::ColdGroup::DFlash2State,
            mlx_paged_attn::ColdCacheFingerprint::from_components([&[seed][..]]),
            None,
            &[u32::from(seed)],
            &[],
            0,
            0,
        ),
        bytes: 160,
        landed: true,
        window_begin: 0,
        chain_owed: false,
    }
}

/// A state on a part page is taken back at whole 32-token blocks plus 16
/// rows, with its earlier stamp, the chain of its whole blocks and the
/// image keys of its part page. A prompt of its lineage resumes on it only
/// with the same image keys on those 16 rows of the page (its keys of the
/// page's later rows do not count); with others, and within a depth below
/// it, the whole state under it resumes. A state with no whole block
/// resumes nothing. Any other row count, a block that is not 32 tokens, a
/// second state on the same rows and a plain disk publish there are refused.
#[test]
fn a_state_on_a_part_page_is_taken_back_and_resumes_with_its_image_keys() {
    let tokens: Vec<u32> = (0..200).collect();
    let keys: [Vec<u64>; 3] = [Vec::new(), vec![7, 1, 9, 3], vec![7, 1, 5, 2]];
    let lineage = |rows: usize| Lineage {
        tokens: &tokens[..rows],
        extra_keys: &keys,
        salt: 4,
    };
    let mut store = Store::new(32);
    let whole = store
        .adopt_disk(lineage(64), 160, StateClass::Ordinary, copy(1), 11)
        .expect("a whole state");
    let part = store
        .adopt_disk_on_part_page(lineage(80), 160, StateClass::Ordinary, copy(2), 12)
        .expect("a state on a part page");
    let entry = store.get(part).expect("the entry");
    assert_eq!(entry.depth(), 80);
    assert_eq!(entry.last_used(), 12);
    assert_eq!(entry.chain(), store.chain(lineage(64)));
    assert_eq!(entry.chain().len(), 2);
    assert_eq!(
        entry.extra_keys(),
        &keys[..],
        "the part page's keys are kept"
    );
    assert_eq!(entry.disk().map(|disk| disk.key), Some(copy(2).key));
    assert_eq!(
        store.get(whole).expect("the whole state").extra_keys(),
        &keys[..2]
    );

    let hit = store.lookup(lineage(200)).expect("a hit");
    assert_eq!((hit.id, hit.depth), (part, 80));
    assert_eq!(
        store.lookup(lineage(81)).map(|hit| hit.depth),
        Some(80),
        "a prompt one row past it"
    );
    assert_eq!(
        store.lookup(lineage(80)).map(|hit| hit.depth),
        Some(64),
        "a prompt that ends on it resumes below it"
    );
    assert_eq!(
        store.lookup_within(lineage(200), 79).map(|hit| hit.depth),
        Some(64)
    );
    let mut offered = Vec::new();
    store.deepest_accepted_within(lineage(200), 199, |id| {
        offered.push(id);
        false
    });
    assert_eq!(offered, vec![part, whole]);
    let with_page_keys = |page: Vec<u64>| {
        let prompt_keys = [keys[0].clone(), keys[1].clone(), page];
        store
            .lookup(Lineage {
                tokens: &tokens,
                extra_keys: &prompt_keys,
                salt: 4,
            })
            .map(|hit| (hit.id, hit.depth))
    };
    assert_eq!(
        with_page_keys(vec![7, 1, 5, 2, 8, 16, 8, 31]),
        Some((part, 80)),
        "an image on the page's later rows"
    );
    for other in [
        Vec::new(),
        vec![7, 1, 6, 2],
        vec![7, 1, 5, 3],
        vec![7, 2, 5, 2],
        vec![7, 1, 5, 2, 8, 15],
        vec![7, 1, 8, 20],
        vec![7, 1, 5],
    ] {
        assert_eq!(
            with_page_keys(other.clone()),
            Some((whole, 64)),
            "other image keys on the part page: {other:?}"
        );
    }
    let mut changed = tokens.clone();
    changed[70] += 1;
    let prompt = Lineage {
        tokens: &changed,
        extra_keys: &keys,
        salt: 4,
    };
    assert_eq!(
        store.lookup(prompt).map(|hit| hit.depth),
        Some(64),
        "another row on the part page"
    );

    for rows in [64, 72, 96, 0] {
        assert!(
            store
                .adopt_disk_on_part_page(lineage(rows), 160, StateClass::Ordinary, copy(3), 1)
                .is_err(),
            "{rows} rows"
        );
    }
    assert!(
        store
            .adopt_disk_on_part_page(lineage(80), 160, StateClass::Ordinary, copy(3), 1)
            .is_err(),
        "one state on the same rows"
    );
    assert!(
        store
            .insert_disk(lineage(112), 160, StateClass::Ordinary, copy(3))
            .is_err(),
        "a publish sits on a whole block"
    );
    assert!(
        store
            .adopt_disk(lineage(112), 160, StateClass::Ordinary, copy(3), 1)
            .is_err()
    );
    assert_eq!(store.bytes().entries, 2);

    let single = store
        .adopt_disk_on_part_page(lineage(16), 160, StateClass::Checkpoint, copy(4), 3)
        .expect("a state of one part page");
    let entry = store.get(single).expect("the entry");
    assert!(entry.chain().is_empty() && entry.checkpoint());
    assert_eq!(entry.extra_keys(), &[] as &[Vec<u64>]);
    assert_eq!(
        store.lookup(lineage(30)).map(|hit| hit.id),
        None,
        "a state with no whole block resumes nothing"
    );

    let mut other = Store::new(16);
    assert!(
        other
            .adopt_disk_on_part_page(
                Lineage::text(&tokens[..24], 0),
                160,
                StateClass::Ordinary,
                copy(5),
                1
            )
            .is_err(),
        "only a 32-token block has a 16-row part page"
    );
}
