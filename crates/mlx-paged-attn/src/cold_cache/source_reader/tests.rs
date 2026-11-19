use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::super::{
    ColdCacheManager, ColdLayerBlock, ColdStateObject, Landed, MakeRoom, object_file_name,
};
use super::*;

/// A fresh directory for one test, removed on drop.
struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mlx-source-reader-{tag}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        Self(path)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn layout() -> ColdCacheLayout {
    ColdCacheLayout {
        block_size: 16,
        num_layers: 2,
        num_kv_heads: 1,
        head_size: 8,
        cache_dtype: "Int8".to_string(),
        key_bytes_per_layer: 160,
        value_bytes_per_layer: 144,
    }
}

fn block(
    fingerprint: ColdCacheFingerprint,
    parent: Option<ColdCacheKey>,
    index: usize,
) -> ColdCacheBlock {
    let layout = layout();
    let tokens: Vec<u32> = (0..16).map(|token| (index * 16 + token) as u32).collect();
    ColdCacheBlock {
        key: ColdCacheKey::chain(ColdGroup::Kv, fingerprint, parent, &tokens, &[], 0, index),
        fingerprint,
        tokens,
        layers: (0..layout.num_layers as u8)
            .map(|layer| ColdLayerBlock {
                keys: vec![layer ^ index as u8; layout.key_bytes_per_layer],
                values: vec![!layer ^ index as u8; layout.value_bytes_per_layer],
            })
            .collect(),
        layout,
    }
}

fn state(fingerprint: ColdCacheFingerprint) -> ColdStateObject {
    let tokens: Vec<u32> = (0..32).collect();
    ColdStateObject {
        key: ColdCacheKey::chain(
            ColdGroup::DFlash2State,
            fingerprint,
            None,
            &tokens,
            &[],
            0,
            0,
        ),
        fingerprint,
        layout: ColdStateLayout {
            boundary_tokens: 32,
            spans: vec![96, 64],
            geometry: "gdn 1 3 14 2 4 3\n".to_string(),
        },
        label: ColdStateLabel {
            tokens,
            extra_keys: Vec::new(),
            salt: 0,
            last_used: 3,
            checkpoint: false,
        },
        payload: (0..160).map(|byte| byte as u8).collect(),
    }
}

/// Every entry of `dir`: name, size and modification time.
fn listing(dir: &Path) -> Vec<(String, u64, SystemTime)> {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .expect("list")
        .map(|entry| {
            let entry = entry.expect("an entry");
            let meta = fs::symlink_metadata(entry.path()).expect("stat");
            (
                entry.file_name().to_string_lossy().into_owned(),
                meta.len(),
                meta.modified().expect("mtime"),
            )
        })
        .collect();
    entries.sort();
    entries
}

/// A reader of a store the product writer left changes nothing in it:
/// names, sizes and times are equal before and after every object is read,
/// a damaged block and a block of another fingerprint are refused and stay,
/// a writer temp file stays, and opening a directory that is not there
/// makes none.
/// Mutation reader-prunes (a refused read removes the object) → red.
#[test]
fn reading_a_source_changes_nothing_in_it() {
    let dir = Dir::new("listing");
    let own = ColdCacheFingerprint::from_components([b"source".as_slice()]);
    let foreign = ColdCacheFingerprint::from_components([b"another load".as_slice()]);
    let first = block(own, None, 0);
    let second = block(own, Some(first.key), 1);
    let third = block(own, Some(second.key), 2);
    let other = block(foreign, None, 0);
    let written_state = state(own);
    let state_key = written_state.key;
    let payload = written_state.payload.clone();
    {
        let manager = ColdCacheManager::open_at(dir.0.clone(), 1 << 30, 0, 8).expect("a writer");
        for block in [&first, &second, &third, &other] {
            assert!(manager.enqueue(block.clone()).expect("a block job"));
        }
        assert!(manager.drain(Duration::from_secs(10)));
        let (tx, rx) = mpsc::channel();
        manager
            .enqueue_state(written_state, MakeRoom::Evict, tx)
            .expect("a state job");
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(10)).expect("landed"),
            Landed::State { ok: true, .. }
        ));
        assert!(manager.close(Duration::from_secs(10)));
    }
    let damaged = dir.0.join(object_file_name(&second.key, ColdGroup::Kv));
    let mut bytes = fs::read(&damaged).expect("the second block");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    fs::write(&damaged, bytes).expect("damage it");
    let temp = dir.0.join(format!(".{}.1.2.tmp", first.key.to_hex()));
    fs::write(&temp, b"half a write").expect("a writer temp file");

    let before = listing(&dir.0);
    let reader = ColdSourceReader::open(&dir.0).expect("a reader");
    let entries = reader.entries().expect("entries");
    assert_eq!(entries.len(), 5, "four blocks and one state; no temp file");
    let mut read = Vec::new();
    for entry in &entries {
        let found = match entry.group {
            ColdGroup::Kv => reader.read_block(entry.key, own, &layout()).is_some(),
            ColdGroup::DFlash2State => {
                let label = reader.state_label(entry.key, own);
                let mut into = Vec::new();
                let full = reader.read_state(entry.key, own, &mut into);
                assert_eq!(label, full);
                assert_eq!(into, payload);
                full.is_some()
            }
            group => panic!("no {group:?} object was written"),
        };
        read.push((entry.key, found));
    }
    read.sort_by_key(|(key, _)| key.to_hex());
    let mut expected = vec![
        (first.key, true),
        (second.key, false),
        (third.key, true),
        (other.key, false),
        (state_key, true),
    ];
    expected.sort_by_key(|(key, _)| key.to_hex());
    assert_eq!(
        read, expected,
        "the damaged and the foreign block are refused"
    );
    let absent = dir.0.join("absent");
    assert!(ColdSourceReader::open(&absent).is_err());
    assert!(!absent.exists(), "opening a missing source makes nothing");
    assert_eq!(
        listing(&dir.0),
        before,
        "names, sizes and times are unchanged after every read"
    );
}
