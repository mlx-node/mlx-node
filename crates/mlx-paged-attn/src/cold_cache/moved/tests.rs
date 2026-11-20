use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::super::{ColdLayerBlock, ColdStateObject};
use super::*;

const SECOND: u128 = 1_000_000_000;

/// A fresh directory for one test, removed on drop.
struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mlx-cold-moved-{tag}-{}-{}",
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

fn fingerprint() -> ColdCacheFingerprint {
    ColdCacheFingerprint::from_components([b"moved".as_slice()])
}

fn block(index: usize) -> ColdCacheBlock {
    let layout = layout();
    let tokens: Vec<u32> = (0..16).map(|token| (index * 16 + token) as u32).collect();
    ColdCacheBlock {
        key: ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &tokens, &[], 0, index),
        fingerprint: fingerprint(),
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

fn state(payload: Vec<u8>) -> ColdStateObject {
    let tokens: Vec<u32> = (0..32).collect();
    ColdStateObject {
        key: ColdCacheKey::chain(
            ColdGroup::DFlash2State,
            fingerprint(),
            None,
            &tokens,
            &[],
            0,
            0,
        ),
        fingerprint: fingerprint(),
        layout: ColdStateLayout {
            boundary_tokens: 32,
            spans: vec![payload.len() as u64 - 64, 64],
            geometry: "gdn 1 3 14 2 4 3\n".to_string(),
        },
        label: ColdStateLabel {
            tokens,
            extra_keys: Vec::new(),
            salt: 0,
            last_used: 3,
            checkpoint: false,
        },
        payload,
    }
}

fn open(dir: &Path, quota: u64) -> ColdCacheManager {
    ColdCacheManager::open_at(dir.to_path_buf(), quota, 0, 8).expect("a cache dir")
}

/// A tracked write of `block`: whether it landed.
fn write(manager: &ColdCacheManager, block: &ColdCacheBlock) -> bool {
    let (done, landed) = mpsc::channel();
    let deadline = Instant::now() + Duration::from_secs(10);
    assert_eq!(
        manager.enqueue_tracked(block.clone(), MakeRoom::Refuse, deadline, done),
        Ok(true)
    );
    match landed.recv_timeout(Duration::from_secs(10)) {
        Ok(Landed::Block { key, ok }) => {
            assert_eq!(key, block.key);
            ok
        }
        other => panic!("no block landing: {other:?}"),
    }
}

fn file_time(dir: &Path, key: &ColdCacheKey) -> u128 {
    fs::metadata(dir.join(object_file_name(key, ColdGroup::Kv)))
        .expect("the object")
        .modified()
        .expect("mtime")
        .duration_since(UNIX_EPOCH)
        .expect("after the epoch")
        .as_nanos()
}

/// A block made on the host lands through the tracked writer and reads
/// back equal with its time untouched; a time set on it is the file's and
/// the index's, and a reopen keeps it. A write that does not fit is refused
/// and writes nothing. A block of another layout is not read back and
/// stays; a damaged one goes.
#[test]
fn a_host_block_lands_reads_back_and_keeps_a_set_time() {
    let dir = Dir::new("tracked");
    let first = block(0);
    let manager = open(&dir.0, 1 << 30);
    assert!(write(&manager, &first));
    let landed_at = file_time(&dir.0, &first.key);
    let indexed = manager.modified(&first.key).expect("an indexed time");
    assert!(
        indexed >= landed_at && indexed - landed_at < SECOND,
        "the index has the time of the landing"
    );
    assert_eq!(
        manager.read_back(first.key, fingerprint(), &layout()),
        Some(first.clone())
    );
    assert_eq!(
        file_time(&dir.0, &first.key),
        landed_at,
        "a read back leaves the time"
    );
    assert!(manager.flush_device());
    #[cfg(target_vendor = "apple")]
    assert!(
        rustix::fs::fcntl_fullfsync(&manager.shared.root.fd).is_ok(),
        "the flush is the device's, not a plain sync"
    );

    let source_time = 1_700_000_000 * SECOND + 123_456_789;
    assert!(manager.set_modified(&first.key, source_time));
    assert_eq!(file_time(&dir.0, &first.key), source_time);
    assert_eq!(manager.modified(&first.key), Some(source_time));
    assert!(manager.close(Duration::from_secs(10)));
    let manager = open(&dir.0, manager.total_bytes());
    assert_eq!(manager.modified(&first.key), Some(source_time));

    let second = block(1);
    assert!(!write(&manager, &second), "no room, no eviction");
    assert!(!manager.contains(&second.key));
    assert!(manager.contains(&first.key));
    assert!(manager.modified(&second.key).is_none());
    assert!(!manager.set_modified(&second.key, source_time));

    let other = ColdCacheLayout {
        head_size: 16,
        ..layout()
    };
    assert_eq!(manager.read_back(first.key, fingerprint(), &other), None);
    assert!(manager.contains(&first.key), "another layout stays");
    let path = dir.0.join(object_file_name(&first.key, ColdGroup::Kv));
    let mut bytes = fs::read(&path).expect("the object");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    fs::write(&path, bytes).expect("damage it");
    assert_eq!(manager.read_back(first.key, fingerprint(), &layout()), None);
    assert!(!manager.contains(&first.key), "a damaged object goes");
    assert!(!path.exists());
}

/// Room is made only among objects older than the given time, oldest
/// first, never one of the kept keys, and nothing goes unless enough of
/// them can.
#[test]
fn room_is_made_only_among_older_objects() {
    let dir = Dir::new("older");
    let blocks: Vec<ColdCacheBlock> = (0..4).map(block).collect();
    let manager = open(&dir.0, 1 << 30);
    for (index, block) in blocks.iter().enumerate() {
        assert!(write(&manager, block));
        assert!(manager.set_modified(&block.key, (10 + 10 * index as u128) * SECOND));
    }
    let total = manager.total_bytes();
    let size: Vec<u64> = blocks
        .iter()
        .map(|block| {
            fs::metadata(dir.0.join(object_file_name(&block.key, ColdGroup::Kv)))
                .expect("the object")
                .len()
        })
        .collect();
    assert_eq!(size.iter().sum::<u64>(), total);
    assert!(manager.close(Duration::from_secs(10)));
    let manager = open(&dir.0, total);
    let held = |manager: &ColdCacheManager| -> Vec<bool> {
        blocks
            .iter()
            .map(|block| manager.contains(&block.key))
            .collect()
    };
    let none = HashSet::new();

    assert!(manager.make_room_among_older(0, 0, &none), "it fits");
    assert!(
        !manager.make_room_among_older(size[0] + size[1] + 1, 25 * SECOND, &none),
        "two older objects are not room for more than their bytes"
    );
    assert_eq!(held(&manager), [true; 4], "nothing goes for no room");
    let keep = HashSet::from([blocks[0].key]);
    assert!(
        !manager.make_room_among_older(size[0] + size[1], 25 * SECOND, &keep),
        "a kept object is no room"
    );
    assert_eq!(held(&manager), [true; 4]);

    assert!(manager.make_room_among_older(size[1], 35 * SECOND, &keep));
    assert_eq!(
        held(&manager),
        [true, false, true, true],
        "the oldest object that is not kept goes, and no more than needed"
    );
    assert!(manager.make_room_among_older(size[1] + size[2], 35 * SECOND, &keep));
    assert_eq!(held(&manager), [true, false, false, true]);
    assert!(
        !manager.make_room_among_older(size[1] + size[2] + 1, 5 * SECOND, &none),
        "an object that is not older stays"
    );
    assert_eq!(held(&manager), [true, false, false, true]);
    assert_eq!(manager.total_bytes(), size[0] + size[3]);
}

/// A source block reads, or is absent, damaged or foreign; only the
/// removal takes an object out, and a removed object stays removed.
#[test]
fn a_source_block_is_told_apart_and_goes_only_by_its_removal() {
    let dir = Dir::new("source");
    let blocks: Vec<ColdCacheBlock> = (0..3).map(block).collect();
    {
        let manager = open(&dir.0, 1 << 30);
        for block in &blocks {
            assert!(write(&manager, block));
        }
        assert!(manager.close(Duration::from_secs(10)));
    }
    let path = dir.0.join(object_file_name(&blocks[1].key, ColdGroup::Kv));
    let mut bytes = fs::read(&path).expect("the second block");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    fs::write(&path, bytes).expect("damage it");

    let reader = ColdSourceReader::open(&dir.0).expect("a reader");
    let probe = |key: ColdCacheKey, expected: &ColdCacheLayout| {
        reader.probe_block(key, fingerprint(), expected)
    };
    assert!(
        matches!(probe(blocks[0].key, &layout()), ColdSourceBlock::Read(block) if *block == blocks[0])
    );
    assert!(matches!(
        probe(blocks[1].key, &layout()),
        ColdSourceBlock::Damaged
    ));
    assert!(matches!(
        probe(block(9).key, &layout()),
        ColdSourceBlock::Absent
    ));
    let wider = ColdCacheLayout {
        head_size: 16,
        key_bytes_per_layer: 320,
        value_bytes_per_layer: 288,
        ..layout()
    };
    let narrower = ColdCacheLayout {
        head_size: 4,
        key_bytes_per_layer: 80,
        value_bytes_per_layer: 72,
        ..layout()
    };
    for other in [&wider, &narrower] {
        assert!(
            matches!(probe(blocks[2].key, other), ColdSourceBlock::Foreign),
            "a sound block of another layout is not damage"
        );
    }
    assert_eq!(reader.entries().expect("entries").len(), 3);

    assert!(reader.remove_moved(blocks[0].key, ColdGroup::Kv));
    assert!(matches!(
        probe(blocks[0].key, &layout()),
        ColdSourceBlock::Absent
    ));
    assert!(reader.remove_moved(blocks[0].key, ColdGroup::Kv));
    assert_eq!(reader.entries().expect("entries").len(), 2);
}

/// A state object reads back when its header carries the layout and label
/// and its bytes are the payload; another payload or label does not, and
/// the object stays; a damaged one goes.
#[test]
fn a_state_reads_back_by_its_header_and_its_bytes() {
    let dir = Dir::new("state");
    let payload: Vec<u8> = (0..3 * STATE_READ_CHUNK + 4096)
        .map(|byte| (byte % 251) as u8)
        .collect();
    let object = state(payload.clone());
    let (key, layout, label) = (object.key, object.layout.clone(), object.label.clone());
    let manager = open(&dir.0, 1 << 30);
    let (done, landed) = mpsc::channel();
    assert!(
        manager
            .enqueue_state(object, MakeRoom::Refuse, done)
            .is_ok()
    );
    assert!(matches!(
        landed.recv_timeout(Duration::from_secs(10)),
        Ok(Landed::State { ok: true, .. })
    ));
    let name = object_file_name(&key, ColdGroup::DFlash2State);
    let time = |dir: &Path| fs::metadata(dir.join(&name)).and_then(|meta| meta.modified());
    let landed_at = time(&dir.0).expect("mtime");

    let reads = |payload: Option<&[u8]>, label: &ColdStateLabel| {
        manager.state_reads_back(key, fingerprint(), &layout, label, payload)
    };
    assert!(reads(Some(&payload), &label));
    assert!(reads(None, &label));
    assert_eq!(time(&dir.0).expect("mtime"), landed_at, "the time is left");
    let mut other = payload.clone();
    other[2 * STATE_READ_CHUNK + 7] ^= 1;
    assert!(!reads(Some(&other), &label));
    assert!(!reads(Some(&payload[1..]), &label));
    let restamped = ColdStateLabel {
        last_used: 4,
        ..label.clone()
    };
    assert!(!reads(None, &restamped));
    assert!(manager.contains_in(&key, ColdGroup::DFlash2State));

    let path = dir.0.join(&name);
    let mut bytes = fs::read(&path).expect("the state");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    fs::write(&path, bytes).expect("damage it");
    assert!(!reads(None, &label));
    assert!(!manager.contains_in(&key, ColdGroup::DFlash2State));
}

/// A write the queue has no slot for until its deadline is a queue drop:
/// nothing is written and nobody is told it landed; the writes the queue
/// took land.
#[test]
fn a_write_the_queue_does_not_take_is_not_written() {
    let dir = Dir::new("queue");
    let manager = ColdCacheManager::open_at(dir.0.clone(), 1 << 30, 0, 1).expect("a cache dir");
    let (release, wait) = mpsc::channel::<()>();
    let (parked, at_gate) = mpsc::channel::<()>();
    let gate = Mutex::new(Some((wait, parked)));
    *manager
        .shared
        .dir_sync_override
        .lock()
        .expect("the sync gate") = Some(Box::new(move || {
        let taken = gate.lock().ok().and_then(|mut gate| gate.take());
        if let Some((wait, parked)) = taken {
            let _ = parked.send(());
            let _ = wait.recv();
        }
        Ok(())
    }));
    let blocks: Vec<ColdCacheBlock> = (0..3).map(block).collect();
    let soon = || Instant::now() + Duration::from_secs(10);
    let (done, landed) = mpsc::channel();
    assert_eq!(
        manager.enqueue_tracked(blocks[0].clone(), MakeRoom::Refuse, soon(), done.clone()),
        Ok(true)
    );
    at_gate
        .recv_timeout(Duration::from_secs(10))
        .expect("the writer is inside its first write");
    assert_eq!(
        manager.enqueue_tracked(blocks[1].clone(), MakeRoom::Refuse, soon(), done),
        Ok(true),
        "the queue's one slot"
    );
    let (dropped, never) = mpsc::channel();
    assert_eq!(
        manager.enqueue_tracked(blocks[2].clone(), MakeRoom::Refuse, Instant::now(), dropped),
        Ok(false)
    );
    assert_eq!(manager.stats().queue_drops, 1);
    release.send(()).expect("the writer waits at the gate");
    for _ in 0..2 {
        assert!(matches!(
            landed.recv_timeout(Duration::from_secs(10)),
            Ok(Landed::Block { ok: true, .. })
        ));
    }
    assert!(manager.drain(Duration::from_secs(10)));
    assert!(never.try_recv().is_err(), "a dropped write never lands");
    assert!(manager.contains(&blocks[0].key) && manager.contains(&blocks[1].key));
    assert!(!manager.contains(&blocks[2].key));
    assert!(
        !dir.0
            .join(object_file_name(&blocks[2].key, ColdGroup::Kv))
            .exists()
    );
}
