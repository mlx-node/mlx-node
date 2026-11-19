//! The paged pool of the indexed layers: page geometry, the slot map, the
//! half-row write read back as flat rows, and the paged indexer state.

use std::path::PathBuf;

use serde_json::Value;

use super::cache::WINDOW_SLACK;
use super::config::Attn;
use super::indexer::{IndexerGeometry, PagedIndexerState};
use super::paged::{BLOCK, PageGeometry, page_ids, pool_layers, slot_ids};
use super::tests_model::tiny_cfg;
use crate::array::MxArray;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash/pool-layout.json");
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

/// The Hy4 page: 12 indexed layers, planes of 288, 8 x 128 pooled keys.
fn real() -> PageGeometry {
    PageGeometry {
        layers: 12,
        half: 288,
        index_rows: 8,
        index_dim: 128,
    }
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn read_i32(x: &MxArray) -> Vec<i32> {
    x.to_int32().unwrap().as_ref().to_vec()
}

/// 466,944 B per page over the 12 layers, a 2,048 B index slot, head 288
/// passes the pool config, and the pool sizes by the fixture's contexts.
/// Mutation: `block_bytes` drops the index plane → red.
#[test]
fn real_page_bytes_match_the_fixture() {
    let f = fixture();
    let bytes = &f["bytes"];
    let g = real();
    assert_eq!(
        g.layout().index_bytes(),
        bytes["page_layout_index_bytes"].as_u64().unwrap()
    );
    assert_eq!(
        g.block_bytes().unwrap(),
        bytes["per_block_all_layers"].as_u64().unwrap()
    );
    assert!(g.layout().values_token_major);
    let cfg = g.config(8192, 262_144).unwrap();
    assert_eq!(
        (cfg.head_size, cfg.num_kv_heads, cfg.num_layers),
        (288, 1, 12)
    );
    assert_eq!(cfg.block_size, BLOCK);
    for c in f["by_ctx"].as_array().unwrap() {
        let blocks = c["blocks"].as_u64().unwrap();
        assert_eq!(
            blocks * g.block_bytes().unwrap(),
            c["pool_bytes"].as_u64().unwrap(),
            "ctx {}",
            c["ctx"]
        );
    }
}

/// The tiny config's page: its indexed trunk layers in order, half its
/// latent row, `32 / kpool` = 8 pooled groups per page; an odd row or a
/// group size that does not tile 32 rows is refused.
#[test]
fn page_geometry_follows_the_config() {
    let cfg = tiny_cfg();
    let want: Vec<Option<u32>> = {
        let mut next = 0;
        cfg.trunk()
            .iter()
            .map(|l| match l.attn {
                Attn::Indexed => {
                    next += 1;
                    Some(next - 1)
                }
                Attn::Sliding => None,
            })
            .collect()
    };
    assert_eq!(pool_layers(&cfg), want);
    let g = PageGeometry::of(&cfg).unwrap();
    assert_eq!(g.layers as usize, want.iter().flatten().count());
    assert_eq!(g.half as usize * 2, cfg.mla.kv_row());
    assert_eq!(g.index_rows, 8);
    assert_eq!(g.index_dim as usize, cfg.indexer.head_dim);

    let mut odd = cfg.clone();
    odd.mla.rope_dim += 1;
    assert!(PageGeometry::of(&odd).is_err());
    let mut three = cfg.clone();
    three.indexer.kpool = 3;
    assert!(PageGeometry::of(&three).is_err());
}

/// A plane width outside the paged config's head sizes is refused.
#[test]
fn page_config_refuses_an_unlisted_plane_width() {
    let g = PageGeometry {
        half: 289,
        ..real()
    };
    assert!(g.config(64, 4096).is_err());
}

/// `slot = page·32 + t % 32`, `-1` kept, for both fixture tables.
/// Mutation: the slot drops `t % 32` → red.
#[test]
fn slot_ids_follow_the_fixture_tables() {
    let f = fixture();
    let map = &f["slot_map"];
    let table: Vec<u32> = ints(&map["block_table"])
        .into_iter()
        .map(|b| b as u32)
        .collect();
    let pages = page_ids(&table).unwrap();
    let mut cells = Vec::new();
    let mut want = Vec::new();
    for c in map["cases"].as_array().unwrap() {
        let Some(slot) = c["slot"].as_i64() else {
            continue;
        };
        cells.push(c["t"].as_i64().unwrap() as i32);
        want.push(slot as i32);
    }
    let idx = MxArray::from_int32(&cells, &[1, cells.len() as i64]).unwrap();
    assert_eq!(read_i32(&slot_ids(&pages, &idx).unwrap()), want);

    let synthetic = &f["synthetic"];
    let table: Vec<u32> = ints(&synthetic["block_table"])
        .into_iter()
        .map(|b| b as u32)
        .collect();
    let gather = &synthetic["gather"];
    let cells: Vec<i32> = ints(&gather["cells"])
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let want: Vec<i32> = ints(&gather["slots"])
        .into_iter()
        .map(|s| s as i32)
        .collect();
    let idx = MxArray::from_int32(&cells, &[cells.len() as i64]).unwrap();
    let got = slot_ids(&page_ids(&table).unwrap(), &idx).unwrap();
    assert_eq!(read_i32(&got), want);
}

/// No pages, or a page whose slots pass i32, give no map.
#[test]
fn page_ids_refuse_an_empty_or_too_large_table() {
    assert!(page_ids(&[]).is_err());
    assert!(page_ids(&[(i32::MAX as u32) / BLOCK + 1]).is_err());
    assert!(page_ids(&[0, 3, 1]).is_ok());
}

/// The paged indexer state holds no pooled keys of its own: its tail keeps
/// `kpool - 1 + WINDOW_SLACK` = 3 + 3 rows, and an empty state's
/// truncate/restore only move lengths.
#[test]
fn paged_indexer_state_keeps_only_the_tail() {
    let g = IndexerGeometry {
        n_heads: 2,
        head_dim: 8,
        rope_dim: 4,
        kpool: 4,
        n_top: 1,
        select_tail: true,
        eps: 1e-5,
        weight_scale: 0.25,
        rope_theta: 1e7,
    };
    let mut st = PagedIndexerState::new(3, &g);
    assert_eq!(st.layer(), 3);
    assert_eq!(st.tail().keep(), 3 + WINDOW_SLACK);
    assert_eq!((st.len(), st.pooled_len(), st.open()), (0, 0, 0..0));
    assert_eq!(st.nbytes(), 0);
    let mark = st.mark();
    st.truncate(0).unwrap();
    assert!(st.truncate(1).is_err());
    st.restore(mark).unwrap();
    assert!(st.is_empty());
}

#[cfg(target_os = "macos")]
mod pool {
    use super::*;
    use crate::array::DType;
    use crate::models::hy_v4_flash::paged::{SlotMap, create, planes, write_latent};
    use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

    const TOKENS: usize = 70;
    const ROW: usize = 576;
    const OWNER: u32 = 1;
    const OTHER: u32 = 9;

    fn row_bits(t: usize, c: usize) -> u16 {
        0x3C00 | ((t * 37 + c * 11) % 1024) as u16
    }

    /// A one-layer Hy4 pool, `None` without Metal.
    fn adapter(pages: u32) -> Option<PagedKVCacheAdapter> {
        let g = PageGeometry {
            layers: 1,
            ..real()
        };
        match create(&g, 4096, pages, pages, u64::MAX) {
            Ok(a) => Some(a),
            Err(e) if crate::test_support::metal_device_absent(&e.to_string()) => {
                eprintln!("skipping Hy4 paged test: {e}");
                None
            }
            Err(e) => panic!("Hy4 paged pool: {e}"),
        }
    }

    /// `OWNER`'s 70 rows recorded with `OTHER`'s pages between its own, so
    /// its table is not `0, 1, 2`.
    fn interleaved(adapter: &mut PagedKVCacheAdapter) {
        adapter.record_tokens_for(OTHER, &[3; 32]).unwrap();
        adapter.record_tokens_for(OWNER, &[3; 32]).unwrap();
        adapter.record_tokens_for(OTHER, &[3; 32]).unwrap();
        adapter.record_tokens_for(OWNER, &[3; TOKENS - 32]).unwrap();
        let table = adapter.block_table().unwrap().block_ids();
        assert_eq!(table.len(), 3);
        assert_ne!(table, vec![0, 1, 2]);
    }

    /// The half-row write of 70 synthetic f16 rows, read back by slot id
    /// through both planes, gives the flat rows bit for bit; `-1` cells read
    /// as masked slot `-1`. A map built for another request is refused.
    /// Mutation: `write_latent` writes cols `0..half` to both planes → red.
    #[test]
    fn half_rows_read_back_as_flat_rows() {
        let Some(mut adapter) = adapter(8) else {
            return;
        };
        interleaved(&mut adapter);
        let bits: Vec<u16> = (0..TOKENS)
            .flat_map(|t| (0..ROW).map(move |c| row_bits(t, c)))
            .collect();
        let flat = MxArray::from_float16(&bits, &[TOKENS as i64, ROW as i64]).unwrap();
        write_latent(&mut adapter, 0, &flat.astype(DType::Float32).unwrap(), 0, 0).unwrap();

        let map = SlotMap::of(&adapter).unwrap();
        let cells: Vec<i32> = (0..TOKENS as i32).chain([-1]).collect();
        let idx = MxArray::from_int32(&cells, &[1, cells.len() as i64]).unwrap();
        let slots = map.slots(&adapter, &idx).unwrap();
        let slot_vec = read_i32(&slots);
        assert_eq!(slot_vec[TOKENS], -1);
        let table = adapter.block_table().unwrap();
        for (t, &slot) in slot_vec[..TOKENS].iter().enumerate() {
            assert_eq!(
                i64::from(slot),
                table.absolute_slot_index(t as u32).unwrap(),
                "token {t}"
            );
        }
        let (k, v) = planes(&adapter, 0).unwrap();
        let gid = MxArray::from_int32(&slot_vec[..TOKENS], &[TOKENS as i64]).unwrap();
        let rows =
            MxArray::concatenate(&k.take(&gid, 0).unwrap(), &v.take(&gid, 0).unwrap(), 1).unwrap();
        assert_eq!(rows.dtype().unwrap(), DType::Float16);
        assert_eq!(rows.to_uint16_native().unwrap(), bits);

        adapter.record_tokens_for(OTHER, &[3]).unwrap();
        assert!(map.slots(&adapter, &idx).is_err());
    }

    /// Rows of the wrong width and rows past the recorded tokens are refused.
    #[test]
    fn write_latent_refuses_bad_rows() {
        let Some(mut adapter) = adapter(4) else {
            return;
        };
        adapter.record_tokens_for(OWNER, &[3; 8]).unwrap();
        let narrow = MxArray::zeros(&[8, 288], Some(DType::Float32)).unwrap();
        assert!(write_latent(&mut adapter, 0, &narrow, 0, 0).is_err());
        let rows = MxArray::zeros(&[8, ROW as i64], Some(DType::Float32)).unwrap();
        assert!(write_latent(&mut adapter, 0, &rows, 4, 0).is_err());
        write_latent(&mut adapter, 0, &rows, 0, 0).unwrap();
    }
}
