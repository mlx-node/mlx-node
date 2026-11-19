//! Owner cells: parts by layer kind, the swap round trip, truncation
//! refusals, and the state at a block row (one seed row, MTP segments
//! shared between states) back into a cell.

use std::sync::Arc;

use super::config::{Attn, HyV4FlashConfig};
use super::indexer::{IndexerGeometry, PagedIndexerState, TokTail};
use super::mtp::Draft;
use super::owner::{CellLayer, OwnerCell, Owners};
use super::paged::pool_layers;
use super::prefix_state::{Key, PrefixState, StateLayer, States, Stored};
use super::tests_model::tiny_cfg;
use crate::array::{DType, MxArray};

fn rows(n: usize, width: usize, from: usize) -> MxArray {
    let v: Vec<f32> = (from * width..(from + n) * width)
        .map(|x| (x % 1024) as f32)
        .collect();
    MxArray::from_float32(&v, &[n as i64, width as i64]).unwrap()
}

fn read(x: &MxArray, n: usize) -> Vec<f32> {
    let w = x.shape_at(1).unwrap();
    x.slice(&[0, 0], &[n as i64, w])
        .unwrap()
        .astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .to_vec()
}

fn geometry(cfg: &HyV4FlashConfig) -> IndexerGeometry {
    let l = cfg
        .trunk()
        .iter()
        .position(|l| l.attn == Attn::Indexed)
        .unwrap();
    IndexerGeometry::of(cfg, l).unwrap()
}

/// A tail of `len` positions holding its last rows (as many as it keeps).
fn tail(g: &IndexerGeometry, len: usize) -> TokTail {
    let held = (g.kpool - 1 + super::cache::WINDOW_SLACK).min(len);
    let r = (held > 0).then(|| rows(held, g.head_dim, len - held));
    TokTail::of_rows(g, r, len).unwrap()
}

/// Appends `ids` to every trunk layer of `cell` with synthetic rows.
fn grow(cfg: &HyV4FlashConfig, cell: &mut OwnerCell, ids: &[u32]) {
    let g = geometry(cfg);
    let (start, n) = (cell.len(), ids.len());
    for p in cell.layers_mut() {
        match p {
            CellLayer::Window(r) => {
                r.append(&rows(n, cfg.mla.kv_row(), start), start).unwrap();
            }
            CellLayer::Index(ix) => {
                *ix = PagedIndexerState::at(ix.layer(), tail(&g, start + n));
            }
        }
    }
    cell.advance(ids).unwrap();
}

fn lens(cell: &OwnerCell) -> Vec<(usize, usize)> {
    cell.layers()
        .iter()
        .map(|p| match p {
            CellLayer::Window(r) => (r.len(), r.kept()),
            CellLayer::Index(ix) => (ix.len(), ix.tail().held()),
        })
        .collect()
}

fn ids(n: usize, from: u32) -> Vec<u32> {
    (0..n as u32).map(|i| from + i).collect()
}

/// A sliding layer holds a window ring, an indexed one its pool layer's
/// tail; the cell keeps `max_ngram - 1` recent ids.
/// Mutation: `layers_of` numbers pool layers by trunk index → red.
#[test]
fn cell_layers_follow_the_config() {
    let cfg = tiny_cfg();
    let cell = OwnerCell::new(&cfg, None).unwrap();
    let pools = pool_layers(&cfg);
    assert_eq!(cell.layers().len(), cfg.trunk().len());
    for ((p, lc), pool) in cell.layers().iter().zip(cfg.trunk()).zip(&pools) {
        match (p, lc.attn) {
            (CellLayer::Window(r), Attn::Sliding) => assert_eq!(r.window(), cfg.window),
            (CellLayer::Index(ix), Attn::Indexed) => assert_eq!(Some(ix.layer()), *pool),
            _ => panic!("layer kind mismatch"),
        }
    }
    assert!(cell.is_empty() && cell.recent().is_empty());
    assert_eq!(cell.nbytes(), 0);
    let mut bad = tiny_cfg();
    bad.indexer.kpool = 3;
    assert!(OwnerCell::new(&bad, None).is_err());
}

/// Out of the table and back: every part keeps its lengths and held rows,
/// the recent ids and the bytes; a second cell for the same owner is
/// refused.
/// Mutation: `take` returns a fresh cell → red.
#[test]
fn swap_round_trip_keeps_lens() {
    let cfg = tiny_cfg();
    let mut owners = Owners::new();
    let a = owners.activate(1, || OwnerCell::new(&cfg, None)).unwrap();
    grow(&cfg, a, &ids(20, 3));
    grow(&cfg, a, &ids(1, 40));
    let before = (lens(a), a.recent().to_vec(), a.nbytes(), a.len());
    assert_eq!(before.1, vec![22, 40]);
    owners.activate(2, || OwnerCell::new(&cfg, None)).unwrap();
    assert_eq!(owners.active(), Some(2));
    owners.park();
    let cell = owners.take(1).unwrap();
    assert!(!owners.contains(1));
    assert_eq!(owners.held_bytes(1), 0);
    owners.put(1, cell).unwrap();
    let a = owners.activate(1, || panic!("owner 1 is held")).unwrap();
    assert_eq!((lens(a), a.recent().to_vec(), a.nbytes(), a.len()), before);
    let dup = OwnerCell::new(&cfg, None).unwrap();
    assert!(owners.put(1, dup).is_err());
    assert_eq!(owners.bytes(), owners.held_bytes(1) + owners.held_bytes(2));
    assert!(owners.release(1).is_some());
    assert_eq!(owners.active(), None);
}

/// A cut past the window's slack is refused with nothing changed; one
/// inside it keeps the recent ids of the history.
#[test]
fn truncate_refusal_changes_nothing() {
    let cfg = tiny_cfg();
    let mut cell = OwnerCell::new(&cfg, None).unwrap();
    let history = ids(40, 100);
    grow(&cfg, &mut cell, &history);
    let before = lens(&cell);
    assert!(!cell.can_truncate(20));
    assert!(cell.truncate(20, &history).is_err());
    assert_eq!(lens(&cell), before);
    assert!(cell.truncate(38, &history[..10]).is_err());
    cell.truncate(38, &history).unwrap();
    assert_eq!(cell.len(), 38);
    assert_eq!(cell.recent(), &history[36..38]);
    assert!(lens(&cell).iter().all(|l| l.0 == 38));
    cell.truncate(0, &history).unwrap();
    assert!(cell.is_empty() && cell.recent().is_empty());
}

/// A draft of `cfg` for a context of 64.
fn draft(cfg: &HyV4FlashConfig) -> Draft {
    Draft::new(cfg.mla.kv_row() as i64, &geometry(cfg), 64)
}

/// Fills `d` with `m` settled MTP rows and the seed of row `m`.
fn settle(cfg: &HyV4FlashConfig, d: &mut Draft, m: usize) {
    let g = geometry(cfg);
    let kv = rows(m, cfg.mla.kv_row(), 0);
    let pooled = rows(m / g.kpool, g.head_dim, 500);
    let seed = rows(1, cfg.dim, 900);
    d.fill(Some(&kv), Some(&pooled), tail(&g, m), seed).unwrap();
}

/// States only at block rows, and only with the draft settled there.
#[test]
fn states_need_a_block_row_and_a_settled_draft() {
    let cfg = tiny_cfg();
    let mut cell = OwnerCell::new(&cfg, None).unwrap();
    grow(&cfg, &mut cell, &ids(31, 1));
    assert!(cell.state().unwrap().is_none());
    grow(&cfg, &mut cell, &ids(1, 1));
    let s = cell.state().unwrap().unwrap();
    assert_eq!(s.row, 32);
    assert!(s.draft.is_none());
    let mut lagged = OwnerCell::new(&cfg, Some(draft(&cfg))).unwrap();
    grow(&cfg, &mut lagged, &ids(32, 1));
    assert!(lagged.state().unwrap().is_none());
    settle(&cfg, lagged.draft_mut().unwrap(), 30);
    assert!(lagged.state().unwrap().is_none());
}

/// A state at row 64 with the MTP layer: one seed row, one segment shared
/// by every state of the cell, 31 open rows; a cell made from it holds the
/// same lengths and MTP rows, and the seed of row 63 only. Two states held
/// count the segment once.
/// Mutation: `draft_state` copies segments on every call → red (ptr_eq).
#[test]
fn state_round_trip_with_the_draft() {
    let cfg = tiny_cfg();
    let g = geometry(&cfg);
    let history = ids(64, 10);
    let mut cell = OwnerCell::new(&cfg, Some(draft(&cfg))).unwrap();
    grow(&cfg, &mut cell, &history);
    settle(&cfg, cell.draft_mut().unwrap(), 63);
    let s = cell.state().unwrap().unwrap();
    let again = cell.state().unwrap().unwrap();
    let (d, d2) = (s.draft.as_ref().unwrap(), again.draft.as_ref().unwrap());
    assert_eq!(d.seed.shape().unwrap().as_ref(), &[1, cfg.dim as i64]);
    assert_eq!(d.segments.len(), 1);
    assert!(Arc::ptr_eq(&d.segments[0], &d2.segments[0]));
    assert_eq!(d.rows().unwrap(), 63);
    assert_eq!(d.tail.len(), 63);
    assert!(s.layers.iter().all(|l| match l {
        StateLayer::Window(r) => r.len() == 64,
        StateLayer::Tail(t) => t.len() == 64,
    }));

    let back = OwnerCell::from_state(&cfg, &s, &history, Some(draft(&cfg))).unwrap();
    assert_eq!(lens(&back), lens(&cell));
    assert_eq!(back.recent(), &history[62..]);
    let (bd, cd) = (back.draft().unwrap(), cell.draft().unwrap());
    assert_eq!(bd.len(), 63);
    assert_eq!(
        read(bd.kv().rows().unwrap(), 63),
        read(cd.kv().rows().unwrap(), 63)
    );
    let groups = 63 / g.kpool;
    assert_eq!(bd.index().pooled_len(), groups);
    assert_eq!(
        read(bd.index().pooled_rows().unwrap(), groups),
        read(cd.index().pooled_rows().unwrap(), groups)
    );
    assert!(bd.seed(63).is_some() && bd.seed(62).is_none());
    assert!(OwnerCell::from_state(&cfg, &s, &history[..63], Some(draft(&cfg))).is_err());
    assert!(OwnerCell::from_state(&cfg, &s, &history, None).is_err());

    let seg = d.segments[0].nbytes() as u64;
    let (b1, b2) = (s.bytes(), again.bytes());
    let mut st = States::<PrefixState>::new();
    let other = ids(64, 5000);
    st.publish(Key::text(&history, 1), s, super::ladder::Kind::Prompt)
        .unwrap();
    st.publish(Key::text(&other, 1), again, super::ladder::Kind::Prompt)
        .unwrap();
    assert_eq!(st.held_bytes(), b1 + b2 - seg);
}
