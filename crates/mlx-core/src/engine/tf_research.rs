//! Research-only teacher forcing for the DSpark/DFlash2 loop
//! (`docs/research/splash-qwen38.md` §6).
//!
//! `MLX_DFLASH2_TF_RECORD=<dir>`: a normal turn writes its generated ids to
//! `<dir>/ref-<prompt key>.json` (kept when an existing file is longer).
//! `MLX_DFLASH2_TF_DIR=<dir>`: a turn whose prompt has a reference file there
//! verifies the REFERENCE ids instead of the draft ids through the real
//! verify path, accepts the leading drafts that match both the reference and
//! the target argmax (`a_live`), commits only reference ids, ends when the
//! reference runs out, and appends one line per turn to
//! `<dir>/cycles-<MLX_DFLASH2_TF_LABEL>.jsonl`:
//! `{"key","maxNew","generated","cycles":[[pos,len,a_ref,a_live,flip,wall_us],..]}`.
//! `flip` is the first committed row where the target argmax differs from the
//! reference (-1 = none); `wall_us` is the loop's propose→commit cycle timer.
//!
//! Both env vars are read once per turn; with neither set every hook is a
//! no-op on an `Off` enum.

use std::path::{Path, PathBuf};
use std::time::Duration;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::backend::DsparkProposal;

pub(crate) struct TfCycle {
    pos: usize,
    len: usize,
    a_ref: usize,
    a_live: usize,
    flip: Option<usize>,
    wall_us: u64,
}

pub(crate) enum TfMode {
    Off,
    Force {
        dir: PathBuf,
        key: String,
        reference: Vec<u32>,
        cycles: Vec<TfCycle>,
    },
    Record {
        dir: PathBuf,
        key: String,
    },
}

fn prompt_key(prompt: &[u32]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &id in prompt {
        for byte in id.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!("{}-{hash:016x}", prompt.len())
}

fn read_reference(path: &Path) -> Result<Vec<u32>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::from_reason(format!("TF reference {}: {e}", path.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::from_reason(format!("TF reference {}: {e}", path.display())))?;
    let tokens = value["tokens"]
        .as_array()
        .ok_or_else(|| Error::from_reason("TF reference has no tokens"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| Error::from_reason("TF reference token is not a u32"))
        })
        .collect::<Result<Vec<u32>>>()?;
    if tokens.is_empty() {
        return Err(Error::from_reason("TF reference is empty"));
    }
    Ok(tokens)
}

/// Reference verify ids `r[pos ..= pos + len]` for the anchor at `pos`.
fn reference_window(reference: &[u32], pos: usize, anchor: u32, len: usize) -> Result<Vec<u32>> {
    if reference.get(pos) != Some(&anchor) {
        return Err(Error::from_reason(format!(
            "TF anchor {anchor} at {pos} is not the reference token {:?}",
            reference.get(pos)
        )));
    }
    reference
        .get(pos..=pos + len)
        .map(<[u32]>::to_vec)
        .ok_or_else(|| Error::from_reason("TF verify window runs past the reference"))
}

/// `(a_ref, a_live, flip)` for `drafts` against the target argmax rows
/// `target[0..=len]` and the reference continuation `next[0..=len]`.
fn count_accept(drafts: &[i32], target: &[u32], next: &[u32]) -> (usize, usize, Option<usize>) {
    let len = drafts.len();
    let a_ref = (0..len)
        .take_while(|&i| drafts[i] as u32 == next[i])
        .count();
    let a_live = (0..len)
        .take_while(|&i| drafts[i] as u32 == target[i] && target[i] == next[i])
        .count();
    let flip = (0..=a_live).find(|&i| target[i] != next[i]);
    (a_ref, a_live, flip)
}

fn keeps_existing(existing: Option<usize>, generated: usize) -> bool {
    existing.is_some_and(|len| len >= generated)
}

impl TfMode {
    pub(crate) fn begin(prompt: &[u32]) -> Result<Self> {
        let key = prompt_key(prompt);
        if let Ok(dir) = std::env::var("MLX_DFLASH2_TF_DIR") {
            let dir = PathBuf::from(dir);
            let reference = read_reference(&dir.join(format!("ref-{key}.json")))?;
            return Ok(TfMode::Force {
                dir,
                key,
                reference,
                cycles: Vec::new(),
            });
        }
        if let Ok(dir) = std::env::var("MLX_DFLASH2_TF_RECORD") {
            return Ok(TfMode::Record {
                dir: PathBuf::from(dir),
                key,
            });
        }
        Ok(TfMode::Off)
    }

    pub(crate) fn is_forcing(&self) -> bool {
        matches!(self, TfMode::Force { .. })
    }

    /// Turn budget: never generate past the reference.
    pub(crate) fn cap_max_new(&self, max_new: usize) -> usize {
        match self {
            TfMode::Force { reference, .. } => max_new.min(reference.len()),
            _ => max_new,
        }
    }

    /// Verify input for this cycle: `None` unless forcing.
    pub(crate) fn verify_ids(
        &self,
        generated: &[u32],
        anchor: u32,
        len: usize,
    ) -> Result<Option<Vec<u32>>> {
        let TfMode::Force { reference, .. } = self else {
            return Ok(None);
        };
        reference_window(reference, generated.len().saturating_sub(1), anchor, len).map(Some)
    }

    /// Teacher-forced acceptance: `(accepted drafts = a_live, boundary = next
    /// reference id)`. Forces the verify graph, then back-fills a device
    /// proposal (its graph is independent of the reference verify).
    pub(crate) fn accept(
        &mut self,
        logits: &MxArray,
        proposal: &mut DsparkProposal,
        generated: &[u32],
    ) -> Result<(usize, u32)> {
        let TfMode::Force {
            reference, cycles, ..
        } = self
        else {
            return Err(Error::from_reason("TF accept without a reference"));
        };
        let pos = generated.len().saturating_sub(1);
        let argmax = logits.argmax(-1, None)?;
        argmax.eval();
        proposal.materialize_draft_ids()?;
        let len = proposal.draft_ids.len();
        let mut target = Vec::with_capacity(len + 1);
        for i in 0..=len {
            target.push(argmax.item_at_int32(i)? as u32);
        }
        let next = reference
            .get(pos + 1..=pos + 1 + len)
            .ok_or_else(|| Error::from_reason("TF accept runs past the reference"))?;
        let (a_ref, a_live, flip) = count_accept(&proposal.draft_ids, &target, next);
        cycles.push(TfCycle {
            pos,
            len,
            a_ref,
            a_live,
            flip,
            wall_us: 0,
        });
        Ok((a_live, next[a_live]))
    }

    pub(crate) fn end_cycle(&mut self, wall: Duration) {
        if let TfMode::Force { cycles, .. } = self
            && let Some(cycle) = cycles.last_mut()
        {
            cycle.wall_us = wall.as_micros().min(u128::from(u64::MAX)) as u64;
        }
    }

    pub(crate) fn finish(self, generated: &[u32], max_new: i32) -> Result<()> {
        match self {
            TfMode::Off => Ok(()),
            TfMode::Force {
                dir, key, cycles, ..
            } => {
                use std::io::Write;
                let label = std::env::var("MLX_DFLASH2_TF_LABEL").unwrap_or_else(|_| "run".into());
                let rows = cycles
                    .iter()
                    .map(|c| {
                        serde_json::json!([
                            c.pos,
                            c.len,
                            c.a_ref,
                            c.a_live,
                            c.flip.map_or(-1, |f| f as i64),
                            c.wall_us
                        ])
                    })
                    .collect::<Vec<_>>();
                let line = serde_json::json!({
                    "key": key,
                    "maxNew": max_new,
                    "generated": generated.len(),
                    "cycles": rows,
                });
                let path = dir.join(format!("cycles-{label}.jsonl"));
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|e| Error::from_reason(format!("TF log {}: {e}", path.display())))?;
                writeln!(file, "{line}")
                    .map_err(|e| Error::from_reason(format!("TF log {}: {e}", path.display())))
            }
            TfMode::Record { dir, key } => {
                let path = dir.join(format!("ref-{key}.json"));
                let existing = read_reference(&path).ok().map(|tokens| tokens.len());
                if keeps_existing(existing, generated.len()) {
                    return Ok(());
                }
                let body = serde_json::json!({ "key": key, "tokens": generated });
                std::fs::write(&path, body.to_string())
                    .map_err(|e| Error::from_reason(format!("TF record {}: {e}", path.display())))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_key_is_length_prefixed_and_order_sensitive() {
        let a = prompt_key(&[1, 2, 3]);
        assert!(a.starts_with("3-"));
        assert_eq!(a, prompt_key(&[1, 2, 3]));
        assert_ne!(a, prompt_key(&[3, 2, 1]));
        assert_ne!(a, prompt_key(&[1, 2, 3, 4]));
    }

    #[test]
    fn reference_window_checks_anchor_and_bounds() {
        let reference = [10, 11, 12, 13, 14];
        assert_eq!(
            reference_window(&reference, 1, 11, 2).unwrap(),
            vec![11, 12, 13]
        );
        assert_eq!(reference_window(&reference, 4, 14, 0).unwrap(), vec![14]);
        assert!(reference_window(&reference, 1, 99, 2).is_err());
        assert!(reference_window(&reference, 5, 14, 0).is_err());
        assert!(reference_window(&reference, 3, 13, 2).is_err());
    }

    #[test]
    fn count_accept_separates_reference_and_live_prefixes() {
        // Drafts match the reference for 3 rows (row 3 is wrong); the
        // target leaves the reference at row 2, so the live prefix is 2 and
        // the boundary row flips.
        let drafts = [5, 6, 7, 80];
        let next = [5, 6, 7, 8, 9];
        let target = [5, 6, 70, 8, 9];
        assert_eq!(count_accept(&drafts, &target, &next), (3, 2, Some(2)));
        // Target tracks the reference: live == ref prefix, boundary agrees.
        assert_eq!(count_accept(&drafts, &next, &next), (3, 3, None));
        // Full agreement: every draft accepted, no flip.
        assert_eq!(count_accept(&[5, 6, 7, 8], &next, &next), (4, 4, None));
        // Draft wrong at row 0 but target still tracks the reference:
        // nothing accepted, no flip.
        let target = [5, 6, 7, 8, 9];
        assert_eq!(count_accept(&[50, 6, 7, 8], &target, &next), (0, 0, None));
        // Draft right for the reference but the target already left it at
        // row 0: a_ref counts, a_live does not, flip at 0.
        let target = [55, 6, 7, 8, 9];
        assert_eq!(count_accept(&drafts, &target, &next), (3, 0, Some(0)));
        // Zero-draft cycle: only the boundary row is judged.
        assert_eq!(count_accept(&[], &[9], &[9]), (0, 0, None));
        assert_eq!(count_accept(&[], &[8], &[9]), (0, 0, Some(0)));
    }

    #[test]
    fn cap_max_new_stops_at_the_reference() {
        let force = TfMode::Force {
            dir: PathBuf::new(),
            key: String::new(),
            reference: vec![1, 2, 3],
            cycles: Vec::new(),
        };
        assert_eq!(force.cap_max_new(10), 3);
        assert_eq!(force.cap_max_new(2), 2);
        assert_eq!(TfMode::Off.cap_max_new(10), 10);
        assert!(TfMode::Off.verify_ids(&[1], 1, 2).unwrap().is_none());
        assert_eq!(force.verify_ids(&[1], 1, 2).unwrap(), Some(vec![1, 2, 3]));
    }

    #[test]
    fn record_keeps_the_longer_reference() {
        assert!(keeps_existing(Some(5), 5));
        assert!(keeps_existing(Some(6), 5));
        assert!(!keeps_existing(Some(4), 5));
        assert!(!keeps_existing(None, 5));

        let dir = std::env::temp_dir().join(format!("mlx-tf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let record = |dir: &Path| TfMode::Record {
            dir: dir.to_path_buf(),
            key: "k".into(),
        };
        let path = dir.join("ref-k.json");
        record(&dir).finish(&[1, 2, 3], 3).unwrap();
        assert_eq!(read_reference(&path).unwrap(), vec![1, 2, 3]);
        record(&dir).finish(&[9, 9], 2).unwrap();
        assert_eq!(read_reference(&path).unwrap(), vec![1, 2, 3]);
        record(&dir).finish(&[4, 5, 6, 7], 4).unwrap();
        assert_eq!(read_reference(&path).unwrap(), vec![4, 5, 6, 7]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
