//! The DFlash2 lane's two GDN cells and its pending tape (Splash
//! `QwenStateStorage` decode parity and `GdnTapeLane`).
//!
//! Every linear layer's conv history and recurrent state live in one blob
//! per kind (Splash `StateLayout`), twice: the current cell, which the
//! per-layer cache slots point at, and the other parity. A verify folds the
//! previous cycle's retained rows (the pending tape) into the other cell
//! and runs its 8 rows from that state for the outputs only (Splash
//! `gdn_decode_batch_phase`); the commit swaps parity and keeps the verify's
//! own tape and keep as the next pending rows (`swapDecodeParity`). No
//! commit kernel and no replay run per cycle. The current cell lags the
//! logical frontier by the pending rows until [`GdnStateBlobs::settle`].

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};

use super::dflash2::DFLASH2_VERIFY_ROWS;
use super::gated_delta::GdnKernelTape;
use super::gated_delta_net::GdnLayerTape;
use super::layer_cache::Qwen3_5LayerCache;

/// The two cells, the decoder-layer index of each blob row and the pending
/// tape.
pub(crate) struct GdnStateBlobs {
    /// Decoder layer index of blob row `i`.
    pub layers: Vec<usize>,
    /// `[L, K-1, conv_dim]` conv history, bf16: the current cell.
    pub conv: MxArray,
    /// `[L, Hv, Dv, Dk]` recurrent state, f32: the current cell.
    pub recurrent: MxArray,
    /// The other parity: the next verify folds into it. Its contents are
    /// stale and must not be read.
    spare: (MxArray, MxArray),
    /// Per-layer views of the spare blobs, in `layers` order.
    destinations: GdnDestinations,
    /// Value head `hv` reads key head `hv % Hk` (GGUF rows), else
    /// `hv / (Hv / Hk)`.
    tiled_heads: bool,
    /// A one-row zero tape `[k, v, g, beta, raw]` and keep 0: the fold of a
    /// lane with nothing pending (Splash `pendingRows = 0` after a prefill).
    empty: [MxArray; 5],
    zero_keep: MxArray,
    pending: Option<PendingGdn>,
}

/// The last commit's retained rows (Splash `GdnTapeLane` + `pendingRows`).
struct PendingGdn {
    /// Per blob row: k, v, g, beta and the raw conv inputs.
    rows: Vec<[MxArray; 5]>,
    /// `int32 [1]`: the retained row count, built on the host.
    keep: MxArray,
    count: usize,
}

/// Destination rows of the spare blobs.
pub(crate) struct GdnDestinations {
    /// `[1, Hv, Dv, Dk]` f32 rows of the spare recurrent blob.
    pub state: Vec<MxArray>,
    /// `[K-1, W]` bf16 rows of the spare conv blob.
    pub conv: Vec<MxArray>,
}

impl GdnStateBlobs {
    /// Pack every linear layer's `(conv_state, recurrent_state)` into blobs
    /// (one concat per kind: a per-turn cost, not a per-cycle one).
    /// `None` when a linear slot is empty, the states are not (bf16 conv,
    /// f32 recurrent), or the conv width is not `2·Hk·Dk + Hv·Dv`.
    pub(crate) fn pack(caches: &[Qwen3_5LayerCache], tiled_heads: bool) -> Result<Option<Self>> {
        let mut layers = Vec::new();
        let mut convs = Vec::new();
        let mut recs = Vec::new();
        for (index, cache) in caches.iter().enumerate() {
            let Qwen3_5LayerCache::Linear(arrays) = cache else {
                continue;
            };
            let (Some(conv), Some(rec)) = (arrays.get(0), arrays.get(1)) else {
                return Ok(None);
            };
            if conv.dtype()? != DType::BFloat16
                || rec.dtype()? != DType::Float32
                || conv.ndim()? != 3
                || rec.ndim()? != 4
                || conv.shape_at(0)? != 1
                || rec.shape_at(0)? != 1
            {
                return Ok(None);
            }
            layers.push(index);
            convs.push(conv);
            recs.push(rec);
        }
        if layers.is_empty() {
            return Ok(None);
        }
        let (width, hv, dv, dk) = (
            convs[0].shape_at(2)?,
            recs[0].shape_at(1)?,
            recs[0].shape_at(2)?,
            recs[0].shape_at(3)?,
        );
        let keys = width - hv * dv;
        if keys <= 0 || dk <= 0 || keys % (2 * dk) != 0 {
            return Ok(None);
        }
        let hk = keys / (2 * dk);
        if hv % hk != 0 {
            return Ok(None);
        }
        let conv = MxArray::concatenate_many(convs, Some(0))?;
        let recurrent = MxArray::concatenate_many(recs, Some(0))?;
        let spare = (
            MxArray::zeros(conv.shape()?.as_ref(), Some(DType::BFloat16))?,
            MxArray::zeros(recurrent.shape()?.as_ref(), Some(DType::Float32))?,
        );
        let empty = [
            MxArray::zeros(&[1, 1, hk, dk], Some(DType::BFloat16))?,
            MxArray::zeros(&[1, 1, hv, dv], Some(DType::BFloat16))?,
            MxArray::zeros(&[1, 1, hv], Some(DType::Float32))?,
            MxArray::zeros(&[1, 1, hv], Some(DType::BFloat16))?,
            MxArray::zeros(&[1, 1, width], Some(DType::BFloat16))?,
        ];
        let destinations = Self::destinations_of(layers.len(), &spare)?;
        Ok(Some(Self {
            layers,
            conv,
            recurrent,
            spare,
            destinations,
            tiled_heads,
            empty,
            zero_keep: MxArray::from_int32(&[0], &[1])?,
            pending: None,
        }))
    }

    fn destinations_of(rows: usize, spare: &(MxArray, MxArray)) -> Result<GdnDestinations> {
        let rows = 0..rows as i64;
        Ok(GdnDestinations {
            state: rows
                .clone()
                .map(|row| spare.1.slice_axis(0, row, row + 1))
                .collect::<Result<Vec<_>>>()?,
            conv: rows
                .map(|row| spare.0.slice_axis(0, row, row + 1)?.squeeze(Some(&[0])))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    pub(crate) fn destinations(&self) -> &GdnDestinations {
        &self.destinations
    }

    /// The fold inputs of blob row `row` (the gated-delta deferred entry's
    /// `pending` order): the pending tape and keep, or the zero tape and
    /// keep 0 when nothing is pending.
    pub(crate) fn pending(&self, row: usize) -> Result<[&MxArray; 6]> {
        let (tape, keep) = match &self.pending {
            Some(pending) => (
                pending.rows.get(row).ok_or_else(|| {
                    Error::from_reason(format!("GDN pending tape has no row {row}"))
                })?,
                &pending.keep,
            ),
            None => (&self.empty, &self.zero_keep),
        };
        Ok([&tape[0], &tape[1], &tape[2], &tape[3], &tape[4], keep])
    }

    /// Splash `QwenStateStorage::swapDecodeParity`: the verify folded the
    /// pending rows into the other cell, which becomes current; the verify's
    /// own tape and `keep` become the pending rows. `keep` is 1..=8 and at
    /// most every layer tape's rows (Splash `GDN::addDecode`).
    pub(crate) fn retain(&mut self, tape: &[Option<GdnLayerTape>], keep: usize) -> Result<()> {
        if !(1..=DFLASH2_VERIFY_ROWS).contains(&keep) {
            return Err(Error::from_reason(format!(
                "GDN decode parity: {keep} retained rows, not 1..={DFLASH2_VERIFY_ROWS}"
            )));
        }
        let mut rows = Vec::with_capacity(self.layers.len());
        for &layer in &self.layers {
            let layer_tape = tape.get(layer).and_then(Option::as_ref).ok_or_else(|| {
                Error::from_reason(format!(
                    "GDN decode parity: layer {layer} has no verify tape"
                ))
            })?;
            let kernel = &layer_tape.kernel;
            let held = kernel.k.shape_at(1)?;
            if keep as i64 > held || layer_tape.qkv.shape_at(1)? != held {
                return Err(Error::from_reason(format!(
                    "GDN decode parity: {keep} retained rows of a {held}-row tape at layer {layer}"
                )));
            }
            rows.push([
                kernel.k.clone(),
                kernel.v.clone(),
                kernel.g.clone(),
                kernel.beta.clone(),
                layer_tape.qkv.clone(),
            ]);
        }
        let destinations = Self::destinations_of(
            self.layers.len(),
            &(self.conv.clone(), self.recurrent.clone()),
        )?;
        std::mem::swap(&mut self.conv, &mut self.spare.0);
        std::mem::swap(&mut self.recurrent, &mut self.spare.1);
        self.destinations = destinations;
        self.pending = Some(PendingGdn {
            rows,
            keep: MxArray::from_int32(&[keep as i32], &[1])?,
            count: keep,
        });
        Ok(())
    }

    /// Fold the pending rows into the cells the cache slots hold, once, so
    /// they reach the logical frontier (the turn tail and a resumed turn
    /// read them). Splash never resumes a decode state (`requireCommitted`);
    /// mlx-node resumes the live cells, so the finishing turn settles them.
    pub(crate) fn settle(&mut self, caches: &mut [Qwen3_5LayerCache]) -> Result<()> {
        let Some(pending) = self.pending.take() else {
            return Ok(());
        };
        for (row, &layer) in self.layers.iter().enumerate() {
            let arrays = caches
                .get_mut(layer)
                .and_then(Qwen3_5LayerCache::as_arrays_cache_mut)
                .ok_or_else(|| {
                    Error::from_reason(format!("GDN blob layer {layer} is not a linear cache slot"))
                })?;
            let (Some(conv), Some(state)) = (arrays.get(0).cloned(), arrays.get(1).cloned()) else {
                return Err(Error::from_reason(format!(
                    "GDN settle: layer {layer} has no state"
                )));
            };
            let [k, v, g, beta, raw] = &pending.rows[row];
            let tape = GdnLayerTape {
                kernel: settle_kernel_tape(k, v, g, beta, self.tiled_heads)?,
                qkv: raw.clone(),
                conv_kernel_dim: conv.shape_at(1)? as i32 + 1,
            };
            tape.replay_into(arrays, Some(&conv), Some(&state), pending.count)?;
        }
        Ok(())
    }

    /// Point every packed layer's cache slots at its blob rows.
    pub(crate) fn apply_views(&self, caches: &mut [Qwen3_5LayerCache]) -> Result<()> {
        for (row, &layer) in self.layers.iter().enumerate() {
            let arrays = caches
                .get_mut(layer)
                .and_then(Qwen3_5LayerCache::as_arrays_cache_mut)
                .ok_or_else(|| {
                    Error::from_reason(format!("GDN blob layer {layer} is not a linear cache slot"))
                })?;
            let row = row as i64;
            arrays.set(0, self.conv.slice_axis(0, row, row + 1)?)?;
            arrays.set(1, self.recurrent.slice_axis(0, row, row + 1)?)?;
        }
        Ok(())
    }
}

/// The replay tape of a pending row: `gated_delta_replay` maps value head
/// `hv` to key head `hv % Hk`, so grouped compact keys are expanded to one
/// per value head first. The replay never reads `q`.
fn settle_kernel_tape(
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    tiled_heads: bool,
) -> Result<GdnKernelTape> {
    let (hk, hv) = (k.shape_at(2)?, v.shape_at(2)?);
    let expand = !tiled_heads && hk < hv;
    let k = if expand {
        k.repeat((hv / hk) as i32, 2)?
    } else {
        k.clone()
    };
    Ok(GdnKernelTape {
        q: k.clone(),
        k,
        v: v.clone(),
        g: g.clone(),
        beta: beta.clone(),
    })
}

/// What a DFlash2 stepper test records of its lane's GDN cells, and the
/// two oracles its finished cells must equal.
#[cfg(test)]
pub(crate) mod trace {
    use super::*;
    use crate::models::qwen3_5::arrays_cache::ArraysCache;
    use crate::models::qwen3_5::gated_delta::{
        GdnPrologue, GdnTail, gated_delta_fused_complete_deferred,
    };

    /// `(conv [1, K-1, W], state [1, Hv, Dv, Dk])` of every packed layer.
    pub(crate) type Cells = Vec<(MxArray, MxArray)>;

    pub(crate) struct GdnTrace {
        pub layers: Vec<usize>,
        pub tiled: bool,
        /// The packed cells at `begin_dspark_decode`.
        pub start: Cells,
        /// Every cycle's verify tape and keep.
        pub cycles: Vec<(Vec<Option<GdnLayerTape>>, usize)>,
        /// The cells and pending fold inputs right before `finish`.
        pub before: Cells,
        pub pending: Vec<[MxArray; 6]>,
        /// The cells after `finish`.
        pub after: Cells,
    }

    impl GdnTrace {
        pub(crate) fn begin(blobs: &GdnStateBlobs, caches: &[Qwen3_5LayerCache]) -> Self {
            Self {
                layers: blobs.layers.clone(),
                tiled: blobs.tiled_heads,
                start: cells(caches, &blobs.layers),
                cycles: Vec::new(),
                before: Vec::new(),
                pending: Vec::new(),
                after: Vec::new(),
            }
        }

        pub(crate) fn before_finish(
            &mut self,
            blobs: &GdnStateBlobs,
            caches: &[Qwen3_5LayerCache],
        ) {
            self.before = cells(caches, &self.layers);
            self.pending = (0..self.layers.len())
                .map(|row| blobs.pending(row).unwrap().map(MxArray::clone))
                .collect();
        }
    }

    /// Owned copies (the blob rows are written in place later).
    pub(crate) fn cells(caches: &[Qwen3_5LayerCache], layers: &[usize]) -> Cells {
        layers
            .iter()
            .map(|&layer| {
                let Qwen3_5LayerCache::Linear(arrays) = &caches[layer] else {
                    panic!("layer {layer} is not linear");
                };
                let copy = |slot: usize| {
                    let x = arrays.get(slot).unwrap().deep_copy().unwrap();
                    x.eval();
                    x
                };
                (copy(0), copy(1))
            })
            .collect()
    }

    fn expanded(k: &MxArray, v: &MxArray, tiled: bool) -> MxArray {
        let (hk, hv) = (k.shape_at(2).unwrap(), v.shape_at(2).unwrap());
        if tiled || hk == hv {
            k.clone()
        } else {
            k.repeat((hv / hk) as i32, 2).unwrap()
        }
    }

    /// The frontier without any fold: every cycle's kept rows replayed from
    /// the start cells (`GdnLayerTape::replay_into`, the T=1 law).
    pub(crate) fn replay_chain(trace: &GdnTrace) -> Cells {
        let mut cells = trace.start.clone();
        for (tape, keep) in &trace.cycles {
            for (row, &layer) in trace.layers.iter().enumerate() {
                let t = tape[layer].as_ref().expect("verify tape");
                let k = expanded(&t.kernel.k, &t.kernel.v, trace.tiled);
                let replay = GdnLayerTape {
                    kernel: GdnKernelTape {
                        q: k.clone(),
                        k,
                        v: t.kernel.v.clone(),
                        g: t.kernel.g.clone(),
                        beta: t.kernel.beta.clone(),
                    },
                    qkv: t.qkv.clone(),
                    conv_kernel_dim: 4,
                };
                let mut arrays = ArraysCache::new(2);
                replay
                    .replay_into(&mut arrays, Some(&cells[row].0), Some(&cells[row].1), *keep)
                    .unwrap();
                cells[row] = (
                    arrays.get(0).unwrap().clone(),
                    arrays.get(1).unwrap().clone(),
                );
            }
        }
        cells
    }

    /// Splash's fold itself: the deferred kernel over the pending tape from
    /// the cells before `finish` (a zero window, fresh destinations). Its
    /// state and history do not read the window.
    pub(crate) fn kernel_fold(trace: &GdnTrace) -> Cells {
        trace
            .before
            .iter()
            .zip(&trace.pending)
            .map(|((conv, state), pending)| {
                let (w, hv) = (conv.shape_at(2).unwrap(), state.shape_at(1).unwrap());
                let zeros = |shape: &[i64], dtype| MxArray::zeros(shape, Some(dtype)).unwrap();
                let (qkv, gate) = (
                    zeros(&[1, 8, w], DType::BFloat16),
                    zeros(&[1, 8, hv], DType::BFloat16),
                );
                let (taps, heads) = (zeros(&[w, 4], DType::Float32), zeros(&[hv], DType::Float32));
                let z = zeros(&[1, 8, hv * 128], DType::BFloat16);
                let norm = MxArray::ones(&[128], Some(DType::BFloat16)).unwrap();
                let history = conv.squeeze(Some(&[0])).unwrap();
                let out = gated_delta_fused_complete_deferred(
                    &GdnPrologue {
                        qkv: &qkv,
                        a: &gate,
                        b: &gate,
                        conv: &taps,
                        history: &history,
                        scale: &heads,
                        dt_bias: &heads,
                        history_dst: None,
                    },
                    state,
                    &GdnTail {
                        z: &z,
                        norm_weight: &norm,
                        eps: 1e-6,
                        state_dst: None,
                    },
                    false,
                    [
                        &pending[0],
                        &pending[1],
                        &pending[2],
                        &pending[3],
                        &pending[4],
                        &pending[5],
                    ],
                    trace.tiled,
                )
                .expect("deferred kernel");
                (out.history, out.state)
            })
            .collect()
    }

    fn bits(x: &MxArray) -> Vec<u32> {
        let f = x.astype(DType::Float32).unwrap();
        f.eval();
        f.to_float32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    pub(crate) fn assert_cells(got: &Cells, want: &Cells, tag: &str) {
        assert_eq!(got.len(), want.len(), "{tag}: layers");
        for (row, ((gc, gs), (wc, ws))) in got.iter().zip(want).enumerate() {
            assert!(bits(gc) == bits(wc), "{tag}: conv history of row {row}");
            assert!(bits(gs) == bits(ws), "{tag}: recurrent state of row {row}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen3_5::arrays_cache::ArraysCache;
    use crate::nn::Activations;
    use mlx_sys as sys;

    fn metal() -> bool {
        unsafe { sys::mlx_metal_is_available() }
    }

    fn rand_bf16(shape: &[i64]) -> MxArray {
        MxArray::random_normal(shape, 0.0, 0.3, Some(DType::Float32))
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    fn bytes_equal(a: &MxArray, b: &MxArray) -> bool {
        let af = a.astype(DType::Float32).unwrap().to_float32().unwrap();
        let bf = b.astype(DType::Float32).unwrap().to_float32().unwrap();
        a.shape().unwrap().as_ref() == b.shape().unwrap().as_ref()
            && af
                .as_ref()
                .iter()
                .zip(bf.as_ref().iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn aliases(a: &MxArray, b: &MxArray) -> bool {
        MxArray::eval_arrays(&[a, b]).unwrap();
        unsafe { sys::mlx_array_aliases(a.as_raw_ptr(), b.as_raw_ptr()) }
    }

    /// Production-shaped (Hk 16, Hv 48, Dk = Dv 128, conv 4 x 10240) model
    /// with `n_linear` GDN layers interleaved with one full-attention slot,
    /// and a verify tape of `window` rows per layer.
    struct Fixture {
        caches: Vec<Qwen3_5LayerCache>,
        tape: Vec<Option<GdnLayerTape>>,
    }

    fn fixture(n_linear: usize, window: i64) -> Fixture {
        let (hk, hv, dk, dv, width) = (16i64, 48i64, 128i64, 128i64, 10240i64);
        let mut caches = Vec::new();
        let mut tape = Vec::new();
        for layer in 0..n_linear + 1 {
            if layer == n_linear / 2 {
                // A full-attention slot in the middle: not packed.
                caches.push(Qwen3_5LayerCache::new_full_attention());
                tape.push(None);
            }
            let mut arrays = ArraysCache::new(2);
            arrays.set(0, rand_bf16(&[1, 3, width])).unwrap();
            arrays
                .set(
                    1,
                    MxArray::random_normal(&[1, hv, dv, dk], 0.0, 0.3, Some(DType::Float32))
                        .unwrap(),
                )
                .unwrap();
            caches.push(Qwen3_5LayerCache::Linear(arrays));
            let gate = |dtype: DType| {
                Activations::sigmoid(
                    &MxArray::random_normal(&[1, window, hv], 0.0, 1.0, Some(DType::Float32))
                        .unwrap(),
                )
                .unwrap()
                .astype(dtype)
                .unwrap()
            };
            tape.push(Some(GdnLayerTape {
                kernel: GdnKernelTape {
                    q: rand_bf16(&[1, window, hk, dk]),
                    k: rand_bf16(&[1, window, hk, dk]),
                    v: rand_bf16(&[1, window, hv, dv]),
                    g: gate(DType::Float32),
                    beta: gate(DType::BFloat16),
                },
                qkv: rand_bf16(&[1, window, width]),
                conv_kernel_dim: 4,
            }));
        }
        // Drop the trailing extra linear layer so the count is exact.
        caches.pop();
        tape.pop();
        Fixture { caches, tape }
    }

    fn linear_states(caches: &[Qwen3_5LayerCache]) -> Vec<(MxArray, MxArray)> {
        caches
            .iter()
            .filter_map(|cache| match cache {
                Qwen3_5LayerCache::Linear(arrays) => Some((
                    arrays.get(0).unwrap().clone(),
                    arrays.get(1).unwrap().clone(),
                )),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn pack_and_views_keep_every_layer_byte_identical() {
        if !metal() {
            return;
        }
        let mut fx = fixture(5, 8);
        let before = linear_states(&fx.caches);
        let blobs = GdnStateBlobs::pack(&fx.caches, true)
            .unwrap()
            .expect("packable");
        assert_eq!(blobs.layers, vec![0, 1, 3, 4, 5]);
        assert_eq!(blobs.conv.shape().unwrap().as_ref(), [5, 3, 10240]);
        assert_eq!(blobs.recurrent.shape().unwrap().as_ref(), [5, 48, 128, 128]);
        blobs.apply_views(&mut fx.caches).unwrap();
        let after = linear_states(&fx.caches);
        for ((c0, r0), (c1, r1)) in before.iter().zip(after.iter()) {
            assert!(bytes_equal(c0, c1), "conv view differs");
            assert!(bytes_equal(r0, r1), "recurrent view differs");
        }
        // Nothing pending after a pack: the zero one-row tape and keep 0.
        let fold = blobs.pending(0).unwrap();
        assert_eq!(fold[0].shape().unwrap().as_ref(), [1, 1, 16, 128]);
        assert_eq!(fold[4].shape().unwrap().as_ref(), [1, 1, 10240]);
        assert_eq!(fold[5].to_int32().unwrap().as_ref(), [0]);
        // An empty slot is not packable.
        fx.caches[0].reset();
        assert!(GdnStateBlobs::pack(&fx.caches, true).unwrap().is_none());
    }

    /// Splash `swapDecodeParity` + the finish settle: `retain` takes keep
    /// 1..=8 within the tape, swaps the two cells (the new current is the
    /// old spare buffer and the next destinations are the old current
    /// rows) and holds the verify tape and keep as the pending fold inputs;
    /// `settle` leaves every layer at `replay_into` of the pending rows from
    /// the cells the slots hold, grouped heads through the expanded keys.
    /// Mutations: R1 (keep 0 accepted) and E1 (no grouped expansion) red.
    #[test]
    fn retain_swaps_parity_and_settle_matches_replay() {
        if !metal() {
            return;
        }
        for tiled in [true, false] {
            for keep in 1..=DFLASH2_VERIFY_ROWS {
                let tag = format!("tiled={tiled} keep={keep}");
                let mut fx = fixture(7, 8);
                let mut blobs = GdnStateBlobs::pack(&fx.caches, tiled)
                    .unwrap()
                    .expect("packable");
                blobs.apply_views(&mut fx.caches).unwrap();
                let (current, spare) = (blobs.recurrent.clone(), blobs.spare.1.clone());
                let (current_conv, spare_conv) = (blobs.conv.clone(), blobs.spare.0.clone());
                for bad in [0, DFLASH2_VERIFY_ROWS + 1] {
                    assert!(blobs.retain(&fx.tape, bad).is_err(), "{tag}: keep {bad}");
                }
                let short = fx
                    .tape
                    .iter()
                    .map(|t| {
                        t.as_ref().map(|t| GdnLayerTape {
                            kernel: GdnKernelTape {
                                q: t.kernel.q.slice_axis(1, 0, 4).unwrap(),
                                k: t.kernel.k.slice_axis(1, 0, 4).unwrap(),
                                v: t.kernel.v.slice_axis(1, 0, 4).unwrap(),
                                g: t.kernel.g.slice_axis(1, 0, 4).unwrap(),
                                beta: t.kernel.beta.slice_axis(1, 0, 4).unwrap(),
                            },
                            qkv: t.qkv.slice_axis(1, 0, 4).unwrap(),
                            conv_kernel_dim: 4,
                        })
                    })
                    .collect::<Vec<_>>();
                assert!(
                    blobs.retain(&short, 5).is_err(),
                    "{tag}: keep past the tape"
                );
                assert!(
                    blobs.pending.is_none(),
                    "{tag}: a refused keep retains nothing"
                );
                blobs.retain(&fx.tape, keep).unwrap();
                assert!(aliases(&blobs.recurrent, &spare), "{tag}: current state");
                assert!(aliases(&blobs.conv, &spare_conv), "{tag}: current conv");
                assert!(aliases(&blobs.spare.1, &current), "{tag}: spare state");
                assert!(aliases(&blobs.spare.0, &current_conv), "{tag}: spare conv");
                assert!(aliases(
                    &blobs.destinations().state[2],
                    &current.slice_axis(0, 2, 3).unwrap()
                ));
                for (row, &layer) in blobs.layers.iter().enumerate() {
                    let t = fx.tape[layer].as_ref().unwrap();
                    let fold = blobs.pending(row).unwrap();
                    for (got, want) in fold.iter().zip([
                        &t.kernel.k,
                        &t.kernel.v,
                        &t.kernel.g,
                        &t.kernel.beta,
                        &t.qkv,
                    ]) {
                        assert!(aliases(got, want), "{tag}: pending row {row}");
                    }
                    assert_eq!(fold[5].to_int32().unwrap().as_ref(), [keep as i32]);
                }
                let before = linear_states(&fx.caches);
                let mut reference = Vec::new();
                for (row, &layer) in blobs.layers.iter().enumerate() {
                    let t = fx.tape[layer].as_ref().unwrap();
                    let k = if tiled {
                        t.kernel.k.clone()
                    } else {
                        t.kernel.k.repeat(3, 2).unwrap()
                    };
                    let expanded = GdnLayerTape {
                        kernel: GdnKernelTape {
                            q: k.clone(),
                            k,
                            v: t.kernel.v.clone(),
                            g: t.kernel.g.clone(),
                            beta: t.kernel.beta.clone(),
                        },
                        qkv: t.qkv.clone(),
                        conv_kernel_dim: 4,
                    };
                    let mut arrays = ArraysCache::new(2);
                    expanded
                        .replay_into(
                            &mut arrays,
                            Some(&before[row].0),
                            Some(&before[row].1),
                            keep,
                        )
                        .unwrap();
                    reference.push((
                        arrays.get(0).unwrap().clone(),
                        arrays.get(1).unwrap().clone(),
                    ));
                }
                blobs.settle(&mut fx.caches).unwrap();
                assert!(
                    blobs.pending.is_none(),
                    "{tag}: settle clears the pending rows"
                );
                for (row, ((rc, rr), (sc, sr))) in reference
                    .iter()
                    .zip(linear_states(&fx.caches).iter())
                    .enumerate()
                {
                    assert!(bytes_equal(rc, sc), "{tag}: conv row {row}");
                    assert!(bytes_equal(rr, sr), "{tag}: state row {row}");
                }
            }
        }
    }
}
