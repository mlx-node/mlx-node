//! Packed GDN state for the DFlash2 stepper (Splash's `StateLayout`: every
//! linear layer's conv history and recurrent state live in one blob per
//! kind, and a commit produces the next pair).
//!
//! The per-layer `ArraysCache` slots stay the public contract: after a pack
//! or a commit they hold axis-0 views of the blobs, so every per-layer
//! consumer (the compiled verify's graph inputs, snapshots, eager fallbacks)
//! keeps working unchanged. The fused commit (`mlx_gdn_commit_all`) replays
//! every layer's accepted prefix from the blobs in a few dispatches instead
//! of one replay plus a conv concat per layer, writing into a spare pair of
//! blobs that then swaps roles with the current pair (`current` / `next`
//! parity, as Splash's `swapParity`): no per-cycle 144 MiB allocation.

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use mlx_sys as sys;

use super::gated_delta_net::GdnLayerTape;
use super::layer_cache::Qwen3_5LayerCache;

/// The two state blobs plus the decoder-layer index each blob row belongs to.
pub(crate) struct GdnStateBlobs {
    /// Decoder layer index of blob row `i`.
    pub layers: Vec<usize>,
    /// `[L, K-1, conv_dim]` conv history, bf16.
    pub conv: MxArray,
    /// `[L, Hv, Dv, Dk]` recurrent state, f32.
    pub recurrent: MxArray,
    /// The other parity: the blobs the next commit writes into. Their
    /// contents are stale (the state before the last commit) and must not
    /// be read.
    spare: (MxArray, MxArray),
}

impl GdnStateBlobs {
    /// Pack every linear layer's `(conv_state, recurrent_state)` into blobs
    /// (one concat per kind: a per-turn cost, not a per-cycle one).
    /// `None` when a linear slot is empty or the states are not (bf16 conv,
    /// f32 recurrent) — the caller keeps the per-layer path.
    pub(crate) fn pack(caches: &[Qwen3_5LayerCache]) -> Result<Option<Self>> {
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
        let conv = MxArray::concatenate_many(convs, Some(0))?;
        let recurrent = MxArray::concatenate_many(recs, Some(0))?;
        let spare = (
            MxArray::zeros(conv.shape()?.as_ref(), Some(DType::BFloat16))?,
            MxArray::zeros(recurrent.shape()?.as_ref(), Some(DType::Float32))?,
        );
        Ok(Some(Self {
            layers,
            conv,
            recurrent,
            spare,
        }))
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

    /// Replay the first `keep` recorded tokens of every packed layer's tape
    /// from these (pre-verify) blobs into a new pair — the fused form of
    /// [`GdnLayerTape::replay_into`] over all layers. `None` when the fused
    /// kernel declines (shape, dtype, no Metal); the caller falls back to
    /// the per-layer replay.
    pub(crate) fn commit(
        &self,
        tape: &[Option<GdnLayerTape>],
        keep: usize,
    ) -> Result<Option<Self>> {
        // Nothing kept: the per-layer path clones the pre-verify state, so
        // decline rather than hand the kernel a zero-length window.
        if keep == 0 {
            return Ok(None);
        }
        let keep = i32::try_from(keep)
            .map_err(|_| Error::from_reason("GDN blob commit keep is too large"))?;
        let mut k = Vec::with_capacity(self.layers.len());
        let mut v = Vec::with_capacity(self.layers.len());
        let mut g = Vec::with_capacity(self.layers.len());
        let mut beta = Vec::with_capacity(self.layers.len());
        let mut qkv = Vec::with_capacity(self.layers.len());
        for &layer in &self.layers {
            let Some(layer_tape) = tape.get(layer).and_then(Option::as_ref) else {
                return Err(Error::from_reason(format!(
                    "GDN blob commit: layer {layer} has no verify tape"
                )));
            };
            if layer_tape.conv_kernel_dim - 1 != self.conv.shape_at(1)? as i32 {
                return Ok(None);
            }
            k.push(layer_tape.kernel.k.as_raw_ptr());
            v.push(layer_tape.kernel.v.as_raw_ptr());
            g.push(layer_tape.kernel.g.as_raw_ptr());
            beta.push(layer_tape.kernel.beta.as_raw_ptr());
            qkv.push(layer_tape.qkv.as_raw_ptr());
        }
        let mut out_rec: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_conv: *mut sys::mlx_array = std::ptr::null_mut();
        // SAFETY: every pointer is a live array handle for the duration of
        // the call; the out pointers receive owned handles or stay null.
        let ok = unsafe {
            sys::mlx_gdn_commit_all(
                self.recurrent.as_raw_ptr(),
                self.conv.as_raw_ptr(),
                self.spare.1.as_raw_ptr(),
                self.spare.0.as_raw_ptr(),
                self.layers.len() as i32,
                k.as_ptr(),
                v.as_ptr(),
                g.as_ptr(),
                beta.as_ptr(),
                qkv.as_ptr(),
                keep,
                &mut out_rec,
                &mut out_conv,
            )
        };
        if !ok {
            return Ok(None);
        }
        // The outputs own the spare buffers; the blobs just read become the
        // spare pair for the next commit.
        Ok(Some(Self {
            layers: self.layers.clone(),
            recurrent: MxArray::from_handle(out_rec, "gdn_commit_all:recurrent")?,
            conv: MxArray::from_handle(out_conv, "gdn_commit_all:conv")?,
            spare: (self.conv.clone(), self.recurrent.clone()),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen3_5::arrays_cache::ArraysCache;
    use crate::models::qwen3_5::gated_delta::GdnKernelTape;
    use crate::nn::Activations;

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

    /// Production-shaped (Hk 16, Hv 48, Dk = Dv 128, conv 4 x 10240) model
    /// with `n_linear` GDN layers interleaved with one full-attention slot.
    struct Fixture {
        caches: Vec<Qwen3_5LayerCache>,
        tape: Vec<Option<GdnLayerTape>>,
        window: i64,
    }

    fn fixture(n_linear: usize, window: i64, qkv_padded: bool) -> Fixture {
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
            // The real tape's `qkv` is an axis-2 slice of the merged
            // projection output (strided rows).
            let qkv = if qkv_padded {
                rand_bf16(&[1, window, width + 96])
                    .slice_axis(2, 0, width)
                    .unwrap()
            } else {
                rand_bf16(&[1, window, width])
            };
            tape.push(Some(GdnLayerTape {
                kernel: GdnKernelTape {
                    q: rand_bf16(&[1, window, hk, dk]),
                    k: rand_bf16(&[1, window, hk, dk]),
                    v: rand_bf16(&[1, window, hv, dv]),
                    g: gate(DType::Float32),
                    beta: gate(DType::BFloat16),
                },
                qkv,
                conv_kernel_dim: 4,
            }));
        }
        // Drop the trailing extra linear layer so the count is exact.
        caches.pop();
        tape.pop();
        Fixture {
            caches,
            tape,
            window,
        }
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
        let mut fx = fixture(5, 8, false);
        let before = linear_states(&fx.caches);
        let blobs = GdnStateBlobs::pack(&fx.caches).unwrap().expect("packable");
        assert_eq!(blobs.layers, vec![0, 1, 3, 4, 5]);
        assert_eq!(blobs.conv.shape().unwrap().as_ref(), [5, 3, 10240]);
        assert_eq!(blobs.recurrent.shape().unwrap().as_ref(), [5, 48, 128, 128]);
        blobs.apply_views(&mut fx.caches).unwrap();
        let after = linear_states(&fx.caches);
        for ((c0, r0), (c1, r1)) in before.iter().zip(after.iter()) {
            assert!(bytes_equal(c0, c1), "conv view differs");
            assert!(bytes_equal(r0, r1), "recurrent view differs");
        }
        // An empty slot keeps the per-layer path.
        fx.caches[0].reset();
        assert!(GdnStateBlobs::pack(&fx.caches).unwrap().is_none());
    }

    /// The fused commit must equal the per-layer `replay_into` path bit for
    /// bit at every accepted prefix, for strided and contiguous `qkv`.
    #[test]
    fn fused_commit_matches_per_layer_replay_every_keep() {
        if !metal() {
            return;
        }
        for qkv_padded in [true, false] {
            let mut fx = fixture(7, 8, qkv_padded);
            let blobs = GdnStateBlobs::pack(&fx.caches).unwrap().expect("packable");
            blobs.apply_views(&mut fx.caches).unwrap();
            let snapshot = linear_states(&fx.caches);
            for keep in 1..=fx.window as usize {
                // Reference: today's per-layer replay from the snapshot.
                let mut reference = Vec::new();
                for (row, &layer) in blobs.layers.iter().enumerate() {
                    let mut arrays = ArraysCache::new(2);
                    fx.tape[layer]
                        .as_ref()
                        .unwrap()
                        .replay_into(
                            &mut arrays,
                            Some(&snapshot[row].0),
                            Some(&snapshot[row].1),
                            keep,
                        )
                        .unwrap();
                    reference.push((
                        arrays.get(0).unwrap().clone(),
                        arrays.get(1).unwrap().clone(),
                    ));
                }
                let next = blobs
                    .commit(&fx.tape, keep)
                    .unwrap()
                    .expect("fused commit accepted the production geometry");
                let mut caches = std::mem::take(&mut fx.caches);
                next.apply_views(&mut caches).unwrap();
                let fused = linear_states(&caches);
                fx.caches = caches;
                for (row, ((rc, rr), (fc, fr))) in reference.iter().zip(fused.iter()).enumerate() {
                    assert!(
                        bytes_equal(rc, fc),
                        "conv state differs at keep={keep} row={row} padded={qkv_padded}"
                    );
                    assert!(
                        bytes_equal(rr, fr),
                        "recurrent state differs at keep={keep} row={row} padded={qkv_padded}"
                    );
                }
                // Restore the pre-verify views for the next keep.
                blobs.apply_views(&mut fx.caches).unwrap();
            }
        }
    }

    /// Full 48-layer geometry: one primitive evaluation per commit, and the
    /// chained commit (which writes back into the first pair's buffers,
    /// ping-pong) keeps feeding the next one.
    #[test]
    fn fused_commit_counts_one_primitive_for_48_layers() {
        if !metal() {
            return;
        }
        let mut fx = fixture(48, 8, true);
        let blobs = GdnStateBlobs::pack(&fx.caches).unwrap().expect("packable");
        blobs.apply_views(&mut fx.caches).unwrap();
        // Reference for the chained pair through the per-layer path, forced
        // BEFORE the fused chain overwrites the first blobs in place.
        let snapshot = linear_states(&fx.caches);
        let rows: Vec<usize> = (0..blobs.layers.len()).step_by(11).collect();
        let mut reference = Vec::new();
        for &row in &rows {
            let tape = fx.tape[blobs.layers[row]].as_ref().unwrap();
            let mut mid = ArraysCache::new(2);
            tape.replay_into(&mut mid, Some(&snapshot[row].0), Some(&snapshot[row].1), 3)
                .unwrap();
            let mut end = ArraysCache::new(2);
            tape.replay_into(&mut end, mid.get(0), mid.get(1), 8)
                .unwrap();
            let (c, r) = (end.get(0).unwrap().clone(), end.get(1).unwrap().clone());
            MxArray::eval_arrays(&[&c, &r]).unwrap();
            reference.push((c, r));
        }
        let count =
            |family: &std::ffi::CStr| unsafe { sys::mlx_test_kquant_family_count(family.as_ptr()) };
        let next = blobs.commit(&fx.tape, 3).unwrap().expect("fused");
        let again = next.commit(&fx.tape, 8).unwrap().expect("fused");
        unsafe { sys::mlx_test_kquant_counting(true) };
        MxArray::eval_arrays(&[&again.recurrent, &again.conv]).unwrap();
        let commits = count(c"gdn_commit_all");
        unsafe { sys::mlx_test_kquant_counting(false) };
        assert_eq!(commits, 2, "one GdnCommit evaluation per commit");
        assert_eq!(
            again.recurrent.shape().unwrap().as_ref(),
            [48, 48, 128, 128]
        );
        // Ping-pong: the second commit landed in the first pair's buffers.
        assert!(
            bytes_equal(&again.recurrent, &blobs.recurrent)
                && bytes_equal(&again.conv, &blobs.conv),
            "chained commit must write back into the first blobs' buffers"
        );
        let mut caches = std::mem::take(&mut fx.caches);
        again.apply_views(&mut caches).unwrap();
        let fused = linear_states(&caches);
        for (i, &row) in rows.iter().enumerate() {
            assert!(
                bytes_equal(&reference[i].0, &fused[row].0),
                "conv row {row}"
            );
            assert!(
                bytes_equal(&reference[i].1, &fused[row].1),
                "recurrent row {row}"
            );
        }
    }
}
