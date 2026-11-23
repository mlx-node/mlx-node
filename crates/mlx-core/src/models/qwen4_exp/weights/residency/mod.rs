//! Admitted hot banks use contiguous quantized storage. Partial residency can
//! assemble fixed projections while experts, PLE/token rows and optional
//! vision/MTP remain demand loaded. All source reads stay bounded.

#[cfg(test)]
mod tests;

use super::*;
use std::sync::atomic::Ordering;

pub(super) fn is_auxiliary(name: &str) -> bool {
    ["mtp.", "visual.", "vision_tower.", "v.", "mm.", "nextn."]
        .iter()
        .any(|p| name.starts_with(p))
}

fn is_hot(name: &str) -> bool {
    name != "token_embd.weight"
        && name != "embed_tokens.weight"
        && name != "per_layer_token_embd.weight"
        && !name.contains(".ngram_embedding")
        && !name.contains(".ple_embedding.")
        && !is_auxiliary(name)
}

fn is_fixed_projection(name: &str, tensor: &Tensor) -> bool {
    is_hot(name)
        && tensor.shape.len() <= 2
        && !name.contains(".experts.")
        && !name.contains("_exps.")
}

impl Tensor {
    // Only fixed linear projections use this representation. Convolution and
    // normalization retain their FP32 arithmetic; routed and optional banks
    // must not silently acquire another dense copy.
    fn dense_projection_cache_bytes(&self, name: &str) -> Result<u64> {
        let f32 = matches!(&self.encoding, Encoding::Safe(t) if t == "F32")
            || matches!(self.encoding, Encoding::Gguf(GgufTensorType::F32));
        if !f32
            || !is_fixed_projection(name, self)
            || name.contains("conv1d")
            || !(self.shape.len() == 2
                || name.ends_with(".ffn_gate_inp_shexp.weight")
                || name.ends_with(".shared_expert_gate.weight"))
        {
            return Ok(0);
        }
        Ok(self.runtime_bytes()? / 2)
    }

    fn bank_bytes(&self, name: &str) -> Result<u64> {
        self.runtime_bytes()?
            .checked_add(self.dense_projection_cache_bytes(name)?)
            .ok_or_else(|| err("Resident compute weight size overflow"))
    }

    pub(super) fn runtime_bytes(&self) -> Result<u64> {
        let n = self
            .shape
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d as u64))
            .ok_or_else(|| err("Hot tensor size overflow"))?;
        // Includes symmetric affine minima reconstructed by the importer.
        let (num, den) = match self.encoding {
            Encoding::Safe(_) => return Ok(self.bytes),
            Encoding::Gguf(t) => match t {
                GgufTensorType::F32 => (4, 1),
                GgufTensorType::F16 | GgufTensorType::BF16 => (2, 1),
                GgufTensorType::Q4_0 | GgufTensorType::Q4_1 => (20, 32),
                GgufTensorType::Q5_0 | GgufTensorType::Q5_1 => (24, 32),
                GgufTensorType::Q8_0 => (36, 32),
                // E8M0 scale byte + 16 nibble bytes, no biases.
                GgufTensorType::MXFP4 => (17, 32),
                // 4 UE4M3 scale bytes + 32 nibble bytes, no biases.
                GgufTensorType::NVFP4 => (36, 64),
                // 64 code bytes + 32 unpacked (sc, m) bytes + (d, dmin).
                GgufTensorType::Q2K => (100, 256),
                GgufTensorType::Q3K => (114, 256),
                GgufTensorType::Q4K => (148, 256),
                GgufTensorType::Q5K => (180, 256),
                GgufTensorType::Q6K => (210, 256),
                GgufTensorType::IQ4NL => (19, 32),
                GgufTensorType::IQ4XS => (138, 256),
                // The grid formats keep ggml's bytes (gguf_kquant.rs); IQ1_M
                // adds the 2-byte f16 d the contract makes explicit, IQ3_S
                // the 4 bytes of its scale nibbles stored one per byte.
                GgufTensorType::IQ3S => (114, 256),
                GgufTensorType::IQ2XXS => (66, 256),
                GgufTensorType::IQ2XS => (74, 256),
                GgufTensorType::IQ2S => (82, 256),
                GgufTensorType::IQ3XXS => (98, 256),
                GgufTensorType::IQ1S => (50, 256),
                GgufTensorType::IQ1M => (58, 256),
                GgufTensorType::PQ2_0 => {
                    return Err(err("Qwen4 does not support PQ2_0 tensor residency"));
                }
                GgufTensorType::MXFP8 => {
                    return Err(err("Qwen4 does not support MXFP8 tensor residency"));
                }
            },
        };
        n.checked_mul(num)
            .map(|n| n / den)
            .ok_or_else(|| err("Hot tensor size overflow"))
    }
}

impl Weight {
    pub(in super::super) fn tiled_expert_rows(
        &self,
        x: &MxArray,
        ids: &MxArray,
        tiles: &MxArray,
        experts: usize,
    ) -> Result<Option<MxArray>> {
        let (Some(scales), Some(biases)) = (&self.scales, &self.biases) else {
            return Ok(None);
        };
        crate::kernels::qwen4::metal::expert_prefill(
            x,
            ids,
            tiles,
            &self.values,
            scales,
            biases,
            experts,
            self.group,
            self.bits,
            &self.mode,
        )
    }

    pub(super) fn slice_rows(&self, start: usize, rows: usize) -> Result<Self> {
        let slice = |a: &MxArray| a.slice_axis(0, start as i64, (start + rows) as i64);
        Ok(Self {
            values: slice(&self.values)?,
            dense_bf16: self.dense_bf16.as_ref().map(slice).transpose()?,
            scales: self.scales.as_ref().map(slice).transpose()?,
            biases: self.biases.as_ref().map(slice).transpose()?,
            qlinear: None,
            ..self.clone()
        }
        .with_cached_qlinear())
    }

    pub(in super::super) fn concatenate_rows(weights: &[Arc<Self>]) -> Result<Self> {
        let first = weights
            .first()
            .ok_or_else(|| err("Cannot concatenate an empty resident weight list"))?;
        if weights.iter().any(|w| {
            w.mode != first.mode
                || w.bits != first.bits
                || w.group != first.group
                || w.scales.is_some() != first.scales.is_some()
                || w.biases.is_some() != first.biases.is_some()
        }) {
            return Err(err("Incompatible resident weight chunks"));
        }
        let values =
            MxArray::concatenate_many(weights.iter().map(|w| &w.values).collect(), Some(0))?;
        let scales = if first.scales.is_some() {
            let arrays = weights
                .iter()
                .map(|w| w.scales.as_ref())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| err("Resident weight chunk is missing scales"))?;
            Some(MxArray::concatenate_many(arrays, Some(0))?)
        } else {
            None
        };
        let biases = if first.biases.is_some() {
            let arrays = weights
                .iter()
                .map(|w| w.biases.as_ref())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| err("Resident weight chunk is missing biases"))?;
            Some(MxArray::concatenate_many(arrays, Some(0))?)
        } else {
            None
        };
        Ok(Self {
            values,
            scales,
            biases,
            // Never retain a cached view of only the first input's rows.
            dense_bf16: weights
                .iter()
                .map(|w| w.dense_bf16.as_ref())
                .collect::<Option<Vec<_>>>()
                .map(|arrays| MxArray::concatenate_many(arrays, Some(0)))
                .transpose()?,
            q8_symmetric_128: weights.iter().all(|w| w.q8_symmetric_128),
            qlinear: None,
            ..first.as_ref().clone()
        }
        .with_cached_qlinear())
    }

    pub(in super::super) fn gather_rows(
        &self,
        x: &MxArray,
        indices: &MxArray,
        experts: usize,
        sorted: bool,
    ) -> Result<MxArray> {
        let shape = self.values.shape()?;
        let view = |a: &MxArray| {
            let s = a.shape()?;
            a.reshape(&[experts as i64, s[0] / experts as i64, s[1]])
        };
        if experts == 0 || shape.len() != 2 || shape[0] % experts as i64 != 0 {
            return Err(err("Invalid resident expert shape"));
        }
        if let Some(scales) = &self.scales {
            let w = view(&self.values)?;
            let scales = view(scales)?;
            let biases = self.biases.as_ref().map(view).transpose()?;
            let mode = std::ffi::CString::new(self.mode.as_str()).map_err(err)?;
            // MLX runs sorted rows through its batched kernel from 16 rows
            // and 4 rows per bank expert, else through a per-row GEMV that
            // rounds differently; a slot piece of a window must keep the
            // window's kernel, so a sorted call pads up to that floor.
            let rows = indices.size()? as i64;
            let floor = (4 * experts as i64).max(16);
            let padded = if sorted && rows > 0 && rows < floor && x.shape_at(0)? == rows {
                let shape = x.shape()?;
                let mut pad = shape.to_vec();
                pad[0] = floor - rows;
                let x = MxArray::concatenate(x, &MxArray::zeros(&pad, Some(x.dtype()?))?, 0)?;
                let last = indices.slice_axis(0, rows - 1, rows)?;
                let indices =
                    MxArray::concatenate(indices, &last.broadcast_to(&[floor - rows])?, 0)?;
                Some((x, indices))
            } else {
                None
            };
            let (x_in, indices_in) = padded.as_ref().map_or((x, indices), |(x, i)| (x, i));
            let out = unsafe {
                mlx_sys::mlx_gather_qmm(
                    x_in.as_raw_ptr(),
                    w.as_raw_ptr(),
                    scales.as_raw_ptr(),
                    biases
                        .as_ref()
                        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
                    std::ptr::null_mut(),
                    indices_in.as_raw_ptr(),
                    true,
                    self.group,
                    self.bits,
                    mode.as_ptr(),
                    sorted,
                )
            };
            let out = MxArray::from_handle(out, "Qwen4 resident gather")?;
            let out = if padded.is_some() {
                out.slice_axis(0, 0, rows)?
            } else {
                out
            };
            if self.mode == "affine" {
                out.astype(x.dtype()?)
            } else {
                Ok(out)
            }
        } else {
            x.gather_mm(
                &view(&self.values)?
                    .transpose(Some(&[0, 2, 1]))?
                    .astype(x.dtype()?)?,
                indices,
                false,
            )
        }
    }
}

impl Store {
    pub(in super::super) fn hot_bytes(&self) -> Result<u64> {
        self.tensors
            .iter()
            .filter(|(name, _)| is_hot(name))
            .try_fold(0u64, |n, (name, t)| {
                n.checked_add(t.bank_bytes(name)?)
                    .ok_or_else(|| err("Hot weight inventory overflow"))
            })
    }

    pub(in super::super) fn resident_bank(&self, name: &str) -> Option<Arc<Weight>> {
        self.banks.get(name).cloned()
    }

    pub(in super::super) fn resident_qlinear(
        &self,
        name: &str,
    ) -> Option<&crate::models::quantized_linear::QuantizedLinear> {
        self.banks.get(name)?.qlinear()
    }

    pub(in super::super) fn prepare_hot(&mut self) -> Result<()> {
        if self.plan.direct || self.slot_bank_budget.is_some() {
            self.prepare_hot_with_policy(&mut super::super::memory::admit_direct)
        } else {
            self.prepare_hot_with_policy(&mut super::super::memory::admit)
        }
    }

    #[cfg(test)]
    pub(in super::super) fn prepare_fixture_hot(&mut self) -> Result<()> {
        self.prepare_hot_with_policy(&mut |_| Ok(()))
    }

    fn prepare_hot_with_policy(&mut self, admit: &mut impl FnMut(u64) -> Result<()>) -> Result<()> {
        self.prepare_hot_with_admission(admit)?;
        if let (Some(experts), Some(top)) = (
            self.metadata
                .get("qwen4exp.expert_count")
                .and_then(GgufMetaValue::as_u64),
            self.metadata
                .get("qwen4exp.expert_used_count")
                .and_then(GgufMetaValue::as_u64),
        ) {
            self.plan.partial_expert_slots =
                self.expert_slot_capacity(experts as usize, top as usize)?;
        }
        Ok(())
    }

    fn prepare_hot_with_admission(
        &mut self,
        admit: &mut impl FnMut(u64) -> Result<()>,
    ) -> Result<()> {
        if !self.banks.is_empty() {
            return Ok(());
        }
        if !self.slots.is_empty() && !self.plan.direct {
            return Err(err(
                "Qwen4 bank residency must be prepared before expert execution",
            ));
        }
        let names = self.projection_bank_names()?;
        if names.is_empty() {
            return Ok(());
        }
        // The store's cap is a plan number the host never grants in one piece
        // (Splash `ExpertStore` starts at its floor; the plan counts the rest
        // as `hostHeldBackBytes`): admit the banks this call builds. Slots,
        // each dense bank and the draft banks pass their own gates.
        if !self.plan.direct {
            let mut banks = 0u64;
            for name in &names {
                banks = banks.saturating_add(self.descriptor(name)?.bank_bytes(name)?);
            }
            admit(banks)?;
        }
        if self.prepare_banks(names, admit)? && admit(super::super::memory::WORKING_BYTES).is_ok() {
            return Ok(());
        }
        // No decoder output can hold these banks during bootstrap. Release the
        // abandoned plan completely before resizing; never allocate a second
        // model alongside it or count another owner's memory as ours.
        self.banks.clear();
        self.paired_banks.clear();
        self.dense_cache.clear();
        self.cache.clear();
        self.window_banks.clear();
        self.row_bank.clear();
        self.order.clear();
        self.recency.clear();
        self.protected.clear();
        self.protected_bytes = 0;
        self.bank_bytes = 0;
        self.mtp_banks_declined = false;
        self.cache_bytes = if self.plan.direct { self.slot_bytes } else { 0 };
        self.growth_bytes = 0;
        self.plan.resident_bank_bytes.store(0, Ordering::Relaxed);
        if !self.plan.direct {
            self.plan.partial_expert_slots = 0;
        }
        crate::array::memory::synchronize_and_clear_cache();
        if !self.plan.direct {
            self.slot_capacity = None;
        }
        let names = self.projection_bank_names()?;
        if !self.prepare_banks(names, admit)? {
            return Err(err(
                "Qwen4 partial residency lost first-request headroom during load",
            ));
        }
        admit(super::super::memory::WORKING_BYTES)
    }

    fn projection_bank_names(&self) -> Result<Vec<String>> {
        // Compare the checkpoint's actual sizes with the bootstrap budget.
        // Fixed projections get at most a quarter, leaving most RAM for experts.
        let mut names = Vec::new();
        let mut bytes = 0u64;
        // Full direct experts are gated separately: fixed banks may take all
        // but the row cache the store plan holds.
        let cap = if self.plan.direct {
            self.cache_limit
                .saturating_sub(self.slot_bytes)
                .saturating_sub(self.store_plan.map_or(0, |store| store.row_cache_bytes))
        } else {
            self.cache_limit / 4
        };
        for (name, tensor) in &self.tensors {
            if !is_fixed_projection(name, tensor) {
                continue;
            }
            let size = tensor.bank_bytes(name)?;
            // An unusual dense/BF16 checkpoint keeps its bounded chunk path.
            if size > 1 << 30 || size > cap.saturating_sub(bytes) {
                return Ok(Vec::new());
            }
            bytes += size;
            names.push(name.clone());
        }
        names.sort();
        Ok(names)
    }

    pub(super) fn prepare_banks(
        &mut self,
        names: Vec<String>,
        admit: &mut impl FnMut(u64) -> Result<()>,
    ) -> Result<bool> {
        for name in &names {
            let d = self.descriptor(name)?;
            // Bound concatenation's temporary duplication independently of the
            // full model. The released quantized expert matrices are <1 GiB.
            if d.bank_bytes(name)? > 1 << 30 {
                return Err(err(format!(
                    "Resident tensor {name} exceeds the 1 GiB bank budget"
                )));
            }
        }
        for name in names {
            let d = self.descriptor(&name)?.clone();
            // Both imported chunks and the concatenated bank coexist until
            // evaluation completes. Preserve the first request's allowance at
            // every growth step, including the final bank.
            if admit(d.bank_bytes(&name)?.saturating_mul(2) + super::super::memory::WORKING_BYTES)
                .is_err()
            {
                return Ok(false);
            }
            let staging_rows = (MAX_READ_BYTES as usize / (d.width()? * 4)).clamp(1, 4096);
            // Share exact prepared-cache keys with streamed expert reads.
            // Otherwise full-bank loading creates overlapping 4096-row files
            // and can fill the disk budget with duplicate representations.
            let chunk = if d.shape.len() == 3 && name.contains("_exps.") {
                d.shape[1].min(staging_rows)
            } else {
                staging_rows
            };
            let mut chunks = Vec::new();
            let rows = d.rows()?;
            let requests: Vec<_> = (0..rows)
                .step_by(chunk)
                .map(|start| (name.clone(), start, (rows - start).min(chunk)))
                .collect();
            for batch in requests.chunks(8) {
                chunks.extend(self.read_expert_batch(batch)?);
            }
            let mut bank = Weight::concatenate_rows(&chunks)?;
            if d.dense_projection_cache_bytes(&name)? != 0 {
                bank.dense_bf16 = Some(bank.values.astype(DType::BFloat16)?);
            }
            let arrays: Vec<_> = [
                Some(&bank.values),
                bank.dense_bf16.as_ref(),
                bank.scales.as_ref(),
                bank.biases.as_ref(),
            ]
            .into_iter()
            .flatten()
            .collect();
            MxArray::eval_arrays_with_context(&arrays, "Qwen4 prepare resident bank")?;
            if super::super::cuda_qwen4_fast_enabled() {
                let mut promoted = false;
                if let Some(dev_v) = bank.values.ensure_cuda_device()? {
                    bank.values = dev_v;
                    promoted = true;
                }
                if let Some(bf16) = &bank.dense_bf16
                    && let Some(dev_b) = bf16.ensure_cuda_device()?
                {
                    bank.dense_bf16 = Some(dev_b);
                    promoted = true;
                }
                if let Some(s) = &bank.scales
                    && let Some(dev_s) = s.ensure_cuda_device()?
                {
                    bank.scales = Some(dev_s);
                    promoted = true;
                }
                if let Some(b) = &bank.biases
                    && let Some(dev_b) = b.ensure_cuda_device()?
                {
                    bank.biases = Some(dev_b);
                    promoted = true;
                }
                if promoted {
                    bank = bank.with_cached_qlinear();
                }
            }
            self.install_bank(name, bank)?;
            drop(chunks);
            super::super::memory::maintain_freelist(self.plan.physical_bytes);
        }
        Ok(true)
    }

    /// Replace the LRU rows of `name` with one resident bank, outside the LRU
    /// but inside the cache budget.
    fn install_bank(&mut self, name: String, bank: Weight) -> Result<()> {
        let keys: Vec<_> = self
            .cache
            .keys()
            .filter(|(n, _, _)| n == &name)
            .cloned()
            .collect();
        for key in keys {
            if let Some((w, freed)) = self.uncache(&key)? {
                self.cache_bytes -= freed;
                if self.protected.remove(&key) {
                    self.protected_bytes -= w.bytes()?;
                }
            }
            self.recency.remove(&key);
        }
        let bytes = bank.bytes()?;
        self.bank_bytes += bytes;
        self.cache_bytes += bytes;
        self.plan
            .resident_bank_bytes
            .store(self.bank_bytes, Ordering::Relaxed);
        self.peak_cache_bytes = self.peak_cache_bytes.max(self.cache_bytes);
        self.dense_cache.remove(&name);
        self.banks.insert(name, Arc::new(bank));
        Ok(())
    }
}

const MTP_EXPERT_BANKS: [&str; 2] = [
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.experts.down_proj",
];

/// A draft bank as the admission builds it: name, descriptor, rows, width,
/// Q8 side dtype (none for a dense gather bank), bytes.
pub(super) type MtpBankSpec = (&'static str, Tensor, usize, usize, Option<DType>, u64);

impl Weight {
    fn quantize_q8(values: &MxArray, sides: DType) -> Result<Self> {
        // Quantize at f16, the GGUF import dtype the affine expert GEMV
        // reads. f32 `sides` hold those same values for a bank it refuses:
        // MLX gather_qmm promotes bf16 x with f16 sides to f32 and would cast
        // the whole bank on every call; an astype to its own dtype is a no-op.
        let source = values.astype(DType::Float16)?;
        let (mut codes, mut scales, mut biases) = (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        let ok = unsafe {
            mlx_sys::mlx_quantize(
                source.as_raw_ptr(),
                32,
                8,
                c"affine".as_ptr(),
                &mut codes,
                &mut scales,
                &mut biases,
            )
        };
        if !ok {
            return Err(err("Qwen4 MTP expert bank quantization failed"));
        }
        let values = MxArray::from_handle(codes, "Qwen4 MTP bank codes")?;
        let scales = MxArray::from_handle(scales, "Qwen4 MTP bank scales")?;
        let biases = MxArray::from_handle(biases, "Qwen4 MTP bank biases")?;
        Ok(Self {
            values,
            dense_bf16: None,
            scales: Some(scales.astype(sides)?),
            biases: Some(biases.astype(sides)?),
            group: 32,
            bits: 8,
            mode: "affine".into(),
            q8_symmetric_128: false,
            qlinear: None,
        })
    }
}

impl Store {
    pub(in super::super) fn mtp_bank_bytes(&self) -> Result<u64> {
        MTP_EXPERT_BANKS
            .iter()
            .filter_map(|name| self.banks.get(*name))
            .try_fold(0u64, |n, w| Ok(n + w.bytes()?))
    }

    /// The MTP head's routed-expert banks in one layout. `wide` gives a bank
    /// the affine expert GEMV refuses f32 sides. Empty without the head.
    fn mtp_bank_specs(&self, wide: bool) -> Result<Vec<MtpBankSpec>> {
        if MTP_EXPERT_BANKS
            .iter()
            .any(|name| !self.tensors.contains_key(*name))
        {
            return Ok(Vec::new());
        }
        MTP_EXPERT_BANKS
            .iter()
            .map(|&name| {
                let d = self.descriptor(name)?.clone();
                let (rows, width) = (d.rows()?, d.width()?);
                // MLX affine quantization needs K % 32 == 0; a narrower expert
                // (fixtures) stays a dense gather bank.
                let f32_sides = wide && !Weight::affine_gemv_takes(width);
                let sides = (width % 32 == 0).then_some(if f32_sides {
                    DType::Float32
                } else {
                    DType::Float16
                });
                // Eighths of a byte per weight: a code byte, and a scale and
                // a bias per 32 weights.
                let eighths = match sides {
                    None => 16,
                    Some(DType::Float32) => 10,
                    Some(_) => 9,
                };
                let bytes = (rows as u64)
                    .saturating_mul(width as u64)
                    .saturating_mul(eighths)
                    / 8;
                Ok((name, d, rows, width, sides, bytes))
            })
            .collect()
    }

    /// What the draft banks may take: the direct plan's charge for them, or
    /// the partial path's auxiliary allowance.
    fn mtp_bank_allowance(&self) -> Result<u64> {
        if self.draft_bank_budget > 0 {
            Ok(self.draft_bank_budget)
        } else {
            self.auxiliary_cache_allowance()
        }
    }

    /// Anonymous bytes the draft banks take once built: what a direct plan
    /// charges beside the dense banks while the MTP head is attached, so the
    /// live-memory gate and the bank admission see the same number. The f32
    /// sides only when the allowance holds them, as in the admission.
    pub(in super::super) fn mtp_bank_estimate(&self) -> Result<u64> {
        let total = |specs: Vec<MtpBankSpec>| specs.iter().fold(0u64, |n, s| n.saturating_add(s.5));
        let wide = total(self.mtp_bank_specs(true)?);
        if wide <= self.mtp_bank_allowance()? {
            Ok(wide)
        } else {
            Ok(total(self.mtp_bank_specs(false)?))
        }
    }

    #[cfg(test)]
    pub(in super::super) fn prepare_mtp_expert_banks(&mut self) -> Result<u64> {
        self.prepare_mtp_expert_banks_with(&mut super::super::memory::admit)
    }

    /// The draft-bank build without the live host admission: it asks for
    /// `WORKING_BYTES` beside the bank, which a 7 GB CI runner refuses.
    #[cfg(test)]
    pub(in super::super) fn prepare_fixture_mtp_expert_banks(&mut self) -> Result<u64> {
        self.prepare_mtp_expert_banks_with(&mut |_| Ok(()))
    }

    /// Build the MTP head's routed experts as resident Q8 (affine, g32) banks
    /// so a draft step gathers them on the device. Runs once, at load; the
    /// whole aux tensor streams through 64 MiB row chunks. Both banks or
    /// none: nothing is installed or charged until both evaluated, and a
    /// cache that cannot stage them keeps the row-reader draft path.
    pub(super) fn prepare_mtp_expert_banks_with(
        &mut self,
        admit: &mut impl FnMut(u64) -> Result<()>,
    ) -> Result<u64> {
        if self.mtp_banks_declined {
            return Ok(0);
        }
        if self.mtp_bank_specs(false)?.is_empty() {
            return Ok(0);
        }
        if MTP_EXPERT_BANKS
            .iter()
            .all(|name| self.banks.contains_key(*name))
        {
            return self.mtp_bank_bytes();
        }
        let Some(specs) = self.mtp_bank_admission()? else {
            self.mtp_banks_declined = true;
            return Ok(0);
        };
        let mut built = self.build_mtp_expert_banks(&specs, admit);
        // The f32 sides need more live headroom than the f16 layout, which
        // the store admits whenever it admits them.
        if built.is_err() && specs.iter().any(|spec| spec.4 == Some(DType::Float32)) {
            let narrow = self.mtp_bank_specs(false)?;
            built = self.build_mtp_expert_banks(&narrow, admit);
        }
        let built = match built {
            Ok(built) => built,
            Err(error) => {
                self.mtp_banks_declined = true;
                return Err(error);
            }
        };
        for (name, bank) in built {
            self.install_bank(name.into(), bank)?;
        }
        super::super::memory::maintain_freelist(self.plan.physical_bytes);
        self.mtp_bank_bytes()
    }

    /// The layout the draft banks are built in now: the f32 sides when they
    /// fit, else the f16 ones, else none.
    pub(super) fn mtp_bank_admission(&self) -> Result<Option<Vec<MtpBankSpec>>> {
        let allowance = self.mtp_bank_allowance()?;
        let pinned = self.protected_bytes + self.bank_bytes + self.slot_bytes;
        let room = self.cache_limit.saturating_sub(pinned);
        let planned_slots = self.planned_slot_bytes()?;
        for wide in [true, false] {
            let specs = self.mtp_bank_specs(wide)?;
            let (mut total, mut peak) = (0u64, 0u64);
            for (.., bytes) in &specs {
                // Imported chunks and the concatenated bank coexist while the
                // earlier banks are already held.
                peak = peak.max(total.saturating_add(bytes.saturating_mul(2)));
                total = total.saturating_add(*bytes);
            }
            // The slot banks a cached capacity has yet to build stay pinned
            // beside the draft banks, and the last one still needs its staging.
            let planned = (self.protected_bytes + self.bank_bytes + planned_slots)
                .saturating_add(total)
                .saturating_add(MAX_READ_BYTES * 3);
            if total <= allowance
                && peak.saturating_add(MAX_READ_BYTES * 3) <= room
                && planned <= self.cache_limit
            {
                return Ok(Some(specs));
            }
        }
        Ok(None)
    }

    fn build_mtp_expert_banks(
        &mut self,
        specs: &[MtpBankSpec],
        admit: &mut impl FnMut(u64) -> Result<()>,
    ) -> Result<Vec<(&'static str, Weight)>> {
        let mut built = Vec::with_capacity(specs.len());
        let mut held = 0u64;
        for (name, d, rows, width, sides, bytes) in specs {
            let (rows, width, bytes) = (*rows, *width, *bytes);
            self.evict_to(
                self.cache_limit
                    .saturating_sub(held + bytes.saturating_mul(2) + MAX_READ_BYTES * 3),
            )?;
            admit(bytes.saturating_mul(2) + super::super::memory::WORKING_BYTES)?;
            let chunk = (MAX_READ_BYTES as usize / (width * 4)).clamp(1, rows);
            let mut chunks = Vec::with_capacity(rows.div_ceil(chunk));
            for start in (0..rows).step_by(chunk) {
                let values = d.import_rows(start, (rows - start).min(chunk))?.values;
                let weight = if let Some(sides) = sides {
                    Weight::quantize_q8(&values, *sides)?
                } else {
                    Weight {
                        values,
                        dense_bf16: None,
                        scales: None,
                        biases: None,
                        group: 0,
                        bits: 0,
                        mode: String::new(),
                        q8_symmetric_128: false,
                        qlinear: None,
                    }
                };
                let arrays: Vec<_> = [
                    Some(&weight.values),
                    weight.scales.as_ref(),
                    weight.biases.as_ref(),
                ]
                .into_iter()
                .flatten()
                .collect();
                MxArray::eval_arrays_with_context(&arrays, "Qwen4 MTP expert bank chunk")?;
                chunks.push(Arc::new(weight));
            }
            let bank = Weight::concatenate_rows(&chunks)?;
            let arrays: Vec<_> = [
                Some(&bank.values),
                bank.scales.as_ref(),
                bank.biases.as_ref(),
            ]
            .into_iter()
            .flatten()
            .collect();
            MxArray::eval_arrays_with_context(&arrays, "Qwen4 MTP expert bank")?;
            drop(chunks);
            held += bank.bytes()?;
            built.push((*name, bank));
        }
        Ok(built)
    }
}

impl Store {
    /// Share one packed projection for small windows when layouts agree.
    /// Original bank entries become views into that allocation, so persistent
    /// weight bytes do not grow. The reference's shared-expert prefill port
    /// joins only the two 640-row projections with unchanged NAX column tiles.
    pub(in super::super) fn linear_pair(
        &mut self,
        x: &MxArray,
        first: &str,
        second: &str,
    ) -> Result<(MxArray, MxArray)> {
        let shape = x.shape()?;
        let width = shape
            .last()
            .copied()
            .filter(|&width| width > 0)
            .ok_or_else(|| {
                err("Qwen4 paired projection input requires a positive final dimension")
            })?;
        let rows = x.size()? / width as u64;
        // mlxfast TrackFastModel.moeForwardShared, MLXFAST-SHAREDFUSE.
        // N=640 and N=1280 both use one K partition for these windows. Each
        // 64-column NAX tile retains its inputs and accumulation order.
        let wide_shared = if (512..=1024).contains(&rows)
            && shape.last() == Some(&2560)
            && x.dtype()? == DType::BFloat16
            && first.ends_with(".ffn_gate_shexp.weight")
            && second.ends_with(".ffn_up_shexp.weight")
            && super::super::tf32_enabled()
            && unsafe { mlx_sys::mlx_metal_is_nax_available() }
        {
            let eligible = |w: &Weight| -> Result<bool> {
                Ok(w.mode == "affine"
                    && w.bits == 8
                    && w.group == 32
                    && w.values.dtype()? == DType::Uint32
                    && *w.values.shape()? == [640, 640]
                    && w.scales.as_ref().is_some_and(|s| {
                        s.dtype().ok() == Some(DType::Float16)
                            && s.shape().is_ok_and(|shape| *shape == [640, 80])
                    })
                    && w.biases.as_ref().is_some_and(|b| {
                        b.dtype().ok() == Some(DType::Float16)
                            && b.shape().is_ok_and(|shape| *shape == [640, 80])
                    }))
            };
            match (self.banks.get(first), self.banks.get(second)) {
                (Some(a), Some(b)) => eligible(a)? && eligible(b)?,
                _ => false,
            }
        } else {
            false
        };
        // The reference groups QKV/Z/B/A projections. Here A/B are F32
        // source matrices with cached BF16 compute weights, unlike Q8 QKV/Z.
        // A wider N changes MLX's matrix-multiply schedule for short windows.
        // This must precede the paired-bank cache lookup: a bank prepared by
        // singleton decoding must not opt a later verification window in.
        if first.ends_with(".ssm_alpha.weight") && second.ends_with(".ssm_beta.weight") {
            return Ok((self.linear(first, x)?, self.linear(second, x)?));
        }
        if rows <= 8
            || wide_shared
            || (crate::accelerator::cuda_ops_available() && rows <= 16)
            || (super::super::cuda_qwen4_fast_enabled() && rows >= 17)
        {
            let key = (first.to_owned(), second.to_owned());
            if !self.paired_banks.contains_key(&key)
                && let (Some(a), Some(b)) = (
                    self.banks.get(first).cloned(),
                    self.banks.get(second).cloned(),
                )
            {
                let storage_compatible = match (&a.scales, &b.scales) {
                    (Some(a), Some(b)) => a.dtype()? == b.dtype()?,
                    _ => false,
                };
                let compatible = storage_compatible
                    && a.mode == b.mode
                    && a.bits == b.bits
                    && a.group == b.group
                    && a.values.shape()?[1] == b.values.shape()?[1]
                    && a.values.dtype()? == b.values.dtype()?
                    && a.biases.is_some() == b.biases.is_some();
                let bytes = a.bytes()? + b.bytes()?;
                if compatible && bytes <= MAX_READ_BYTES * 3 {
                    // Pairing is optional; existing projections remain
                    // valid when transient concatenation cannot be admitted.
                    let admitted = if self.plan.direct || self.slot_bank_budget.is_some() {
                        super::super::memory::admit_direct(bytes)
                    } else {
                        super::super::memory::admit(bytes)
                    };
                    if admitted.is_err() {
                        return Ok((self.linear(first, x)?, self.linear(second, x)?));
                    }
                    let joined = Arc::new(
                        Weight::concatenate_rows(&[a.clone(), b.clone()])?.with_cached_qlinear(),
                    );
                    let arrays: Vec<_> = [
                        Some(&joined.values),
                        joined.dense_bf16.as_ref(),
                        joined.scales.as_ref(),
                        joined.biases.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    .collect();
                    MxArray::eval_arrays_with_context(
                        &arrays,
                        "Qwen4 paired projection preparation",
                    )?;
                    let n = a.values.shape()?[0] as usize;
                    self.banks.insert(
                        first.into(),
                        Arc::new(joined.slice_rows(0, n)?.with_cached_qlinear()),
                    );
                    self.banks.insert(
                        second.into(),
                        Arc::new(
                            joined
                                .slice_rows(n, b.values.shape()?[0] as usize)?
                                .with_cached_qlinear(),
                        ),
                    );
                    self.paired_banks.insert(key.clone(), joined);
                }
            }
            if let Some(bank) = self.paired_banks.get(&key) {
                let output = bank.linear(x)?;
                let n = self.descriptor(first)?.rows()? as i64;
                let axis = output.ndim()? as usize - 1;
                return Ok((
                    output.slice_axis(axis, 0, n)?,
                    output.slice_axis(axis, n, output.shape()?[axis])?,
                ));
            }
        }
        Ok((self.linear(first, x)?, self.linear(second, x)?))
    }
}

impl Weight {
    /// The input widths the affine expert GEMV takes: the generic fast path
    /// uses a different accumulation for K%512=0.
    fn affine_gemv_takes(k: usize) -> bool {
        !k.is_multiple_of(512)
    }

    pub(in super::super) fn expert_rows(
        &self,
        x: &MxArray,
        ids: &MxArray,
        experts: usize,
        sorted: bool,
    ) -> Result<MxArray> {
        if self.mode == "affine"
            && matches!(self.bits, 5 | 8)
            && self.group == 32
            && let (Some(scales), Some(biases)) = (&self.scales, &self.biases)
            && scales.dtype().ok() == Some(DType::Float16)
            && ids.size()? <= 80
            && x.shape()?
                .last()
                .is_some_and(|&k| Self::affine_gemv_takes(k as usize))
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            return crate::kernels::qwen4::metal::affine_expert_gemv(
                x,
                ids,
                &self.values,
                scales,
                biases,
                experts,
                self.group,
                self.bits,
            );
        }
        if self.mode != "affine"
            && ids.size()? <= 80
            && let (Some(scales), Some(biases)) = (&self.scales, &self.biases)
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            return crate::kernels::qwen4::metal::expert_gemv(
                x,
                ids,
                &self.values,
                scales,
                biases,
                experts,
                self.group,
                self.bits,
                &self.mode,
            );
        }
        self.gather_rows(x, ids, experts, sorted)
    }
}
