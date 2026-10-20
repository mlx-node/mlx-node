//! Admitted hot banks use contiguous quantized storage. Partial residency can
//! assemble fixed projections while experts, PLE/token rows and optional
//! vision/MTP remain demand loaded. All source reads stay bounded.
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
        let mode = std::ffi::CString::new(self.mode.as_str()).map_err(err)?;
        let out = unsafe {
            mlx_sys::mlx_qwen4_expert_prefill(
                x.as_raw_ptr(),
                ids.as_raw_ptr(),
                tiles.as_raw_ptr(),
                self.values.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
                experts as i32,
                self.group,
                self.bits,
                mode.as_ptr(),
            )
        };
        if out.is_null() {
            Ok(None)
        } else {
            MxArray::from_handle(out, "Qwen4 expert-aligned prefill").map(Some)
        }
    }

    pub(super) fn slice_rows(&self, start: usize, rows: usize) -> Result<Self> {
        let slice = |a: &MxArray| a.slice_axis(0, start as i64, (start + rows) as i64);
        Ok(Self {
            values: slice(&self.values)?,
            dense_bf16: self.dense_bf16.as_ref().map(slice).transpose()?,
            scales: self.scales.as_ref().map(slice).transpose()?,
            biases: self.biases.as_ref().map(slice).transpose()?,
            ..self.clone()
        })
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
            ..first.as_ref().clone()
        })
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
            let out = unsafe {
                mlx_sys::mlx_gather_qmm(
                    x.as_raw_ptr(),
                    w.as_raw_ptr(),
                    scales.as_raw_ptr(),
                    biases
                        .as_ref()
                        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
                    std::ptr::null_mut(),
                    indices.as_raw_ptr(),
                    true,
                    self.group,
                    self.bits,
                    mode.as_ptr(),
                    sorted,
                )
            };
            let out = MxArray::from_handle(out, "Qwen4 resident gather")?;
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

    /// Fixed projections with their dense compute copies: the hot set minus
    /// routed experts, what the mapped budget must hold as banks.
    pub(super) fn fixed_projection_bytes(&self) -> Result<u64> {
        self.tensors
            .iter()
            .filter(|(name, tensor)| is_fixed_projection(name, tensor))
            .try_fold(0u64, |n, (name, t)| {
                n.checked_add(t.bank_bytes(name)?)
                    .ok_or_else(|| err("Fixed projection inventory overflow"))
            })
    }

    pub(in super::super) fn prepare_hot(&mut self) -> Result<()> {
        self.prepare_hot_with_policy(
            &mut super::super::memory::admit,
            &mut super::super::memory::refresh_plan,
        )
    }

    #[cfg(test)]
    pub(in super::super) fn prepare_fixture_hot(&mut self) -> Result<()> {
        self.prepare_hot_with_policy(&mut |_| Ok(()), &mut |_| Ok(()))
    }

    fn prepare_hot_with_policy(
        &mut self,
        admit: &mut impl FnMut(u64) -> Result<()>,
        refresh: &mut impl FnMut(&mut super::super::memory::Plan) -> Result<()>,
    ) -> Result<()> {
        self.prepare_hot_with_admission(admit, refresh)?;
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
        refresh: &mut impl FnMut(&mut super::super::memory::Plan) -> Result<()>,
    ) -> Result<()> {
        if !self.banks.is_empty() {
            return Ok(());
        }
        if !self.slots.is_empty() {
            return Err(err(
                "Qwen4 bank residency must be prepared before expert execution",
            ));
        }
        // Pool creation happens after metadata bootstrap and is invisible to
        // MLX allocation counters. Reconcile against live headroom before IO.
        if self.cache_bytes == 0 && self.plan.available_bytes.is_some() {
            refresh(&mut self.plan)?;
            self.cache_limit = self.plan.budget;
        }
        let names = self.projection_bank_names()?;
        if names.is_empty() {
            return Ok(());
        }
        // Admit the whole set before reading any payload, then recheck live
        // headroom during growth in case another application starts allocating.
        admit(self.cache_limit)?;
        if self.prepare_banks(names, admit)? && admit(super::super::memory::WORKING_BYTES).is_ok() {
            return Ok(());
        }
        // No decoder output can hold these banks during bootstrap. Release the
        // abandoned plan completely before resizing; never allocate a second
        // model alongside it or count another owner's memory as ours.
        self.banks.clear();
        self.paired_banks.clear();
        self.cache.clear();
        self.window_banks.clear();
        self.row_bank.clear();
        self.order.clear();
        self.recency.clear();
        self.protected.clear();
        self.protected_bytes = 0;
        self.bank_bytes = 0;
        self.mtp_banks_declined = false;
        self.cache_bytes = 0;
        self.growth_bytes = 0;
        self.plan.resident_bank_bytes.store(0, Ordering::Relaxed);
        self.plan.partial_expert_slots = 0;
        crate::array::memory::synchronize_and_clear_cache();
        refresh(&mut self.plan)?;
        self.cache_limit = self.plan.budget;
        self.slot_capacity = None;
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
        // The mapped budget holds no experts: banks may take all but the
        // LRU room memory::mapped_desired reserved.
        let cap = if self.plan.mapped {
            self.cache_limit.saturating_sub(2 << 30)
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

    /// What the draft banks may take: the mapped plan's charge for them, or
    /// the slot path's auxiliary allowance.
    fn mtp_bank_allowance(&self) -> Result<u64> {
        if self.draft_bank_budget > 0 {
            Ok(self.draft_bank_budget)
        } else {
            self.auxiliary_cache_allowance()
        }
    }

    /// Anonymous bytes the draft banks take once built: what a mapped plan
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

    /// Build the MTP head's routed experts as resident Q8 (affine, g32) banks
    /// so a draft step gathers them on the device. Runs once, on the first MTP
    /// owner; the whole aux tensor streams through 64 MiB row chunks. Both
    /// banks or none: nothing is installed or charged until both evaluated,
    /// and a cache that cannot stage them keeps the row-reader draft path.
    pub(in super::super) fn prepare_mtp_expert_banks(&mut self) -> Result<u64> {
        self.prepare_mtp_expert_banks_with(&mut super::super::memory::admit)
    }

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
        if rows <= 8 || wide_shared {
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
                    if super::super::memory::admit(bytes).is_err() {
                        return Ok((self.linear(first, x)?, self.linear(second, x)?));
                    }
                    let joined = Arc::new(Weight::concatenate_rows(&[a.clone(), b.clone()])?);
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
                    self.banks
                        .insert(first.into(), Arc::new(joined.slice_rows(0, n)?));
                    self.banks.insert(
                        second.into(),
                        Arc::new(joined.slice_rows(n, b.values.shape()?[0] as usize)?),
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
            let out = unsafe {
                mlx_sys::mlx_qwen4_affine_expert_gemv(
                    x.as_raw_ptr(),
                    ids.as_raw_ptr(),
                    self.values.as_raw_ptr(),
                    scales.as_raw_ptr(),
                    biases.as_raw_ptr(),
                    experts as i32,
                    self.group,
                    self.bits,
                )
            };
            return MxArray::from_handle(out, "Qwen4 affine expert GEMV");
        }
        if self.mode != "affine"
            && ids.size()? <= 80
            && let (Some(scales), Some(biases)) = (&self.scales, &self.biases)
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            let mode = std::ffi::CString::new(self.mode.as_str()).map_err(err)?;
            let out = unsafe {
                mlx_sys::mlx_qwen4_expert_gemv(
                    x.as_raw_ptr(),
                    ids.as_raw_ptr(),
                    self.values.as_raw_ptr(),
                    scales.as_raw_ptr(),
                    biases.as_raw_ptr(),
                    experts as i32,
                    self.group,
                    self.bits,
                    mode.as_ptr(),
                )
            };
            return MxArray::from_handle(out, "Qwen4 direct expert GEMV");
        }
        self.gather_rows(x, ids, experts, sorted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_resident_weight_concatenation_returns_an_error() {
        assert!(Weight::concatenate_rows(&[]).is_err());
    }
    #[test]
    fn affine_direct_experts_preserve_promoted_arithmetic_and_tail_rows() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        for bits in [5, 8] {
            for (k, n, rows) in [(32, 17, 1), (640, 2560, 10), (672, 19, 80)] {
                let experts = 3;
                let codes = (0..experts * n * k * bits / 32)
                    .map(|i| (i as u32).wrapping_mul(2654435761))
                    .collect::<Vec<_>>();
                let sides = (0..experts * n * k / 32)
                    .map(|i| ((i % 29) as f32 + 1.) / 1024.)
                    .collect::<Vec<_>>();
                let scales = MxArray::from_float32(&sides, &[experts * n, k / 32])
                    .unwrap()
                    .astype(DType::Float16)
                    .unwrap();
                let biases = scales.mul_scalar(-7.).unwrap();
                let bank = Weight {
                    dense_bf16: None,
                    values: MxArray::from_uint32(&codes, &[experts * n, k * bits / 32]).unwrap(),
                    scales: Some(scales),
                    biases: Some(biases),
                    group: 32,
                    bits: bits as i32,
                    mode: "affine".into(),
                };
                let ids = MxArray::from_uint32(
                    &(0..rows).map(|i| (i % experts) as u32).collect::<Vec<_>>(),
                    &[rows],
                )
                .unwrap();
                let raw = (0..rows * k)
                    .map(|i| (i as f32 * 0.13).sin())
                    .collect::<Vec<_>>();
                for dtype in [DType::BFloat16, DType::Float16, DType::Float32] {
                    let x = MxArray::from_float32(&raw, &[rows, 1, k])
                        .unwrap()
                        .astype(dtype)
                        .unwrap();
                    let want = bank
                        .gather_rows(&x, &ids, experts as usize, false)
                        .unwrap()
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap();
                    let got = bank
                        .expert_rows(&x, &ids, experts as usize, false)
                        .unwrap()
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap();
                    let error = got
                        .iter()
                        .zip(want.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert_eq!(
                        error, 0.,
                        "bits={bits} K={k} N={n} rows={rows} dtype={dtype:?}"
                    );
                }
            }
        }
    }

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp")
    }

    #[test]
    fn dense_compute_cache_preserves_source_dtypes_and_row_views() {
        let values = MxArray::from_float32(
            &(0..19 * 32)
                .map(|i| (i as f32 * 0.131).sin() * 0.27)
                .collect::<Vec<_>>(),
            &[19, 32],
        )
        .unwrap();
        let original = Weight {
            values,
            dense_bf16: None,
            scales: None,
            biases: None,
            group: 0,
            bits: 0,
            mode: String::new(),
        };
        let mut cached = original.clone();
        cached.dense_bf16 = Some(cached.values.astype(DType::BFloat16).unwrap());
        MxArray::eval_arrays_with_context(
            &[cached.dense_bf16.as_ref().unwrap()],
            "test cached projection",
        )
        .unwrap();
        assert_eq!(cached.bytes().unwrap(), 19 * 32 * 6);
        assert_eq!(
            cached.dense().unwrap().to_float32().unwrap().to_vec(),
            original.values.to_float32().unwrap().to_vec()
        );
        let parts = [
            Arc::new(cached.slice_rows(0, 7).unwrap()),
            Arc::new(cached.slice_rows(7, 12).unwrap()),
        ];
        let joined = Weight::concatenate_rows(&parts).unwrap();
        for dtype in [DType::BFloat16, DType::Float16, DType::Float32] {
            for rows in [1, 8] {
                let x = MxArray::from_float32(
                    &(0..rows * 32)
                        .map(|i| (i as f32 * 0.19).cos())
                        .collect::<Vec<_>>(),
                    &[1, rows, 32],
                )
                .unwrap()
                .astype(dtype)
                .unwrap();
                let want = original
                    .linear(&x)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .to_vec();
                for bank in [&cached, &joined] {
                    let got = bank.linear(&x).unwrap();
                    assert_eq!(got.dtype().unwrap(), dtype);
                    assert_eq!(
                        got.astype(DType::Float32)
                            .unwrap()
                            .to_float32()
                            .unwrap()
                            .to_vec(),
                        want
                    );
                }
            }
        }
        // Mixed cached/uncached concatenations must never inherit a stale
        // derived representation of only the first matrix.
        assert!(
            Weight::concatenate_rows(&[Arc::new(cached), Arc::new(original)])
                .unwrap()
                .dense_bf16
                .is_none()
        );
    }

    #[test]
    fn dense_gate_pair_preserves_projection_rows_and_cache_bytes() {
        for phase in (0..16).map(|i| 0.3 + i as f32 * 0.7) {
            let make = |shift: f32| {
                let values = MxArray::from_float32(
                    &(0..48 * 2560)
                        .map(|i| (i as f32 * 0.131 + shift).sin() * 0.27)
                        .collect::<Vec<_>>(),
                    &[48, 2560],
                )
                .unwrap();
                Weight {
                    dense_bf16: Some(values.astype(DType::BFloat16).unwrap()),
                    values,
                    scales: None,
                    biases: None,
                    group: 0,
                    bits: 0,
                    mode: String::new(),
                }
            };
            let a = Arc::new(make(phase));
            let b = Arc::new(make(phase + 0.7));
            let joined = Weight::concatenate_rows(&[a.clone(), b.clone()]).unwrap();
            assert_eq!(
                joined.bytes().unwrap(),
                a.bytes().unwrap() + b.bytes().unwrap()
            );
            for rows in [1] {
                let x = MxArray::from_float32(
                    &(0..rows * 2560)
                        .map(|i| (i as f32 * 0.017 + phase).cos())
                        .collect::<Vec<_>>(),
                    &[1, rows, 2560],
                )
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
                let want = MxArray::concatenate_many(
                    vec![&a.linear(&x).unwrap(), &b.linear(&x).unwrap()],
                    Some(-1),
                )
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec();
                let got = joined.linear(&x).unwrap();
                assert_eq!(got.dtype().unwrap(), DType::BFloat16);
                assert_eq!(
                    got.to_float32().unwrap().to_vec(),
                    want,
                    "phase={phase} rows={rows}"
                );
                for (start, original) in [(0, &a), (48, &b)] {
                    let view = joined.slice_rows(start, 48).unwrap();
                    assert_eq!(
                        view.linear(&x).unwrap().to_float32().unwrap().to_vec(),
                        original.linear(&x).unwrap().to_float32().unwrap().to_vec()
                    );
                    assert_eq!(
                        view.values.to_float32().unwrap().to_vec(),
                        original.values.to_float32().unwrap().to_vec()
                    );
                }
            }
            // Reproduce the actual cache state after a singleton projection.
            // Short verification windows must use the two original N=48
            // matmuls even though the joined N=96 bank is already cached.
            let mut store = Store::open_metadata(
                &fixture().join("model.gguf"),
                Some(&Store::fixture_pack_root()),
            )
            .unwrap();
            let first = "blk.0.ssm_alpha.weight";
            let second = "blk.0.ssm_beta.weight";
            for (name, start) in [(first, 0), (second, 48)] {
                store
                    .banks
                    .insert(name.into(), Arc::new(joined.slice_rows(start, 48).unwrap()));
                store.tensors.insert(
                    name.into(),
                    Tensor {
                        // Cached banks satisfy every read; no tensor payload is opened.
                        path: PathBuf::new(),
                        shape: vec![48, 2560],
                        offset: 0,
                        bytes: 48 * 2560 * 4,
                        encoding: Encoding::Gguf(GgufTensorType::F32),
                    },
                );
            }
            store
                .paired_banks
                .insert((first.into(), second.into()), Arc::new(joined));
            for rows in [2, 8, 16] {
                let x = MxArray::from_float32(
                    &(0..rows * 2560)
                        .map(|i| (i as f32 * 0.017 + phase).cos())
                        .collect::<Vec<_>>(),
                    &[1, rows, 2560],
                )
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
                let (ga, gb) = store.linear_pair(&x, first, second).unwrap();
                for (want, got) in [(&a, &ga), (&b, &gb)] {
                    let want = want.linear(&x).unwrap().to_float32().unwrap().to_vec();
                    let got = got.to_float32().unwrap().to_vec();
                    let differences = want.iter().zip(&got).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        differences, 0,
                        "cached pair fallback phase={phase} rows={rows}"
                    );
                }
            }
        }
    }

    #[test]
    fn failed_bank_load_releases_banks_before_the_retry() {
        let mut store = Store::open_metadata(
            &fixture().join("model.gguf"),
            Some(&Store::fixture_pack_root()),
        )
        .unwrap();
        {
            store.plan.hot_bytes = store.hot_bytes().unwrap();
            let mut refused = false;
            let mut refreshed = false;
            let result = store.prepare_hot_with_admission(
                &mut |bytes| {
                    // Simulate headroom loss only after the bank set is
                    // complete. No large allocation or OS settings change needed.
                    if bytes == super::super::super::memory::WORKING_BYTES && !refused {
                        refused = true;
                        Err(err("simulated post-load headroom loss"))
                    } else {
                        Ok(())
                    }
                },
                &mut |plan| {
                    refreshed = true;
                    plan.budget = 1 << 30;
                    Ok(())
                },
            );
            assert!(refused);
            assert!(refreshed);
            result.unwrap();
        }
        assert!(!store.banks.is_empty());
        {
            {
                assert!(
                    store
                        .banks
                        .keys()
                        .all(|name| is_fixed_projection(name, &store.tensors[name]))
                );
                assert_eq!(store.cache_bytes, store.bank_bytes);
                assert_eq!(
                    store.plan.resident_bank_bytes.load(Ordering::Relaxed),
                    store.bank_bytes
                );
                assert!(store.bank_bytes < store.plan.hot_bytes);
                assert!(
                    store.cache.is_empty()
                        && store.recency.is_empty()
                        && store.protected.is_empty()
                );
            }
        }
    }

    #[test]
    fn mtp_banks_decline_when_the_cache_cannot_stage_both() {
        // 3 x MAX_READ_BYTES lets the first bank's eviction target saturate
        // to zero and pass; the second bank's cannot once the first is pinned.
        let mut s = Store::open_fixture(&fixture().join("bf16/paged"), None).unwrap();
        s.cache_limit = MAX_READ_BYTES * 3;
        let before = s.cache_bytes;
        assert_eq!(s.prepare_mtp_expert_banks().unwrap(), 0);
        assert_eq!(s.mtp_bank_bytes().unwrap(), 0);
        assert!(MTP_EXPERT_BANKS.iter().all(|n| !s.banks.contains_key(*n)));
        assert_eq!(s.cache_bytes, before);
        assert_eq!(s.bank_bytes, 0);
        assert_eq!(s.plan.resident_bank_bytes.load(Ordering::Relaxed), 0);
        assert!(s.mtp_banks_declined, "a declined build must not retry");
        s.cache_limit = CACHE_BYTES;
        assert_eq!(s.prepare_mtp_expert_banks().unwrap(), 0);
        s.mtp_banks_declined = false;
        let bytes = s.prepare_mtp_expert_banks().unwrap();
        assert!(bytes > 0);
        assert_eq!(s.cache_bytes, before + bytes);
    }

    #[test]
    fn failed_second_mtp_bank_leaves_no_partial_bank() {
        let mut s = Store::open_fixture(&fixture().join("bf16/paged"), None).unwrap();
        let before = (s.cache_bytes, s.bank_bytes);
        let mut admissions = 0;
        let error = s
            .prepare_mtp_expert_banks_with(&mut |_| {
                admissions += 1;
                if admissions == 2 {
                    Err(err("simulated headroom loss before the second bank"))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert!(error.reason.contains("second bank"), "{error}");
        assert_eq!(admissions, 2);
        assert!(MTP_EXPERT_BANKS.iter().all(|n| !s.banks.contains_key(*n)));
        assert_eq!((s.cache_bytes, s.bank_bytes), before);
        assert_eq!(s.mtp_bank_bytes().unwrap(), 0);
        assert_eq!(s.plan.resident_bank_bytes.load(Ordering::Relaxed), 0);
        assert!(s.mtp_banks_declined);
        assert_eq!(s.prepare_mtp_expert_banks().unwrap(), 0);
        s.mtp_banks_declined = false;
        let bytes = s.prepare_mtp_expert_banks_with(&mut |_| Ok(())).unwrap();
        assert!(bytes > 0);
        assert!(MTP_EXPERT_BANKS.iter().all(|n| s.banks.contains_key(*n)));
        assert_eq!(s.bank_bytes, before.1 + bytes);
        assert_eq!(
            s.plan.resident_bank_bytes.load(Ordering::Relaxed),
            s.bank_bytes
        );
    }

    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const MTP_TEST_EXPERTS: usize = 4;

    /// The MTP head's routed-expert banks at their real expert shapes (four
    /// experts, BF16 in a temporary file), with each bank's source rows.
    fn real_shape_mtp_experts() -> (Vec<(String, Tensor)>, Vec<MxArray>, Cleanup) {
        let dir = std::env::temp_dir().join(format!("qwen4-mtp-sides-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cleanup = Cleanup(dir.clone());
        let path = dir.join("experts.bin");
        let banks = [
            (MTP_EXPERT_BANKS[0], 1280, 2560),
            (MTP_EXPERT_BANKS[1], 2560, 640),
        ];
        let (mut payload, mut tensors, mut sources) = (Vec::new(), Vec::new(), Vec::new());
        for (k, (name, n, width)) in banks.into_iter().enumerate() {
            let rows = MTP_TEST_EXPERTS * n;
            let bits: Vec<u16> = (0..rows * width)
                .map(|i| {
                    let hash = (i as u32 ^ k as u32).wrapping_mul(2654435761) >> 8;
                    let row = 0.05 + (i / width % 7) as f32 * 0.04;
                    let v = (hash as f32 / (1u32 << 24) as f32 - 0.5) * row;
                    (v.to_bits() >> 16) as u16
                })
                .collect();
            tensors.push((
                name.into(),
                Tensor {
                    path: path.clone(),
                    shape: vec![MTP_TEST_EXPERTS, n, width],
                    offset: payload.len() as u64,
                    bytes: bits.len() as u64 * 2,
                    encoding: Encoding::Safe("BF16".into()),
                },
            ));
            payload.extend(bits.iter().flat_map(|b| b.to_le_bytes()));
            sources.push(MxArray::from_bfloat16(&bits, &[rows as i64, width as i64]).unwrap());
        }
        std::fs::write(&path, &payload).unwrap();
        (tensors, sources, cleanup)
    }

    /// The draft gate_up bank at its real expert shape (1280 x 2560, a K the
    /// affine expert GEMV refuses) is built with f32 sides once, and its
    /// gather keeps the bits of the f16-sided bank it replaces; the down bank
    /// keeps f16 sides for the GEMV, and the estimate is the built bytes.
    /// MUTATIONS: M-M1-f16 (`quantize_q8` keeps f16 sides) -> the gate_up
    /// dtype assertion; M-M1-estimate (9/8 bytes for f32 sides) -> the bytes;
    /// M-M1-bits (`quantize_q8` quantizes at the side dtype) -> `rows=10`.
    #[test]
    fn mtp_gate_up_bank_holds_f32_sides_with_the_f16_gather_bits() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let (experts, sources, _cleanup) = real_shape_mtp_experts();
        let mut s = Store::open_fixture(&fixture().join("bf16/paged"), None).unwrap();
        s.tensors.extend(experts);
        let estimate = s.mtp_bank_estimate().unwrap();
        let installed = s.prepare_mtp_expert_banks_with(&mut |_| Ok(())).unwrap();
        let sides = |name: &str| {
            let bank = s.resident_bank(name).unwrap();
            [&bank.scales, &bank.biases].map(|a| a.as_ref().unwrap().dtype().unwrap())
        };
        assert_eq!(
            sides(MTP_EXPERT_BANKS[0]),
            [DType::Float32; 2],
            "the gather casts f16 gate_up sides on every draft"
        );
        assert_eq!(
            sides(MTP_EXPERT_BANKS[1]),
            [DType::Float16; 2],
            "the affine expert GEMV reads f16 down sides"
        );
        assert_eq!(installed, s.mtp_bank_bytes().unwrap());
        assert_eq!(
            installed,
            (MTP_TEST_EXPERTS * 1280 * 2560 * 10 / 8 + MTP_TEST_EXPERTS * 2560 * 640 * 9 / 8)
                as u64
        );
        assert_eq!(estimate, installed, "the estimate charges the built banks");
        let gate_up = s.resident_bank(MTP_EXPERT_BANKS[0]).unwrap();
        let before = Weight::quantize_q8(&sources[0], DType::Float16).unwrap();
        for rows in [10usize, 40, 120] {
            let ids = MxArray::from_uint32(
                &(0..rows)
                    .map(|i| ((i * 7 + i / 3) % MTP_TEST_EXPERTS) as u32)
                    .collect::<Vec<_>>(),
                &[rows as i64],
            )
            .unwrap();
            let x = MxArray::from_float32(
                &(0..rows * 2560)
                    .map(|i| (i as f32 * 0.013).cos())
                    .collect::<Vec<_>>(),
                &[rows as i64, 1, 2560],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let bits = |bank: &Weight| {
                bank.expert_rows(&x, &ids, MTP_TEST_EXPERTS, false)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            };
            let want = bits(&before);
            assert!(want.iter().any(|&b| b != 0), "rows={rows}");
            assert_eq!(bits(&gate_up), want, "rows={rows}");
        }
    }

    /// The f32 sides need more live headroom than the f16 layout: a refused
    /// f32 build builds the f16 one once instead of declining the draft banks,
    /// and declines them only when that is refused too.
    /// MUTATIONS: M-M1-live (no f16 retry) -> "the f32 gate_up staging";
    /// M-M1-live-any (the retry for any layout) ->
    /// `failed_second_mtp_bank_leaves_no_partial_bank`.
    #[test]
    fn refused_f32_draft_banks_fall_back_to_the_f16_layout() {
        let (experts, _, _cleanup) = real_shape_mtp_experts();
        let bytes = |n: usize, width: usize, eighths: usize| {
            (MTP_TEST_EXPERTS * n * width * eighths / 8) as u64
        };
        let (wide, narrow, down) = (
            bytes(1280, 2560, 10),
            bytes(1280, 2560, 9),
            bytes(2560, 640, 9),
        );
        let [wide_up, narrow_up, down_up] =
            [wide, narrow, down].map(|bank| 2 * bank + super::super::super::memory::WORKING_BYTES);
        let cases: [(&str, &[usize], &[u64], bool); 3] = [
            (
                "the f32 gate_up staging",
                &[0],
                &[wide_up, narrow_up, down_up],
                true,
            ),
            (
                "the down staging beside the f32 gate_up bank",
                &[1],
                &[wide_up, down_up, narrow_up, down_up],
                true,
            ),
            (
                "both gate_up stagings",
                &[0, 1],
                &[wide_up, narrow_up],
                false,
            ),
        ];
        for (why, refused, want, built) in cases {
            let mut s = Store::open_fixture(&fixture().join("bf16/paged"), None).unwrap();
            s.tensors.extend(experts.iter().cloned());
            assert_eq!(s.mtp_bank_estimate().unwrap(), wide + down, "{why}");
            let mut seen = Vec::new();
            let result = s.prepare_mtp_expert_banks_with(&mut |bytes| {
                seen.push(bytes);
                if refused.contains(&(seen.len() - 1)) {
                    Err(err("simulated headroom loss"))
                } else {
                    Ok(())
                }
            });
            if built {
                assert_eq!(result.ok(), Some(narrow + down), "{why}");
                let bank = s.resident_bank(MTP_EXPERT_BANKS[0]).unwrap();
                assert_eq!(
                    [&bank.scales, &bank.biases].map(|a| a.as_ref().unwrap().dtype().unwrap()),
                    [DType::Float16; 2],
                    "{why}"
                );
                assert!(!s.mtp_banks_declined, "{why}");
            } else {
                assert!(result.is_err(), "{why}");
                assert!(s.mtp_banks_declined, "{why}");
                assert_eq!(s.mtp_bank_bytes().unwrap(), 0, "{why}");
            }
            assert_eq!(seen, want, "{why}");
        }
    }

    #[test]
    fn partial_projection_banks_use_actual_sizes_and_preserve_expert_lru() {
        for path in [fixture(), fixture().join("model.gguf")] {
            let mut s = Store::open_metadata(&path, Some(&Store::fixture_pack_root())).unwrap();
            let bytes: u64 = s
                .tensors
                .iter()
                .filter(|(name, t)| is_fixed_projection(name, t))
                .map(|(name, t)| t.bank_bytes(name).unwrap())
                .sum();
            assert!(bytes > 0);
            s.cache_limit = bytes * 4 - 1;
            assert!(s.projection_bank_names().unwrap().is_empty());
            s.cache_limit = bytes * 4;
            let names = s.projection_bank_names().unwrap();
            assert!(!names.is_empty());
            assert_eq!(
                s.bytes_read, 0,
                "budget selection must read only descriptors"
            );

            // A metadata-only fixture can exercise the bank preparation without
            // requesting tens of GiB from the machine running these tests.
            s.cache_limit = CACHE_BYTES;
            s.prepare_fixture_hot().unwrap();
            assert_eq!(s.bank_bytes, bytes);
            assert_eq!(s.cache_bytes, bytes);
            assert_eq!(s.plan.resident_bank_bytes.load(Ordering::Relaxed), bytes);
            assert!(s.cache.is_empty());
            assert_eq!(s.banks.len(), names.len());
            assert!(
                s.banks
                    .keys()
                    .all(|n| is_fixed_projection(n, &s.tensors[n]))
            );
            let loaded = (s.bytes_read, s.packed_bytes_read);
            let mut control =
                Store::open_metadata(&path, Some(&Store::fixture_pack_root())).unwrap();
            for name in names {
                let rows = s.descriptor(&name).unwrap().rows().unwrap();
                for start in [0, rows - 1] {
                    let got = s
                        .read(&name, start, 1)
                        .unwrap()
                        .dense()
                        .unwrap()
                        .to_float32()
                        .unwrap();
                    let want = control
                        .read(&name, start, 1)
                        .unwrap()
                        .dense()
                        .unwrap()
                        .to_float32()
                        .unwrap();
                    assert_eq!(&*got, &*want, "partial bank changed {name}, row {start}");
                }
            }
            assert_eq!((s.bytes_read, s.packed_bytes_read), loaded);
            let expert = s
                .tensors
                .keys()
                .find(|n| n.contains(".experts.") || n.contains("_exps."))
                .unwrap()
                .clone();
            assert!(!s.banks.contains_key(&expert));
            s.read(&expert, 0, 1).unwrap();
            assert!(s.cache_bytes > bytes);
            s.evict_to(bytes).unwrap();
            assert!(s.cache.is_empty(), "expert rows must remain evictable");
            assert_eq!(s.cache_bytes, s.bank_bytes);
        }
    }

    #[test]
    fn projection_banks_preserve_rows_and_leave_lookup_and_auxiliary_weights_cold() {
        for path in [fixture(), fixture().join("bf16/paged/model.gguf")] {
            let mut s = Store::open_metadata(&path, Some(&Store::fixture_pack_root())).unwrap();
            assert_eq!(s.bytes_read, 0);
            s.prepare_fixture_hot().unwrap();
            assert_eq!(s.cache_bytes, s.bank_bytes);
            assert!(s.cache.is_empty());
            assert!(s.banks.keys().all(|name| is_hot(name)));
            assert!(
                s.banks
                    .keys()
                    .all(|name| is_fixed_projection(name, &s.tensors[name]))
            );
            assert!(!s.banks.is_empty());
            let reads = (s.bytes_read, s.packed_hits, s.packed_bytes_read);
            let mut reference =
                Store::open_metadata(&path, Some(&Store::fixture_pack_root())).unwrap();
            for name in s.banks.keys().cloned().collect::<Vec<_>>() {
                let rows = s.descriptor(&name).unwrap().rows().unwrap();
                for start in [0, rows - 1] {
                    let a = s
                        .read(&name, start, 1)
                        .unwrap()
                        .dense()
                        .unwrap()
                        .to_float32()
                        .unwrap();
                    let b = reference
                        .read(&name, start, 1)
                        .unwrap()
                        .dense()
                        .unwrap()
                        .to_float32()
                        .unwrap();
                    assert_eq!(&*a, &*b, "resident row changed: {name}, {start}");
                }
                assert!(s.read(&name, rows, 1).is_err());
            }
            assert_eq!((s.bytes_read, s.packed_hits, s.packed_bytes_read), reads);
            for name in [
                "token_embd.weight",
                "embed_tokens.weight",
                "per_layer_token_embd.weight",
                "layers.0.ple_embedding.ngram_embedding.weight",
                "mtp.fc_embedding.weight",
                "visual.patch_embed.proj.weight",
            ] {
                assert!(!is_hot(name), "cold tensor was selected: {name}");
            }
        }
    }

    #[test]
    fn oversized_bank_is_rejected_before_any_payload_read() {
        let mut s = Store::open_metadata(&fixture(), None).unwrap();
        let d = s.tensors.get_mut("lm_head.weight").unwrap();
        d.shape = vec![1 << 30, 1];
        d.bytes = 4 << 30;
        // Budget selection leaves such a checkpoint on its chunk path...
        assert!(s.projection_bank_names().unwrap().is_empty());
        s.prepare_fixture_hot().unwrap();
        assert!(s.banks.is_empty());
        // ...and an explicit bank request is refused before any read.
        assert!(
            s.prepare_banks(vec!["lm_head.weight".into()], &mut |_| Ok(()))
                .unwrap_err()
                .to_string()
                .contains("1 GiB bank budget")
        );
        assert_eq!(s.bytes_read, 0);
        assert!(s.banks.is_empty());
    }
}
