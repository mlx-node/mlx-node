//! Admitted hot banks use contiguous quantized storage. Partial residency can
//! assemble fixed projections while experts, PLE/token rows and optional
//! vision/MTP remain demand loaded. All source reads stay bounded.
use super::*;
use crate::models::qwen4_exp::runtime_flags;

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
        if runtime_flags::is_zero(c"MLX_QWEN4_DENSE_BF16_CACHE")
            || !f32
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
                GgufTensorType::Q5_1 => (24, 32),
                GgufTensorType::Q8_0 => (36, 32),
                GgufTensorType::Q3K => (114, 256),
                GgufTensorType::Q4K => (148, 256),
                GgufTensorType::Q5K => (180, 256),
                GgufTensorType::Q6K => (210, 256),
                GgufTensorType::IQ4NL => (19, 32),
                GgufTensorType::IQ3S => (266, 256),
                GgufTensorType::IQ4XS => (138, 256),
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
        self.plan.resident_expert_layers = self
            .tensors
            .keys()
            .filter(|name| {
                name.ends_with(".ffn_gate_exps.weight")
                    && !is_auxiliary(name)
                    && ["gate", "up", "down"].iter().all(|part| {
                        self.banks.contains_key(
                            &name.replace("ffn_gate_exps", &format!("ffn_{part}_exps")),
                        )
                    })
            })
            .count();
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
        let names = if self.plan.resident {
            let mut names: Vec<_> = self.tensors.keys().filter(|n| is_hot(n)).cloned().collect();
            names.sort();
            names
        } else if self.plan.policy != "stream" && !runtime_flags::is_zero(c"MLX_QWEN4_DENSE_BANKS")
        {
            self.projection_bank_names()?
        } else {
            Vec::new()
        };
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
        self.order.clear();
        self.recency.clear();
        self.protected.clear();
        self.protected_bytes = 0;
        self.bank_bytes = 0;
        self.cache_bytes = 0;
        self.growth_bytes = 0;
        self.plan.resident_bank_bytes = 0;
        self.plan.resident_expert_layers = 0;
        self.plan.partial_expert_slots = 0;
        crate::array::memory::synchronize_and_clear_cache();
        if self.plan.policy == "full" {
            return Err(err(
                "Qwen4 full residency cannot preserve first-request headroom",
            ));
        }
        refresh(&mut self.plan)?;
        self.cache_limit = self.plan.budget;
        self.plan.resident = false;
        self.slot_capacity = None;
        let names = if runtime_flags::is_zero(c"MLX_QWEN4_DENSE_BANKS") {
            Vec::new()
        } else {
            self.projection_bank_names()?
        };
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
        for (name, tensor) in &self.tensors {
            if !is_fixed_projection(name, tensor) {
                continue;
            }
            let size = tensor.bank_bytes(name)?;
            // An unusual dense/BF16 checkpoint keeps its bounded chunk path.
            if size > 1 << 30 || size > (self.cache_limit / 4).saturating_sub(bytes) {
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
            let keys: Vec<_> = self
                .cache
                .keys()
                .filter(|(n, _, _)| n == &name)
                .cloned()
                .collect();
            for key in keys {
                if let Some(w) = self.cache.remove(&key) {
                    self.cache_bytes -= w.bytes()?;
                    if self.protected.remove(&key) {
                        self.protected_bytes -= w.bytes()?;
                    }
                }
                self.recency.remove(&key);
            }
            let bytes = bank.bytes()?;
            self.bank_bytes += bytes;
            self.cache_bytes += bytes;
            self.plan.resident_bank_bytes = self.bank_bytes;
            self.peak_cache_bytes = self.peak_cache_bytes.max(self.cache_bytes);
            self.banks.insert(name, Arc::new(bank));
            drop(chunks);
            super::super::memory::maintain_freelist(self.plan.physical_bytes);
        }
        Ok(true)
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
            && !runtime_flags::is_zero(c"MLX_QWEN4_PREFILL_SHARED_PAIR")
            && !runtime_flags::is_zero(c"MLX_ENABLE_TF32")
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
        let gate_pair =
            first.ends_with(".ssm_alpha.weight") && second.ends_with(".ssm_beta.weight");
        let dense_gates = gate_pair
            && rows == 1
            && x.dtype()? == DType::BFloat16
            && shape.last() == Some(&2560)
            && runtime_flags::is_one(c"MLX_QWEN4_GDN_GATE_PAIR");
        // A wider N changes MLX's matrix-multiply schedule for short windows.
        // This must precede the paired-bank cache lookup: a bank prepared by
        // singleton decoding must not opt a later verification window in.
        if gate_pair && !dense_gates {
            return Ok((self.linear(first, x)?, self.linear(second, x)?));
        }
        if (rows <= 8 || wide_shared) && !runtime_flags::is_zero(c"MLX_QWEN4_PAIRED_PROJECTIONS") {
            let key = (first.to_owned(), second.to_owned());
            if !self.paired_banks.contains_key(&key)
                && let (Some(a), Some(b)) = (
                    self.banks.get(first).cloned(),
                    self.banks.get(second).cloned(),
                )
            {
                let storage_compatible = match (&a.scales, &b.scales) {
                    (Some(a), Some(b)) => a.dtype()? == b.dtype()?,
                    (None, None) => {
                        dense_gates
                            && a.values.dtype()? == DType::Float32
                            && b.values.dtype()? == DType::Float32
                            && *a.values.shape()? == [48, 2560]
                            && *b.values.shape()? == [48, 2560]
                            && a.dense_bf16.is_some()
                            && b.dense_bf16.is_some()
                            && a.biases.is_none()
                            && b.biases.is_none()
                    }
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
            // The generic fast path uses a different accumulation for K%512=0.
            && x.shape()?.last().is_some_and(|k| k % 512 != 0)
            && crate::engine::persistence::compiled_forward_backend_available()
            && !runtime_flags::is_zero(c"MLX_QWEN4_AFFINE_GEMV")
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
            && !runtime_flags::is_zero(c"MLX_QWEN4_DIRECT_GEMV")
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
            let mut store = Store::open_metadata(&fixture().join("model.gguf"), None).unwrap();
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
    fn failed_full_load_releases_banks_before_partial_retry() {
        for policy in ["auto", "full"] {
            let mut store = Store::open_metadata(&fixture().join("model.gguf"), None).unwrap();
            store.plan.hot_bytes = store.hot_bytes().unwrap();
            store.plan.resident = true;
            store.plan.policy = policy.into();
            let mut refused = false;
            let mut refreshed = false;
            let result = store.prepare_hot_with_admission(
                &mut |bytes| {
                    // Simulate headroom loss only after the full bank set is
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
            assert_eq!(refreshed, policy == "auto");
            if policy == "full" {
                assert!(result.is_err());
                assert!(store.banks.is_empty());
                assert_eq!(store.cache_bytes, 0);
            } else {
                result.unwrap();
                assert!(!store.plan.resident);
                assert!(!store.banks.is_empty());
                assert!(
                    store
                        .banks
                        .keys()
                        .all(|name| is_fixed_projection(name, &store.tensors[name]))
                );
                assert_eq!(store.cache_bytes, store.bank_bytes);
                assert_eq!(store.plan.resident_bank_bytes, store.bank_bytes);
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
    fn partial_projection_banks_use_actual_sizes_and_preserve_expert_lru() {
        for path in [fixture(), fixture().join("model.gguf")] {
            let mut s = Store::open_metadata(&path, None).unwrap();
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

            // A metadata-only fixture can exercise the partial policy without
            // requesting tens of GiB from the machine running these tests.
            s.cache_limit = CACHE_BYTES;
            s.plan.policy = "auto".into();
            s.prepare_fixture_hot().unwrap();
            assert!(!s.plan.resident);
            assert_eq!(s.bank_bytes, bytes);
            assert_eq!(s.cache_bytes, bytes);
            assert_eq!(s.plan.resident_bank_bytes, bytes);
            assert!(s.cache.is_empty());
            assert_eq!(s.banks.len(), names.len());
            assert!(
                s.banks
                    .keys()
                    .all(|n| is_fixed_projection(n, &s.tensors[n]))
            );
            let loaded = (s.bytes_read, s.packed_bytes_read);
            let mut control = Store::open_metadata(&path, None).unwrap();
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
    fn resident_banks_preserve_rows_and_leave_lookup_and_auxiliary_weights_cold() {
        for path in [fixture(), fixture().join("bf16/paged/model.gguf")] {
            let mut s = Store::open_metadata(&path, None).unwrap();
            s.plan.resident = true;
            let hot = s.hot_bytes().unwrap();
            assert_eq!(s.bytes_read, 0);
            s.prepare_fixture_hot().unwrap();
            assert_eq!(s.cache_bytes, hot);
            assert!(s.cache.is_empty());
            assert!(s.banks.keys().all(|name| is_hot(name)));
            assert!(!s.banks.is_empty());
            let reads = (s.bytes_read, s.packed_hits, s.packed_bytes_read);
            let mut reference = Store::open_metadata(&path, None).unwrap();
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
        s.plan.resident = true;
        let d = s.tensors.get_mut("lm_head.weight").unwrap();
        d.shape = vec![1 << 30, 1];
        d.bytes = 4 << 30;
        assert!(
            s.prepare_fixture_hot()
                .unwrap_err()
                .to_string()
                .contains("1 GiB bank budget")
        );
        assert_eq!(s.bytes_read, 0);
        assert!(s.banks.is_empty());
    }
}
