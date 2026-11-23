use super::*;

impl Decoder {
    pub(super) fn hyper(
        &mut self,
        x: &MxArray,
        hf: &str,
        gg: &str,
        inject: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        self.hyper_with_norm(x, hf, gg, inject, None)
    }

    fn cuda_hc_inject_weight(
        bank: &super::super::weights::Weight,
    ) -> Option<crate::kernels::qwen4::Qwen4HcInjectWeight<'_>> {
        if bank.mode == "affine"
            && bank.bits == 8
            && bank.group == 32
            && let (Some(scales), Some(biases)) = (&bank.scales, &bank.biases)
        {
            return Some(crate::kernels::qwen4::Qwen4HcInjectWeight::Q8_0Bias {
                weight: &bank.values,
                scales,
                biases,
            });
        }
        if bank.scales.is_some() {
            return None;
        }
        if let Some(bf16) = &bank.dense_bf16 {
            return Some(crate::kernels::qwen4::Qwen4HcInjectWeight::DenseBf16(bf16));
        }
        match bank.values.dtype().ok()? {
            DType::BFloat16 => Some(crate::kernels::qwen4::Qwen4HcInjectWeight::DenseBf16(
                &bank.values,
            )),
            DType::Float16 => Some(crate::kernels::qwen4::Qwen4HcInjectWeight::DenseFp16(
                &bank.values,
            )),
            DType::Float32 => Some(crate::kernels::qwen4::Qwen4HcInjectWeight::DenseFp32(
                &bank.values,
            )),
            _ => None,
        }
    }

    pub(super) fn hyper_with_norm(
        &mut self,
        x: &MxArray,
        hf: &str,
        gg: &str,
        inject: bool,
        normed: Option<&MxArray>,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let c = self.config.clone();
        let rows = x.shape_at(1)?;
        let n = if let Some(normed) = normed {
            normed.clone()
        } else if super::super::cuda_qwen4_fast_enabled()
            && c.hc_count == 4
            && c.hidden_size == 2560
            && x.dtype()? == DType::BFloat16
        {
            let w = self.dense(
                &format!("{hf}.hc_norm.weight"),
                &format!("{gg}_norm.weight"),
            )?;
            let weight_offset = if self.gguf() { 0.0 } else { 1.0 };
            if let Some((_, normed)) = crate::kernels::qwen4::inject_norm_cuda(
                x,
                None,
                &w,
                c.rms_norm_eps as f32,
                weight_offset,
            )? {
                normed
            } else {
                self.norm(
                    x,
                    &format!("{hf}.hc_norm.weight"),
                    &format!("{gg}_norm.weight"),
                    c.hidden_size,
                    true,
                )?
            }
        } else {
            self.norm(
                x,
                &format!("{hf}.hc_norm.weight"),
                &format!("{gg}_norm.weight"),
                c.hidden_size,
                true,
            )?
        };
        let mut gate_ready = false;
        let cuda_down = if super::super::cuda_qwen4_fast_enabled()
            && c.hc_count == 4
            && c.hidden_size == 2560
            && c.hc_lowrank == 320
            && (rows <= 16 || rows > 512)
            && rows >= 1
            && n.dtype()? == DType::BFloat16
        {
            let down_key = self.key(
                &format!("{hf}.input_mix_weight_down.weight"),
                &format!("{gg}_down.weight"),
            );
            let inj_key = self.key(
                &format!("{hf}.block_inject_weight.weight"),
                &format!("{gg}_inject.weight"),
            );
            if let Some(down_bank) = self.weights.resident_bank(&down_key)
                && down_bank.mode == "affine"
                && down_bank.bits == 8
                && down_bank.group == 32
                && (rows <= 16 || down_bank.q8_symmetric_128)
                && let (Some(s_down), Some(b_down)) = (&down_bank.scales, &down_bank.biases)
            {
                let inj_bank = if inject {
                    self.weights.resident_bank(&inj_key)
                } else {
                    None
                };
                let inj_w = match inj_bank.as_deref() {
                    Some(bank) => Self::cuda_hc_inject_weight(bank)
                        .unwrap_or(crate::kernels::qwen4::Qwen4HcInjectWeight::None),
                    None => crate::kernels::qwen4::Qwen4HcInjectWeight::None,
                };
                let fused_inj = !matches!(inj_w, crate::kernels::qwen4::Qwen4HcInjectWeight::None);
                if let Some((down_act, inj_out)) = crate::kernels::qwen4::hc_down_act_cuda(
                    &n,
                    &down_bank.values,
                    s_down,
                    b_down,
                    inj_w,
                )? {
                    let gate = if inject {
                        if fused_inj {
                            gate_ready = true;
                            inj_out
                        } else {
                            Some(self.linear_rows(
                                &n,
                                &format!("{hf}.block_inject_weight.weight"),
                                &format!("{gg}_inject.weight"),
                            )?)
                        }
                    } else {
                        None
                    };
                    Some((down_act, gate))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        let (is_activated, down, inject_projection) = if let Some((down_act, gate)) = cuda_down {
            (true, down_act, gate)
        } else {
            let activated = if self.gguf()
                && c.hc_count == 4
                && c.hidden_size == 2560
                && (rows >= 1024 || rows == 1)
            {
                let down_key = self.key(
                    &format!("{hf}.input_mix_weight_down.weight"),
                    &format!("{gg}_down.weight"),
                );
                self.weights
                    .resident_bank(&down_key)
                    .map(|w| w.mixer_act(&n))
                    .transpose()?
                    .flatten()
            } else {
                None
            };
            let is_activated = activated.is_some();
            let (down, inject_projection) = if let Some(activated) = activated {
                let gate = if inject {
                    Some(self.linear(
                        &n,
                        &format!("{hf}.block_inject_weight.weight"),
                        &format!("{gg}_inject.weight"),
                    )?)
                } else {
                    None
                };
                (activated, gate)
            } else if inject {
                let (down, gate) = self.linear_pair(
                    &n,
                    (
                        &format!("{hf}.input_mix_weight_down.weight"),
                        &format!("{gg}_down.weight"),
                    ),
                    (
                        &format!("{hf}.block_inject_weight.weight"),
                        &format!("{gg}_inject.weight"),
                    ),
                )?;
                (down, Some(gate))
            } else {
                (
                    self.linear(
                        &n,
                        &format!("{hf}.input_mix_weight_down.weight"),
                        &format!("{gg}_down.weight"),
                    )?,
                    None,
                )
            };
            (is_activated, down, inject_projection)
        };
        let down = if is_activated {
            down
        } else {
            Activations::silu(&down.div_scalar(c.hc_count as f64)?)?
        };
        let key = self.key(
            &format!("{hf}.input_mix_weight_up.weight"),
            &format!("{gg}_up.weight"),
        );
        let cuda_up = if super::super::cuda_qwen4_fast_enabled()
            && c.hc_count == 4
            && c.hidden_size == 2560
            && c.hc_lowrank == 320
            && down.dtype()? == DType::BFloat16
            && n.dtype()? == DType::BFloat16
        {
            if let Some(up_bank) = self.weights.resident_bank(&key)
                && up_bank.mode == "affine"
                && up_bank.bits == 8
                && up_bank.group == 32
                && let (Some(s_up), Some(b_up)) = (&up_bank.scales, &up_bank.biases)
            {
                if (1..=128).contains(&rows) {
                    crate::kernels::qwen4::hc_up_mix_cuda(&down, &n, &up_bank.values, s_up, b_up)?
                } else if rows > 128 && up_bank.q8_symmetric_128 {
                    crate::kernels::qwen4::hc_up_mix_prefill_cuda(
                        &down,
                        &n,
                        &up_bank.values,
                        s_up,
                        b_up,
                    )?
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        let fused = if cuda_up.is_some() {
            cuda_up
        } else if self.gguf() && c.hc_count == 4 && c.hidden_size == 2560 {
            self.weights
                .resident_bank(&key)
                .map(|w| w.hyper_up(&down, &n))
                .transpose()?
                .flatten()
        } else {
            None
        };
        let mixed = if let Some(mixed) = fused {
            mixed
        } else {
            let up = self.weights.linear(&key, &down)?;
            if super::super::cuda_qwen4_fast_enabled()
                && c.hc_count == 4
                && c.hidden_size == 2560
                && up.dtype()? == DType::BFloat16
                && n.dtype()? == DType::BFloat16
                && let Some(out) = crate::kernels::qwen4::prefill_hc_mix_cuda(&up, &n)?
            {
                out
            } else {
                let combined = if self.gguf()
                    && crate::engine::persistence::compiled_forward_backend_available()
                {
                    crate::kernels::qwen4::metal::prefill_hc_mix(&up, &n)?
                } else {
                    None
                };
                if let Some(combined) = combined {
                    combined
                } else {
                    math::sigmoid_mul(&up, &n)?
                        .reshape(&[1, rows, c.hc_count as i64, c.hidden_size as i64])?
                        .mean(Some(&[-2]), Some(false))?
                }
            }
        };
        let gate = if inject {
            let w = inject_projection.ok_or_else(|| {
                Error::from_reason("Qwen4 hyper-connection is missing its injection projection")
            })?;
            if gate_ready {
                Some(w)
            } else {
                Some(Activations::sigmoid(&w.div_scalar(c.hc_count as f64)?)?.mul_scalar(2.0)?)
            }
        } else {
            None
        };
        Ok((mixed, gate))
    }

    // Normalized results stay on the forward's call stack, with no
    // cache/frontier state to restore after a device-route replay.
    pub(super) fn inject_for_mlp(
        &mut self,
        x: &MxArray,
        branch: &MxArray,
        gate: &MxArray,
        layer: usize,
        normalize: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        self.inject_for_hyper(
            x,
            branch,
            gate,
            &format!("layers.{layer}.mlp_hyper_connection.hc_norm.weight"),
            &format!("blk.{layer}.hc_ffn_norm.weight"),
            normalize,
        )
    }

    pub(super) fn inject_for_hyper(
        &mut self,
        x: &MxArray,
        branch: &MxArray,
        gate: &MxArray,
        hf: &str,
        gg: &str,
        normalize: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        if normalize
            && super::super::cuda_qwen4_fast_enabled()
            && self.config.hidden_size == 2560
            && self.config.hc_count == 4
            && x.dtype()? == DType::BFloat16
        {
            let w = self.dense(hf, gg)?;
            let weight_offset = if self.gguf() { 0.0 } else { 1.0 };
            if let Some((stream, normed)) = crate::kernels::qwen4::inject_norm_cuda(
                x,
                Some((branch, gate)),
                &w,
                self.config.rms_norm_eps as f32,
                weight_offset,
            )? {
                return Ok((stream, Some(normed)));
            }
        }
        if normalize
            && self.gguf()
            && self.config.hidden_size == 2560
            && self.config.hc_count == 4
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            let w = self.dense(hf, gg)?;
            if let Some((stream, normed)) = crate::kernels::qwen4::metal::inject_norm(
                x,
                branch,
                gate,
                &w,
                self.config.rms_norm_eps,
            )? {
                return Ok((stream, Some(normed)));
            }
        }
        Ok((Self::inject(x, branch, gate)?, None))
    }

    pub(super) fn inject(x: &MxArray, y: &MxArray, g: &MxArray) -> Result<MxArray> {
        if super::super::cuda_qwen4_fast_enabled()
            && x.dtype()? == DType::BFloat16
            && y.dtype()? == DType::BFloat16
            && g.dtype()? == DType::BFloat16
            && x.ndim()? >= 2
            && x.shape_at(x.ndim()? - 1)? == 10240
            && let Some(out) = crate::kernels::qwen4::inject_only_cuda(x, y, g)?
        {
            return Ok(out);
        }
        if crate::engine::persistence::compiled_forward_backend_available() {
            return crate::kernels::qwen4::common::inject(x, y, g);
        }
        x.add(
            &y.expand_dims(-2)?
                .mul(&g.expand_dims(-1)?)?
                .reshape(&x.shape()?)?,
        )
    }

    /// `hyper_with_norm` over a `[1, R, 4H]` verify window. The admitted
    /// mixer banks stream once through the row-batched kernels; any other
    /// configuration runs the single-row mixer per row.
    pub(super) fn hyper_rows(
        &mut self,
        x: &MxArray,
        hf: &str,
        gg: &str,
        inject: bool,
        normed: Option<&MxArray>,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let rows = x.shape_at(1)?;
        if rows == 1 {
            return self.hyper_with_norm(x, hf, gg, inject, normed);
        }
        let c = self.config.clone();
        if super::super::cuda_qwen4_fast_enabled()
            && (2..=16).contains(&rows)
            && c.hc_count == 4
            && c.hidden_size == 2560
            && x.dtype()? == DType::BFloat16
        {
            return self.hyper_with_norm(x, hf, gg, inject, normed);
        }
        let down_key = self.key(
            &format!("{hf}.input_mix_weight_down.weight"),
            &format!("{gg}_down.weight"),
        );
        let up_key = self.key(
            &format!("{hf}.input_mix_weight_up.weight"),
            &format!("{gg}_up.weight"),
        );
        if self.gguf()
            && c.hc_count == 4
            && c.hidden_size == 2560
            && let (Some(down_bank), Some(up_bank)) = (
                self.weights.resident_bank(&down_key),
                self.weights.resident_bank(&up_key),
            )
        {
            let n = match normed {
                Some(normed) => normed.clone(),
                None => self.norm(
                    x,
                    &format!("{hf}.hc_norm.weight"),
                    &format!("{gg}_norm.weight"),
                    c.hidden_size,
                    true,
                )?,
            };
            if let Some(down) = math::rows::mixer_act(&down_bank, &n)? {
                let injection = if inject {
                    Some(self.linear_rows(
                        &n,
                        &format!("{hf}.block_inject_weight.weight"),
                        &format!("{gg}_inject.weight"),
                    )?)
                } else {
                    None
                };
                if let Some(mixed) = math::rows::hyper_up(&up_bank, &down, &n)? {
                    let gate = injection
                        .map(|w| -> Result<MxArray> {
                            Activations::sigmoid(&w.div_scalar(c.hc_count as f64)?)?.mul_scalar(2.0)
                        })
                        .transpose()?;
                    return Ok((mixed, gate));
                }
            }
        }
        let mut mixed = Vec::with_capacity(rows as usize);
        let mut gates = Vec::with_capacity(rows as usize);
        for j in 0..rows {
            let normed = normed.map(|n| n.slice_axis(1, j, j + 1)).transpose()?;
            let (m, gate) =
                self.hyper_with_norm(&x.slice_axis(1, j, j + 1)?, hf, gg, inject, normed.as_ref())?;
            mixed.push(m);
            gates.extend(gate);
        }
        let gate = if inject {
            Some(MxArray::concatenate_many(gates.iter().collect(), Some(1))?)
        } else {
            None
        };
        Ok((
            MxArray::concatenate_many(mixed.iter().collect(), Some(1))?,
            gate,
        ))
    }
}
