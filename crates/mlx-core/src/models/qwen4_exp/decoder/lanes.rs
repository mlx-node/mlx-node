//! One decode step over 2-4 sequences, one row each, on the row-exact
//! kernels of a verify pass: projections, mixers, router, experts and head
//! read their weights once for all rows, and row `j` carries the bits of its
//! sequence's one-row step. The per-row part (PLE mix, GDN state, attention
//! with its page write) runs with that row's own state and page table. The
//! rows' page writes chain through the adapter's pool arrays, which every
//! request shares, so the step waits once, at its end.
use super::*;
use crate::transformer::paged_kv_cache_adapter::SeqId;

/// The most rows one lane step takes: the row kernels serve 1-4 rows.
pub(in super::super) const LANE_ROWS: usize = 4;

/// One sequence's row of a lane step: its next token and its state, out of
/// the decoder while the step runs.
pub(in super::super) struct Lane {
    pub(in super::super) seq: SeqId,
    pub(in super::super) token: u32,
    pub(in super::super) state: DecoderState,
}

impl Decoder {
    fn swap_lane(&mut self, state: &mut DecoderState) {
        std::mem::swap(&mut self.caches, &mut state.caches);
        std::mem::swap(&mut self.history, &mut state.history);
        std::mem::swap(&mut self.positions, &mut state.positions);
        std::mem::swap(&mut self.rope_delta, &mut state.rope_delta);
        std::mem::swap(&mut self.media_digests, &mut state.media_digests);
        std::mem::swap(&mut self.last_chunk_hidden, &mut state.last_chunk_hidden);
        std::mem::swap(&mut self.cancelled, &mut state.cancelled);
        std::mem::swap(&mut self.route_guard, &mut state.route_guard);
    }

    /// `run` with `lane`'s state in the decoder and, with `pages`, its page
    /// table active.
    fn as_lane<R>(
        &mut self,
        lane: &mut Lane,
        pages: bool,
        run: impl FnOnce(&mut Self) -> Result<R>,
    ) -> Result<R> {
        if pages {
            self.paged
                .as_mut()
                .ok_or_else(|| Error::from_reason("Qwen4 lane step has no paged cache"))?
                .activate_request(lane.seq)
                .map_err(Error::from_reason)?;
        }
        self.swap_lane(&mut lane.state);
        let out = run(self);
        self.swap_lane(&mut lane.state);
        out
    }

    /// One decode step of `lanes` (2 through [`LANE_ROWS`] distinct
    /// sequences): `[rows, 1, vocabulary]` logits, row `j` for `lanes[j]`.
    /// Each lane's state comes back advanced by its token, with every array
    /// and page write evaluated by the step's one blocking eval.
    pub(in super::super) fn lanes_step(&mut self, lanes: &mut [Lane]) -> Result<MxArray> {
        if !(2..=LANE_ROWS).contains(&lanes.len()) {
            return Err(Error::from_reason(format!(
                "Qwen4 lane step takes 2-{LANE_ROWS} rows, not {}",
                lanes.len()
            )));
        }
        self.rotary_tables.get_mut().clear();
        self.verification = Some(Vec::new());
        if self.verify_routes_on_device() {
            self.device_routes = Some(Vec::with_capacity(self.config.num_hidden_layers));
        }
        self.index_planes = Some(Vec::new());
        let result = self.lanes_pass(lanes);
        self.verification = None;
        self.device_routes = None;
        self.index_planes = None;
        self.rotary_tables.get_mut().clear();
        if result.is_ok() {
            self.weights
                .live
                .lane_steps
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            #[cfg(test)]
            {
                self.lane_steps += 1;
            }
        }
        result
    }

    fn lanes_pass(&mut self, lanes: &mut [Lane]) -> Result<MxArray> {
        let rows = lanes.len();
        let mut starts = Vec::with_capacity(rows);
        for lane in lanes.iter_mut() {
            let token = lane.token;
            starts.push(self.as_lane(lane, true, |d| d.lane_start(token))?);
        }
        let mut x = MxArray::concatenate_many(starts.iter().collect(), Some(1))?;
        drop(starts);
        for i in 0..self.config.num_hidden_layers {
            let next = self.lanes_layer(&x, lanes, i)?;
            self.submit(&[&next], "qwen4::decoder::lanes")?;
            x = next;
            super::super::memory::maintain_freelist(self.weights.plan.physical_bytes);
        }
        let (mixed, _) = self.hyper_rows(&x, "hyper_connection_mixer", "output_hc", false, None)?;
        let logits = self
            .linear_rows(&mixed, "lm_head.weight", "output.weight")?
            .reshape(&[rows as i64, 1, -1])?;
        let routes = self.device_routes.take();
        let tape = routes
            .as_ref()
            .filter(|routes| !routes.is_empty())
            .map(|routes| {
                MxArray::concatenate_many(routes.iter().map(|(_, ids)| ids).collect(), Some(0))
            })
            .transpose()?;
        let planes = self.index_planes.take().unwrap_or_default();
        let completed: Vec<_> = std::iter::once(&logits)
            .chain(std::iter::once(&x))
            .chain(
                lanes
                    .iter()
                    .flat_map(|lane| lane.state.caches.iter().flat_map(LayerCache::arrays)),
            )
            .chain(tape.iter())
            .chain(planes.iter())
            .collect();
        MxArray::eval_arrays_with_context(&completed, "qwen4::decoder::lanes")?;
        // The K/V writes are inputs of the last row's page reads and the
        // index stores are in `completed`: this only clears their marks.
        self.paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen4 lane step has no paged cache"))?
            .eval_pending_pool_writes()
            .map_err(Error::from_reason)?;
        if let (Some(routes), Some(tape)) = (routes, tape) {
            let layers: Vec<_> = routes.iter().map(|(layer, _)| *layer).collect();
            if !self.weights.commit_device_routes(
                &layers,
                &tape.to_uint32()?,
                rows * self.config.num_experts_per_tok,
            )? {
                return Err(Error::from_reason(
                    "Qwen4 lane step routes lost their expert banks",
                ));
            }
        }
        for (j, lane) in lanes.iter_mut().enumerate() {
            lane.state.history.push(lane.token);
            lane.state.last_chunk_hidden = vec![x.slice_axis(1, j as i64, j as i64 + 1)?];
        }
        Ok(logits)
    }

    /// The lane's frontier check, its page cursor's record, its row
    /// requests and its embedding, as the stream entering layer 0.
    fn lane_start(&mut self, token: u32) -> Result<MxArray> {
        let base = self.history.len();
        if base + 1 > self.context_window || token as usize >= self.config.vocab_size {
            return Err(Error::from_reason(
                "Qwen4 lane token exceeds vocabulary/context",
            ));
        }
        self.paged_chunk_start = base;
        let adapter = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen4 lane step has no paged cache"))?;
        if adapter.request_tokens() != self.history {
            return Err(Error::from_reason("Qwen4 lane frontier mismatch"));
        }
        adapter
            .record_tokens(&[token])
            .map_err(Error::from_reason)?;
        self.prefetch_window_rows(&[token], &[], true)?;
        MxArray::tile(
            &self.embed_token(token)?,
            &[1, 1, self.config.hc_count as i32],
        )
    }

    /// Layer `i` over the lanes' `[1, rows, 4H]` streams.
    fn lanes_layer(&mut self, x: &MxArray, lanes: &mut [Lane], i: usize) -> Result<MxArray> {
        let row = |a: &MxArray, j: usize| a.slice_axis(1, j as i64, j as i64 + 1);
        let x = if self.config.ple_layer_ids.contains(&(i + 1)) {
            let dtype = x.dtype()?;
            let mut embeddings = Vec::with_capacity(lanes.len());
            for lane in lanes.iter_mut() {
                let token = lane.token;
                embeddings
                    .push(self.as_lane(lane, false, |d| d.ple_embeddings(&[token], i, dtype))?);
            }
            let embeddings = MxArray::concatenate_many(embeddings.iter().collect(), Some(1))?;
            let (key, value) = self.ple_projections(&embeddings, i)?;
            let mut parts = Vec::with_capacity(lanes.len());
            for (j, lane) in lanes.iter_mut().enumerate() {
                let input = row(x, j)?;
                let (key, value) = (row(&key, j)?, row(&value, j)?);
                let ple = self.as_lane(lane, false, |d| {
                    let mut cache = std::mem::take(&mut d.caches[i]);
                    let ple = d.ple_mix(&input, &key, &value, i, &mut cache);
                    d.caches[i] = cache;
                    ple
                })?;
                parts.push(input.add(&ple)?);
            }
            MxArray::concatenate_many(parts.iter().collect(), Some(1))?
        } else {
            x.clone()
        };
        let (mixed, gate) = self.hyper_rows(
            &x,
            &format!("layers.{i}.attn_hyper_connection"),
            &format!("blk.{i}.hc_attn"),
            true,
            None,
        )?;
        let branch = if self.config.linear(i) {
            let p = format!("layers.{i}.linear_attn");
            let g = format!("blk.{i}");
            let projections = self.gdn_projections(&mixed, &p, &g)?;
            let dtype = mixed.dtype()?;
            let mut outputs = Vec::with_capacity(lanes.len());
            for (j, lane) in lanes.iter_mut().enumerate() {
                let projections = projections.row(j)?;
                outputs.push(self.as_lane(lane, false, |d| {
                    let mut cache = std::mem::take(&mut d.caches[i]);
                    let out = d.gdn_mix(dtype, &projections, i, &mut cache);
                    d.caches[i] = cache;
                    out
                })?);
            }
            let out = MxArray::concatenate_many(outputs.iter().collect(), Some(1))?;
            self.linear_rows(
                &out,
                &format!("{p}.out_proj.weight"),
                &format!("{g}.ssm_out.weight"),
            )?
        } else {
            let projections = self.attention_projections(&mixed, i)?;
            let mut outputs = Vec::with_capacity(lanes.len());
            for (j, lane) in lanes.iter_mut().enumerate() {
                let input = row(&mixed, j)?;
                let projections = projections.row(j)?;
                outputs.push(self.as_lane(lane, true, |d| {
                    d.paged_chunk_start = d.history.len();
                    let mut cache = std::mem::take(&mut d.caches[i]);
                    let out =
                        d.attention_projected(&input, i, &mut cache, &projections, false, true);
                    d.caches[i] = cache;
                    out
                })?);
            }
            let out = MxArray::concatenate_many(outputs.iter().collect(), Some(1))?;
            self.linear_rows(
                &out,
                &format!("layers.{i}.self_attn.o_proj.weight"),
                &format!("blk.{i}.attn_output.weight"),
            )?
        };
        let gate = gate
            .ok_or_else(|| Error::from_reason("Qwen4 attention is missing its injection gate"))?;
        let (x, normed) = self.inject_for_mlp(&x, &branch, &gate, i, true)?;
        let (mixed, gate) = self.hyper_rows(
            &x,
            &format!("layers.{i}.mlp_hyper_connection"),
            &format!("blk.{i}.hc_ffn"),
            true,
            normed.as_ref(),
        )?;
        let branch = self.moe_rows(&mixed, i)?;
        let gate =
            gate.ok_or_else(|| Error::from_reason("Qwen4 MLP is missing its injection gate"))?;
        Self::inject(&x, &branch, &gate)
    }
}
