use super::super::tapped_tree_rows;
use super::{DraftStepper, Verified};
use crate::array::MxArray;
use crate::engine::backend::DsparkTreeGreedyResult;
use crate::models::qwen3_5::verify_tree::cuda::{
    TreeLookupSource, finalize_tree_greedy_walk, prepare_tree_lookup, select_tree_gpu,
};
use crate::models::qwen3_5::verify_tree::{
    DraftLattice, LookupAgreement, MAX_TREE_NODES, TreeCostModel, TreeDeviceViews, VerifyTree,
    build_verify_tree, build_verify_tree_cost_aware,
};
use napi::{Error, Result};

pub(super) fn env_flag_default(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let trimmed = v.trim();
            if trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no")
            {
                false
            } else if trimmed.eq_ignore_ascii_case("1")
                || trimmed.eq_ignore_ascii_case("true")
                || trimmed.eq_ignore_ascii_case("on")
                || trimmed.eq_ignore_ascii_case("yes")
            {
                true
            } else {
                default
            }
        }
        Err(_) => default,
    }
}

pub(super) fn env_qwen4_tree_cost_model() -> TreeCostModel {
    let mut model = TreeCostModel::QWEN4_MOE_DEFAULT;
    if let Ok(v) = std::env::var("MLX_QWEN4_TREE_COST_BASE")
        && let Ok(parsed) = v.trim().parse::<f64>()
        && parsed.is_finite()
        && parsed >= 0.0
    {
        model.base_ms = parsed;
    }
    if let Ok(v) = std::env::var("MLX_QWEN4_TREE_COST_SLOPE")
        && let Ok(parsed) = v.trim().parse::<f64>()
        && parsed.is_finite()
        && parsed > 0.0
    {
        model.slope_ms = parsed;
    }
    if let Ok(v) = std::env::var("MLX_QWEN4_TREE_MIN_NODES")
        && let Ok(parsed) = v.trim().parse::<usize>()
    {
        model.min_nodes = parsed.clamp(1, MAX_TREE_NODES);
    }
    model.bucketed = env_flag_default("MLX_QWEN4_TREE_BUCKETED", model.bucketed);
    model
}

pub(super) fn env_qwen4_tree_fixed_max_nodes() -> Option<usize> {
    std::env::var("MLX_QWEN4_TREE_MAX_NODES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|n| n.clamp(1, MAX_TREE_NODES))
}

impl DraftStepper<'_> {
    pub(super) fn print_tree_stats(&self) {
        if self.tree_cycles > 0
            && (std::env::var("MLX_QWEN4_TREE_STATS").is_ok()
                || std::env::var("MLX_DFLASH2_PHASE_TIME").is_ok())
        {
            let c = self.tree_cycles as f64;
            eprintln!(
                "[qwen4-tree-stats] cycles={} mean_T={:.2} mean_acc={:.2} noncontig={} lookup_acc={} propose_ms={:.2} verify_ms={:.2} cost=({:.1}+{:.2}*T) fixed={:?}",
                self.tree_cycles,
                self.tree_nodes_sum as f64 / c,
                self.tree_accepted_sum as f64 / c,
                self.tree_noncontig_commits,
                self.tree_lookup_accepted,
                (self.tree_propose_us as f64 / c) / 1000.0,
                (self.tree_verify_us as f64 / c) / 1000.0,
                self.tree_cost_model.base_ms,
                self.tree_cost_model.slope_ms,
                self.tree_fixed_max_nodes,
            );
        }
    }

    pub(super) fn verify_tree_greedy_impl(
        &mut self,
        anchor: u32,
        max_nodes: usize,
        max_depth: usize,
    ) -> Result<DsparkTreeGreedyResult> {
        if !self.tree_enabled {
            return Err(Error::from_reason(
                "Qwen4 DFlash2 tree verification is not enabled on this stepper",
            ));
        }
        if self.verified.is_some() {
            return Err(Error::from_reason(
                "Qwen4 draft: the last verify was not committed",
            ));
        }
        let position = self.decoder.history.len();
        if usize::try_from(self.context.logical_len()).ok() != Some(position) {
            return Err(Error::from_reason(format!(
                "Qwen4 draft tree: context ends at {}, target at {position}",
                self.context.logical_len()
            )));
        }

        let t_propose = std::time::Instant::now();
        let draft_slots = self.draft.config.block_size;
        let (candidates, scores) = self.draft.propose_tree_lattice_arrays_embedded(
            &mut self.decoder.draft_view_with_limit(None),
            self.context,
            anchor,
            draft_slots,
        )?;

        let lookup = prepare_tree_lookup(&self.lookup_index, self.lookup_enabled, max_depth);
        let max_cap = max_nodes.clamp(1, MAX_TREE_NODES);

        enum TreeLookupBacking {
            Lattice(DraftLattice),
            Agreements(Vec<LookupAgreement>),
        }

        // Optional GPU `select_tree_cuda` when a fixed tree width is pinned and
        // `MLX_QWEN4_GPU_TREE=1`; otherwise evaluate the 7.6 KB lattice on host
        // and run MoE-cost-aware adaptive sizing `T* = argmax E[T] / Cost(T)`
        // (or fixed-width `build_verify_tree` when `MLX_QWEN4_TREE_MAX_NODES` is set).
        let (tree, views, lookup_backing) = if let Some(fixed) = self.tree_fixed_max_nodes
            && self.gpu_tree_enabled
            && crate::accelerator::cuda_ops_available()
        {
            let node_budget = max_cap.min(fixed);
            if let Some(sel) = select_tree_gpu(
                &candidates,
                &scores,
                &lookup,
                &self.lookup_acceptance,
                &self.verify_budget,
                anchor,
                node_budget,
            )? {
                MxArray::eval_arrays_with_context(
                    &[&sel.tree_tokens, &sel.parent, &sel.node_meta],
                    "qwen4::gpu_tree_select",
                )?;
                let raw_tokens = sel.tree_tokens.to_uint32()?.to_vec();
                let raw_parent = sel.parent.to_int32()?.to_vec();
                let raw_meta = sel.node_meta.to_float32()?.to_vec();
                let (tree, agreements, _) =
                    VerifyTree::from_gpu_readback(&raw_tokens, &raw_parent, &raw_meta)?;
                let views = if tree.len() == node_budget {
                    TreeDeviceViews {
                        depths_i32: sel.depths,
                        mask_u32: sel.ancestors_mask,
                        parent_i32: sel.parent,
                        conv_taps_i32: sel.conv_taps,
                        tree_len: node_budget,
                    }
                } else {
                    tree.to_device_views()?
                };
                (tree, views, TreeLookupBacking::Agreements(agreements))
            } else {
                let lattice = self
                    .draft
                    .lattice_from_arrays(&candidates, &scores, draft_slots)?;
                let tree = build_verify_tree(
                    anchor,
                    &lattice,
                    &lookup.proposals,
                    lookup.match_len,
                    node_budget,
                    Some(lookup.eff_max_depth),
                    false,
                    &self.verify_budget,
                    &self.lookup_acceptance,
                );
                let views = tree.to_device_views()?;
                (tree, views, TreeLookupBacking::Lattice(lattice))
            }
        } else {
            let lattice = self
                .draft
                .lattice_from_arrays(&candidates, &scores, draft_slots)?;
            let tree = if let Some(fixed) = self.tree_fixed_max_nodes {
                build_verify_tree(
                    anchor,
                    &lattice,
                    &lookup.proposals,
                    lookup.match_len,
                    max_cap.min(fixed),
                    Some(lookup.eff_max_depth),
                    false,
                    &self.verify_budget,
                    &self.lookup_acceptance,
                )
            } else {
                build_verify_tree_cost_aware(
                    anchor,
                    &lattice,
                    &lookup.proposals,
                    lookup.match_len,
                    max_cap,
                    Some(lookup.eff_max_depth),
                    false,
                    &self.verify_budget,
                    &self.lookup_acceptance,
                    &self.tree_cost_model,
                )
            };
            let views = tree.to_device_views()?;
            (tree, views, TreeLookupBacking::Lattice(lattice))
        };
        let propose_us = t_propose.elapsed().as_micros() as u64;

        let t_verify = std::time::Instant::now();
        let (logits, ticket) = self.decoder.verify_tree_deferred(&tree, &views)?;
        let argmax = logits.argmax(-1, None)?;
        MxArray::eval_arrays_with_context(&[&argmax], "qwen4::tree_greedy_accept")?;
        let verify_us = t_verify.elapsed().as_micros() as u64;

        let target_argmax: Vec<u32> = argmax.to_uint32()?.to_vec();
        if target_argmax.len() != tree.len() {
            return Err(Error::from_reason(format!(
                "Qwen4 DFlash2 tree argmax length {} != tree length {}",
                target_argmax.len(),
                tree.len()
            )));
        }
        let lookup_source = match &lookup_backing {
            TreeLookupBacking::Lattice(lattice) => TreeLookupSource::Lattice(lattice),
            TreeLookupBacking::Agreements(agreements) => TreeLookupSource::Agreements(agreements),
        };
        let (walk, lookup_accepted) = finalize_tree_greedy_walk(
            &tree,
            &target_argmax,
            &lookup,
            lookup_source,
            &mut self.verify_budget,
            &mut self.lookup_acceptance,
        );
        self.tree_lookup_accepted += lookup_accepted;

        let is_chain = tree.is_chain();
        let draft_count = tree.depth.iter().copied().max().unwrap_or(0).max(0) as usize;
        let accepted_draft_count = walk.accepted_nodes.len().saturating_sub(1);
        let tree_nodes = tree.len();
        self.tree_cycles += 1;
        self.tree_nodes_sum += tree_nodes;
        self.tree_accepted_sum += walk.emitted_tokens.len();
        self.tree_propose_us += propose_us;
        self.tree_verify_us += verify_us;

        self.verified = Some(Verified::TreeDeferred {
            position,
            is_chain,
            ticket,
        });
        Ok(DsparkTreeGreedyResult {
            accepted_nodes: walk.accepted_nodes,
            emitted_tokens: walk.emitted_tokens,
            is_contiguous_prefix: walk.is_contiguous_prefix,
            draft_count,
            tree_nodes,
            accepted_draft_count,
        })
    }

    pub(super) fn commit_tree_impl(
        &mut self,
        accepted_nodes: &[usize],
        is_contiguous_prefix: bool,
    ) -> Result<()> {
        let verified = self
            .verified
            .take()
            .ok_or_else(|| Error::from_reason("Qwen4 draft tree commit has no verify"))?;
        let Verified::TreeDeferred {
            position,
            is_chain,
            ticket,
        } = verified
        else {
            return Err(Error::from_reason(
                "Qwen4 draft tree commit called on a non-tree verify",
            ));
        };
        let keep = accepted_nodes.len();
        let total = ticket.tree_tokens.len();
        if keep == 0 || keep > total || accepted_nodes[0] != 0 {
            return Err(Error::from_reason(format!(
                "Qwen4 draft tree commit invalid accepted_nodes {accepted_nodes:?} for T={total}"
            )));
        }
        let accepted_tokens: Vec<u32> = accepted_nodes
            .iter()
            .map(|&idx| ticket.tree_tokens[idx])
            .collect();
        if !is_contiguous_prefix {
            self.tree_noncontig_commits += 1;
        }
        let passes = self.decoder.take_hidden_taps();
        let rows = tapped_tree_rows(&passes, position, accepted_nodes, is_contiguous_prefix)?;
        drop(passes);
        self.decoder.commit_tree_deferred(
            ticket,
            accepted_nodes,
            is_contiguous_prefix,
            is_chain,
        )?;
        let fused = self.draft.fuse_context(&rows)?;
        self.context
            .append(self.draft, &fused, position as i32, &accepted_tokens)?;
        // Leave commit + context append lazy on CUDA: `eval_boundary` flushes
        // caches + context + boundary token asynchronously via `cuda_async_eval`,
        // overlapping commit with the next cycle's draft proposal.
        Ok(())
    }
}
