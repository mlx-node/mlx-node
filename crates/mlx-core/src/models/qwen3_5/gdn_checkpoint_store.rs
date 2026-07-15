use std::collections::VecDeque;

/// One resident GDN sidecar for the root Pi session plus the four child loops
/// the subagent extension can run concurrently. Sidecars are large (about
/// 75 MiB for the dense 27B checkpoint), so keep this a hard bound rather than
/// making it scale with the paged KV pool or the eight-task submission limit.
pub(crate) const GDN_PREFIX_CHECKPOINT_LIMIT: usize = 5;

/// A linear owner may keep its current head plus one recent branch point.
pub(crate) const GDN_PREFIX_CHECKPOINTS_PER_OWNER: usize = 2;

/// Minimal identity needed to decide whether two materialized GDN sidecars
/// belong to the same exact paged-cache lineage.
///
/// The block-hash chain is load-bearing: token ancestry alone is insufficient
/// when VLM extra keys or a cache salt differ. Each hash already commits to the
/// preceding hash, the block's tokens, and that block's extra keys (including
/// the first-block salt), so a chain prefix is the same identity used by the
/// paged allocator itself.
pub(crate) trait GdnCheckpointLineage {
    fn owner_id(&self) -> &str;
    fn prefix_len(&self) -> u32;
    fn block_size(&self) -> u32;
    fn final_block_hash(&self) -> u64;
    fn tokens(&self) -> &[u32];
    fn block_hashes(&self) -> &[u64];
}

/// Replay into a staged GDN cache and publish it only after the replay succeeds.
///
/// Materialized replay is fallible after it has already mutated recurrent
/// state. Keeping that state local prevents a failed prefix preparation from
/// leaving the model's active cache partially advanced.
pub(crate) fn replay_gdn_cache_and_commit<T, E, F>(
    active: &mut Option<T>,
    mut staged: T,
    replay: F,
) -> Result<(), E>
where
    F: FnOnce(&mut T) -> Result<(), E>,
{
    replay(&mut staged)?;
    *active = Some(staged);
    Ok(())
}

fn is_strict_ancestor<T: GdnCheckpointLineage>(ancestor: &T, descendant: &T) -> bool {
    ancestor.block_size() == descendant.block_size()
        && ancestor.tokens().len() < descendant.tokens().len()
        && ancestor.block_hashes().len() < descendant.block_hashes().len()
        && descendant.tokens().starts_with(ancestor.tokens())
        && descendant
            .block_hashes()
            .starts_with(ancestor.block_hashes())
}

fn is_same_owner_ancestor<T: GdnCheckpointLineage>(ancestor: &T, descendant: &T) -> bool {
    ancestor.owner_id() == descendant.owner_id() && is_strict_ancestor(ancestor, descendant)
}

/// Enforce the bounded branch-aware retention policy.
///
/// First enforce the two-entry owner cap, preferring an exact older ancestor.
/// Then enforce the global cap: replace a same-owner ancestor, otherwise an
/// owner's extra entry, otherwise the least-recent non-root owner. The root's
/// sole checkpoint is the final eviction candidate so child turns cannot
/// displace the interactive parent merely by cycling through task owners.
pub(crate) fn prune_gdn_checkpoints<T: GdnCheckpointLineage>(
    checkpoints: &mut VecDeque<T>,
    limit: usize,
    per_owner_limit: usize,
    root_owner_id: &str,
) {
    let owners: Vec<String> = checkpoints
        .iter()
        .map(|checkpoint| checkpoint.owner_id().to_owned())
        .collect();
    for owner in owners {
        while checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.owner_id() == owner)
            .count()
            > per_owner_limit
        {
            let same_owner_ancestor = (0..checkpoints.len()).find(|&idx| {
                checkpoints[idx].owner_id() == owner
                    && checkpoints.iter().enumerate().any(|(other_idx, other)| {
                        other_idx != idx && is_same_owner_ancestor(&checkpoints[idx], other)
                    })
            });
            let oldest_for_owner =
                (0..checkpoints.len()).find(|&idx| checkpoints[idx].owner_id() == owner);
            checkpoints.remove(same_owner_ancestor.or(oldest_for_owner).unwrap());
        }
    }

    while checkpoints.len() > limit {
        let same_owner_ancestor = (0..checkpoints.len()).find(|&idx| {
            checkpoints.iter().enumerate().any(|(other_idx, other)| {
                other_idx != idx && is_same_owner_ancestor(&checkpoints[idx], other)
            })
        });
        if let Some(idx) = same_owner_ancestor {
            checkpoints.remove(idx);
            continue;
        }

        let owner_with_extra = (0..checkpoints.len()).find(|&idx| {
            let owner = checkpoints[idx].owner_id();
            checkpoints
                .iter()
                .filter(|checkpoint| checkpoint.owner_id() == owner)
                .count()
                > 1
        });
        if let Some(idx) = owner_with_extra {
            checkpoints.remove(idx);
            continue;
        }

        let non_root =
            (0..checkpoints.len()).find(|&idx| checkpoints[idx].owner_id() != root_owner_id);
        checkpoints.remove(non_root.unwrap_or(0));
    }
}

/// Compute every allocator-compatible block hash through `prefix_len`.
pub(crate) fn compute_paged_prefix_block_hashes(
    tokens: &[u32],
    prefix_len: u32,
    block_size: u32,
    extra_keys_per_block: &[Vec<u64>],
    cache_salt: u64,
) -> Option<Vec<u64>> {
    if prefix_len == 0 || block_size == 0 || !prefix_len.is_multiple_of(block_size) {
        return None;
    }

    let prefix_len = prefix_len as usize;
    let block_size = block_size as usize;
    if prefix_len > tokens.len() {
        return None;
    }

    let num_blocks = prefix_len / block_size;
    let mut hashes = Vec::with_capacity(num_blocks);
    let mut parent_hash = 0;
    for block_idx in 0..num_blocks {
        let extra_keys = extra_keys_per_block.get(block_idx)?;
        let start = block_idx * block_size;
        let end = start + block_size;
        parent_hash = if block_idx == 0 && cache_salt != 0 {
            let mut salted_keys = Vec::with_capacity(extra_keys.len() + 1);
            salted_keys.extend_from_slice(extra_keys);
            salted_keys.push(cache_salt);
            mlx_paged_attn::hash_tokens(&tokens[start..end], parent_hash, &salted_keys)
        } else {
            mlx_paged_attn::hash_tokens(&tokens[start..end], parent_hash, extra_keys)
        };
        hashes.push(parent_hash);
    }

    Some(hashes)
}

#[cfg(test)]
pub(crate) fn compute_paged_prefix_block_hash(
    tokens: &[u32],
    prefix_len: u32,
    block_size: u32,
    extra_keys_per_block: &[Vec<u64>],
    cache_salt: u64,
) -> Option<u64> {
    compute_paged_prefix_block_hashes(
        tokens,
        prefix_len,
        block_size,
        extra_keys_per_block,
        cache_salt,
    )
    .and_then(|hashes| hashes.last().copied())
}

/// Find the most recent longest checkpoint on the requested paged-cache
/// lineage.
///
/// PagedAttention may back a cache hit up by one or more complete blocks when
/// the live tail cannot be reused. A sidecar captured at the former live head
/// is then a descendant of the requested prefix and cannot be restored, but an
/// older retained sidecar can still be a valid ancestor. Match that ancestor
/// using the allocator's complete hash chain (which commits to tokens, cache
/// salt, and per-block extra keys), then let the caller replay only the gap.
pub(crate) fn find_longest_valid_gdn_checkpoint_index<T, F>(
    checkpoints: &VecDeque<T>,
    owner_id: &str,
    tokens: &[u32],
    requested_prefix_len: u32,
    block_size: u32,
    extra_keys_per_block: &[Vec<u64>],
    cache_salt: u64,
    mut caches_ready: F,
) -> Option<usize>
where
    T: GdnCheckpointLineage,
    F: FnMut(&T) -> bool,
{
    let requested_block_hashes = compute_paged_prefix_block_hashes(
        tokens,
        requested_prefix_len,
        block_size,
        extra_keys_per_block,
        cache_salt,
    )?;

    let mut best: Option<(usize, u32)> = None;
    for (idx, checkpoint) in checkpoints.iter().enumerate() {
        let prefix_len = checkpoint.prefix_len();
        let prefix_len_usize = prefix_len as usize;
        let expected_blocks = prefix_len.checked_div(block_size)? as usize;
        let valid = checkpoint.owner_id() == owner_id
            && prefix_len > 0
            && prefix_len <= requested_prefix_len
            && prefix_len.is_multiple_of(block_size)
            && checkpoint.block_size() == block_size
            && checkpoint.tokens().len() == prefix_len_usize
            && tokens.get(..prefix_len_usize) == Some(checkpoint.tokens())
            && checkpoint.block_hashes().len() == expected_blocks
            && requested_block_hashes.get(..expected_blocks) == Some(checkpoint.block_hashes())
            && checkpoint.block_hashes().last().copied() == Some(checkpoint.final_block_hash())
            && caches_ready(checkpoint);
        if !valid {
            continue;
        }

        // The queue is LRU ordered, so prefer the later entry when duplicate
        // lineage lengths exist.
        if best.is_none_or(|(best_idx, best_len)| {
            prefix_len > best_len || (prefix_len == best_len && idx > best_idx)
        }) {
            best = Some((idx, prefix_len));
        }
    }
    best.map(|(idx, _)| idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_commit_is_transactional() {
        let mut active = Some(vec![1, 2]);
        let failed: Result<(), &'static str> =
            replay_gdn_cache_and_commit(&mut active, vec![10], |staged| {
                staged.push(11);
                Err("replay failed")
            });
        assert_eq!(failed, Err("replay failed"));
        assert_eq!(active, Some(vec![1, 2]));

        replay_gdn_cache_and_commit(&mut active, vec![20], |staged| {
            staged.push(21);
            Ok::<_, &'static str>(())
        })
        .unwrap();
        assert_eq!(active, Some(vec![20, 21]));
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Lineage {
        name: &'static str,
        owner: &'static str,
        prefix_len: u32,
        block_size: u32,
        final_block_hash: u64,
        tokens: Vec<u32>,
        hashes: Vec<u64>,
    }

    impl GdnCheckpointLineage for Lineage {
        fn owner_id(&self) -> &str {
            self.owner
        }

        fn prefix_len(&self) -> u32 {
            self.prefix_len
        }

        fn block_size(&self) -> u32 {
            self.block_size
        }

        fn final_block_hash(&self) -> u64 {
            self.final_block_hash
        }

        fn tokens(&self) -> &[u32] {
            &self.tokens
        }

        fn block_hashes(&self) -> &[u64] {
            &self.hashes
        }
    }

    fn lineage(name: &'static str, owner: &'static str, hashes: &[u64]) -> Lineage {
        Lineage {
            name,
            owner,
            prefix_len: hashes.len() as u32,
            block_size: 16,
            final_block_hash: hashes.last().copied().unwrap_or_default(),
            tokens: (0..hashes.len() as u32).collect(),
            hashes: hashes.to_vec(),
        }
    }

    #[test]
    fn linear_history_replaces_the_oldest_ancestor() {
        let mut checkpoints: VecDeque<_> = (1..=10)
            .map(|len| {
                let hashes: Vec<u64> = (1..=len as u64).collect();
                lineage("linear", "root", &hashes)
            })
            .collect();

        prune_gdn_checkpoints(
            &mut checkpoints,
            GDN_PREFIX_CHECKPOINT_LIMIT,
            GDN_PREFIX_CHECKPOINTS_PER_OWNER,
            "root",
        );

        assert_eq!(checkpoints.len(), GDN_PREFIX_CHECKPOINTS_PER_OWNER);
        assert_eq!(checkpoints.front().unwrap().hashes.len(), 9);
        assert_eq!(checkpoints.back().unwrap().hashes.len(), 10);
    }

    #[test]
    fn root_and_four_active_child_owners_fit_without_eviction() {
        let mut checkpoints = VecDeque::from([lineage("parent", "root", &[1, 2])]);
        for branch in 0..4u64 {
            let owner = match branch {
                0 => "child-0",
                1 => "child-1",
                2 => "child-2",
                _ => "child-3",
            };
            checkpoints.push_back(lineage("child", owner, &[10 + branch]));
        }

        prune_gdn_checkpoints(
            &mut checkpoints,
            GDN_PREFIX_CHECKPOINT_LIMIT,
            GDN_PREFIX_CHECKPOINTS_PER_OWNER,
            "root",
        );

        assert_eq!(checkpoints.len(), 5);
        assert!(
            checkpoints
                .iter()
                .any(|checkpoint| checkpoint.name == "parent")
        );
    }

    #[test]
    fn sixth_owner_evicts_oldest_child_but_preserves_root() {
        let mut checkpoints = VecDeque::from([lineage("parent", "root", &[1, 2])]);
        for branch in 0..5u64 {
            let owner = match branch {
                0 => "child-0",
                1 => "child-1",
                2 => "child-2",
                3 => "child-3",
                _ => "child-4",
            };
            checkpoints.push_back(lineage("child", owner, &[100 + branch]));
        }

        prune_gdn_checkpoints(
            &mut checkpoints,
            GDN_PREFIX_CHECKPOINT_LIMIT,
            GDN_PREFIX_CHECKPOINTS_PER_OWNER,
            "root",
        );

        assert_eq!(checkpoints.len(), 5);
        assert!(
            checkpoints
                .iter()
                .any(|checkpoint| checkpoint.name == "parent")
        );
        assert!(
            !checkpoints
                .iter()
                .any(|checkpoint| checkpoint.owner == "child-0")
        );
    }

    #[test]
    fn rotating_root_evicts_old_root_and_preserves_new_root_with_four_children() {
        let mut checkpoints = VecDeque::from([lineage("old-root", "root-0", &[1])]);
        checkpoints.push_back(lineage("new-root", "root-1", &[2]));
        for (name, owner, hash) in [
            ("child-0", "child-0", 10),
            ("child-1", "child-1", 11),
            ("child-2", "child-2", 12),
            ("child-3", "child-3", 13),
        ] {
            checkpoints.push_back(lineage(name, owner, &[hash]));
            prune_gdn_checkpoints(
                &mut checkpoints,
                GDN_PREFIX_CHECKPOINT_LIMIT,
                GDN_PREFIX_CHECKPOINTS_PER_OWNER,
                "root-1",
            );
        }

        assert_eq!(checkpoints.len(), GDN_PREFIX_CHECKPOINT_LIMIT);
        assert!(
            !checkpoints
                .iter()
                .any(|checkpoint| checkpoint.owner == "root-0")
        );
        assert!(
            checkpoints
                .iter()
                .any(|checkpoint| checkpoint.owner == "root-1")
        );
    }

    #[test]
    fn equal_tokens_with_different_hash_chains_are_not_same_lineage() {
        let tokens = vec![1, 2];
        let mut checkpoints = VecDeque::from([
            Lineage {
                name: "salt-a",
                owner: "root",
                prefix_len: 2,
                block_size: 16,
                final_block_hash: 12,
                tokens: tokens.clone(),
                hashes: vec![11, 12],
            },
            Lineage {
                name: "salt-b",
                owner: "root",
                prefix_len: 2,
                block_size: 16,
                final_block_hash: 22,
                tokens,
                hashes: vec![21, 22],
            },
        ]);

        prune_gdn_checkpoints(&mut checkpoints, 1, 2, "root");

        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints.front().unwrap().name, "salt-b");
    }

    #[test]
    fn owner_cap_prefers_replacing_an_exact_ancestor() {
        let mut checkpoints = VecDeque::from([
            lineage("ancestor", "child", &[1]),
            lineage("other-branch", "child", &[9]),
            lineage("head", "child", &[1, 2]),
        ]);

        prune_gdn_checkpoints(&mut checkpoints, 5, 2, "root");

        assert_eq!(checkpoints.len(), 2);
        assert!(
            !checkpoints
                .iter()
                .any(|checkpoint| checkpoint.name == "ancestor")
        );
        assert!(
            checkpoints
                .iter()
                .any(|checkpoint| checkpoint.name == "other-branch")
        );
        assert!(
            checkpoints
                .iter()
                .any(|checkpoint| checkpoint.name == "head")
        );
    }

    #[test]
    fn lineage_ancestry_rejects_different_block_sizes_and_owners() {
        let ancestor = lineage("ancestor", "root", &[1]);
        let mut other_block_size = lineage("descendant", "root", &[1, 2]);
        other_block_size.block_size = 32;
        assert!(!is_strict_ancestor(&ancestor, &other_block_size));

        let other_owner = lineage("descendant", "child", &[1, 2]);
        assert!(is_strict_ancestor(&ancestor, &other_owner));
        assert!(!is_same_owner_ancestor(&ancestor, &other_owner));
    }

    #[test]
    fn block_hash_chain_commits_salt_and_extra_keys() {
        let tokens: Vec<u32> = (1..=8).collect();
        let keys = vec![vec![11], vec![22]];
        let unsalted = compute_paged_prefix_block_hashes(&tokens, 8, 4, &keys, 0).unwrap();
        let salted = compute_paged_prefix_block_hashes(&tokens, 8, 4, &keys, 99).unwrap();
        let different_keys =
            compute_paged_prefix_block_hashes(&tokens, 8, 4, &[vec![12], vec![22]], 0).unwrap();

        assert_ne!(unsalted, salted);
        assert_ne!(unsalted, different_keys);
    }

    fn paged_lineage(
        name: &'static str,
        owner: &'static str,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        keys: &[Vec<u64>],
        salt: u64,
    ) -> Lineage {
        let hashes =
            compute_paged_prefix_block_hashes(tokens, prefix_len, block_size, keys, salt).unwrap();
        Lineage {
            name,
            owner,
            prefix_len,
            block_size,
            final_block_hash: *hashes.last().unwrap(),
            tokens: tokens[..prefix_len as usize].to_vec(),
            hashes,
        }
    }

    #[test]
    fn lookup_prefers_exact_then_longest_same_owner_ancestor() {
        let tokens: Vec<u32> = (1..=12).collect();
        let keys = vec![vec![11], vec![22], vec![33]];
        let mut checkpoints = VecDeque::from([
            paged_lineage("four", "root", &tokens, 4, 4, &keys, 7),
            paged_lineage("eight", "root", &tokens, 8, 4, &keys, 7),
            paged_lineage("other-owner", "child", &tokens, 12, 4, &keys, 7),
            paged_lineage("twelve", "root", &tokens, 12, 4, &keys, 7),
        ]);

        let exact = find_longest_valid_gdn_checkpoint_index(
            &checkpoints,
            "root",
            &tokens,
            12,
            4,
            &keys,
            7,
            |_| true,
        )
        .unwrap();
        assert_eq!(checkpoints[exact].name, "twelve");

        checkpoints.remove(exact);
        let ancestor = find_longest_valid_gdn_checkpoint_index(
            &checkpoints,
            "root",
            &tokens,
            12,
            4,
            &keys,
            7,
            |_| true,
        )
        .unwrap();
        assert_eq!(checkpoints[ancestor].name, "eight");

        checkpoints.remove(ancestor);
        checkpoints.push_back(paged_lineage(
            "descendant",
            "root",
            &tokens,
            12,
            4,
            &keys,
            7,
        ));
        let backed_up = find_longest_valid_gdn_checkpoint_index(
            &checkpoints,
            "root",
            &tokens,
            8,
            4,
            &keys,
            7,
            |_| true,
        )
        .unwrap();
        assert_eq!(checkpoints[backed_up].name, "four");
    }

    #[test]
    fn lookup_rejects_token_salt_extra_key_and_cache_readiness_mismatches() {
        let tokens: Vec<u32> = (1..=8).collect();
        let keys = vec![vec![11], vec![22]];
        let checkpoint = paged_lineage("candidate", "root", &tokens, 8, 4, &keys, 7);
        let checkpoints = VecDeque::from([checkpoint]);

        let mut divergent_tokens = tokens.clone();
        divergent_tokens[6] = 99;
        assert!(
            find_longest_valid_gdn_checkpoint_index(
                &checkpoints,
                "root",
                &divergent_tokens,
                8,
                4,
                &keys,
                7,
                |_| true,
            )
            .is_none()
        );
        assert!(
            find_longest_valid_gdn_checkpoint_index(
                &checkpoints,
                "root",
                &tokens,
                8,
                4,
                &keys,
                8,
                |_| true,
            )
            .is_none()
        );
        assert!(
            find_longest_valid_gdn_checkpoint_index(
                &checkpoints,
                "root",
                &tokens,
                8,
                4,
                &[vec![11], vec![23]],
                7,
                |_| true,
            )
            .is_none()
        );
        assert!(
            find_longest_valid_gdn_checkpoint_index(
                &checkpoints,
                "root",
                &tokens,
                8,
                4,
                &keys,
                7,
                |_| false,
            )
            .is_none()
        );
    }
}
