//! Dense expert IDs permit linear assignment planning without hash lookups.
//! Keep the existing last-use group order and assignment order exactly.

pub fn expert_order(ids: &[u32], experts: usize) -> Result<Vec<u32>, &'static str> {
    let mut last = vec![usize::MAX; experts];
    for (assignment, &expert) in ids.iter().enumerate() {
        *last
            .get_mut(expert as usize)
            .ok_or("Expert ID exceeds configured count")? = assignment;
    }
    let mut order = last
        .iter()
        .enumerate()
        .filter_map(|(e, &position)| (position != usize::MAX).then_some(e as u32))
        .collect::<Vec<_>>();
    order.sort_unstable_by_key(|&e| last[e as usize]);
    Ok(order)
}

pub fn assignment_groups(
    ids: &[u32],
    order: &[u32],
    experts: usize,
    capacity: usize,
) -> Result<Vec<Vec<usize>>, &'static str> {
    if capacity == 0 {
        return Err("Expert slot capacity must be positive");
    }
    let mut group_of = vec![usize::MAX; experts];
    for (position, &expert) in order.iter().enumerate() {
        let group = group_of
            .get_mut(expert as usize)
            .ok_or("Expert ID exceeds configured count")?;
        if *group != usize::MAX {
            return Err("Expert order contains a duplicate");
        }
        *group = position / capacity;
    }
    let mut groups = vec![Vec::new(); order.len().div_ceil(capacity)];
    for (assignment, &expert) in ids.iter().enumerate() {
        let &group = group_of
            .get(expert as usize)
            .ok_or("Expert ID exceeds configured count")?;
        groups
            .get_mut(group)
            .ok_or("Assignment missing from expert order")?
            .push(assignment);
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashSet};

    fn reference(ids: &[u32], capacity: usize) -> (Vec<u32>, Vec<Vec<usize>>) {
        let mut last = BTreeMap::new();
        for (position, &expert) in ids.iter().enumerate() {
            last.insert(expert, position);
        }
        let mut order = last.keys().copied().collect::<Vec<_>>();
        order.sort_by_key(|e| last[e]);
        let groups = order
            .chunks(capacity)
            .map(|experts| {
                let wanted: HashSet<_> = experts.iter().copied().collect();
                ids.iter()
                    .enumerate()
                    .filter(|(_, e)| wanted.contains(e))
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();
        (order, groups)
    }

    #[test]
    fn preserves_last_use_groups_and_router_order() {
        let mut state = 0x5f37_u64;
        for experts in [1usize, 9, 32, 257, 512] {
            for assignments in [0usize, 1, 10, 270, 6920, 10240] {
                let ids = (0..assignments)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((state >> 32) % experts as u64) as u32
                    })
                    .collect::<Vec<_>>();
                for capacity in [1usize, 8, 10, 32, 441, 512] {
                    let expected = reference(&ids, capacity);
                    let order = expert_order(&ids, experts).unwrap();
                    let groups = assignment_groups(&ids, &order, experts, capacity).unwrap();
                    assert_eq!(
                        (order, groups),
                        expected,
                        "experts={experts}, assignments={assignments}, capacity={capacity}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_ids_and_incomplete_group_maps() {
        assert!(expert_order(&[4], 4).is_err());
        assert!(assignment_groups(&[4], &[4], 4, 1).is_err());
        assert!(assignment_groups(&[1], &[0], 4, 1).is_err());
        assert!(assignment_groups(&[1], &[1, 1], 4, 1).is_err());
        assert!(assignment_groups(&[], &[], 4, 0).is_err());
    }
}
