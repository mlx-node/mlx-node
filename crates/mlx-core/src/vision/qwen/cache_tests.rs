use super::*;

fn digest(word: u64) -> engine::ImageCacheDigest {
    engine::ImageCacheDigest::from_test_word(word)
}

fn key(image_hash: u64, grid_thw: [i32; 3]) -> VisionFeatureCacheKey {
    VisionFeatureCacheKey {
        image_hash: digest(image_hash),
        grid_thw,
    }
}

fn request(key: VisionFeatureCacheKey, patch_start: i64, patch_count: i64) -> VisionImageRequest {
    VisionImageRequest {
        key,
        patch_start,
        patch_count,
        feature_count: patch_count,
    }
}

fn feature(value: f32) -> MxArray {
    let array = MxArray::from_float32(&[value], &[1]).unwrap();
    array.eval();
    array
}

fn snapshot(
    total: u64,
    mlx_limit: u64,
    metal_limit: u64,
    active: u64,
    metal_current: u64,
    allocator_cache: u64,
) -> VisionMemorySnapshot {
    VisionMemorySnapshot {
        total_system_memory_bytes: total,
        mlx_memory_limit_bytes: mlx_limit,
        metal_working_set_bytes: metal_limit,
        allocator_active_bytes: active,
        allocator_active_probe_ok: active > 0,
        metal_current_allocated_bytes: metal_current,
        metal_current_probe_ok: metal_current > 0,
        allocator_cache_bytes: allocator_cache,
    }
}

fn miss(key: VisionFeatureCacheKey, patch_count: i64) -> VisionCacheMiss {
    VisionCacheMiss {
        request: request(key, 0, patch_count),
        request_indices: vec![0],
    }
}

#[test]
fn plans_per_image_ranges_and_rejects_invalid_geometry() {
    let planned = plan_vision_image_requests(&[1, 4, 4, 2, 2, 4], &[digest(10), digest(20)], 32, 2)
        .expect("valid image geometry");
    assert_eq!(planned[0].patch_start, 0);
    assert_eq!(planned[0].patch_count, 16);
    assert_eq!(planned[0].feature_count, 4);
    assert_eq!(planned[1].patch_start, 16);
    assert_eq!(planned[1].patch_count, 16);
    assert_eq!(planned[1].feature_count, 4);

    assert!(plan_vision_image_requests(&[1, 4, 4], &[digest(1), digest(2)], 16, 2).is_err());
    assert!(plan_vision_image_requests(&[1, 3, 4], &[digest(1)], 12, 2).is_err());
    assert!(plan_vision_image_requests(&[1, 4, 4], &[digest(1)], 15, 2).is_err());
    assert!(plan_vision_image_requests(&[0, 4, 4], &[digest(1)], 0, 2).is_err());
}

#[test]
fn partitions_at_boundary_over_boundary_and_single_oversize_image() {
    let a = key(1, [1, 1, 1]);
    let b = key(2, [1, 1, 1]);
    let exact = vec![miss(a, 16_000), miss(b, 16_768)];
    assert_eq!(partition_vision_cache_misses(&exact, 32_768), vec![0..2]);

    let over = vec![miss(a, 16_000), miss(b, 16_769)];
    assert_eq!(
        partition_vision_cache_misses(&over, 32_768),
        vec![0..1, 1..2]
    );

    let oversize = vec![miss(a, 40_000)];
    assert_eq!(partition_vision_cache_misses(&oversize, 32_768), vec![0..1]);
}

#[test]
fn cache_reuses_appended_images_preserves_order_and_deduplicates_misses() {
    let a = key(10, [1, 2, 2]);
    let b = key(20, [1, 2, 2]);
    let c = key(30, [1, 2, 2]);
    let mut cache = VisionCacheInner::new();
    let no_protection = HashSet::new();
    cache.insert(a, feature(1.0), 4, &no_protection, usize::MAX);
    cache.insert(b, feature(2.0), 4, &no_protection, usize::MAX);

    let appended = [request(a, 0, 4), request(b, 4, 4), request(c, 8, 4)];
    let (_, misses, hits) = lookup_vision_feature_cache(&mut cache, &appended);
    assert_eq!(hits, 2);
    assert_eq!(misses.len(), 1);
    assert_eq!(misses[0].request.key, c);

    let reordered = [request(b, 0, 4), request(a, 4, 4)];
    let (ordered, misses, hits) = lookup_vision_feature_cache(&mut cache, &reordered);
    assert_eq!(hits, 2);
    assert!(misses.is_empty());
    assert_eq!(
        ordered[0].as_ref().unwrap().to_float32().unwrap().to_vec(),
        [2.0]
    );
    assert_eq!(
        ordered[1].as_ref().unwrap().to_float32().unwrap().to_vec(),
        [1.0]
    );

    let mut duplicate_cache = VisionCacheInner::new();
    duplicate_cache.insert(b, feature(2.0), 4, &no_protection, usize::MAX);
    let duplicated = [request(a, 0, 4), request(a, 4, 4), request(b, 8, 4)];
    let (_, misses, hits) = lookup_vision_feature_cache(&mut duplicate_cache, &duplicated);
    assert_eq!(hits, 1);
    assert_eq!(misses.len(), 1, "duplicate A should be encoded once");
    assert_eq!(misses[0].request_indices, vec![0, 1]);
}

#[test]
fn cache_key_includes_grid_for_identical_content_hash() {
    let requests = [
        request(key(42, [1, 2, 2]), 0, 4),
        request(key(42, [1, 4, 4]), 4, 16),
    ];
    let mut cache = VisionCacheInner::new();
    let (_, misses, hits) = lookup_vision_feature_cache(&mut cache, &requests);
    assert_eq!(hits, 0);
    assert_eq!(misses.len(), 2);
    assert_ne!(misses[0].request.key, misses[1].request.key);
}

#[test]
fn mixed_hits_and_duplicate_miss_fill_exact_request_order() {
    let a = key(10, [1, 1, 1]);
    let b = key(20, [1, 1, 1]);
    let c = key(30, [1, 1, 1]);
    let mut cache = VisionCacheInner::new();
    let no_protection = HashSet::new();
    cache.insert(b, feature(2.0), 4, &no_protection, usize::MAX);
    cache.insert(c, feature(3.0), 4, &no_protection, usize::MAX);

    let requests = [
        request(b, 0, 1),
        request(a, 1, 1),
        request(a, 2, 1),
        request(c, 3, 1),
    ];
    let (mut ordered, misses, hits) = lookup_vision_feature_cache(&mut cache, &requests);
    assert_eq!(hits, 2);
    assert_eq!(misses.len(), 1);
    assert_eq!(misses[0].request_indices, vec![1, 2]);
    let encoded_a = feature(1.0);
    for &request_index in &misses[0].request_indices {
        ordered[request_index] = Some(encoded_a.clone());
    }
    let ordered: Vec<MxArray> = ordered.into_iter().map(Option::unwrap).collect();
    let refs: Vec<&MxArray> = ordered.iter().collect();
    let concatenated = MxArray::concatenate_many(refs, Some(0)).unwrap();
    assert_eq!(
        concatenated.to_float32().unwrap().to_vec(),
        [2.0, 1.0, 1.0, 3.0]
    );
}

#[test]
fn duplicate_images_count_once_for_cache_floor_but_once_per_request_for_concat() {
    let a = key(10, [1, 1, 1]);
    let b = key(20, [1, 1, 1]);
    let requests = [request(a, 0, 1), request(a, 1, 1), request(b, 2, 1)];
    let (request_bytes, protected_bytes) = projected_vision_feature_bytes(&requests, 8, 2);
    assert_eq!(request_bytes, 48, "concat contains A, A, and B");
    assert_eq!(protected_bytes, 32, "cache stores only unique A and B");
}

#[test]
fn replacing_cache_key_updates_retained_byte_accounting() {
    let a = key(1, [1, 1, 1]);
    let no_protection = HashSet::new();
    let mut cache = VisionCacheInner::new();
    cache.insert(a, feature(1.0), 4, &no_protection, usize::MAX);
    cache.insert(a, feature(2.0), 12, &no_protection, usize::MAX);
    assert_eq!(cache.entries.len(), 1);
    assert_eq!(cache.retained_bytes, 12);
    assert_eq!(cache.get(&a).unwrap().to_float32().unwrap().to_vec(), [2.0]);
}

#[test]
fn later_all_hit_subset_prunes_inactive_oversized_floor() {
    let a = key(1, [1, 1, 1]);
    let b = key(2, [1, 1, 1]);
    let c = key(3, [1, 1, 1]);
    let all_protected: HashSet<_> = [a, b, c].into_iter().collect();
    let mut cache = VisionCacheInner::new();
    cache.insert(a, feature(1.0), 4, &all_protected, 4);
    cache.insert(b, feature(2.0), 4, &all_protected, 4);
    cache.insert(c, feature(3.0), 4, &all_protected, 4);
    assert_eq!(cache.entries.len(), 3, "active request is the floor");
    assert_eq!(cache.retained_bytes, 12);

    let current = [request(a, 0, 1)];
    let (_, misses, hits) = lookup_vision_feature_cache(&mut cache, &current);
    assert_eq!(hits, 1);
    assert!(misses.is_empty());
    let current_protected: HashSet<_> = [a].into_iter().collect();
    let eviction = cache.evict_to_limits(&current_protected, 1, 4);
    assert_eq!(eviction.entries, 2);
    assert_eq!(eviction.bytes, 8);
    assert_eq!(cache.entries.len(), 1);
    assert_eq!(cache.retained_bytes, 4);
    assert!(cache.entries.contains_key(&a));
}

#[test]
fn memory_budget_scales_across_16_32_and_128_gib_caps() {
    let b16 = resolve_vision_memory_budget(
        snapshot(16 * VISION_GIB, 0, 0, 8 * VISION_GIB, 8 * VISION_GIB, 0),
        0,
        0,
        1152,
        4304,
        4,
        3_072,
        0,
    );
    let b32 = resolve_vision_memory_budget(
        snapshot(32 * VISION_GIB, 0, 0, 10 * VISION_GIB, 10 * VISION_GIB, 0),
        0,
        0,
        1152,
        4304,
        4,
        3_072,
        0,
    );
    let b128 = resolve_vision_memory_budget(
        snapshot(128 * VISION_GIB, 0, 0, 40 * VISION_GIB, 40 * VISION_GIB, 0),
        0,
        0,
        1152,
        4304,
        4,
        3_072,
        0,
    );
    assert!(b16.effective_cap_bytes < b32.effective_cap_bytes);
    assert!(b32.effective_cap_bytes < b128.effective_cap_bytes);
    assert!(b16.miss_batch_patch_budget < b32.miss_batch_patch_budget);
    assert!(b32.miss_batch_patch_budget <= b128.miss_batch_patch_budget);
    assert_eq!(b128.miss_batch_patch_budget, 32_768);
    assert!(b16.cache_budget_bytes <= b32.cache_budget_bytes);
    assert!(b32.cache_budget_bytes <= b128.cache_budget_bytes);
    assert_eq!(b128.cache_budget_bytes, VISION_CACHE_MAX_BYTES);
}

#[test]
fn memory_budget_uses_lower_haircut_cap_and_external_metal_usage() {
    let budget = resolve_vision_memory_budget(
        snapshot(
            128 * VISION_GIB,
            64 * VISION_GIB,
            96 * VISION_GIB,
            10 * VISION_GIB,
            18 * VISION_GIB,
            VISION_GIB,
        ),
        0,
        0,
        1152,
        4304,
        2,
        1_536,
        0,
    );
    assert_eq!(budget.effective_cap_bytes, 64 * VISION_GIB);
    assert_eq!(budget.cap_source, VisionMemoryCapSource::MlxMemoryLimit);
    assert_eq!(budget.metal_nonreclaimable_bytes, 17 * VISION_GIB);
    assert_eq!(budget.used_memory_bytes, 17 * VISION_GIB);
}

#[test]
fn memory_budget_fails_closed_for_missing_usage_and_near_cap() {
    let missing =
        resolve_vision_memory_budget(snapshot(0, 0, 0, 0, 0, 0), 0, 0, 1152, 4304, 4, 3_072, 0);
    assert_eq!(
        missing.cap_source,
        VisionMemoryCapSource::ConservativeFallback
    );
    assert!(!missing.usage_probe_available);
    assert_eq!(missing.headroom_bytes, 0);
    assert_eq!(missing.cache_budget_bytes, 0);
    assert_eq!(missing.miss_batch_patch_budget, 0);

    let near_cap = resolve_vision_memory_budget(
        snapshot(0, 16 * VISION_GIB, 0, 15 * VISION_GIB, 15 * VISION_GIB, 0),
        0,
        0,
        1152,
        4304,
        4,
        3_072,
        0,
    );
    assert!(near_cap.usage_probe_available);
    assert_eq!(near_cap.headroom_bytes, 0);
    assert_eq!(near_cap.miss_batch_patch_budget, 0);
}

#[test]
fn memory_budget_rejects_outputs_larger_than_post_reserve_headroom() {
    let budget = resolve_vision_memory_budget(
        snapshot(0, 16 * VISION_GIB, 0, 12 * VISION_GIB, 12 * VISION_GIB, 0),
        0,
        0,
        1152,
        4304,
        4,
        3_072,
        3 * VISION_GIB,
    );
    assert_eq!(budget.output_headroom_bytes, 2 * VISION_GIB);
    assert!(!budget.projected_output_fits);
    assert_eq!(budget.headroom_bytes, 0);
    assert_eq!(budget.miss_batch_patch_budget, 0);
}

#[test]
fn memory_budget_saturates_extreme_geometry_without_overflow() {
    let budget = resolve_vision_memory_budget(
        snapshot(u64::MAX, u64::MAX, u64::MAX, 1, 1, 0),
        0,
        0,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
    );
    assert_eq!(budget.peak_bytes_per_patch, u64::MAX);
    assert_eq!(budget.miss_batch_patch_budget, 0);
    assert!(budget.cache_budget_bytes <= VISION_CACHE_MAX_BYTES);
}
