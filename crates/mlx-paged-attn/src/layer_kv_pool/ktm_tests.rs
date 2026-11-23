use super::*;

const H: usize = 4;
const D: usize = 256;
const BS: usize = 32;
const DATA: usize = H * D * BS;
const SCALES: usize = H * BS * 4;

fn config(num_layers: u32, block_size: u32) -> PagedAttentionConfig {
    PagedAttentionConfig {
        block_size,
        gpu_memory_mb: 256,
        head_size: D as u32,
        num_kv_heads: H as u32,
        num_layers,
        max_seq_len: Some(block_size * 64),
        max_batch_size: Some(1),
    }
}

fn keys_token_major() -> PageLayout {
    PageLayout {
        keys_token_major: true,
        ..PageLayout::default()
    }
}

fn metal() -> bool {
    crate::metal::MetalState::get().is_ok()
}

/// One block's host blob of one side: every byte differs per block, layer
/// and side.
fn blob(block: usize, layer: usize, side: usize) -> Vec<u8> {
    (0..DATA + SCALES)
        .map(|i| ((i * 7 + block * 131 + layer * 61 + side * 29) % 251) as u8)
        .collect()
}

fn blobs(block: usize, layers: usize) -> Vec<BlockLayerBytes> {
    (0..layers)
        .map(|layer| (blob(block, layer, 0), blob(block, layer, 1)))
        .collect()
}

fn upload(block: u32, payload: &[BlockLayerBytes]) -> BlockUpload<'_> {
    (
        block,
        payload
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect(),
    )
}

fn shape_and_dtype(raw: *mut mlx_sys::mlx_array) -> (Vec<i64>, i32) {
    assert!(!raw.is_null());
    unsafe {
        let ndim = mlx_sys::mlx_array_ndim(raw);
        let mut dims = vec![0i64; ndim];
        mlx_sys::mlx_array_shape(raw, dims.as_mut_ptr());
        let dtype = mlx_sys::mlx_array_dtype(raw);
        mlx_sys::mlx_array_delete(raw);
        (dims, dtype)
    }
}

/// The keys-token-major page: plane bytes equal to the 32-token vLLM page,
/// K viewed `[nb, H, 32, D]`, V and scales as the vLLM page; 16 layers of
/// blocks cross the host whole (write, read, single write, move, grow) on
/// the shared and the private path; the raw host paths and raw kernels
/// refuse it. Mutation: `require_vllm_page` accepts every layout -> red at
/// the first raw-path assert.
#[test]
fn keys_token_major_pages_cross_the_host_and_refuse_raw_paths() {
    if !metal() {
        return;
    }
    assert!(PageLayout::default().is_vllm());
    assert!(!keys_token_major().is_vllm());
    for shared in [true, false] {
        let pool = LayerKVPool::new_inner_with_layout(
            config(16, BS as u32),
            4,
            8,
            MetalDtype::Int8,
            keys_token_major(),
            MetalDtype::BFloat16,
            shared,
        )
        .unwrap();
        let vllm = LayerKVPool::new(config(16, BS as u32), 4, 8, MetalDtype::Int8).unwrap();
        assert_eq!(pool.layout(), keys_token_major());
        assert_eq!(pool.plane_bytes(), vllm.plane_bytes());
        assert_eq!(
            pool.block_blob_bytes(),
            Ok(((DATA + SCALES) as u64, (DATA + SCALES) as u64))
        );
        assert_eq!(pool.bytes_per_block(), vllm.bytes_per_block());
        for (raw, shape, code) in [
            (pool.key_cache_array_raw(3).unwrap(), vec![4, 4, 32, 256], 6),
            (
                pool.value_cache_array_raw(3).unwrap(),
                vec![4, 4, 256, 32],
                6,
            ),
            (pool.key_scale_array_raw(3).unwrap(), vec![4, 4, 32], 0),
            (pool.value_scale_array_raw(3).unwrap(), vec![4, 4, 32], 0),
        ] {
            assert_eq!(shape_and_dtype(raw), (shape, code));
        }

        let payloads: Vec<Vec<BlockLayerBytes>> = (0..4).map(|block| blobs(block, 16)).collect();
        let uploads: Vec<BlockUpload<'_>> = payloads
            .iter()
            .enumerate()
            .map(|(block, payload)| upload(block as u32, payload))
            .collect();
        pool.write_blocks_all_layers(&uploads).unwrap();
        for (block, payload) in payloads.iter().enumerate() {
            assert_eq!(
                &pool.read_block_all_layers(block as u32).unwrap(),
                payload,
                "block {block} shared={shared}"
            );
        }
        let single = blobs(9, 16);
        pool.write_block_all_layers(
            3,
            &single
                .iter()
                .map(|(k, v)| (k.as_slice(), v.as_slice()))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(pool.read_block_all_layers(3).unwrap(), single);
        pool.move_blocks(&[(2, 0)]).unwrap();
        assert_eq!(pool.read_block_all_layers(0).unwrap(), payloads[2]);
        assert!(pool.grow_to(8).unwrap());
        assert_eq!(pool.read_block_all_layers(1).unwrap(), payloads[1]);
        assert_eq!(pool.read_block_all_layers(3).unwrap(), single);
        assert_eq!(
            shape_and_dtype(pool.key_cache_array_raw(0).unwrap()).0,
            [8, 4, 32, 256]
        );

        let read = pool.read_blocks_to_host(0, &[0]);
        assert!(
            read.as_ref()
                .is_err_and(|error| error.contains("has no host blob")),
            "read_blocks_to_host: {read:?}"
        );
        let (k, v) = &payloads[0][0];
        assert!(pool.write_blocks_from_host(0, &[0], k, v).is_err());
    }
}

/// A pool refuses the keys-token-major order beside token-major values, an
/// index plane, a bf16 cache or a 16-token block. Mutation: the pool takes
/// every combination -> red.
#[test]
fn keys_token_major_pages_are_int8_32_token_pages_only() {
    if !metal() {
        return;
    }
    let refused = |layout: PageLayout, dtype: MetalDtype, block_size: u32| {
        LayerKVPool::new_with_layout(config(2, block_size), 2, 2, dtype, layout)
            .is_err_and(|error| error.contains("keys-token-major"))
    };
    let with_values = PageLayout {
        values_token_major: true,
        ..keys_token_major()
    };
    let with_index = PageLayout {
        index_rows: 4,
        index_dimension: 128,
        ..keys_token_major()
    };
    assert!(
        refused(with_values, MetalDtype::Int8, 32),
        "values_token_major"
    );
    assert!(refused(with_index, MetalDtype::Int8, 32), "index plane");
    assert!(
        refused(keys_token_major(), MetalDtype::BFloat16, 32),
        "bf16"
    );
    assert!(
        refused(keys_token_major(), MetalDtype::Int8, 16),
        "block 16"
    );
    assert!(
        LayerKVPool::new_with_layout(config(2, 32), 2, 2, MetalDtype::Int8, keys_token_major())
            .is_ok()
    );
}

/// A pool built without a layout keeps the vLLM page and views at block
/// 32 and 16. Mutation: `LayerKVPool::new` takes the keys-token-major
/// layout -> red.
#[test]
fn default_pools_keep_the_vllm_page() {
    if !metal() {
        return;
    }
    let pool = LayerKVPool::new(config(2, 32), 3, 3, MetalDtype::Int8).unwrap();
    assert!(pool.layout().is_vllm(), "{:?}", pool.layout());
    for (block_size, dtype, x) in [
        (32u32, MetalDtype::Int8, 16i64),
        (16, MetalDtype::Int8, 16),
        (16, MetalDtype::BFloat16, 8),
    ] {
        let pool = LayerKVPool::new(config(2, block_size), 3, 3, dtype).unwrap();
        let bs = i64::from(block_size);
        assert_eq!(
            shape_and_dtype(pool.key_cache_array_raw(1).unwrap()).0,
            [3, 4, 256 / x, bs, x]
        );
        assert_eq!(
            shape_and_dtype(pool.value_cache_array_raw(1).unwrap()).0,
            [3, 4, 256, bs]
        );
    }
}
