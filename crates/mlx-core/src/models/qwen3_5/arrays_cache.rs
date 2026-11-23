pub use crate::transformer::arrays_cache::ArraysCache;
pub(crate) use crate::transformer::arrays_cache::forward_rows_independently;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::MxArray;

    #[test]
    fn kv_cache_and_arrays_cache_broadcast_batch() {
        use super::super::attention::Qwen3_5Attention;
        use super::super::config::Qwen3_5Config;
        use super::super::gated_delta_net::GatedDeltaNet;
        use crate::array::DType;
        use crate::transformer::KVCache;

        let cfg = Qwen3_5Config {
            rope_scaling: None,
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
            vocab_size: 32,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 128,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            partial_rotary_factor: 0.5,
            rope_theta: 10_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            n_mtp_layers: 0,
        };

        let mut attn = Qwen3_5Attention::new(&cfg).unwrap();
        let mut gdn = GatedDeltaNet::new(&cfg).unwrap();

        // Deterministic weights so parity assertions are reproducible across runs.
        let det = |shape: &[i64], salt: f32, scale: f32| -> MxArray {
            let n: usize = shape.iter().map(|&d| d as usize).product();
            let vals: Vec<f32> = (0..n)
                .map(|i| ((i as f32) * 0.3719 + salt).sin() * scale)
                .collect();
            MxArray::from_float32(&vals, shape).unwrap()
        };
        let hidden = cfg.hidden_size as i64;
        let h = cfg.num_heads as i64;
        let kv = cfg.num_kv_heads as i64;
        let d = cfg.head_dim as i64;
        attn.set_q_proj_weight(&det(&[2 * h * d, hidden], 1.0, 0.12))
            .unwrap();
        attn.set_k_proj_weight(&det(&[kv * d, hidden], 2.0, 0.12))
            .unwrap();
        attn.set_v_proj_weight(&det(&[kv * d, hidden], 3.0, 0.12))
            .unwrap();
        attn.set_o_proj_weight(&det(&[hidden, h * d], 4.0, 0.12))
            .unwrap();

        let key_dim = cfg.linear_key_dim() as i64;
        let val_dim = cfg.linear_value_dim() as i64;
        let conv_dim = cfg.linear_conv_dim() as i64;
        let hv = cfg.linear_num_value_heads as i64;
        gdn.set_in_proj_qkvz_weight(&det(&[2 * key_dim + 2 * val_dim, hidden], 5.0, 0.12))
            .unwrap();
        gdn.set_in_proj_ba_weight(&det(&[2 * hv, hidden], 6.0, 0.12))
            .unwrap();
        gdn.set_conv1d_weight(&det(&[conv_dim, 4, 1], 7.0, 0.2), DType::Float32)
            .unwrap();
        gdn.set_out_proj_weight(&det(&[hidden, val_dim], 8.0, 0.12))
            .unwrap();

        // Shared prefix (B=1, T=5)
        let prefix_vals: Vec<f32> = (0..(5 * 64))
            .map(|i| ((i as f32) * 0.037).sin() * 0.25)
            .collect();
        let prefix = MxArray::from_float32(&prefix_vals, &[1, 5, 64]).unwrap();

        let mut kv_prefix = KVCache::new();
        let mut gdn_prefix = ArraysCache::new(2);
        let _ = attn
            .forward(&prefix, None, Some(&mut kv_prefix), None)
            .unwrap();
        let _ = gdn
            .forward(&prefix, None, Some(&mut gdn_prefix), false)
            .unwrap();

        // Validation: batch < 1 fails
        assert!(kv_prefix.broadcast_batch(0).is_err());
        assert!(gdn_prefix.broadcast_batch(0).is_err());

        // Broadcast B=1 -> B=3
        let mut kv_batched = kv_prefix.broadcast_batch(3).unwrap();
        let mut gdn_batched = gdn_prefix.broadcast_batch(3).unwrap();
        assert_eq!(kv_batched.get_offset(), 5);
        assert_eq!(
            kv_batched.keys_ref().unwrap().shape().unwrap().as_ref(),
            [3, 2, 5, 16]
        );
        assert_eq!(
            gdn_batched.get(0).unwrap().shape().unwrap().as_ref(),
            [3, 3, conv_dim]
        );
        assert_eq!(
            gdn_batched.get(1).unwrap().shape().unwrap().as_ref(),
            [3, 4, 16, 16]
        );

        // Broadcasted rows must be bit-identical to the B=1 prefix before divergence.
        let prefix_k = kv_prefix
            .keys_ref()
            .unwrap()
            .slice_axis(2, 0, 5)
            .unwrap()
            .to_float32()
            .unwrap();
        let prefix_conv = gdn_prefix.get(0).unwrap().to_float32().unwrap();
        let prefix_rec = gdn_prefix.get(1).unwrap().to_float32().unwrap();
        for b in 0..3i64 {
            let row_k = kv_batched
                .keys_ref()
                .unwrap()
                .slice_axis(0, b, b + 1)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(row_k.as_ref(), prefix_k.as_ref());
            let row_conv = gdn_batched
                .get(0)
                .unwrap()
                .slice_axis(0, b, b + 1)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(row_conv.as_ref(), prefix_conv.as_ref());
            let row_rec = gdn_batched
                .get(1)
                .unwrap()
                .slice_axis(0, b, b + 1)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(row_rec.as_ref(), prefix_rec.as_ref());
        }

        // Validation: broadcasting a B=3 cache fails
        assert!(kv_batched.broadcast_batch(2).is_err());
        assert!(gdn_batched.broadcast_batch(2).is_err());

        // Independent B=1 clones to diverge separately
        let mut kv_singles: Vec<KVCache> = (0..3)
            .map(|_| kv_prefix.broadcast_batch(1).unwrap())
            .collect();
        let mut gdn_singles: Vec<ArraysCache> = (0..3)
            .map(|_| gdn_prefix.broadcast_batch(1).unwrap())
            .collect();

        // Run 2 distinct decode steps (T=1 each) and assert parity across every row
        for step in 0..2 {
            let mut step_rows = Vec::with_capacity(3);
            for b in 0..3 {
                let data: Vec<f32> = (0..64)
                    .map(|i| (((b * 97 + step * 31 + i) as f32) * 0.11).cos() * 0.3)
                    .collect();
                step_rows.push(MxArray::from_float32(&data, &[1, 1, 64]).unwrap());
            }
            let x_batch = MxArray::concatenate_many(step_rows.iter().collect(), Some(0)).unwrap();

            let attn_batch_out = attn
                .forward(&x_batch, None, Some(&mut kv_batched), None)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            let gdn_batch_out = gdn
                .forward(&x_batch, None, Some(&mut gdn_batched), false)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();

            for b in 0..3 {
                let attn_single = attn
                    .forward(&step_rows[b], None, Some(&mut kv_singles[b]), None)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let gdn_single = gdn
                    .forward(&step_rows[b], None, Some(&mut gdn_singles[b]), false)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();

                let offset = b * 64;
                let attn_diff = attn_single
                    .as_ref()
                    .iter()
                    .zip(&attn_batch_out.as_ref()[offset..offset + 64])
                    .map(|(a, c)| (a - c).abs())
                    .fold(0.0f32, f32::max);
                let gdn_diff = gdn_single
                    .as_ref()
                    .iter()
                    .zip(&gdn_batch_out.as_ref()[offset..offset + 64])
                    .map(|(a, c)| (a - c).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    attn_diff < 1e-4,
                    "KVCache broadcast_batch mismatch at step={step}, row={b}: max_diff={attn_diff}"
                );
                assert!(
                    gdn_diff < 1e-4,
                    "ArraysCache broadcast_batch mismatch at step={step}, row={b}: max_diff={gdn_diff}"
                );
            }
        }

        // Original B=1 prefix caches remain unmutated
        assert_eq!(kv_prefix.get_offset(), 5);
        assert_eq!(gdn_prefix.get(0).unwrap().shape_at(0).unwrap(), 1);
    }
}
