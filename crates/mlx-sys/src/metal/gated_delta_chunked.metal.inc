// Chunked gated delta recurrence kernel for prefill (SIMD-accelerated).
// Processes BT tokens per chunk with 5-phase algorithm.
// Uses (32, DV_PER_TG, 1) threadgroups with simd_sum for Dk reduction.
//
// Native layout (same as per-step kernel):
//   q, k: [B, S, Hv, Dk]  (GQA-expanded, Hk == Hv by caller)
//   v:    [B, S, Hv, Dv]
//   g:    [B, S, Hv]       (log-space decay gate)
//   beta: [B, S, Hv]       (sigmoid already applied)
//   state_in:  [B, Hv, Dv, Dk]  (recurrent state)
//   state_out: [B, Hv, Dv, Dk]  (f32 to avoid bf16 drift across chunks)
//   y:    [B, S, Hv, Dv]
//
// Grid:  (32, Dv, B * Hv)
// Block: (32, DV_PER_TG, 1)
//
// Template params: InT, BT (chunk size=32), BK (=Dk), DV_PER_TG, Dv, Hv
//
// Each SIMD group of 32 threads splits BK=128 into n_per_t=4 elements each.
// DV_PER_TG V-columns share threadgroup memory for k_chunk, kk_dot, etc.
R"(
    auto n = thread_position_in_grid.z;              // b_idx * Hv + hv_idx
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto v_idx = (int)thread_position_in_grid.y;     // global V column
    auto simd_lane = (int)thread_position_in_threadgroup.x;   // 0..31
    auto v_col_in_tg = (int)thread_position_in_threadgroup.y; // 0..DV_PER_TG-1

    if (v_idx >= Dv) return;

    constexpr int n_per_t = BK / 32;   // 4 for BK=128
    int linear_tid = simd_lane * DV_PER_TG + v_col_in_tg;  // 0..127 for cooperative loads
    int num_threads = 32 * DV_PER_TG;  // 128

    int num_chunks = (S + BT - 1) / BT;

    // --- Stride helpers for native [B, S, Hv, D] layout ---
    int qk_batch_stride = S * Hv * BK;
    int qk_time_stride  = Hv * BK;
    int qk_head_off     = hv_idx * BK;
    auto q_base = q + b_idx * qk_batch_stride + qk_head_off;
    auto k_base = k + b_idx * qk_batch_stride + qk_head_off;

    int v_batch_stride = S * Hv * Dv;
    int v_time_stride  = Hv * Dv;
    int v_head_off     = hv_idx * Dv;
    auto v_base = v + b_idx * v_batch_stride + v_head_off;
    auto y_base = y + b_idx * v_batch_stride + v_head_off;

    int g_batch_stride = S * Hv;
    int g_time_stride  = Hv;
    auto g_base    = g    + b_idx * g_batch_stride + hv_idx;
    auto beta_base = beta + b_idx * g_batch_stride + hv_idx;

    // state element at [b, h, v, d]: state[b*Hv*Dv*BK + h*Dv*BK + v*BK + d]
    int state_bh_off = (b_idx * Hv + hv_idx) * Dv * BK;
    auto si_base = state_in  + state_bh_off + v_idx * BK;
    auto so_base = state_out + state_bh_off + v_idx * BK;

    // --- Threadgroup memory ---
    threadgroup float k_chunk[BT * BK];       // 32*128 = 4096 floats = 16 KB
    threadgroup float kk_dot[BT * BT];        // 32*32  = 1024 floats = 4 KB
    threadgroup float gcum[BT];               // 32 floats
    threadgroup float beta_s[BT];             // 32 floats
    threadgroup float q_buf[BK];              // 128 floats

    // --- Load state into registers: 4 elements per thread ---
    float s[n_per_t];
    for (int i = 0; i < n_per_t; i++) {
        s[i] = static_cast<float>(si_base[simd_lane * n_per_t + i]);
    }

    float delta_arr[BT];

    for (int c = 0; c < num_chunks; c++) {
        int chunk_start = c * BT;
        int chunk_len   = min(BT, S - chunk_start);

        // ========== Phase 1: Cooperative load k, beta, g ==========
        // 128 threads cooperatively load chunk_len * BK elements
        for (int j = linear_tid; j < chunk_len * BK; j += num_threads) {
            int t = j / BK;
            int d = j % BK;
            k_chunk[j] = static_cast<float>(k_base[(chunk_start + t) * qk_time_stride + d]);
        }
        // Load beta and gcum — only first chunk_len threads
        if (linear_tid < chunk_len) {
            beta_s[linear_tid] = static_cast<float>(beta_base[(chunk_start + linear_tid) * g_time_stride]);
            gcum[linear_tid]   = static_cast<float>(g_base[(chunk_start + linear_tid) * g_time_stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ========== Phase 1b: Hillis-Steele prefix sum of g ==========
        for (int stride = 1; stride < BT; stride <<= 1) {
            float prev = 0.0f;
            if (linear_tid < chunk_len && linear_tid >= stride)
                prev = gcum[linear_tid - stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (linear_tid < chunk_len && linear_tid >= stride)
                gcum[linear_tid] += prev;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ========== Phase 2: kk_dot lower-triangular (SIMD-accelerated) ==========
        // Each of DV_PER_TG v-columns handles different (i,j) pairs
        for (int i = 0; i < chunk_len; i++) {
            for (int j = v_col_in_tg; j < i; j += DV_PER_TG) {
                float dot = 0.0f;
                for (int p = 0; p < n_per_t; p++) {
                    int d = simd_lane * n_per_t + p;
                    dot = fma(k_chunk[i * BK + d], k_chunk[j * BK + d], dot);
                }
                dot = simd_sum(dot);
                if (simd_lane == 0) kk_dot[i * BT + j] = dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ========== Phase 3: Forward substitution (SIMD reduction for kv_mem) ==========
        // Sequential over i (data dependency), but Dk dot uses simd_sum
        for (int i = 0; i < chunk_len; i++) {
            float v_i     = static_cast<float>(v_base[(chunk_start + i) * v_time_stride + v_idx]);
            float decay_i = exp(gcum[i]);
            float beta_i  = beta_s[i];

            // kv_mem = sum_d(s[d] * decay_i * k[i,d]) — SIMD reduced
            float kv_mem = 0.0f;
            for (int p = 0; p < n_per_t; p++) {
                int d = simd_lane * n_per_t + p;
                kv_mem = fma(s[p] * decay_i, k_chunk[i * BK + d], kv_mem);
            }
            kv_mem = simd_sum(kv_mem);  // broadcast to all 32 lanes

            float rhs = beta_i * (v_i - kv_mem);
            for (int j = 0; j < i; j++) {
                float a_ij = beta_i * kk_dot[i * BT + j] * exp(gcum[i] - gcum[j]);
                rhs -= a_ij * delta_arr[j];
            }
            delta_arr[i] = rhs;
        }

        // ========== Phase 4: Output (SIMD reduction for q@state and q@k) ==========
        for (int i = 0; i < chunk_len; i++) {
            // Cooperative load q[chunk_start+i, :] into shared memory
            auto q_t = q_base + (chunk_start + i) * qk_time_stride;
            if (linear_tid < BK) {
                q_buf[linear_tid] = static_cast<float>(q_t[linear_tid]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float decay_i = exp(gcum[i]);

            // Inter-chunk: q @ (state * decay) — SIMD reduced
            float o_val = 0.0f;
            for (int p = 0; p < n_per_t; p++) {
                int d = simd_lane * n_per_t + p;
                o_val = fma(q_buf[d], s[p] * decay_i, o_val);
            }
            o_val = simd_sum(o_val);  // broadcast to all 32 lanes

            // Intra-chunk: sum_j(qk_dot * delta * decay) — SIMD reduced per j
            for (int j = 0; j <= i; j++) {
                float qk_dot = 0.0f;
                for (int p = 0; p < n_per_t; p++) {
                    int d = simd_lane * n_per_t + p;
                    qk_dot = fma(q_buf[d], k_chunk[j * BK + d], qk_dot);
                }
                qk_dot = simd_sum(qk_dot);  // broadcast
                o_val += qk_dot * delta_arr[j] * exp(gcum[i] - gcum[j]);
            }

            // Only lane 0 writes output
            if (simd_lane == 0) {
                y_base[(chunk_start + i) * v_time_stride + v_idx] = static_cast<InT>(o_val);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ========== Phase 5: State update (direct register update, no reduction) ==========
        float g_total = gcum[chunk_len - 1];
        for (int p = 0; p < n_per_t; p++) {
            int d = simd_lane * n_per_t + p;
            float s_new = s[p] * exp(g_total);
            for (int t = 0; t < chunk_len; t++) {
                s_new += k_chunk[t * BK + d] * delta_arr[t] * exp(g_total - gcum[t]);
            }
            s[p] = s_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Write final state (f32 output to avoid drift) ---
    for (int p = 0; p < n_per_t; p++) {
        so_base[simd_lane * n_per_t + p] = s[p];
    }
)"
